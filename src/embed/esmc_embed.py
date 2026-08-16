"""ESM-C embedding extraction for the OpenCell taggability grid.

One script, two axes:  --model {300m,6b}  x  --seqtype {protein,construct}
Everything else is identical across passes so that cell-to-cell deltas are attributable
to the varied axis alone.

Caches POOLED features only (mean / terminal windows / junction window). Residue-level
tensors for the 6B model would be ~11 GB per pass; pooled vectors are ~100 MB.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

REPO = "/workspace/fluoresceAnything"
TABLE = f"{REPO}/data/modeling_table.csv"
CACHE_DIR = f"{REPO}/cache/emb"

MAX_LEN = 2046  # + <cls> + <eos> = 2048 = ESM-C max_position_embeddings
WINDOWS = [25, 50]  # terminal / junction window sizes to cache
SAVE_EVERY = 200


def middle_truncate(seq, junction=None):
    """Keep both termini (and therefore the junction) when a sequence is too long."""
    n = len(seq)
    if n <= MAX_LEN:
        return seq, junction
    half = MAX_LEN // 2
    out = seq[:half] + seq[-half:]
    if junction is None:
        return out, None
    if junction <= half:
        return out, junction
    if junction >= n - half:
        return out, junction - (n - MAX_LEN)
    raise ValueError(f"junction {junction} falls in the truncated middle of a {n} aa sequence")


def pool(residues, junction):
    """residues: [L, D] float32 on cpu. Returns dict of pooled vectors."""
    feats = {"mean": residues.mean(0)}
    for k in WINDOWS:
        feats[f"first_{k}"] = residues[:k].mean(0)
        feats[f"last_{k}"] = residues[-k:].mean(0)
        if junction is not None:
            lo, hi = max(0, junction - k), min(len(residues), junction + k)
            feats[f"junction_{k}"] = residues[lo:hi].mean(0)
    return {k: v.numpy().astype(np.float32) for k, v in feats.items()}


def load_cache(path):
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    names = [n for n in z.files if n != "keys"]
    return {key: {n: z[n][i] for n in names} for i, key in enumerate(keys)}


def save_cache(path, cache):
    keys = sorted(cache)
    names = sorted(cache[keys[0]])
    arrays = {n: np.stack([cache[k][n] for k in keys]) for n in names}
    tmp = path + ".tmp.npz"  # np.savez appends .npz unless the name already ends in it
    np.savez(tmp, keys=np.array(keys), **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ model backends
class Esmc300M:
    """Native evolutionaryscale SDK."""

    name = "300m"

    def __init__(self, device="cuda"):
        from esm.models.esmc import ESMC

        self.device = device
        self.model = ESMC.from_pretrained("esmc_300m").to(device).eval()
        self.tokenizer = self.model.tokenizer

    def describe(self):
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[300m] native esm SDK  params={n / 1e6:.1f}M  tokenizer={type(self.tokenizer).__name__}")

    def tokenize(self, seqs):
        ids = self.model._tokenize(seqs)
        mask = ids != self.tokenizer.pad_token_id
        return ids, mask

    @torch.no_grad()
    def forward(self, ids, mask):
        out = self.model(ids.to(self.device))
        return out.embeddings


class Esmc6B:
    """multimolecule HF repack."""

    name = "6b"
    hf_repo = "multimolecule/esmc-6b"

    def __init__(self, device="cuda"):
        from multimolecule.models.esmc import EsmCModel
        from transformers import AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo)
        self.model = (
            EsmCModel.from_pretrained(self.hf_repo, dtype=torch.bfloat16).to(device).eval()
        )

    def describe(self):
        c = self.model.config
        n = sum(p.numel() for p in self.model.parameters())
        print(
            f"[6b] {self.hf_repo}  params={n / 1e9:.2f}B  hidden={c.hidden_size} "
            f"layers={c.num_hidden_layers} heads={c.num_attention_heads} "
            f"max_pos={c.max_position_embeddings}  tokenizer={type(self.tokenizer).__name__}"
        )

    def tokenize(self, seqs):
        enc = self.tokenizer(seqs, return_tensors="pt", padding=True)
        return enc["input_ids"], enc["attention_mask"].bool()

    @torch.no_grad()
    def forward(self, ids, mask):
        out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        return out.last_hidden_state


def show_tokenization(backend, seqs):
    """Special-token placement is VERIFIED here, not assumed."""
    tok = backend.tokenizer
    for label, s in seqs:
        ids, _ = backend.tokenize([s])
        head = tok.convert_ids_to_tokens(ids[0][:5].tolist())
        tail = tok.convert_ids_to_tokens(ids[0][-3:].tolist())
        print(
            f"  [{backend.name}/{label}] len={len(s)} -> {tuple(ids.shape)}  "
            f"head={head} tail={tail}  residues={list(s[:3])}...{list(s[-2:])}"
        )
        assert ids.shape[1] == len(s) + 2, "expected exactly one leading and one trailing token"
    print(f"  [{backend.name}] verified layout: <cls> + residues + <eos>  ->  slice emb[1:1+L]")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["300m", "6b"], required=True)
    ap.add_argument("--seqtype", choices=["protein", "construct"], required=True)
    ap.add_argument("--batch-tokens", type=int, default=None)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="dry-run on the first N items")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    df = pd.read_csv(TABLE)

    if args.seqtype == "protein":
        items = df.drop_duplicates("ENSG")[["ENSG", "protein"]].copy()
        items["key"] = items["ENSG"]
        items["seq"] = items["protein"]
        items["junction"] = None
    else:
        items = df.drop_duplicates(["ENSG", "terminus"])[
            ["ENSG", "terminus", "construct", "junction_idx"]
        ].copy()
        items["key"] = items["ENSG"] + "_" + items["terminus"]
        items["seq"] = items["construct"]
        items["junction"] = items["junction_idx"]

    lengths = items["seq"].str.len()
    print(f"=== {args.model} / {args.seqtype}: {len(items)} unique sequences")
    print(
        f"    length: min={lengths.min()} median={int(lengths.median())} "
        f"p95={int(lengths.quantile(0.95))} max={lengths.max()}"
    )
    for m in [1022, MAX_LEN, 4096]:
        print(f"    > {m}: {(lengths > m).sum()}")
    print(f"    middle-truncating {(lengths > MAX_LEN).sum()} sequences to {MAX_LEN}")

    if args.limit:
        items = items.head(args.limit)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = f"{CACHE_DIR}/{args.model}_{args.seqtype}{args.out_suffix}.npz"
    cache = load_cache(out_path)
    print(f"    cache {out_path}: {len(cache)} already embedded")

    todo = items[~items["key"].isin(cache)].copy()
    if todo.empty:
        print("    nothing to do")
        return

    backend = Esmc300M() if args.model == "300m" else Esmc6B()
    backend.describe()
    show_tokenization(
        backend,
        [(args.seqtype, items.iloc[0]["seq"][:60]), (args.seqtype + "-long", items.iloc[-1]["seq"][:200])],
    )

    default_tokens = 32768 if args.model == "300m" else 16384
    batch_tokens = args.batch_tokens or default_tokens

    # truncate up front, then sort by length so batches are near-uniform (less padding)
    prepared = []
    for r in todo.itertuples(index=False):
        j = None if r.junction is None or pd.isna(r.junction) else int(r.junction)
        seq, j = middle_truncate(r.seq, j)
        prepared.append((r.key, seq, j))
    prepared.sort(key=lambda t: len(t[1]))

    batches, cur = [], []
    for item in prepared:
        longest = max(len(cur[0][1]) if cur else 0, len(item[1]))
        if cur and ((len(cur) + 1) * longest > batch_tokens or len(cur) >= args.max_batch):
            batches.append(cur)
            cur = []
        cur.append(item)
    if cur:
        batches.append(cur)
    print(f"    {len(prepared)} sequences in {len(batches)} batches (<= {batch_tokens} tokens each)")

    dim, t0, done = None, time.time(), 0
    for bi, batch in enumerate(batches):
        # batches are length-sorted, so pad to the longest member
        batch = sorted(batch, key=lambda t: -len(t[1]))
        ids, mask = backend.tokenize([b[1] for b in batch])
        emb = backend.forward(ids, mask).float().cpu()
        if dim is None:
            dim = emb.shape[-1]
            print(f"    hidden dim read off tensor: {dim}")
        assert emb.shape[-1] == dim, f"dim changed: {emb.shape[-1]} != {dim}"
        for i, (key, seq, junction) in enumerate(batch):
            residues = emb[i, 1 : 1 + len(seq)]
            assert residues.shape[0] == len(seq)
            cache[key] = pool(residues, junction)
        done += len(batch)
        if (bi + 1) % 20 == 0 or bi == len(batches) - 1:
            rate = done / max(time.time() - t0, 1e-9)
            print(
                f"    {done}/{len(prepared)}  {rate:.1f} seq/s  "
                f"gpu={torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                flush=True,
            )
        if done % SAVE_EVERY < len(batch):
            save_cache(out_path, cache)

    save_cache(out_path, cache)
    z = np.load(out_path)
    print(f"\nsaved {len(z['keys'])} entries -> {out_path}")
    print(f"    arrays: {[(n, z[n].shape) for n in z.files if n != 'keys']}")
    print(f"    elapsed {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
