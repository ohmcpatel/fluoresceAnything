"""Check that the serving path reproduces the training path.

The failure this exists to catch is silent: a served feature vector that is built slightly
differently from the cached one -- wrong block order, a missed truncation, an off-by-one on
the <cls> token -- still returns a plausible-looking probability. So re-embed genes that are
already in `cache/emb/`, compare vectors and scores, and print the difference.

    python src/serve/verify.py --n 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modeling"))

import joblib
import numpy as np
import pandas as pd

from data import load_embeddings, load_table
from featurize import Featurizer, clean_sequence

REPO = "/workspace/fluoresceAnything"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{REPO}/models/taggability_6b_terminal_logreg.joblib")
    ap.add_argument("--n", type=int, default=8, help="genes to re-embed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    spec, est = bundle["spec"], bundle["estimator"]
    print(f"=== {os.path.basename(args.model)}")
    print(f"    backbone {spec['backbone']}  blocks {spec['blocks']}  features {spec['n_features']}")

    df = load_table()
    index, arrays = load_embeddings(spec["backbone"], "protein")

    rng = np.random.default_rng(0)
    picks = df.drop_duplicates("ENSG")
    # cover the length range, including at least one sequence past the 2046 truncation
    long_ones = picks[picks["protein_length"] > 2046].head(2)
    rest = picks[picks["protein_length"] <= 2046].sample(
        max(args.n - len(long_ones), 1), random_state=int(rng.integers(1e6))
    )
    picks = pd.concat([long_ones, rest])

    fz = Featurizer(spec["backbone"], spec["blocks"], device=args.device)
    print(f"    loading backbone onto {args.device} ...", flush=True)
    fz.load()

    rows = []
    for r in picks.itertuples(index=False):
        seq, _ = clean_sequence(r.protein)
        live, truncated = fz.embed(seq)

        i = index[r.ENSG]
        cached = np.concatenate([arrays[b][i] for b in spec["blocks"]]).astype(np.float32).reshape(1, -1)

        p_live = float(est.predict_proba(live)[0, 1])
        p_cached = float(est.predict_proba(cached)[0, 1])
        rows.append(
            {
                "ENSG": r.ENSG,
                "gene": r.gene_name,
                "len": r.protein_length,
                "trunc": truncated,
                "max_abs_dvec": float(np.abs(live - cached).max()),
                "cosine": float(
                    (live @ cached.T).item() / (np.linalg.norm(live) * np.linalg.norm(cached))
                ),
                "p_live": p_live,
                "p_cached": p_cached,
                "dp": abs(p_live - p_cached),
            }
        )
        print(f"    {r.gene_name:12s} len={r.protein_length:5d} "
              f"cos={rows[-1]['cosine']:.6f} p_live={p_live:.4f} p_cached={p_cached:.4f}", flush=True)

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\n    worst score difference live vs cached: {out['dp'].max():.5f}")
    print(f"    lowest cosine:                         {out['cosine'].min():.6f}")

    # bf16 (6B) and batch-shape differences move the last digits, not the answer; a real
    # featurizer bug moves the score by tenths.
    tol = 0.01
    if out["dp"].max() > tol:
        print(f"\nFAIL: serving features do not reproduce the training cache (> {tol})")
        return 1
    print(f"\nOK: serving path reproduces the training cache within {tol} of probability")
    return 0


if __name__ == "__main__":
    sys.exit(main())
