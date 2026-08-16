"""Shared data plumbing: the modeling table, the fold assignment, and feature builders.

The fold assignment is computed ONCE here and reused verbatim by every cell, so that
cell-to-cell deltas reflect the varied axis and not a different data split.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = "/workspace/fluoresceAnything"
TABLE = f"{REPO}/data/modeling_table.csv"
CACHE_DIR = f"{REPO}/cache/emb"
RESULTS = f"{REPO}/results"

N_FOLDS = 5
SEED = 42
K_TERMINAL = 25  # terminal / junction window size used for features

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

CELLS = [
    ("300m", "protein"),
    ("300m", "construct"),
    ("6b", "protein"),
    ("6b", "construct"),
]


def available_cells():
    """Cells whose embedding cache exists. Missing cells are announced, never silent."""
    out = []
    for model, seqtype in CELLS:
        if os.path.exists(f"{CACHE_DIR}/{model}_{seqtype}.npz"):
            out.append((model, seqtype))
        else:
            print(f"  !! SKIPPING cell {model}/{seqtype}: no embedding cache yet")
    return out


def load_table():
    df = pd.read_csv(TABLE)
    df["row_id"] = np.arange(len(df))
    return df


# ------------------------------------------------------------------ folds
def _assign(y, groups, n_rows):
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(n_rows, -1)
    for f, (_, test_idx) in enumerate(splitter.split(np.zeros(n_rows), y, groups)):
        fold[test_idx] = f
    assert (fold >= 0).all()
    return fold


def get_folds(df, task):
    """task='filter' -> all rows, y=library_success. task='terminus' -> successful rows, y=N."""
    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/folds_{task}.csv"
    sub = df if task == "filter" else df[df["label"] == 1].reset_index(drop=True)
    y = sub["label"].values if task == "filter" else (sub["terminus"] == "N").astype(int).values

    if os.path.exists(path):
        saved = pd.read_csv(path)
        assert list(saved["row_id"]) == list(sub["row_id"]), f"{path} is stale -- delete it"
        fold = saved["fold"].values
    else:
        fold = _assign(y, sub["ENSG"].values, len(sub))
        pd.DataFrame({"row_id": sub["row_id"], "ENSG": sub["ENSG"], "y": y, "fold": fold}).to_csv(
            path, index=False
        )

    # no gene may straddle folds
    per_gene = pd.DataFrame({"ENSG": sub["ENSG"], "fold": fold}).groupby("ENSG")["fold"].nunique()
    assert per_gene.max() == 1, f"{(per_gene > 1).sum()} genes span folds"
    return sub, y, fold


# ------------------------------------------------------------------ features
def aa_composition(seqs):
    idx = {a: i for i, a in enumerate(AMINO_ACIDS)}
    out = np.zeros((len(seqs), len(AMINO_ACIDS)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for ch in s:
            j = idx.get(ch)
            if j is not None:
                out[r, j] += 1
        out[r] /= max(len(s), 1)
    return out


def floor_features(sub, with_terminus=False):
    X = np.column_stack(
        [np.log10(sub["protein_length"].values).astype(np.float32), aa_composition(sub["protein"])]
    )
    names = ["log_length"] + [f"aa_{a}" for a in AMINO_ACIDS]
    if with_terminus:
        X = np.column_stack([X, (sub["terminus"] == "N").astype(np.float32).values])
        names = names + ["terminus_is_N"]
    return X.astype(np.float32), names


def terminus_only(sub):
    return (sub["terminus"] == "N").astype(np.float32).values.reshape(-1, 1), ["terminus_is_N"]


_emb_cache = {}


def load_embeddings(model, seqtype):
    key = (model, seqtype)
    if key not in _emb_cache:
        path = f"{CACHE_DIR}/{model}_{seqtype}.npz"
        z = np.load(path)
        keys = [str(k) for k in z["keys"]]
        index = {k: i for i, k in enumerate(keys)}
        _emb_cache[key] = (index, {n: z[n] for n in z.files if n != "keys"})
    return _emb_cache[key]


def embedding_features(sub, model, seqtype, feature_set):
    """feature_set in {'meanpool', 'terminal'}."""
    index, arrays = load_embeddings(model, seqtype)
    if seqtype == "protein":
        lookup = sub["ENSG"].values
    else:
        lookup = (sub["ENSG"] + "_" + sub["terminus"]).values
    rows = np.array([index[k] for k in lookup])

    blocks, names = [arrays["mean"][rows]], ["mean"]
    if feature_set == "terminal":
        for part in [f"first_{K_TERMINAL}", f"last_{K_TERMINAL}"]:
            blocks.append(arrays[part][rows])
            names.append(part)
        if seqtype == "construct":
            blocks.append(arrays[f"junction_{K_TERMINAL}"][rows])
            names.append(f"junction_{K_TERMINAL}")
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    assert np.isfinite(X).all()
    return X, names
