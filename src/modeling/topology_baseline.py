"""Topology baseline for the terminus selector (Model 2).

Answers: is the terminus selector just a signal-peptide detector?

Fetches UniProt signal-peptide / transmembrane / topology annotations for the accessions
already in data/opencell_sequences.csv, builds two baselines --

  TOPOLOGY_SP    signal peptide only (the literal "is it just SignalP?" control)
  TOPOLOGY_FULL  + transmembrane counts and N/C-terminal hydropathy

-- and scores them under exactly the same folds and classifiers as the embedding cells,
appending the rows to results/master_results.csv.
"""

import io
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score

from data import RESULTS, load_table, get_folds
from run_grid import run_cv

REPO = "/workspace/fluoresceAnything"
SEQS = f"{REPO}/data/opencell_sequences.csv"
TOPO = f"{REPO}/data/uniprot_topology.csv"
MASTER = f"{RESULTS}/master_results.csv"
OOF = f"{RESULTS}/oof_predictions.csv"

URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,ft_signal,ft_transmem,ft_topo_dom,cc_subcellular_location"
CHUNK = 80

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def fetch_topology(accessions):
    session = requests.Session()
    frames = []
    for i in range(0, len(accessions), CHUNK):
        chunk = accessions[i : i + CHUNK]
        query = " OR ".join(f"accession:{a}" for a in chunk)
        for attempt in range(4):
            r = session.get(
                URL,
                params={"format": "tsv", "fields": FIELDS, "query": query, "size": "500"},
                timeout=60,
            )
            if r.status_code == 200:
                break
            time.sleep(0.5 + attempt)
        else:
            raise RuntimeError(f"UniProt request failed for chunk {i}")
        frames.append(pd.read_csv(io.StringIO(r.text), sep="\t"))
        if (i // CHUNK) % 5 == 0:
            print(f"    fetched {min(i + CHUNK, len(accessions))}/{len(accessions)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.lower().replace(" ", "_") for c in out.columns]
    return out.rename(columns={"entry": "uniprot_accession"})


def kd_window_max(seq, k=19):
    if len(seq) < k:
        return np.nan
    vals = np.array([KD.get(c, 0.0) for c in seq])
    window = np.convolve(vals, np.ones(k) / k, mode="valid")
    return float(window.max())


def parse_features(topo, seqs):
    df = seqs.merge(topo, on="uniprot_accession", how="left")
    sig = df["signal_peptide"].fillna("")
    df["has_signal"] = sig.str.contains("SIGNAL").astype(float)
    df["signal_length"] = [
        float(m.group(1)) if (m := re.search(r"SIGNAL\s+\d+\.\.(\d+)", s)) else 0.0 for s in sig
    ]
    tm = df["transmembrane"].fillna("")
    df["n_transmem"] = tm.str.count("TRANSMEM").astype(float)
    df["has_transmem"] = (df["n_transmem"] > 0).astype(float)
    df["first_tm_start"] = [
        float(m.group(1)) if (m := re.search(r"TRANSMEM\s+(\d+)", s)) else 0.0 for s in tm
    ]
    df["nterm_kd_max"] = [kd_window_max(s[:60]) for s in df["sequence"]]
    df["cterm_kd_max"] = [kd_window_max(s[-60:]) for s in df["sequence"]]
    df["log_length"] = np.log10(df["sequence_length"])
    for c in ["nterm_kd_max", "cterm_kd_max"]:
        df[c] = df[c].fillna(df[c].median())
    return df


SP_COLS = ["has_signal", "signal_length"]
FULL_COLS = SP_COLS + [
    "n_transmem",
    "has_transmem",
    "first_tm_start",
    "nterm_kd_max",
    "cterm_kd_max",
    "log_length",
]


def main():
    seqs = pd.read_csv(SEQS)
    if os.path.exists(TOPO):
        topo = pd.read_csv(TOPO)
        print(f"using cached {TOPO} ({len(topo)} rows)")
    else:
        print(f"fetching UniProt topology for {len(seqs)} accessions")
        topo = fetch_topology(sorted(seqs["uniprot_accession"].unique()))
        topo.to_csv(TOPO, index=False)
        print(f"wrote {TOPO} ({len(topo)} rows)")

    feats = parse_features(topo, seqs)
    missing = feats["has_signal"].isna().sum()
    print(
        f"annotation coverage: {len(feats) - missing}/{len(feats)}  "
        f"signal peptides={int(feats['has_signal'].sum())}  "
        f"multi-pass TM={int((feats['n_transmem'] > 1).sum())}  "
        f"single-pass TM={int((feats['n_transmem'] == 1).sum())}"
    )

    df = load_table()
    rows, oof_frames = [], []

    for task in ["terminus", "filter"]:
        sub, y, fold = get_folds(df, task)
        groups = sub["ENSG"].values
        merged = sub[["ENSG"]].merge(
            feats[["ENSG"] + FULL_COLS].drop_duplicates("ENSG"), on="ENSG", how="left"
        )
        assert merged[FULL_COLS].notna().all().all()
        print(f"\n--- task={task}  n={len(y)}  base rate={y.mean():.3f}")
        if task == "terminus":
            print(
                "signal peptide vs tagged terminus among successful targets:\n"
                f"{pd.crosstab(merged['has_signal'], pd.Series(y, name='terminus_is_N'))}"
            )
        else:
            print(
                "membrane annotation vs library success:\n"
                f"{pd.crosstab(merged['has_transmem'], pd.Series(y, name='successful'), normalize='index').round(3)}"
            )

        for fs, cols in [("TOPOLOGY_SP", SP_COLS), ("TOPOLOGY_FULL", FULL_COLS)]:
            X = merged[cols].values.astype(np.float32)
            for name in ["logreg", "xgboost"]:
                for weighted in [False, True]:
                    t0 = time.time()
                    oof, per_fold, chosen = run_cv(X, y, groups, fold, name, weighted, tune=True)
                    row = {
                        "task": task,
                        "cell": "baseline",
                        "framing": "reference",
                        "feature_set": fs,
                        "n_features": X.shape[1],
                        "classifier": name,
                        "weighted": weighted,
                        "n_train": len(y),
                        "n_pos": int(y.sum()),
                        "n_neg": int((y == 0).sum()),
                        "base_rate": float(y.mean()),
                    }
                    for m in per_fold.columns:
                        row[f"{m}_mean"] = per_fold[m].mean()
                        row[f"{m}_std"] = per_fold[m].std(ddof=1)
                    row["oof_pr_auc"] = average_precision_score(y, oof)
                    row["chosen_C"] = ",".join(str(c) for c in chosen) if chosen else ""
                    row["seconds"] = time.time() - t0
                    rows.append(row)
                    oof_frames.append(
                        pd.DataFrame(
                            {
                                "config_id": f"{task}|baseline|{fs}|{name}|{'w' if weighted else 'u'}",
                                "task": task,
                                "cell": "baseline",
                                "feature_set": fs,
                                "classifier": name,
                                "weighted": weighted,
                                "row_id": sub["row_id"].values,
                                "fold": fold,
                                "y": y,
                                "p": oof,
                            }
                        )
                    )
                    print(
                        f"  {fs:14s} {name:8s} w={int(weighted)}  "
                        f"PR-AUC={row['pr_auc_mean']:.3f}+/-{row['pr_auc_std']:.3f}  "
                        f"minority={row['pr_auc_minority_mean']:.3f}  "
                        f"bal-acc={row['balanced_accuracy_mean']:.3f}",
                        flush=True,
                    )

    new = pd.DataFrame(rows)
    master = pd.read_csv(MASTER)
    master = master[~master["feature_set"].str.startswith("TOPOLOGY")]
    pd.concat([master, new], ignore_index=True).to_csv(MASTER, index=False)

    new_oof = pd.concat(oof_frames)
    old_oof = pd.read_csv(OOF)
    old_oof = old_oof[~old_oof["feature_set"].astype(str).str.startswith("TOPOLOGY")]
    pd.concat([old_oof, new_oof], ignore_index=True).to_csv(OOF, index=False)
    print(f"\nappended {len(new)} rows -> {MASTER}")


if __name__ == "__main__":
    main()
