"""Fit and persist the deployment head for the taggability score.

The configuration is not chosen here -- it is the one locked by
`notebooks/decision_to_classifier.ipynb`, decision for decision:

    1 ESM-C embeddings   (not the 21-feature FLOOR)
    2 TERMINAL pooling   (mean + first_25 + last_25)
    3 bare protein       (not the tag-fusion construct)
    4 6B backbone        (2560-d blocks -> 7,680 features)
    5 logistic regression (not XGBoost)
    6 class_weight=None  -- the *probability* branch of decision 6, because this
                           endpoint ships a continuous score whose value has to mean
                           something, not just a rank
    7 C=0.001, 25-residue windows -- decision 7 tested C down to 1e-6 (no change) and
                           50-residue windows (solidly worse on every fold)

Training is on all rows with no holdout: cross-validation is already spent, and the
held-out numbers reported in the bundle come from that frozen 5-fold grid, not from
anything re-measured here.

    python src/serve/train_final.py               # 6B, the locked spec
    python src/serve/train_final.py --model 300m  # cheaper backbone, see README
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modeling"))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import RESULTS, SEED, get_folds, load_embeddings, load_table

REPO = "/workspace/fluoresceAnything"
MODEL_DIR = f"{REPO}/models"

# The locked spec. TERMINAL pooling is these three blocks in this order -- the same
# order `embedding_features()` builds during the sweep, and the order the serving
# featurizer must reproduce.
BLOCKS = ["mean", "first_25", "last_25"]
C = 0.001
MAX_ITER = 5000


def build_features(sub, model):
    """TERMINAL pooling for the bare protein, keyed by ENSG."""
    index, arrays = load_embeddings(model, "protein")
    rows = np.array([index[k] for k in sub["ENSG"].values])
    X = np.concatenate([arrays[b][rows] for b in BLOCKS], axis=1).astype(np.float32)
    assert np.isfinite(X).all()
    return X


def held_out_reference(model):
    """The frozen-fold numbers for this exact cell, read from the sweep -- never recomputed
    on the training data, where they would be meaningless."""
    oof = pd.read_csv(f"{RESULTS}/oof_predictions.csv")
    cid = f"filter|OpenCell/protein/{model}|TERMINAL|logreg|u"  # u = unweighted
    g = oof[oof["config_id"] == cid]
    if g.empty:
        raise KeyError(f"{cid} not in oof_predictions.csv")
    y, p = g["y"].values, g["p"].values
    per_fold = [
        average_precision_score(1 - g[g.fold == f]["y"], 1 - g[g.fold == f]["p"])
        for f in sorted(g["fold"].unique())
    ]
    return {
        "config_id": cid,
        "pr_auc_pooled": float(average_precision_score(y, p)),
        "pr_auc_minority_pooled": float(average_precision_score(1 - y, 1 - p)),
        "pr_auc_minority_per_fold": [float(v) for v in per_fold],
        "pr_auc_minority_mean": float(np.mean(per_fold)),
        "pr_auc_minority_std": float(np.std(per_fold, ddof=1)),
        "brier_pooled": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
        # the shape of the score distribution, so a served score can be reported as a
        # percentile of held-out predictions rather than as a bare number
        "oof_score_quantiles": {
            str(q): float(np.quantile(p, q)) for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        },
        "oof_scores": [float(v) for v in np.sort(p)],
    }


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["300m", "6b"], default="6b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_table()
    sub, y, fold = get_folds(df, "filter")  # all 1,757 rows, y = library_success
    X = build_features(sub, args.model)
    print(f"=== training the deployment head on all {len(y)} rows, no holdout")
    print(f"    backbone   ESM-C {args.model}, bare protein, TERMINAL {BLOCKS}")
    print(f"    features   {X.shape[1]}   positives {int(y.sum())}  negatives {int((y == 0).sum())}")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C, max_iter=MAX_ITER, class_weight=None, random_state=SEED
        ),
    )
    clf.fit(X, y)
    print(f"    estimator  StandardScaler -> LogisticRegression(C={C}, max_iter={MAX_ITER}, "
          f"class_weight=None, random_state={SEED})")

    ref = held_out_reference(args.model)
    print(f"\n    held-out reference (frozen 5-fold sweep, {ref['config_id']}):")
    print(f"      PR-AUC          {ref['pr_auc_pooled']:.3f}   (no-skill {ref['base_rate']:.3f})")
    print(f"      PR-AUC minority {ref['pr_auc_minority_mean']:.3f} +/- "
          f"{ref['pr_auc_minority_std']:.3f}   (no-skill {1 - ref['base_rate']:.3f})")
    print(f"      Brier           {ref['brier_pooled']:.3f}")

    # A resubstitution score is not a performance number -- it is here only so the
    # bundle records that the fit converged onto the expected side of the data.
    p_train = clf.predict_proba(X)[:, 1]
    print(f"      (resubstitution PR-AUC {average_precision_score(y, p_train):.3f} -- not a "
          f"performance estimate, optimistic by construction)")

    bundle = {
        "estimator": clf,
        "spec": {
            "task": "filter",
            "target": "P(library_success) for a knock-in tagging attempt",
            "backbone": args.model,
            "seqtype": "protein",
            "pooling": "TERMINAL",
            "blocks": BLOCKS,
            "n_features": int(X.shape[1]),
            "C": C,
            "max_iter": MAX_ITER,
            "class_weight": None,
            "seed": SEED,
            "max_len": 2046,
            "source": "notebooks/decision_to_classifier.ipynb",
        },
        "training": {
            "n_rows": int(len(y)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "table": "data/modeling_table.csv",
            "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "held_out": ref,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = args.out or f"{MODEL_DIR}/taggability_{args.model}_terminal_logreg.joblib"
    joblib.dump(bundle, out, compress=3)
    print(f"\nwrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    meta = {k: v for k, v in bundle.items() if k != "estimator"}
    meta["held_out"] = {k: v for k, v in ref.items() if k != "oof_scores"}
    with open(out.replace(".joblib", ".json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out.replace('.joblib', '.json')}")


if __name__ == "__main__":
    main()
