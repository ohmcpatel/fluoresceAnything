"""Master sweep: every (cell, framing, feature_set, classifier, weighted) under one shared
fold assignment.

No SMOTE, no subsampling, no resampling. The only imbalance knob is class_weight /
scale_pos_weight, and both variants are always run so the knob can be judged on CV PR-AUC.
Thresholds come from the validation PR curve, never from 0.5.
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import (
    available_cells,
    N_FOLDS,
    RESULTS,
    SEED,
    embedding_features,
    floor_features,
    get_folds,
    load_table,
    terminus_only,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
PRECISION_TARGET = 0.90
RECALL_TARGET = 0.50


# ------------------------------------------------------------------ metrics
def recall_at_precision(y, p, target):
    """Highest recall whose precision still clears the target."""
    precision, recall, _ = precision_recall_curve(y, p)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


def threshold_at_precision(y, p, target):
    """Highest-recall threshold whose precision clears the target.

    sklearn aligns precision[i]/recall[i] with thresholds[i] (the trailing
    precision=1, recall=0 point has no threshold), so the arrays are trimmed to
    thresholds' length -- getting this off by one silently returns a cutoff one step
    too loose.
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[: len(thresholds)], recall[: len(thresholds)]
    ok = np.where(precision >= target)[0]
    if len(ok) == 0:
        return float(np.max(p))
    return float(thresholds[ok[np.argmax(recall[ok])]])


def precision_at_recall(y, p, target):
    precision, recall, _ = precision_recall_curve(y, p)
    ok = recall >= target
    return float(precision[ok].max()) if ok.any() else 0.0


def fold_metrics(y, p):
    return {
        "pr_auc": average_precision_score(y, p),
        "pr_auc_minority": average_precision_score(1 - y, 1 - p),
        "recall_at_prec90": recall_at_precision(y, p, PRECISION_TARGET),
        "precision_at_recall50": precision_at_recall(y, p, RECALL_TARGET),
        "logloss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        # 0.5 is meaningful only for the near-balanced terminus task; the filter reads the
        # operating-point version instead
        "balanced_accuracy": _balanced_accuracy(y, p, 0.5),
        "balanced_accuracy_at_op": _balanced_accuracy(
            y, p, threshold_at_precision(y, p, PRECISION_TARGET)
        ),
    }


def _balanced_accuracy(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    return 0.5 * (tp / npos + tn / nneg)


# ------------------------------------------------------------------ classifiers
def build_classifier(name, weighted, y_train, groups_train, tune):
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    if name == "logreg":
        base = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced" if weighted else None,
                random_state=SEED,
            ),
        )
        if not tune:
            base.set_params(logisticregression__C=1.0)
            return base, None
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        return (
            GridSearchCV(
                base,
                {"logisticregression__C": C_GRID},
                scoring="neg_log_loss",
                cv=inner,
                n_jobs=5,
                refit=True,
            ),
            groups_train,
        )
    if name == "xgboost":
        return (
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.5,
                min_child_weight=5,
                reg_lambda=1.0,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=16,
                random_state=SEED,
                scale_pos_weight=pos_weight if weighted else 1.0,
            ),
            None,
        )
    raise ValueError(name)


def run_cv(X, y, groups, fold, name, weighted, tune):
    oof = np.zeros(len(y), dtype=float)
    per_fold, chosen = [], []
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        clf, fit_groups = build_classifier(name, weighted, y[tr], groups[tr], tune)
        if fit_groups is not None:
            clf.fit(X[tr], y[tr], groups=fit_groups)
            chosen.append(clf.best_params_["logisticregression__C"])
        else:
            clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te] = p
        per_fold.append(fold_metrics(y[te], p))
    return oof, pd.DataFrame(per_fold), chosen


# ------------------------------------------------------------------ sweep
def configurations(task, sub):
    """Yield (cell, framing, feature_set, X, names)."""
    if task == "filter":
        X, _ = floor_features(sub)
        yield "baseline", "reference", "FLOOR", X
        X, _ = floor_features(sub, with_terminus=True)
        yield "baseline", "reference", "FLOOR+terminus", X
        X, _ = terminus_only(sub)
        yield "baseline", "reference", "TERMINUS_ONLY", X
        for model, seqtype in available_cells():
            framing = "A_model1_filter" if seqtype == "protein" else "B_construct"
            cell = f"OpenCell/{'protein' if seqtype == 'protein' else 'protein+tag'}/{model}"
            for fs in ["MEANPOOL", "TERMINAL"]:
                X, _ = embedding_features(sub, model, seqtype, fs.lower())
                yield cell, framing, fs, X
    else:
        X, _ = floor_features(sub)
        yield "baseline", "reference", "FLOOR", X
        for model, seqtype in available_cells():
            if seqtype != "protein":  # tag placement would encode the label
                continue
            cell = f"OpenCell/protein/{model}"
            for fs in ["MEANPOOL", "TERMINAL"]:
                X, _ = embedding_features(sub, model, seqtype, fs.lower())
                yield cell, "A_model2_terminus", fs, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["filter", "terminus"])
    ap.add_argument("--classifiers", nargs="+", default=["logreg", "xgboost"])
    ap.add_argument("--no-tune", action="store_true", help="skip inner C selection (fast smoke test)")
    ap.add_argument("--out", default=f"{RESULTS}/master_results.csv")
    ap.add_argument("--oof-out", default=f"{RESULTS}/oof_predictions.csv")
    args = ap.parse_args()

    df = load_table()
    os.makedirs(RESULTS, exist_ok=True)
    rows, oof_frames = [], []

    for task in args.tasks:
        sub, y, fold = get_folds(df, task)
        groups = sub["ENSG"].values
        base_rate = y.mean()
        print(
            f"\n=== task={task}  n={len(y)}  pos={y.sum()} neg={(y == 0).sum()}  "
            f"base rate={base_rate:.3f}  folds={np.bincount(fold)}"
        )
        for cell, framing, fs, X in configurations(task, sub):
            for name in args.classifiers:
                for weighted in [False, True]:
                    t0 = time.time()
                    oof, per_fold, chosen = run_cv(
                        X, y, groups, fold, name, weighted, tune=not args.no_tune
                    )
                    row = {
                        "task": task,
                        "cell": cell,
                        "framing": framing,
                        "feature_set": fs,
                        "n_features": X.shape[1],
                        "classifier": name,
                        "weighted": weighted,
                        "n_train": len(y),
                        "n_pos": int(y.sum()),
                        "n_neg": int((y == 0).sum()),
                        "base_rate": base_rate,
                    }
                    for m in per_fold.columns:
                        row[f"{m}_mean"] = per_fold[m].mean()
                        row[f"{m}_std"] = per_fold[m].std(ddof=1)
                    row["oof_pr_auc"] = average_precision_score(y, oof)
                    row["chosen_C"] = ",".join(str(c) for c in chosen) if chosen else ""
                    row["seconds"] = time.time() - t0
                    rows.append(row)
                    config_id = f"{task}|{cell}|{fs}|{name}|{'w' if weighted else 'u'}"
                    oof_frames.append(
                        pd.DataFrame(
                            {
                                "config_id": config_id,
                                "task": task,
                                "cell": cell,
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
                        f"  {cell:32s} {fs:14s} {name:8s} w={int(weighted)}  "
                        f"PR-AUC={row['pr_auc_mean']:.3f}+/-{row['pr_auc_std']:.3f}  "
                        f"minority={row['pr_auc_minority_mean']:.3f}  "
                        f"({row['seconds']:.0f}s)",
                        flush=True,
                    )
                    pd.DataFrame(rows).to_csv(args.out, index=False)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.concat(oof_frames).to_csv(args.oof_out, index=False)
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    print(f"wrote out-of-fold predictions -> {args.oof_out}")


if __name__ == "__main__":
    main()
