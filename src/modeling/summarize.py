"""Compact markdown views of results/master_results.csv, for the report and the terminal."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from data import RESULTS

MASTER = f"{RESULTS}/master_results.csv"

CELL_ORDER = [
    "baseline",
    "OpenCell/protein/300m",
    "OpenCell/protein+tag/300m",
    "OpenCell/protein/6b",
    "OpenCell/protein+tag/6b",
]


def pm(row, metric):
    return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"


def best_rows(df, keys):
    return df.loc[df.groupby(keys, dropna=False)["pr_auc_mean"].idxmax()]


def table(df, task, columns):
    sel = df[df["task"] == task].copy()
    sel["order"] = sel["cell"].map({c: i for i, c in enumerate(CELL_ORDER)}).fillna(99)
    sel = best_rows(sel, ["cell", "feature_set"]).sort_values(["order", "feature_set"])
    header = "| cell | feature set | classifier | weighted | " + " | ".join(c[1] for c in columns) + " |"
    rule = "|" + "---|" * (4 + len(columns))
    lines = [header, rule]
    for _, r in sel.iterrows():
        cells = [pm(r, c[0]) if c[0] != "n_features" else str(int(r["n_features"])) for c in columns]
        lines.append(
            f"| {r['cell']} | {r['feature_set']} | {r['classifier']} | "
            f"{'yes' if r['weighted'] else 'no'} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def knob_effect(df):
    """Did class_weight / scale_pos_weight help? Paired over every other axis."""
    keys = ["task", "cell", "feature_set", "classifier"]
    piv = df.pivot_table(index=keys, columns="weighted", values="pr_auc_mean")
    piv = piv.dropna()
    delta = piv[True] - piv[False]
    return delta


def main():
    df = pd.read_csv(MASTER)
    print("## Filter task (library_success), positive = successful\n")
    print(
        table(
            df,
            "filter",
            [
                ("pr_auc", "PR-AUC"),
                ("pr_auc_minority", "PR-AUC (minority)"),
                ("recall_at_prec90", "recall @ P>=0.90"),
                ("precision_at_recall50", "precision @ R=0.50"),
                ("brier", "Brier"),
            ],
        )
    )
    print("\n## Terminus selector (Model 2), positive = N-terminus\n")
    print(
        table(
            df,
            "terminus",
            [
                ("pr_auc", "PR-AUC"),
                ("balanced_accuracy", "bal. acc @ 0.5"),
                ("brier", "Brier"),
            ],
        )
    )
    d = knob_effect(df)
    print(
        f"\n## Imbalance knob (weighted - unweighted PR-AUC), {len(d)} paired configurations\n"
        f"mean {d.mean():+.4f}   median {d.median():+.4f}   "
        f"min {d.min():+.4f}   max {d.max():+.4f}   helped in {(d > 0).sum()}/{len(d)}"
    )


if __name__ == "__main__":
    main()
