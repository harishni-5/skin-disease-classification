"""
Statistical tests requested by the reviewers:
  - McNemar's test comparing two models' paired predictions on the same val set
    (reviewer DAI comment #13).
  - Exact binomial confidence interval for the small external test set
    (reviewer comments about the 12-image external test being too small to
    interpret without an uncertainty estimate).

Usage:
    python stats.py mcnemar <results_dir_A>/val_predictions.csv <results_dir_B>/val_predictions.csv
    python stats.py binomial_ci <n_correct> <n_total>

Requires the two val_predictions.csv files to be predictions on the SAME val
set (same `path` column, same order or at least same set of paths) -- this is
guaranteed if both models were evaluated with the same --data_root.
"""
import sys
import csv
import json

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binomtest


def load_predictions(path):
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["path"]] = row
    return rows


def run_mcnemar(path_a, path_b):
    preds_a = load_predictions(path_a)
    preds_b = load_predictions(path_b)
    common = sorted(set(preds_a) & set(preds_b))
    if len(common) < len(preds_a) or len(common) < len(preds_b):
        print(f"WARNING: only {len(common)} images in common "
              f"(A has {len(preds_a)}, B has {len(preds_b)}). "
              f"Are these predictions on the same val set?")

    # 2x2 contingency table: [[both correct, A correct only],[B correct only, both wrong]]
    both_correct = a_only = b_only = both_wrong = 0
    for p in common:
        a_correct = preds_a[p]["correct"] == "1"
        b_correct = preds_b[p]["correct"] == "1"
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif b_correct and not a_correct:
            b_only += 1
        else:
            both_wrong += 1

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=(a_only + b_only < 25), correction=True)

    out = {
        "n_common_images": len(common),
        "contingency_table": {
            "both_correct": both_correct,
            "a_correct_b_wrong": a_only,
            "b_correct_a_wrong": b_only,
            "both_wrong": both_wrong,
        },
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
    print(json.dumps(out, indent=2))
    return out


def run_binomial_ci(n_correct, n_total, confidence=0.95):
    n_correct, n_total = int(n_correct), int(n_total)
    result = binomtest(n_correct, n_total, p=0.5)
    ci = result.proportion_ci(confidence_level=confidence, method="exact")
    out = {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "confidence": confidence,
        "ci_low": ci.low,
        "ci_high": ci.high,
        "note": "Clopper-Pearson exact binomial CI",
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "mcnemar":
        run_mcnemar(sys.argv[2], sys.argv[3])
    elif cmd == "binomial_ci":
        run_binomial_ci(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
