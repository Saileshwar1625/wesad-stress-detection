"""
Does S14's temperature reversal survive a control for protocol ORDER?

WESAD subjects went through the protocol in one of two orders (see
segment_order.py): stress EARLY (baseline > stress > meditation > amusement)
or stress LATE (baseline > amusement > meditation > stress). If skin
temperature drifts over the session, the stress-minus-non-stress temperature
difference depends on order, so the population average mixes two situations.
This script splits subjects by order group and shows each subject's
stress-minus-non-stress change for a few features, so you can see whether S14
is an outlier WITHIN its own order group.

Usage (needs the raw pickles for the order, and the features table):
    python pipeline/order_check.py --input data/raw --features data/processed/features.parquet
Writes reports/order_check.txt
"""

import argparse
from pathlib import Path

import pandas as pd

from load_wesad import load_subject_pickle
from segment_order import order_string, segment_sequence

FEATURES = ["temp_mean", "hr_mean_bpm", "eda_mean", "acc_mag_std"]


def stress_response_table(df: pd.DataFrame, cols: list[str] = FEATURES) -> pd.DataFrame:
    """Per subject: mean(stress) - mean(non_stress) for each feature."""
    m = df.groupby(["subject", "label"])[cols].mean().unstack("label")
    return pd.DataFrame({c: m[(c, "stress")] - m[(c, "non_stress")] for c in cols})


def add_order_groups(table: pd.DataFrame, orders: dict[str, str]) -> pd.DataFrame:
    """Label each subject with a group letter: 'A' = the most common order."""
    freq = pd.Series(list(orders.values())).value_counts()
    letter = {order: chr(ord("A") + i) for i, order in enumerate(freq.index)}
    out = table.copy()
    out["order_group"] = [letter[orders[s]] for s in out.index]
    out["order"] = [orders[s] for s in out.index]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--report", default="reports/order_check.txt")
    args = parser.parse_args()

    orders = {}
    for f in sorted(Path(args.input).rglob("S*.pkl"), key=lambda p: p.name):
        orders[f.stem] = order_string(segment_sequence(load_subject_pickle(f)["label"]))

    df = pd.read_parquet(args.features)
    table = add_order_groups(stress_response_table(df), orders)

    pd.set_option("display.width", 200)
    lines = ["=== Order groups ==="]
    for g, sub in table.groupby("order_group"):
        lines.append(f"Group {g}: {sub['order'].iloc[0]}  -> {sorted(sub.index)}")
    lines += ["", "=== Stress minus non-stress, per subject (sorted by group) ===",
              table.sort_values(["order_group"]).drop(columns="order").round(3).to_string(), "",
              "=== Group means / std of those differences ===",
              table.groupby("order_group")[FEATURES].agg(["mean", "std"]).round(3).to_string(), "",
              "How to read this: look at temp_mean. If S14's value sits far from the other subjects "
              "IN ITS OWN GROUP, the reversal is not explained by protocol order. If its group as a "
              "whole shows a similar sign, order (session drift) is a plausible explanation."]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
