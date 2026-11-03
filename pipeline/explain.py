"""
Computed statistics on how one held-out subject's wrist-sensor features differ
from the other subjects (per class, plus the stress-minus-non-stress response
compared with the population). No API call, no LLM.

The optional LLM description of these statistics lives in explain_llm.py, which
adds the per-subject model diagnostics, a pinned model, a number-grounding check
and saved audit metadata. An earlier version of this file asked an LLM to give
"reasons the model failed"; that framing was dropped because the reasons were
never tested (see REPORT.md).

Usage:
    python pipeline/explain.py --features data/processed/features.parquet --subject S14
"""

import argparse
import json

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hr_mean_bpm", "hrv_proxy_std_ms",
    "eda_mean", "eda_std", "eda_slope",
    "temp_mean", "temp_slope",
    "acc_mag_mean", "acc_mag_std",
]


WEAK_RESPONSE_RATIO = 0.25


def build_subject_comparison(df: pd.DataFrame, subject: str,
                              feature_columns: list[str] = FEATURE_COLUMNS) -> dict:
    """For each class, z-score the target subject's mean feature values
    against the distribution of OTHER subjects' per-subject means. Uses
    per-subject means (not pooled windows) so one subject with many
    windows can't dominate the reference distribution."""
    others = df[df["subject"] != subject]
    target = df[df["subject"] == subject]
    if target.empty:
        raise ValueError(f"Subject {subject} not found in features.")

    comparison = {"subject": subject, "per_class": {}}
    for label in sorted(df["label"].unique()):
        target_cls = target[target["label"] == label]
        others_cls = others[others["label"] == label]
        if target_cls.empty or others_cls.empty:
            continue
        ref_means = others_cls.groupby("subject")[feature_columns].mean()
        target_mean = target_cls[feature_columns].mean()

        feats = {}
        for col in feature_columns:
            mu, sd = ref_means[col].mean(), ref_means[col].std()
            z = (target_mean[col] - mu) / sd if sd and sd > 0 else np.nan
            feats[col] = {
                "subject_mean": round(float(target_mean[col]), 4),
                "others_mean": round(float(mu), 4),
                "z_score": None if np.isnan(z) else round(float(z), 2),
            }
        comparison["per_class"][label] = feats

    # The single most diagnostic fact: does this subject's stress-vs-
    # non_stress DIRECTION match everyone else's, per feature? A sign
    # flip means the model learned a relationship this subject doesn't follow.
    direction = {}
    if {"stress", "non_stress"} <= set(comparison["per_class"]):
        for col in feature_columns:
            s = comparison["per_class"]["stress"][col]
            n = comparison["per_class"]["non_stress"][col]
            subj_delta = s["subject_mean"] - n["subject_mean"]
            pop_delta = s["others_mean"] - n["others_mean"]
            same_dir = bool(np.sign(subj_delta) == np.sign(pop_delta))
            # Direction alone is a weak test: a subject whose EDA rises 2% as
            # much as everyone else's still "matches". The ratio captures
            # how much of the population-typical response this subject shows
            # (1.0 = typical, ~0 = flat, negative = reversed).
            ratio = subj_delta / pop_delta if pop_delta else np.nan
            direction[col] = {
                "subject_stress_minus_nonstress": round(subj_delta, 4),
                "population_stress_minus_nonstress": round(pop_delta, 4),
                "direction_matches_population": same_dir,
                "response_ratio": None if np.isnan(ratio) else round(float(ratio), 2),
                # 0.25 is a judgment call, not a standard.
                "weak_response": bool(same_dir and not np.isnan(ratio) and ratio < WEAK_RESPONSE_RATIO),
            }
    comparison["stress_response_direction"] = direction
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    comparison = build_subject_comparison(df, args.subject)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
