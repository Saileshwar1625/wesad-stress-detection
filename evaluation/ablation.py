"""
Feature-group ablation under LOSO: which kinds of signal is each model
leaning on, and does a hard subject fail because of one feature group?

Background: explain.py shows HOW a failing subject differs (e.g. S14's
EDA barely moves; S5 looked like it hardly moved during the stress task). Those are
hypotheses. This script tests them: re-run LOSO with feature groups removed
or isolated and see whether the failing subject's balanced accuracy changes.

Feature sets:
  all           every feature (the main result)
  no_acc        drop accelerometer features
  no_hrv_proxy  drop the crude interval-variability feature
  no_temp / no_eda / no_eda_no_temp   drop those groups (see S14 note below)
  acc_only      movement alone -- if this scores well, movement is a shortcut
                (the stress task involves speaking/arithmetic, so people move)
  cardiac_only  hr_mean_bpm + hrv_proxy_std_ms
  hr_only       hr_mean_bpm alone (no noisy HRV proxy)
  eda_only      EDA features
  temp_only     temperature features

Usage:
    python evaluation/ablation.py --features data/processed/features.parquet
Writes reports/ablation_report.txt and reports/ablation_per_subject.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from train_eval import ALL_MODEL_NAMES, FEATURE_COLUMNS, MODEL_NAMES, run_loso

ACC = ["acc_mag_mean", "acc_mag_std"]
FEATURE_SETS = {
    "all": FEATURE_COLUMNS,
    "no_acc": [c for c in FEATURE_COLUMNS if c not in ACC],
    "no_hrv_proxy": [c for c in FEATURE_COLUMNS if c != "hrv_proxy_std_ms"],
    # Drop-one-group sets: S14 has flat EDA AND a reversed temperature response
    # at the same time; other subjects have only one of the two and do fine.
    # These test whether removing either (or both) rescues the tree models.
    "no_temp": [c for c in FEATURE_COLUMNS if not c.startswith("temp_")],
    "no_eda": [c for c in FEATURE_COLUMNS if not c.startswith("eda_")],
    "no_eda_no_temp": [c for c in FEATURE_COLUMNS if not c.startswith(("eda_", "temp_"))],
    "acc_only": ACC,
    "cardiac_only": ["hr_mean_bpm", "hrv_proxy_std_ms"],
    # hr_only exists because cardiac_only includes the unreliable HRV proxy,
    # which can drown out a clean heart-rate signal (seen on S14).
    "hr_only": ["hr_mean_bpm"],
    "eda_only": ["eda_mean", "eda_std", "eda_slope"],
    "temp_only": ["temp_mean", "temp_slope"],
}


def run_ablation(df: pd.DataFrame, feature_sets: dict = FEATURE_SETS,
                 model_names: list[str] = MODEL_NAMES) -> pd.DataFrame:
    """Long table: one row per (model, feature_set, subject) with balanced accuracy."""
    rows = []
    for model_name in model_names:
        for set_name, cols in feature_sets.items():
            res = run_loso(df, feature_columns=cols, model_name=model_name)
            for _, r in res.iterrows():
                rows.append({"model": model_name, "feature_set": set_name,
                             "subject": r["subject"],
                             "balanced_accuracy": r["balanced_accuracy"]})
    return pd.DataFrame(rows)


def summarize_ablation(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean balanced accuracy per (model, feature_set)."""
    out = (long_df.groupby(["model", "feature_set"])["balanced_accuracy"]
           .agg(mean_bal_acc="mean", min_bal_acc="min").round(4).reset_index())
    return out


def subject_view(long_df: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    """Rows = feature sets, columns = (model, subject): how the chosen
    subjects fare under each feature set."""
    sub = long_df[long_df["subject"].isin(subjects)]
    return (sub.pivot_table(index="feature_set", columns=["model", "subject"],
                            values="balanced_accuracy").round(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--subjects", nargs="*", default=["S14", "S5"],
                        help="subjects to show in detail")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="run only these feature sets (default: all of them)")
    parser.add_argument("--models", nargs="*", default=MODEL_NAMES, choices=ALL_MODEL_NAMES)
    parser.add_argument("--report", default="reports/ablation_report.txt")
    parser.add_argument("--csv", default="reports/ablation_per_subject.csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    sets = FEATURE_SETS if not args.sets else {k: FEATURE_SETS[k] for k in args.sets}
    long_df = run_ablation(df, feature_sets=sets, model_names=args.models)

    pd.set_option("display.width", 200)
    lines = [
        "=== Mean / minimum LOSO balanced accuracy per feature set ===",
        summarize_ablation(long_df).to_string(index=False), "",
        f"=== Detail for {args.subjects} (balanced accuracy) ===",
        subject_view(long_df, args.subjects).to_string(), "",
        "How to read this: compare each row to 'all'. If removing a group "
        "RAISES a failing subject's score, the model was leaning on that group "
        "in a way that doesn't hold for them. If a single group ('acc_only') "
        "scores well on its own, that group may be a shortcut rather than a "
        "physiological signal. Differences under ~0.10 for one subject are "
        "within noise (about 35 independent windows per person).",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))
    long_df.to_csv(args.csv, index=False)
    print("\n".join(lines))
    print(f"\nWrote {args.report} and {args.csv}")


if __name__ == "__main__":
    main()
