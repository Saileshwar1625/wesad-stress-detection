"""
Leave-one-subject-out (LOSO) cross-validation for the binary stress
classifier, run with TWO models side by side:

  - random_forest      (RandomForestClassifier)
  - gradient_boosting  (HistGradientBoostingClassifier)

Why two models: the research question is whether good average accuracy
hides people the model fails on -- and if so, WHY. Running two different
model families answers a key part of "why":

  - If both models fail on the SAME people, the problem is in those
    people's data (unusual stress response, motion artifact, poor sensor
    contact), not in the choice of model.
  - If they fail on DIFFERENT people, at least part of the failure is
    model-specific.

Reports per-model mean metrics, the per-subject breakdown for both models
in one table, and the overlap between each model's struggling subjects.

Deliberately NOT a neural net -- see README for why 15 subjects makes that
a credibility risk, not a strength.

Usage:
    python evaluation/train_eval.py --features data/processed/features.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "hr_mean_bpm", "hrv_proxy_std_ms",
    "eda_mean", "eda_std", "eda_slope",
    "temp_mean", "temp_slope",
    "acc_mag_mean", "acc_mag_std",
]

MODEL_NAMES = ["random_forest", "gradient_boosting"]  # the default pair
ALL_MODEL_NAMES = MODEL_NAMES + ["logistic_regression", "ensemble"]  # opt in with --models

# A subject counts as "struggling" below this BALANCED accuracy (mean of
# recall on stress and recall on non-stress; chance = 0.5 no matter how
# imbalanced the classes are). Plain accuracy is NOT used for this decision:
# WESAD windows are ~70% non-stress, so a model that always says "non-stress"
# scores ~0.70 plain accuracy while having learned nothing about stress.
# 0.7 is a judgment call, not a standard -- state it in the write-up and
# check the main conclusion doesn't flip at 0.6 or 0.8.
STRUGGLE_METRIC = "balanced_accuracy"
STRUGGLE_THRESHOLD = 0.7


def make_model(name: str = "random_forest", random_state: int = 0):
    # class_weight='balanced' in both: non_stress windows substantially
    # outnumber stress windows in WESAD (the stress segment is shorter than
    # baseline), and an unweighted model would lean toward always predicting
    # the majority class while still posting a deceptively OK accuracy.
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=random_state)
    if name == "gradient_boosting":
        # Defaults kept deliberately modest -- with 15 subjects, aggressive
        # tuning would just overfit the LOSO folds. If you tune, tune inside
        # each training fold (nested CV), never on the held-out subject.
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_depth=3, class_weight="balanced",
                                               random_state=random_state)
    if name == "logistic_regression":
        # A linear model: structurally different from the two tree ensembles,
        # so agreement between it and them is stronger evidence that a
        # subject's failure comes from their data, not from tree-model quirks.
        # Scaling matters here (trees don't need it, logistic regression does).
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=1000,
                                                random_state=random_state))
    if name == "ensemble":
        # Soft-voting average of the three families' predicted probabilities.
        # POST HOC: added after seeing that the tree models and logistic
        # regression fail on different people. Report it as an extra row,
        # never as the headline, and say it was motivated by the results.
        return VotingClassifier(
            estimators=[("rf", make_model("random_forest", random_state)),
                        ("gb", make_model("gradient_boosting", random_state)),
                        ("lr", make_model("logistic_regression", random_state))],
            voting="soft")
    raise ValueError(f"Unknown model name: {name}")


def run_loso(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
             model_name: str = "random_forest") -> pd.DataFrame:
    """Per-subject results for one model: accuracy, balanced accuracy, F1,
    n_windows, and the majority-class share (what plain accuracy would be
    for a model that always predicts the majority class)."""
    subjects = sorted(df["subject"].unique())
    if len(subjects) < 2:
        raise ValueError(
            f"LOSO needs at least 2 subjects; got {len(subjects)}."
        )

    results = []
    for held_out in subjects:
        train_df = df[df["subject"] != held_out]
        test_df = df[df["subject"] == held_out]

        if test_df["label"].nunique() < 2:
            # Single-class test set -> accuracy/F1 not meaningful; flag, don't score.
            results.append({
                "subject": held_out, "n_windows": len(test_df),
                "accuracy": np.nan, "balanced_accuracy": np.nan, "f1": np.nan,
                "majority_baseline": np.nan,
                "note": "held-out subject has only one class in test windows -- metric not meaningful",
            })
            continue

        model = make_model(model_name)
        model.fit(train_df[feature_columns], train_df["label"])
        preds = model.predict(test_df[feature_columns])

        results.append({
            "subject": held_out,
            "n_windows": len(test_df),
            "accuracy": round(accuracy_score(test_df["label"], preds), 4),
            "balanced_accuracy": round(balanced_accuracy_score(test_df["label"], preds), 4),
            "majority_baseline": round(float(test_df["label"].value_counts(normalize=True).max()), 4),
            "f1": round(f1_score(test_df["label"], preds, pos_label="stress", zero_division=0), 4),
            "note": "",
        })

    return pd.DataFrame(results)


def summarize(results_df: pd.DataFrame) -> dict:
    valid = results_df.dropna(subset=["accuracy"])
    if not len(valid):
        return {"n_subjects_evaluated": 0}
    return {
        "n_subjects_evaluated": len(valid),
        "n_subjects_skipped": len(results_df) - len(valid),
        "mean_accuracy": round(float(valid["accuracy"].mean()), 4),
        "mean_balanced_accuracy": round(float(valid["balanced_accuracy"].mean()), 4),
        "std_balanced_accuracy_across_subjects": round(float(valid["balanced_accuracy"].std()), 4),
        "mean_f1": round(float(valid["f1"].mean()), 4),
        "worst_subject": valid.loc[valid["balanced_accuracy"].idxmin(), "subject"],
        "worst_subject_balanced_accuracy": float(valid["balanced_accuracy"].min()),
    }


def struggling_subjects(results_df: pd.DataFrame, threshold: float = STRUGGLE_THRESHOLD,
                        metric: str = STRUGGLE_METRIC) -> set:
    valid = results_df.dropna(subset=[metric])
    return set(valid.loc[valid[metric] < threshold, "subject"])


def compare_models(df: pd.DataFrame, model_names: list[str] = MODEL_NAMES,
                   threshold: float = STRUGGLE_THRESHOLD) -> dict:
    """Run LOSO for every model and return per-model results, a merged
    per-subject accuracy table, and the overlap of struggling subjects."""
    per_model = {name: run_loso(df, model_name=name) for name in model_names}

    merged = None
    for name, res in per_model.items():
        part = res[["subject", "accuracy", "balanced_accuracy"]].rename(
            columns={"accuracy": f"acc_{name}", "balanced_accuracy": f"bal_acc_{name}"})
        merged = part if merged is None else merged.merge(part, on="subject")

    struggling = {name: struggling_subjects(res, threshold) for name, res in per_model.items()}
    sets = list(struggling.values())
    shared = set.intersection(*sets) if sets else set()
    only = {name: s - set.union(*(o for n, o in struggling.items() if n != name))
            for name, s in struggling.items()}

    # Who struggles in WHICH models. The old "all" / "only one" split hid
    # subjects flagged by two of three models (e.g. both tree models but not
    # logistic regression), so list every flagged subject with its models.
    by_subject = {}
    for name, flagged in struggling.items():
        for subj in flagged:
            by_subject.setdefault(subj, set()).add(name)

    # How much does the choice of model change each person's score?
    bal_cols = [f"bal_acc_{n}" for n in model_names]
    spread = (merged[bal_cols].max(axis=1) - merged[bal_cols].min(axis=1))
    spread.index = merged["subject"]

    return {
        "per_model": per_model,
        "summaries": {name: summarize(res) for name, res in per_model.items()},
        "per_subject_table": merged,
        "struggling": struggling,
        "struggling_in_all_models": shared,
        "struggling_in_one_model_only": only,
        "struggling_by_subject": by_subject,
        # Pearson (pandas default) is dominated by a shared outlier at n=15, so
        # report Spearman alongside it. Neither is a "rank" statistic unless it
        # is the Spearman one -- label them accordingly.
        "balanced_accuracy_correlation": merged[bal_cols].corr().round(2),
        "balanced_accuracy_spearman": merged[bal_cols].corr(method="spearman").round(2),
        "per_subject_model_spread": spread.round(3).sort_values(ascending=False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--report", default="reports/loso_report.txt")
    parser.add_argument("--threshold", type=float, default=STRUGGLE_THRESHOLD)
    parser.add_argument("--models", nargs="*", default=MODEL_NAMES, choices=ALL_MODEL_NAMES,
                        help="models to run (default: random_forest gradient_boosting)")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    out = compare_models(df, model_names=args.models, threshold=args.threshold)

    lines = ["=== Per-subject LOSO results, all models (acc = plain accuracy, bal_acc = balanced accuracy) ===",
             out["per_subject_table"].to_string(index=False), ""]
    for name, summary in out["summaries"].items():
        lines.append(f"{name}: {summary}")
    lines += [
        "",
        f"Struggling threshold: balanced accuracy < {args.threshold} (chance = 0.5)",
        f"Struggling in ALL {len(out['struggling'])} models: {sorted(out['struggling_in_all_models'])}",
        "Struggling in at least one model (subject: models flagged):",
    ]
    for subj, models in sorted(out["struggling_by_subject"].items()):
        lines.append(f"  {subj}: {len(models)} of {len(out['struggling'])} -> {sorted(models)}")
    lines += [
        "",
        "Pearson correlation of per-subject balanced accuracy between models "
        "(n = number of subjects, so imprecise; one subject that fails in two "
        "models can dominate it):",
        out["balanced_accuracy_correlation"].to_string(),
        "",
        "Spearman (rank) correlation of the same values "
        "(1.0 = models order people identically):",
        out["balanced_accuracy_spearman"].to_string(),
        "",
        "Per-subject spread of balanced accuracy across models (max - min); "
        "large values mean the person's score depends on the model chosen:",
        out["per_subject_model_spread"].head(8).to_string(),
    ]
    lines += [
        "",
        "How to read this: subjects that struggle in ALL models point to "
        "something about that person's data (reversed stress response, "
        "motion, sensor contact). Subjects flagged by only some models point "
        "to an interaction between the person's data and those models. A "
        "spread above ~0.2 means the per-person number is not model-invariant.",
    ]

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
