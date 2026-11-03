"""
Sanity tests for train_eval.py's LOSO logic, against a synthetic feature
set where the "correct" answer is known in advance: four subjects with a
clean, consistent stress-vs-non_stress signal, and one deliberately
atypical subject whose signal is inverted (simulating a genuine
physiological non-responder). The point of this test is to confirm LOSO
actually identifies the atypical subject as the worst performer --
exactly the kind of result the real analysis is supposed to surface.

Run directly: python evaluation/test_train_eval.py
"""

import numpy as np
import pandas as pd

from train_eval import (run_loso, summarize, compare_models, struggling_subjects,
                        make_model, FEATURE_COLUMNS)


def build_synthetic_features(n_per_class=30, n_normal_subjects=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def make_rows(subject, flip=False):
        # Normal subjects: non_stress features cluster near 0, stress near 1.
        # The atypical subject has the two labels' feature signatures
        # SWAPPED, not just dampened -- a model trained on normal subjects
        # should then actively mispredict both classes for this subject
        # (confusing stress-for-non_stress and vice versa), landing below
        # chance accuracy. That's what "a genuine physiological
        # non-responder" should look like to the classifier, not just a
        # washed-out version of the normal signal.
        centers = {"non_stress": 0, "stress": 1}
        if flip:
            centers = {"non_stress": 1, "stress": 0}
        for label, shift in centers.items():
            for _ in range(n_per_class):
                row = {"subject": subject, "label": label}
                for col in FEATURE_COLUMNS:
                    row[col] = rng.normal(loc=shift, scale=0.3)
                rows.append(row)

    for i in range(n_normal_subjects):
        make_rows(f"S{i}", flip=False)
    make_rows("S_ATYPICAL", flip=True)  # inverted signal -- a genuine non-responder

    return pd.DataFrame(rows)


def test_loso_runs_and_covers_all_subjects():
    df = build_synthetic_features()
    results = run_loso(df)
    assert set(results["subject"]) == set(df["subject"].unique())
    print("test_loso_runs_and_covers_all_subjects: OK")


def test_normal_subjects_score_well():
    df = build_synthetic_features()
    results = run_loso(df)
    normal = results[results["subject"] != "S_ATYPICAL"]
    assert (normal["accuracy"] >= 0.8).all(), normal
    print("test_normal_subjects_score_well: OK", normal["accuracy"].tolist())


def test_atypical_subject_is_identified_as_worst():
    """This is the actual point of the module: LOSO trained only on the
    four 'normal' subjects should fail to generalize to the inverted
    S_ATYPICAL subject, and summarize() should flag it as the worst."""
    df = build_synthetic_features()
    results = run_loso(df)
    summary = summarize(results)
    assert summary["worst_subject"] == "S_ATYPICAL", summary
    atypical_acc = results.loc[results["subject"] == "S_ATYPICAL", "balanced_accuracy"].iloc[0]
    assert atypical_acc < 0.5, atypical_acc
    print("test_atypical_subject_is_identified_as_worst: OK", summary)


def test_single_class_held_out_subject_flagged_not_scored():
    df = build_synthetic_features(n_normal_subjects=3)
    # Force one subject to have only one class in its data
    df.loc[df["subject"] == "S0", "label"] = "non_stress"
    results = run_loso(df)
    row = results[results["subject"] == "S0"].iloc[0]
    assert pd.isna(row["accuracy"]), row
    assert "only one class" in row["note"]
    print("test_single_class_held_out_subject_flagged_not_scored: OK")


def test_gradient_boosting_runs_and_finds_atypical():
    df = build_synthetic_features()
    results = run_loso(df, model_name="gradient_boosting")
    summary = summarize(results)
    assert summary["worst_subject"] == "S_ATYPICAL", summary
    normal = results[results["subject"] != "S_ATYPICAL"]
    assert (normal["accuracy"] >= 0.8).all(), normal
    print("test_gradient_boosting_runs_and_finds_atypical: OK", summary["mean_accuracy"])


def test_compare_models_shared_failure():
    """The data-driven failure (S_ATYPICAL's reversed response) should show
    up as struggling in BOTH models -- that's the signal that the problem
    lives in the person's data, not the model choice."""
    df = build_synthetic_features()
    out = compare_models(df)
    assert out["struggling_in_all_models"] == {"S_ATYPICAL"}, out["struggling"]
    for name, only in out["struggling_in_one_model_only"].items():
        assert only == set(), (name, only)
    table = out["per_subject_table"]
    assert set(table.columns) == {
        "subject", "acc_random_forest", "acc_gradient_boosting",
        "bal_acc_random_forest", "bal_acc_gradient_boosting"}, table.columns
    assert len(table) == df["subject"].nunique()
    assert out["struggling_by_subject"]["S_ATYPICAL"] == {"random_forest", "gradient_boosting"}, out["struggling_by_subject"]
    assert out["balanced_accuracy_correlation"].shape == (2, 2)
    sp = out["balanced_accuracy_spearman"]
    assert sp.shape == (2, 2) and (sp.values.diagonal() == 1.0).all(), sp
    assert out["per_subject_model_spread"].index.is_unique
    print("test_compare_models_shared_failure: OK")


def test_struggling_threshold_respected():
    res = pd.DataFrame({"subject": ["A", "B", "C"],
                        "accuracy": [0.95, 0.69, np.nan],
                        "balanced_accuracy": [0.95, 0.69, np.nan],
                        "f1": [0.9, 0.6, np.nan]})
    assert struggling_subjects(res, threshold=0.7) == {"B"}
    assert struggling_subjects(res, threshold=0.6) == set()
    print("test_struggling_threshold_respected: OK")


def test_always_majority_class_is_flagged_despite_ok_accuracy():
    """The reason struggling is judged on balanced accuracy: a subject whose
    plain accuracy is 0.71 only because the model always said 'non_stress'
    must still be flagged."""
    res = pd.DataFrame({"subject": ["X"], "accuracy": [0.71], "balanced_accuracy": [0.5], "f1": [0.0]})
    assert struggling_subjects(res, threshold=0.7) == {"X"}
    print("test_always_majority_class_is_flagged_despite_ok_accuracy: OK")


def test_balanced_accuracy_and_majority_baseline_reported():
    df = build_synthetic_features()
    res = run_loso(df)
    assert {"balanced_accuracy", "majority_baseline"} <= set(res.columns)
    assert (res["majority_baseline"] == 0.5).all(), res["majority_baseline"].tolist()  # fixture is balanced
    print("test_balanced_accuracy_and_majority_baseline_reported: OK")


def test_logistic_regression_runs_and_finds_atypical():
    df = build_synthetic_features()
    results = run_loso(df, model_name="logistic_regression")
    summary = summarize(results)
    assert summary["worst_subject"] == "S_ATYPICAL", summary
    normal = results[results["subject"] != "S_ATYPICAL"]
    assert (normal["accuracy"] >= 0.8).all(), normal
    out = compare_models(df, model_names=["random_forest", "logistic_regression"])
    assert out["struggling_in_all_models"] == {"S_ATYPICAL"}, out["struggling"]
    print("test_logistic_regression_runs_and_finds_atypical: OK", summary["mean_balanced_accuracy"])


def test_ensemble_runs_and_scores_normal_subjects():
    df = build_synthetic_features()
    results = run_loso(df, model_name="ensemble")
    normal = results[results["subject"] != "S_ATYPICAL"]
    assert (normal["accuracy"] >= 0.8).all(), normal
    assert summarize(results)["worst_subject"] == "S_ATYPICAL"
    print("test_ensemble_runs_and_scores_normal_subjects: OK")


def test_unknown_model_name_raises():
    try:
        make_model("svm")
    except ValueError:
        print("test_unknown_model_name_raises: OK")
        return
    raise AssertionError("expected ValueError for unknown model name")


if __name__ == "__main__":
    test_loso_runs_and_covers_all_subjects()
    test_normal_subjects_score_well()
    test_atypical_subject_is_identified_as_worst()
    test_single_class_held_out_subject_flagged_not_scored()
    test_gradient_boosting_runs_and_finds_atypical()
    test_compare_models_shared_failure()
    test_struggling_threshold_respected()
    test_always_majority_class_is_flagged_despite_ok_accuracy()
    test_balanced_accuracy_and_majority_baseline_reported()
    test_logistic_regression_runs_and_finds_atypical()
    test_ensemble_runs_and_scores_normal_subjects()
    test_unknown_model_name_raises()
    print("\nAll train_eval.py sanity tests passed.")
