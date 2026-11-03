"""
Sanity tests for auroc_norm.py against synthetic data where the right answer
is known in advance.

The two cases the diagnostic exists to tell apart:
  - THRESHOLD failure: a subject whose features are all shifted by a constant.
    A linear model ranks their windows perfectly (AUROC ~1) but classifies them
    all as one class (balanced accuracy ~0.5). Per-subject normalization removes
    the shift.
  - RANKING failure: a subject whose stress signature is inverted. AUROC is far
    below 0.5 and normalization cannot fix it.

Also tested: normalization arithmetic, the "no future data" property of earlyN,
zero-IQR handling, single-class subjects, and the reference check.

Run directly: python evaluation/test_auroc_norm.py
"""

import numpy as np
import pandas as pd

from auroc_norm import (FEATURE_COLUMNS, best_threshold_balanced_accuracy,
                        check_against_reference, classify_status, cross_model_correlations,
                        normalize_per_subject, parse_norm, run_loso_with_scores)


def build(n_per_class=30, n_subjects=5, offsets=None, flip=(), seed=0):
    """Each subject: n_per_class non_stress windows first (like a baseline
    segment), then n_per_class stress windows, in time order."""
    rng = np.random.default_rng(seed)
    offsets = offsets or {}
    rows = []
    for i in range(n_subjects):
        name = f"S{i}"
        centers = {"non_stress": 0.0, "stress": 1.0}
        if name in flip:
            centers = {"non_stress": 1.0, "stress": 0.0}
        t = 0
        for label in ["non_stress", "stress"]:
            for _ in range(n_per_class):
                row = {"subject": name, "label": label, "window_start_sec": t}
                for col in FEATURE_COLUMNS:
                    row[col] = rng.normal(loc=centers[label], scale=0.3) + offsets.get(name, 0.0)
                rows.append(row)
                t += 30
    return pd.DataFrame(rows)


def _row(results, subject):
    return results[results["subject"] == subject].iloc[0]


def test_threshold_failure_has_high_auroc_low_balanced_accuracy():
    df = build(offsets={"S4": 3.0})
    res = run_loso_with_scores(df, model_name="logistic_regression", norm="none")
    r = _row(res, "S4")
    assert r["auroc"] >= 0.95, r
    assert r["balanced_accuracy"] <= 0.6, r
    assert r["threshold_gap"] >= 0.3, r
    assert r["status"] == "threshold", r
    # scores are shifted: the model predicts mostly "stress" although prevalence is 0.5
    assert r["mean_p_stress"] > 0.8, r
    print("test_threshold_failure_has_high_auroc_low_balanced_accuracy: OK",
          r[["auroc", "balanced_accuracy", "threshold_gap"]].to_dict())


def test_per_subject_normalization_removes_the_shift():
    df = build(offsets={"S4": 3.0})
    for norm in ["full", "early50"]:
        res = run_loso_with_scores(df, model_name="logistic_regression", norm=norm)
        r = _row(res, "S4")
        assert r["balanced_accuracy"] >= 0.9, (norm, r)
        assert r["auroc"] >= 0.95, (norm, r)
        assert r["status"] == "ok", (norm, r)
    print("test_per_subject_normalization_removes_the_shift: OK")


def test_ranking_failure_survives_normalization():
    df = build(flip=("S4",))
    for norm in ["none", "full"]:
        res = run_loso_with_scores(df, model_name="logistic_regression", norm=norm)
        r = _row(res, "S4")
        assert r["auroc"] < 0.3, (norm, r)
        assert r["status"] == "ranking", (norm, r)
    print("test_ranking_failure_survives_normalization: OK")


def test_normal_subjects_score_well_without_normalization():
    df = build()
    res = run_loso_with_scores(df, model_name="random_forest", norm="none")
    assert (res["auroc"] >= 0.95).all(), res[["subject", "auroc"]]
    assert (res["balanced_accuracy"] >= 0.9).all()
    assert (res["status"] == "ok").all()
    print("test_normal_subjects_score_well_without_normalization: OK")


def test_best_threshold_is_an_upper_bound_and_perfect_when_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert best_threshold_balanced_accuracy(y, perfect) == 1.0
    df = build(offsets={"S4": 3.0})
    res = run_loso_with_scores(df, model_name="logistic_regression", norm="none")
    assert (res["bal_acc_best_threshold"] >= res["balanced_accuracy"] - 1e-9).all()
    print("test_best_threshold_is_an_upper_bound_and_perfect_when_separable: OK")


def test_full_normalization_gives_zero_median_unit_iqr_per_subject():
    df = build(n_subjects=3)
    out = normalize_per_subject(df, FEATURE_COLUMNS, "full")
    for _, g in out.groupby("subject"):
        med = g[FEATURE_COLUMNS].median()
        iqr = g[FEATURE_COLUMNS].quantile(0.75) - g[FEATURE_COLUMNS].quantile(0.25)
        assert np.allclose(med, 0, atol=1e-9)
        assert np.allclose(iqr, 1, atol=1e-9)
    print("test_full_normalization_gives_zero_median_unit_iqr_per_subject: OK")


def test_early_normalization_uses_only_early_windows():
    """Changing windows AFTER the early block must not change how the early
    block is normalized (no future information)."""
    df = build(n_subjects=3)
    base = normalize_per_subject(df, FEATURE_COLUMNS, "early20")
    changed = df.copy()
    late = (changed["subject"] == "S1") & (changed["window_start_sec"] >= 30 * 20)  # after the first 20 windows
    changed.loc[late, FEATURE_COLUMNS] = 1000.0
    again = normalize_per_subject(changed, FEATURE_COLUMNS, "early20")
    early_mask = (df["subject"] == "S1") & (df["window_start_sec"] < 30 * 12)  # first 12 of 60 = 20%
    assert np.allclose(base.loc[early_mask, FEATURE_COLUMNS], again.loc[early_mask, FEATURE_COLUMNS])
    # other subjects untouched entirely
    other = df["subject"] != "S1"
    assert np.allclose(base.loc[other, FEATURE_COLUMNS], again.loc[other, FEATURE_COLUMNS])
    print("test_early_normalization_uses_only_early_windows: OK")


def test_early_window_count_has_a_floor():
    # 40 windows * 10% = 4, below the floor of 5, so the first 5 windows define the statistics
    df = build(n_per_class=20, n_subjects=2)
    out = normalize_per_subject(df, FEATURE_COLUMNS, "early10")
    s0 = out[out["subject"] == "S0"].sort_values("window_start_sec").head(5)
    assert np.allclose(s0[FEATURE_COLUMNS].median(), 0, atol=1e-9)
    print("test_early_window_count_has_a_floor: OK")


def test_none_leaves_features_unchanged():
    df = build(n_subjects=3)
    out = normalize_per_subject(df, FEATURE_COLUMNS, "none")
    assert np.allclose(out[FEATURE_COLUMNS], df[FEATURE_COLUMNS])
    print("test_none_leaves_features_unchanged: OK")


def test_constant_feature_does_not_produce_nan_or_inf():
    df = build(n_subjects=3)
    df["temp_slope"] = 5.0  # zero IQR in every subject
    out = normalize_per_subject(df, FEATURE_COLUMNS, "full")
    assert np.isfinite(out[FEATURE_COLUMNS].to_numpy()).all()
    assert (out["temp_slope"] == 0).all()
    print("test_constant_feature_does_not_produce_nan_or_inf: OK")


def test_single_class_subject_is_flagged_not_scored():
    df = build(n_subjects=4)
    df = df[~((df["subject"] == "S3") & (df["label"] == "stress"))]
    res = run_loso_with_scores(df, model_name="logistic_regression", norm="none")
    r = _row(res, "S3")
    assert np.isnan(r["auroc"]) and np.isnan(r["balanced_accuracy"]) and r["status"] == "n/a"
    assert "one class" in r["note"]
    print("test_single_class_subject_is_flagged_not_scored: OK")


def test_unknown_normalization_and_missing_time_column_raise():
    for bad in ["zscore", "early", "early0", "early100", "early5x"]:
        try:
            parse_norm(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_norm accepted {bad!r}")
    df = build(n_subjects=3).drop(columns=["window_start_sec"])
    try:
        normalize_per_subject(df, FEATURE_COLUMNS, "full")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without window_start_sec")
    # 'none' does not need the time column
    normalize_per_subject(df, FEATURE_COLUMNS, "none")
    print("test_unknown_normalization_and_missing_time_column_raise: OK")


def test_classify_status_rules():
    assert classify_status(0.99, 0.80, 0.99) == "ok"
    assert classify_status(0.40, 0.50, 0.55) == "ranking"
    assert classify_status(0.95, 0.50, 0.95) == "threshold"
    assert classify_status(0.95, 0.65, 0.70) == "mixed"  # gap 0.05 < 0.15
    assert classify_status(0.75, 0.60, 0.80) == "mixed"  # AUROC between the two cutoffs
    assert classify_status(np.nan, np.nan, np.nan) == "n/a"
    print("test_classify_status_rules: OK")


def test_reference_check_flags_a_mismatch():
    ok = pd.DataFrame({"model": ["random_forest"] * 3, "norm": ["none"] * 3,
                       "balanced_accuracy": [0.846, 0.846, 0.846]})
    bad = ok.assign(balanced_accuracy=[0.5, 0.5, 0.5])
    assert any("OK" in line for line in check_against_reference(ok))
    assert any("DIFFERS" in line for line in check_against_reference(bad))
    print("test_reference_check_flags_a_mismatch: OK")


def test_cross_model_correlations_pearson_and_spearman_differ_on_a_monotone_curve():
    # Same ordering of subjects in both models (Spearman = 1) but a strongly
    # nonlinear relationship (Pearson < 1).
    vals = [0.5, 0.52, 0.55, 0.6, 0.7, 0.95]
    subjects = [f"S{i}" for i in range(6)]
    rows = []
    for subj, a in zip(subjects, vals):
        rows.append({"model": "random_forest", "norm": "none", "subject": subj,
                     "auroc": a, "balanced_accuracy": a})
        rows.append({"model": "gradient_boosting", "norm": "none", "subject": subj,
                     "auroc": a ** 8, "balanced_accuracy": a ** 8})
    res = pd.DataFrame(rows)
    corr = cross_model_correlations(res, "none")
    assert corr[("AUROC", "Spearman")].loc["rf", "gb"] == 1.0
    assert corr[("AUROC", "Pearson")].loc["rf", "gb"] < 0.99
    assert set(k[0] for k in corr) == {"AUROC", "balanced accuracy"}
    # a single model gives nothing to correlate
    assert cross_model_correlations(res[res["model"] == "random_forest"], "none") == {}
    print("test_cross_model_correlations_pearson_and_spearman_differ_on_a_monotone_curve: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
