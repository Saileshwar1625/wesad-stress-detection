"""
Sanity tests for ablation.py. Fixture: only the EDA features carry the stress
signal; all others are pure noise. Expected: eda_only ~ all >> chance, while
acc_only / temp_only sit near chance (0.5).

Run directly: python evaluation/test_ablation.py
"""

import numpy as np
import pandas as pd

from ablation import FEATURE_SETS, run_ablation, summarize_ablation, subject_view
from train_eval import FEATURE_COLUMNS


def build_eda_only_signal(n_subjects=5, n_per_class=30, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for label, shift in (("non_stress", 0.0), ("stress", 1.5)):
            for _ in range(n_per_class):
                row = {"subject": f"S{i}", "label": label}
                for col in FEATURE_COLUMNS:
                    informative = col.startswith("eda_")
                    row[col] = rng.normal(loc=shift if informative else 0.0, scale=0.5 if informative else 1.0)
                rows.append(row)
    return pd.DataFrame(rows)


def test_feature_sets_only_use_known_columns():
    for name, cols in FEATURE_SETS.items():
        assert set(cols) <= set(FEATURE_COLUMNS), name
        assert len(cols) > 0, name
    print("test_feature_sets_only_use_known_columns: OK")


def test_ablation_isolates_the_informative_group():
    df = build_eda_only_signal()
    long_df = run_ablation(df, model_names=["random_forest"])
    summary = summarize_ablation(long_df).set_index("feature_set")["mean_bal_acc"]
    assert summary["eda_only"] > 0.85, summary
    assert summary["all"] > 0.85, summary
    assert summary["acc_only"] < 0.65, summary
    assert summary["temp_only"] < 0.65, summary
    print("test_ablation_isolates_the_informative_group: OK", summary.to_dict())


def test_subject_view_shape():
    df = build_eda_only_signal()
    long_df = run_ablation(df, feature_sets={"all": FEATURE_SETS["all"], "eda_only": FEATURE_SETS["eda_only"]},
                           model_names=["random_forest"])
    view = subject_view(long_df, ["S0", "S1"])
    assert list(view.index) == ["all", "eda_only"] or set(view.index) == {"all", "eda_only"}, view.index
    assert view.shape == (2, 2), view.shape
    print("test_subject_view_shape: OK")


if __name__ == "__main__":
    test_feature_sets_only_use_known_columns()
    test_ablation_isolates_the_informative_group()
    test_subject_view_shape()
    print("\nAll ablation.py sanity tests passed.")
