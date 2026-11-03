"""Run directly: python pipeline/test_order_check.py"""
import pandas as pd
from order_check import add_order_groups, stress_response_table


def test_table_and_groups():
    rows = []
    for subj, (ns, st) in {"S1": (30.0, 29.0), "S2": (31.0, 30.0), "S3": (30.0, 31.0)}.items():
        for _ in range(3):
            rows.append({"subject": subj, "label": "non_stress", "temp_mean": ns, "hr_mean_bpm": 60.0, "eda_mean": 1.0, "acc_mag_std": 1.0})
            rows.append({"subject": subj, "label": "stress", "temp_mean": st, "hr_mean_bpm": 70.0, "eda_mean": 2.0, "acc_mag_std": 3.0})
    t = stress_response_table(pd.DataFrame(rows))
    assert t.loc["S1", "temp_mean"] == -1.0 and t.loc["S3", "temp_mean"] == 1.0, t
    assert (t["hr_mean_bpm"] == 10.0).all(), t
    g = add_order_groups(t, {"S1": "x", "S2": "x", "S3": "y"})
    assert g.loc["S1", "order_group"] == "A" and g.loc["S3", "order_group"] == "B", g
    print("test_table_and_groups: OK")


if __name__ == "__main__":
    test_table_and_groups()
    print("\nAll order_check.py sanity tests passed.")
