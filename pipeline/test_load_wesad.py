"""
Sanity tests for load_wesad.py against a synthetic fixture matching WESAD's
documented pickle schema. Run directly: python pipeline/test_load_wesad.py
"""

import pickle
import numpy as np
from pathlib import Path
import tempfile
import shutil

from load_wesad import (
    extract_windows_for_subject, majority_label_for_window,
    BASELINE, STRESS, AMUSEMENT, MEDITATION, LABEL_SAMPLING_HZ,
)


def build_synthetic_subject(subject_id="S99", segment_seconds=120):
    """Four back-to-back segments: baseline, stress, amusement, meditation,
    each `segment_seconds` long -- matches the documented WESAD schema."""
    total_seconds = segment_seconds * 4
    label = np.concatenate([
        np.full(segment_seconds * LABEL_SAMPLING_HZ, BASELINE),
        np.full(segment_seconds * LABEL_SAMPLING_HZ, STRESS),
        np.full(segment_seconds * LABEL_SAMPLING_HZ, AMUSEMENT),
        np.full(segment_seconds * LABEL_SAMPLING_HZ, MEDITATION),
    ])

    rng = np.random.default_rng(0)
    wrist = {
        "BVP": rng.normal(size=total_seconds * 64),
        "ACC": rng.normal(size=(total_seconds * 32, 3)),
        "EDA": rng.normal(size=total_seconds * 4),
        "TEMP": rng.normal(size=total_seconds * 4) + 32,
    }
    return {"subject": subject_id, "signal": {"wrist": wrist, "chest": {}}, "label": label}


def test_majority_label_for_window_pure_segment():
    label = np.full(700 * 60, BASELINE)
    code = majority_label_for_window(label, 0, 700 * 60)
    assert code == BASELINE, code
    print("test_majority_label_for_window_pure_segment: OK")


def test_majority_label_for_window_mixed_rejects():
    # 40% baseline, 35% stress, 25% transient -- no class has >=90%, must reject
    label = np.concatenate([
        np.full(400, BASELINE), np.full(350, STRESS), np.full(250, 0),
    ])
    code = majority_label_for_window(label, 0, len(label))
    assert code is None, code
    print("test_majority_label_for_window_mixed_rejects: OK")


def test_meditation_excluded_from_output():
    data = build_synthetic_subject(segment_seconds=120)
    windows = extract_windows_for_subject(data)
    assert len(windows) > 0, "expected some windows"
    labels = {w["label"] for w in windows}
    assert labels <= {"stress", "non_stress"}, labels

    # No window should be centered in the meditation segment (seconds 360-480)
    meditation_windows = [w for w in windows if 360 <= w["window_start_sec"] < 480 - 60]
    assert len(meditation_windows) == 0, (
        f"meditation segment leaked into output: {meditation_windows}"
    )
    print(f"test_meditation_excluded_from_output: OK ({len(windows)} windows)")


def test_window_label_matches_segment():
    data = build_synthetic_subject(segment_seconds=120)
    windows = extract_windows_for_subject(data)
    by_start = {w["window_start_sec"]: w["label"] for w in windows}

    # A window fully inside the baseline segment (0-120s) should be non_stress
    assert by_start.get(30.0) == "non_stress", by_start.get(30.0)
    # A window fully inside the stress segment (120-240s) should be stress
    assert by_start.get(150.0) == "stress", by_start.get(150.0)
    # A window fully inside the amusement segment (240-360s) should be non_stress
    assert by_start.get(270.0) == "non_stress", by_start.get(270.0)
    print("test_window_label_matches_segment: OK")


def test_window_signal_shapes():
    data = build_synthetic_subject(segment_seconds=120)
    windows = extract_windows_for_subject(data)
    w = windows[0]
    sig = w["raw_signals"]
    assert len(sig["BVP"]) == 60 * 64, len(sig["BVP"])
    assert sig["ACC"].shape == (60 * 32, 3), sig["ACC"].shape
    assert len(sig["EDA"]) == 60 * 4, len(sig["EDA"])
    print("test_window_signal_shapes: OK")


def test_end_to_end_main():
    """Smoke test the full CLI: write a synthetic .pkl, run main(), check output."""
    import load_wesad as lw

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        raw_dir = tmp_dir / "raw"
        raw_dir.mkdir()
        data = build_synthetic_subject(subject_id="S99", segment_seconds=120)
        with open(raw_dir / "S99.pkl", "wb") as f:
            pickle.dump(data, f)

        out_path = tmp_dir / "processed" / "windows.pkl"
        report_path = tmp_dir / "reports" / "load_report.txt"

        import sys
        old_argv = sys.argv
        sys.argv = ["load_wesad.py", "--input", str(raw_dir),
                    "--output", str(out_path), "--report", str(report_path)]
        try:
            lw.main()
        finally:
            sys.argv = old_argv

        assert out_path.exists()
        with open(out_path, "rb") as f:
            windows = pickle.load(f)
        assert len(windows) > 0
        assert report_path.exists()
        assert "S99" in report_path.read_text()
        print(f"test_end_to_end_main: OK ({len(windows)} windows written)")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_majority_label_for_window_pure_segment()
    test_majority_label_for_window_mixed_rejects()
    test_meditation_excluded_from_output()
    test_window_label_matches_segment()
    test_window_signal_shapes()
    test_end_to_end_main()
    print("\nAll load_wesad.py sanity tests passed.")
