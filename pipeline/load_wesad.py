"""
Parse WESAD per-subject .pkl files into fixed-length, labeled windows of
wrist-only signal, collapsed to binary stress/non-stress.

Written against WESAD's documented pickle structure (corroborated across
multiple independent sources describing the dataset, not verified against
a live downloaded file in this sandbox — no network access here):

    {
        "subject": "S2",
        "signal": {
            "wrist": {"BVP": <64 Hz array>, "ACC": <32 Hz, 3-axis array>,
                       "EDA": <4 Hz array>, "TEMP": <4 Hz array>},
            "chest": {"ECG": <700 Hz array>, "Resp": <700 Hz array>, ...}
        },
        "label": <700 Hz array of ints>  # 0=undefined/transient, 1=baseline,
                                           # 2=stress, 3=amusement,
                                           # 4=meditation, 5-7=undefined
    }

Run this yourself against a real subject file FIRST and read
reports/load_report.txt before trusting anything downstream — if the real
schema differs (WESAD's exact key casing/nesting has been described
inconsistently across sources even in documentation found for this repo),
fix the field access here rather than silently coercing.

Usage:
    python pipeline/load_wesad.py --input data/raw --output data/processed/windows.parquet
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

LABEL_SAMPLING_HZ = 700  # the 'label' array is sampled at the chest/ECG rate

# Only these label codes are usable; everything else (0, 5, 6, 7) is
# transient/undefined per the WESAD protocol and must be dropped, not
# coerced into one of the real classes.
BASELINE, STRESS, AMUSEMENT, MEDITATION = 1, 2, 3, 4
NONSTRESS_CODES = {BASELINE, AMUSEMENT}
STRESS_CODES = {STRESS}
# MEDITATION is deliberately excluded from both classes for the binary task
# -- it's a distinct physiological/affective state, not "non-stress" in the
# same sense as a resting baseline, and folding it in without justification
# would quietly change what "non-stress" means.

WRIST_RATES_HZ = {"BVP": 64, "ACC": 32, "EDA": 4, "TEMP": 4}

WINDOW_SECONDS = 60
WINDOW_STRIDE_SECONDS = 30  # 50% overlap, standard for this kind of signal


def load_subject_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        # WESAD pickles were written under Python 2/older numpy in the
        # original release; latin1 encoding avoids a common unpickling
        # failure on signal arrays saved that way.
        return pickle.load(f, encoding="latin1")


def majority_label_for_window(label_array: np.ndarray, start_sample: int, end_sample: int) -> int | None:
    """Return the majority label code within [start_sample, end_sample) of
    the 700Hz label array, restricted to BASELINE/STRESS/AMUSEMENT -- or
    None if the window doesn't have a clear majority in an allowed class
    (e.g. it straddles a transition, or is mostly transient/undefined)."""
    segment = label_array[start_sample:end_sample]
    if len(segment) == 0:
        return None
    values, counts = np.unique(segment, return_counts=True)
    allowed = {BASELINE, STRESS, AMUSEMENT}
    best_code, best_count = None, 0
    for v, c in zip(values, counts):
        if v in allowed and c > best_count:
            best_code, best_count = v, c
    if best_code is None:
        return None
    # Require a real majority within the window, not just a plurality --
    # a window that's 40% baseline / 35% stress / 25% transient is not a
    # clean example of either class and should be dropped, not forced in.
    if best_count / len(segment) < 0.9:
        return None
    return int(best_code)


def extract_windows_for_subject(subject_data: dict) -> list[dict]:
    subject_id = subject_data.get("subject", "unknown")
    label_array = np.asarray(subject_data["label"])
    wrist = subject_data["signal"]["wrist"]

    total_seconds = len(label_array) / LABEL_SAMPLING_HZ
    window_starts = np.arange(0, total_seconds - WINDOW_SECONDS, WINDOW_STRIDE_SECONDS)

    windows = []
    for w_start_sec in window_starts:
        w_end_sec = w_start_sec + WINDOW_SECONDS
        label_start = int(w_start_sec * LABEL_SAMPLING_HZ)
        label_end = int(w_end_sec * LABEL_SAMPLING_HZ)

        code = majority_label_for_window(label_array, label_start, label_end)
        if code is None:
            continue
        if code not in NONSTRESS_CODES and code not in STRESS_CODES:
            continue  # meditation or anything else -- excluded by design, see module docstring

        window_signals = {}
        valid = True
        for channel, rate in WRIST_RATES_HZ.items():
            arr = np.asarray(wrist[channel])
            s = int(w_start_sec * rate)
            e = int(w_end_sec * rate)
            if channel == "ACC":
                # 3-axis, shape (n_samples, 3) -- confirmed on real S2.pkl.
                seg = arr[s:e, :]
            else:
                # Real WESAD stores BVP/EDA/TEMP as (n_samples, 1), not flat
                # (n_samples,) -- confirmed on real S2.pkl. Flatten so the
                # feature code always receives 1-D signals.
                seg = arr.reshape(-1)[s:e]
            if seg is None or len(seg) == 0:
                valid = False
                break
            window_signals[channel] = seg
        if not valid:
            continue

        windows.append({
            "subject": subject_id,
            "window_start_sec": float(w_start_sec),
            "label": "stress" if code in STRESS_CODES else "non_stress",
            "raw_signals": window_signals,
        })
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed/windows.pkl")
    parser.add_argument("--report", default="reports/load_report.txt")
    args = parser.parse_args()

    in_dir = Path(args.input)
    # rglob, not glob: the WESAD download unzips as WESAD/S2/S2.pkl,
    # WESAD/S3/S3.pkl, ... so the files sit one or two folders deep.
    # Searching recursively means the unzipped folder can be dropped into
    # data/raw/ as-is, without moving 15 files out by hand.
    subject_files = sorted(in_dir.rglob("S*.pkl"), key=lambda p: p.name)
    if not subject_files:
        log.error(f"No S*.pkl files found under {in_dir} -- download WESAD, "
                   f"unzip it, and put the extracted WESAD folder inside it.")
        return
    log.info(f"Found {len(subject_files)} subject files (WESAD should have 15).")

    all_windows = []
    per_subject_counts = {}
    for f in subject_files:
        try:
            data = load_subject_pickle(f)
            windows = extract_windows_for_subject(data)
        except Exception as e:
            log.error(f"Failed to parse {f.name}: {e}. If this is a schema "
                      f"mismatch, fix load_subject_pickle/extract_windows_for_subject "
                      f"against your real file's actual structure.")
            continue
        all_windows.extend(windows)
        labels = [w["label"] for w in windows]
        per_subject_counts[f.stem] = {
            "n_windows": len(windows),
            "n_stress": labels.count("stress"),
            "n_non_stress": labels.count("non_stress"),
        }
        log.info(f"{f.stem}: {len(windows)} windows "
                 f"({labels.count('stress')} stress / {labels.count('non_stress')} non-stress)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_windows, f)

    report_lines = [
        f"Subjects parsed: {len(per_subject_counts)}",
        f"Total windows: {len(all_windows)}",
        "",
        "Per-subject window/class counts:",
    ]
    for sid, counts in per_subject_counts.items():
        report_lines.append(f"  {sid}: {counts}")
    report_lines.append("")
    report_lines.append(
        "Sanity check: WESAD has 15 subjects. A per-subject class imbalance "
        "(e.g. far more non-stress than stress windows, since the stress "
        "protocol segment is shorter than baseline) is EXPECTED -- note it, "
        "don't 'fix' it by discarding data without saying so in the writeup."
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    log.info(f"Wrote {len(all_windows)} windows to {out_path}")
    log.info(f"Wrote sanity report to {report_path} -- READ IT before Phase 2.")


if __name__ == "__main__":
    main()
