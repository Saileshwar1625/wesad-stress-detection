"""
Hand-engineered features per 60s window, from wrist-only WESAD signals
(BVP, EDA, TEMP, ACC). Classical features, not learned representations --
deliberate, given 15 subjects (see README for why deep learning is the
wrong tool here).

Feature groups:
  - BVP -> heart rate + HRV proxies (peak-to-peak interval statistics).
    NOTE: this is a simple peak detector, not a validated HRV pipeline
    (no artifact rejection, no bandpass filtering first) -- good enough
    for a portfolio-scale demonstration, not good enough to cite as
    clinical-grade HRV. Say so in the writeup, don't imply otherwise.
  - EDA -> tonic/phasic-ish summary stats (mean, std, slope) -- a real
    tonic/phasic decomposition needs a proper deconvolution method
    (e.g. cvxEDA); this uses simple statistics as a lighter-weight stand-in.
  - TEMP -> mean, slope.
  - ACC -> movement magnitude stats (mean, std of the 3-axis vector norm)
    -- motion artifact is a primary EDA/BVP confound on a wrist device, so
    this doubles as a basic data-quality signal, not just an activity proxy.

Usage:
    python pipeline/features.py --input data/processed/windows.pkl \
        --output data/processed/features.parquet
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BVP_HZ = 64
EDA_HZ = 4
TEMP_HZ = 4
ACC_HZ = 32


def simple_peak_interval_features(bvp: np.ndarray, hz: int = BVP_HZ) -> dict:
    """Very simple peak-to-peak interval stats from raw BVP, as an HRV
    proxy. Real HRV pipelines bandpass-filter and reject artifacted beats
    first -- this is intentionally simpler; see module docstring.

    Caught by testing against a clean synthetic 60bpm sine wave: the first
    version of this function (raw-signal peak detection, 300ms refractory)
    overestimated HR by ~25% (75.7 vs 60 bpm) because sample-level noise
    created spurious local maxima between true beats. Fixed with a light
    moving-average smoothing pass and a wider (400ms) refractory window
    before re-testing against the same synthetic signal.
    """
    if len(bvp) < hz * 2:
        return {"hr_mean_bpm": np.nan, "hrv_proxy_std_ms": np.nan}

    smooth_win = max(3, int(hz * 0.08))  # ~80ms smoothing window
    kernel = np.ones(smooth_win) / smooth_win
    smoothed = np.convolve(bvp, kernel, mode="same")

    threshold = np.percentile(smoothed, 75)
    refractory_samples = int(hz * 0.4)  # ~400ms -> caps detectable HR at 150bpm
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if not peaks or (i - peaks[-1]) > refractory_samples:
                peaks.append(i)

    if len(peaks) < 3:
        return {"hr_mean_bpm": np.nan, "hrv_proxy_std_ms": np.nan}

    intervals_sec = np.diff(peaks) / hz
    intervals_ms = intervals_sec * 1000
    hr_bpm = 60.0 / intervals_sec

    return {
        "hr_mean_bpm": float(np.mean(hr_bpm)),
        "hrv_proxy_std_ms": float(np.std(intervals_ms)),
    }


def eda_features(eda: np.ndarray) -> dict:
    if len(eda) < 2:
        return {"eda_mean": np.nan, "eda_std": np.nan, "eda_slope": np.nan}
    x = np.arange(len(eda))
    slope = float(np.polyfit(x, eda, 1)[0]) if len(eda) > 1 else np.nan
    return {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "eda_slope": slope,
    }


def temp_features(temp: np.ndarray) -> dict:
    if len(temp) < 2:
        return {"temp_mean": np.nan, "temp_slope": np.nan}
    x = np.arange(len(temp))
    slope = float(np.polyfit(x, temp, 1)[0])
    return {"temp_mean": float(np.mean(temp)), "temp_slope": slope}


def acc_features(acc: np.ndarray) -> dict:
    if acc.shape[0] < 1:
        return {"acc_mag_mean": np.nan, "acc_mag_std": np.nan}
    magnitude = np.linalg.norm(acc, axis=1)
    return {
        "acc_mag_mean": float(np.mean(magnitude)),
        "acc_mag_std": float(np.std(magnitude)),
    }


def extract_features_for_window(window: dict) -> dict:
    sig = window["raw_signals"]
    row = {
        "subject": window["subject"],
        "window_start_sec": window["window_start_sec"],
        "label": window["label"],
    }
    row.update(simple_peak_interval_features(np.asarray(sig["BVP"])))
    row.update(eda_features(np.asarray(sig["EDA"])))
    row.update(temp_features(np.asarray(sig["TEMP"])))
    row.update(acc_features(np.asarray(sig["ACC"])))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/windows.pkl")
    parser.add_argument("--output", default="data/processed/features.parquet")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        windows = pickle.load(f)

    rows = [extract_features_for_window(w) for w in windows]
    df = pd.DataFrame(rows)

    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        print(f"[WARN] Dropped {n_before - n_after} windows with NaN features "
              f"(usually too few BVP peaks detected) -- check this isn't "
              f"concentrated in one subject before proceeding, that would "
              f"bias the LOSO comparison.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path}")
    print(df.groupby(["subject", "label"]).size())


if __name__ == "__main__":
    main()
