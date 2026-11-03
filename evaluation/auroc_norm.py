"""
Does a "failing" subject fail to RANK stress above non-stress, or only fail
at the fixed 0.5 decision threshold?  And does per-subject normalization
change that?

Background
----------
train_eval.py scores each held-out subject with balanced accuracy at the
model's default threshold. A later paper (Akkaya 2026, BMC Med Inform
Decis Mak) reports per-subject WESAD AUROC >= 0.927 for every subject, using
per-subject normalization. AUROC does not change if a subject's scores are all
shifted up or down, but balanced accuracy at a fixed threshold does. So some
of the near-chance subjects in train_eval.py may be THRESHOLD failures (the
subject's windows are ranked well but sit on the wrong side of the cutoff) and
not RANKING failures. This script measures both, per subject, side by side.

What it reports per (model, normalization, held-out subject)
------------------------------------------------------------
  auroc                    ranking quality; 0.5 = chance; threshold-free
  balanced_accuracy        at the model's default threshold (same number as
                           train_eval.py when normalization is "none")
  bal_acc_best_threshold   best balanced accuracy over ALL thresholds on that
                           subject's own scores. This uses the subject's
                           labels, so it is an oracle upper bound, a diagnostic
                           and not a result you could achieve without labels
  threshold_gap            bal_acc_best_threshold - balanced_accuracy
  mean_p_stress, prevalence  mean predicted P(stress) vs the true stress share;
                           a large mismatch means the scores are shifted

Normalization options (--norms), all PER SUBJECT, label-free, median/IQR
(a zero IQR is replaced by 1):
  none      raw features (what train_eval.py does)
  full      statistics over the subject's WHOLE recording. Not available
            in deployment (it needs future windows). Same choice as Akkaya
  earlyN    statistics over the subject's first N percent of windows (at
            least 5), so only data from the start of the recording is used.
            CAVEAT: every WESAD recording starts with a rest/baseline
            segment, so early windows are baseline-dominated. In deployment
            you would have to know the person is at rest during that period.

The "status" column labels each subject. These rules are judgment calls,
declared before looking at results; the raw numbers are always printed too:
  ok         balanced accuracy >= STRUGGLE_THRESHOLD (0.7)
  ranking    below 0.7 and AUROC < 0.70 (scores do not rank the classes)
  threshold  below 0.7, AUROC >= 0.85 and gap >= 0.15 (ranks well, wrong cutoff)
  mixed      everything else below 0.7
Per subject there are only ~35 independent windows, so every per-subject
number here is noisy; treat differences of ~0.1 as noise.

Usage:
    python evaluation/auroc_norm.py --features data/processed/features.parquet
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from train_eval import FEATURE_COLUMNS, STRUGGLE_THRESHOLD, make_model

POSITIVE_LABEL = "stress"
MIN_EARLY_WINDOWS = 5

# Pattern rules (judgment calls; see module docstring).
RANK_FAIL_AUROC = 0.70
THRESHOLD_OK_AUROC = 0.85
THRESHOLD_GAP = 0.15

DEFAULT_MODELS = ["random_forest", "gradient_boosting", "logistic_regression"]
DEFAULT_NORMS = ["none", "full", "early50", "early20", "early10"]
DEFAULT_FOCUS = ["S14", "S5", "S15", "S2", "S9"]
SHORT = {"random_forest": "rf", "gradient_boosting": "gb", "logistic_regression": "lr",
         "ensemble": "ens"}

# Mean balanced accuracy of the earlier train_eval.py run on the real data
# (no normalization). The "none" rows here must reproduce these; if they do
# not, something differs (different features.parquet, library versions).
REFERENCE_MEAN_BAL_ACC = {"random_forest": 0.846, "gradient_boosting": 0.833,
                          "logistic_regression": 0.806}
REFERENCE_TOLERANCE = 0.005


def parse_norm(name: str):
    """'none' | 'full' | 'earlyN' (N = 1..99 percent) -> (mode, fraction)."""
    if name == "none":
        return "none", None
    if name == "full":
        return "full", None
    m = re.fullmatch(r"early(\d{1,2})", name)
    if m and 0 < int(m.group(1)) < 100:
        return "early", int(m.group(1)) / 100
    raise ValueError(f"Unknown normalization '{name}'. Use none, full or earlyN (e.g. early20).")


def _robust_stats(values: pd.DataFrame):
    med = values.median()
    iqr = values.quantile(0.75) - values.quantile(0.25)
    iqr = iqr.where(iqr > 0, 1.0)  # zero (or NaN) IQR -> 1, so a constant column does not blow up
    return med, iqr


def normalize_per_subject(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
                          norm: str = "none") -> pd.DataFrame:
    """Return a copy of df whose feature columns are (x - median) / IQR, with the
    median and IQR computed separately for each subject. No labels are used."""
    mode, frac = parse_norm(norm)
    out = df.reset_index(drop=True).copy()
    if mode == "none":
        return out
    if "window_start_sec" not in out.columns:
        raise ValueError("Per-subject normalization needs a 'window_start_sec' column "
                         "(it orders each subject's windows in time).")
    out[feature_columns] = out[feature_columns].astype(float)
    for _, idx in out.groupby("subject").groups.items():
        sub = out.loc[idx].sort_values("window_start_sec")
        if mode == "full":
            ref = sub
        else:
            k = min(len(sub), max(MIN_EARLY_WINDOWS, math.ceil(frac * len(sub))))
            ref = sub.iloc[:k]
        med, iqr = _robust_stats(ref[feature_columns])
        out.loc[sub.index, feature_columns] = (sub[feature_columns] - med) / iqr
    return out


def stress_probability(model, X) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(POSITIVE_LABEL)]


def best_threshold_balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    """max over thresholds of (TPR + TNR) / 2 = (1 + max(TPR - FPR)) / 2. Oracle: uses labels."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float((1.0 + np.max(tpr - fpr)) / 2.0)


def classify_status(auroc: float, bal_acc: float, bal_acc_best: float) -> str:
    if np.isnan(auroc) or np.isnan(bal_acc):
        return "n/a"
    if bal_acc >= STRUGGLE_THRESHOLD:
        return "ok"
    if auroc < RANK_FAIL_AUROC:
        return "ranking"
    if auroc >= THRESHOLD_OK_AUROC and (bal_acc_best - bal_acc) >= THRESHOLD_GAP:
        return "threshold"
    return "mixed"


def run_loso_with_scores(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
                         model_name: str = "random_forest", norm: str = "none") -> pd.DataFrame:
    """LOSO for one (model, normalization). Each subject, training and held-out,
    is normalized only with its own statistics."""
    subjects = sorted(df["subject"].unique())
    if len(subjects) < 2:
        raise ValueError(f"LOSO needs at least 2 subjects; got {len(subjects)}.")
    data = normalize_per_subject(df, feature_columns, norm)

    rows = []
    for held_out in subjects:
        train = data[data["subject"] != held_out]
        test = data[data["subject"] == held_out]
        y = (test["label"] == POSITIVE_LABEL).astype(int).to_numpy()
        base = {"model": model_name, "norm": norm, "subject": held_out,
                "n_windows": len(test), "prevalence": round(float(y.mean()), 4)}
        if y.min() == y.max():
            rows.append({**base, "auroc": np.nan, "balanced_accuracy": np.nan,
                         "bal_acc_best_threshold": np.nan, "threshold_gap": np.nan,
                         "mean_p_stress": np.nan, "status": "n/a",
                         "note": "held-out subject has only one class -- metrics not meaningful"})
            continue

        model = make_model(model_name)
        model.fit(train[feature_columns], train["label"])
        preds = model.predict(test[feature_columns])  # same call train_eval.py uses
        p = stress_probability(model, test[feature_columns])

        auroc = float(roc_auc_score(y, p))
        bal = float(balanced_accuracy_score(test["label"], preds))
        best = best_threshold_balanced_accuracy(y, p)
        rows.append({**base, "auroc": round(auroc, 4), "balanced_accuracy": round(bal, 4),
                     "bal_acc_best_threshold": round(best, 4),
                     "threshold_gap": round(best - bal, 4),
                     "mean_p_stress": round(float(p.mean()), 4),
                     "status": classify_status(auroc, bal, best), "note": ""})
    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame, models: list[str] = DEFAULT_MODELS,
            norms: list[str] = DEFAULT_NORMS,
            feature_columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    parts = []
    for model_name in models:
        for norm in norms:
            print(f"  running {model_name} / {norm} ...", flush=True)
            parts.append(run_loso_with_scores(df, feature_columns, model_name, norm))
    return pd.concat(parts, ignore_index=True)


def summarize_variants(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, norm), g in results.groupby(["model", "norm"], sort=False):
        v = g.dropna(subset=["auroc"])
        if not len(v):
            continue
        lo_a, lo_b = v.loc[v["auroc"].idxmin()], v.loc[v["balanced_accuracy"].idxmin()]
        rows.append({
            "model": SHORT.get(model, model), "norm": norm,
            "mean_auroc": v["auroc"].mean(), "min_auroc": lo_a["auroc"],
            "min_auroc_subj": lo_a["subject"],
            "mean_bal_acc": v["balanced_accuracy"].mean(), "min_bal_acc": lo_b["balanced_accuracy"],
            "min_bal_acc_subj": lo_b["subject"],
            "n_below_0.7": int((v["status"] != "ok").sum()),
            "n_threshold": int((v["status"] == "threshold").sum()),
            "n_ranking": int((v["status"] == "ranking").sum()),
            "n_mixed": int((v["status"] == "mixed").sum()),
            "mean_gap": v["threshold_gap"].mean(),
        })
    return pd.DataFrame(rows)


def cross_model_correlations(results: pd.DataFrame, norm: str = "none") -> dict:
    """Pearson and Spearman correlation of per-subject AUROC and of per-subject
    balanced accuracy across models, for one normalization. Pearson is pulled
    around by a single subject that fails in several models; Spearman uses only
    the ordering. With 15 subjects both are imprecise."""
    g = results[results["norm"] == norm]
    out = {}
    for metric, label in [("auroc", "AUROC"), ("balanced_accuracy", "balanced accuracy")]:
        wide = g.pivot(index="subject", columns="model", values=metric).dropna()
        wide.columns = [SHORT.get(c, c) for c in wide.columns]
        if wide.shape[1] < 2 or len(wide) < 3:
            continue
        out[(label, "Pearson")] = wide.corr().round(2)
        out[(label, "Spearman")] = wide.corr(method="spearman").round(2)
    return out


def check_against_reference(results: pd.DataFrame) -> list[str]:
    lines = []
    for model, ref in REFERENCE_MEAN_BAL_ACC.items():
        g = results[(results["model"] == model) & (results["norm"] == "none")]
        if g.empty:
            continue
        got = float(g["balanced_accuracy"].mean())
        status = "OK" if abs(got - ref) <= REFERENCE_TOLERANCE else "DIFFERS"
        lines.append(f"  {SHORT[model]}: mean balanced accuracy {got:.3f} vs earlier train_eval.py "
                     f"{ref:.3f} -> {status}")
        if status == "DIFFERS":
            lines.append("    (check that features.parquet is the same file and the library "
                         "versions match; the other numbers below are not comparable to the "
                         "earlier report until this matches)")
    return lines or ["  (no reference models in this run)"]


def _cell(row) -> str:
    return f"{row['auroc']:.2f}/{row['balanced_accuracy']:.2f}"


def build_report(results: pd.DataFrame, focus: list[str]) -> list[str]:
    fmt = {"float_format": lambda x: f"{x:.3f}"}
    lines = ["=== Per-subject AUROC vs balanced accuracy, and per-subject normalization ===", "",
             "Reproducibility check (norm = none) against the earlier train_eval.py run:"]
    lines += check_against_reference(results)

    lines += ["", "=== Summary per model and normalization ===",
              "(mean_gap = mean of best-threshold balanced accuracy minus balanced accuracy at the "
              "default threshold; n_below_0.7 counts subjects with balanced accuracy < "
              f"{STRUGGLE_THRESHOLD})",
              summarize_variants(results).to_string(index=False, **fmt)]

    # All subjects, no normalization: the direct test of the threshold hypothesis.
    none = results[results["norm"] == "none"]
    if not none.empty:
        lines += ["", "=== Every subject, no normalization: AUROC / balanced accuracy at the default threshold ==="]
        table = none.assign(cell=none.apply(_cell, axis=1)).pivot(
            index="subject", columns="model", values="cell")
        table.columns = [SHORT.get(c, c) for c in table.columns]
        lines.append(table.to_string())

    corr = cross_model_correlations(results, "none")
    if corr:
        lines += ["", "=== Cross-model correlation of per-subject scores, no normalization "
                      "(n = subjects, imprecise; Pearson can hinge on one subject) ==="]
        for (label, kind), table in corr.items():
            lines += ["", f"{label}, {kind}:", table.to_string()]

    # Focus subjects across normalizations.
    present = [s for s in focus if s in set(results["subject"])]
    if present:
        for model, g in results.groupby("model", sort=False):
            lines += ["", f"=== {SHORT.get(model, model)}: focus subjects, AUROC / balanced accuracy by normalization ==="]
            sub = g[g["subject"].isin(present)]
            t = sub.assign(cell=sub.apply(_cell, axis=1)).pivot(
                index="subject", columns="norm", values="cell")
            order = [n for n in results["norm"].drop_duplicates() if n in t.columns]
            lines.append(t.loc[present, order].to_string())

    # Subjects below the struggle threshold, with the diagnostics that explain them.
    lines += ["", f"=== Subjects with balanced accuracy < {STRUGGLE_THRESHOLD}, per model and normalization ===",
              "(status rules: ranking = AUROC < 0.70; threshold = AUROC >= 0.85 and gap >= 0.15; "
              "mixed = otherwise; judgment calls)"]
    cols = ["subject", "auroc", "balanced_accuracy", "bal_acc_best_threshold",
            "threshold_gap", "mean_p_stress", "prevalence", "status"]
    for (model, norm), g in results.groupby(["model", "norm"], sort=False):
        flagged = g[(g["status"] != "ok") & (g["status"] != "n/a")]
        lines.append("")
        lines.append(f"-- {SHORT.get(model, model)} / {norm}: {len(flagged)} subject(s)")
        if len(flagged):
            lines.append(flagged[cols].to_string(index=False, **fmt))

    lines += ["", "How to read this:",
              "  - Flagged subject with HIGH AUROC and a large gap, or mean_p_stress far from prevalence: "
              "the model ranks that person's windows well but the scores sit on the wrong side of the "
              "cutoff. That is a threshold/shift failure, not a failure to separate stress from non-stress.",
              "  - Flagged subject with LOW AUROC, still low after normalization: the model cannot "
              "separate that person's stress windows from non-stress ones at all (ranking failure).",
              "  - Tree models (rf, gb) can saturate: if a subject's features fall outside the range seen in "
              "training, every window lands in the same leaves and the scores tie, so AUROC drops toward 0.5 "
              "and the subject is labelled 'ranking' even though the cause is a between-subject offset. "
              "The tell is that normalization restores AUROC. Linear models (lr) shift smoothly, so there the "
              "gap shows up as a threshold failure instead.",
              "  - 'none' vs 'full' vs 'earlyN': if scores improve with 'full' but not with 'earlyN', the gain "
              "depends on knowing the whole recording and would not be available in deployment. "
              "'earlyN' is only meaningful because every WESAD recording starts at rest.",
              "  - bal_acc_best_threshold uses the subject's own labels; it shows what a perfect per-person "
              "cutoff could achieve, not what a deployed model gets.",
              "  - ~35 independent windows per subject: single-subject differences under ~0.1 are noise."]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        choices=list(SHORT), help="default: rf, gb, lr")
    parser.add_argument("--norms", nargs="*", default=DEFAULT_NORMS,
                        help="none, full, earlyN (e.g. early20); default: none full early50 early20 early10")
    parser.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS)
    parser.add_argument("--report", default="reports/auroc_norm_report.txt")
    parser.add_argument("--csv", default="reports/auroc_norm_per_subject.csv")
    args = parser.parse_args()

    for n in args.norms:  # fail fast on a typo, before the slow part
        parse_norm(n)

    df = pd.read_parquet(args.features)
    print(f"Loaded {len(df)} windows, {df['subject'].nunique()} subjects. Running LOSO ...")
    results = run_all(df, args.models, args.norms)

    lines = build_report(results, args.focus)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.csv, index=False)
    print("\n".join(lines))
    print(f"\nWrote {report_path} and {args.csv}")


if __name__ == "__main__":
    main()
