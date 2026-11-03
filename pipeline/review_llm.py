"""
Print an LLM description next to the data it was given, so a person can check
it quickly. Reads reports/llm_explanations/<subject>.json (written by
explain_llm.py). Makes no API call and decides nothing: it only lays out
(1) the model table, (2) the feature table with a plain-word category computed
from the same rules the prompt describes, and both the subject's and the population's change expressed in
between-person SDs (recovered from the z-scores in the facts; approximate), and (3) each sentence of the
description that mentions a feature or a model, under that feature or model.

Usage:
    python pipeline/review_llm.py                 # S14 S5 S9
    python pipeline/review_llm.py --subjects S14
"""

import argparse
import json
import re
from pathlib import Path

WEAK_RATIO = 0.25
FEATURES = ["hr_mean_bpm", "hrv_proxy_std_ms", "eda_mean", "eda_std", "eda_slope",
            "temp_mean", "temp_slope", "acc_mag_mean", "acc_mag_std"]
MODELS = ["random_forest", "gradient_boosting", "logistic_regression"]
LARGE_RATIO = 1.5


def category(row: dict) -> str:
    """Plain-word label from the facts. Mirrors the prompt's wording."""
    ratio = row.get("response_ratio")
    if ratio is None:
        return "no ratio"
    if not row["direction_matches_population"]:
        return "FLAT (|ratio|<0.05)" if abs(ratio) < 0.05 else "reversed"
    if row["weak_response"]:
        return "weak"
    return "larger than typical" if ratio >= LARGE_RATIO else "typical"


SMALL_POP_EFFECT_SD = 0.25  # judgment call: population change below this many between-person SDs


def between_subject_sd(comparison: dict, feature: str):
    """Approximate between-person SD of a feature, recovered from the facts:
    z = (subject_mean - others_mean) / sd, so sd = (subject_mean - others_mean) / z.
    Uses the class with the larger |z| (>= 0.3, to limit rounding error) and
    returns None if neither class qualifies."""
    best = None
    for label in ("non_stress", "stress"):
        row = comparison["per_class"].get(label, {}).get(feature)
        if not row or row["z_score"] is None or abs(row["z_score"]) < 0.3:
            continue
        sd = abs((row["subject_mean"] - row["others_mean"]) / row["z_score"])
        if best is None or abs(row["z_score"]) > best[0]:
            best = (abs(row["z_score"]), sd)
    return None if best is None else best[1]


def sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Za-z(])', re.sub(r"\s+", " ", text))
    return [p.strip() for p in parts if p.strip()]


def review(result: dict) -> str:
    facts, text = result["facts"], result["text"]
    sents = sentences(text)
    out = [f"{'=' * 78}", f"{result['subject']}   served model: {result['model_served']}   "
           f"temperature: {result['temperature']}   numbers checked: "
           f"{result['grounding']['n_numbers']}, ungrounded: {result['grounding']['ungrounded']}",
           f"{'=' * 78}", "", "1. MODELS (check against what the text says about each model)",
           f"{'model':<20}{'auroc':>7}{'bal.acc':>9}{'best-thr':>9}{'mean P':>8}{'true':>7}  status"]
    for m in MODELS:
        d = facts["model_diagnostics"][m]["raw_features_default_threshold"]
        out.append(f"{m:<20}{d['auroc']:>7.3f}{d['balanced_accuracy']:>9.3f}"
                   f"{d['best_threshold_balanced_accuracy']:>9.3f}{d['mean_predicted_p_stress']:>8.3f}"
                   f"{d['true_stress_share']:>7.3f}  {d['status_label']}")
        for s in sents:
            if m in s:
                out.append(f"      TEXT: {s}")
    out += ["", "2. FEATURES (check each stated direction, size word and number)",
            f"{'feature':<18}{'subj chg':>10}{'pop chg':>10}{'ratio':>8}{'subj chg':>10}{'pop chg':>10}  category",
            f"{'':<18}{'':>10}{'':>10}{'':>8}{'in SDs':>10}{'in SDs':>10}"]
    comp = facts["feature_comparison"]
    for f in FEATURES:
        row = comp["stress_response_direction"].get(f)
        if row is None:
            continue
        sd = between_subject_sd(comp, f)
        pop_sd = abs(row["population_stress_minus_nonstress"]) / sd if sd else None
        note = ""
        if pop_sd is not None and pop_sd < SMALL_POP_EFFECT_SD:
            note = "   <- population change is small: ratio unstable"
        shown = f"{pop_sd:>10.2f}" if pop_sd is not None else f"{'n/a':>10}"
        subj_sd = row["subject_stress_minus_nonstress"] / sd if sd else None
        subj_shown = f"{subj_sd:>10.2f}" if subj_sd is not None else f"{'n/a':>10}"
        ratio = row["response_ratio"] if row["response_ratio"] is not None else float("nan")
        out.append(f"{f:<18}{row['subject_stress_minus_nonstress']:>10.3f}"
                   f"{row['population_stress_minus_nonstress']:>10.3f}{ratio:>8.2f}{subj_shown}{shown}  "
                   f"{category(row)}{note}")
        for s in sents:
            if f in s:
                out.append(f"      TEXT: {s}")
    out += ["", "3. SENTENCES THAT MENTION NEITHER A MODEL NOR A FEATURE (check these are only caveats)"]
    for s in sents:
        if not any(k in s for k in MODELS + FEATURES):
            out.append(f"      TEXT: {s}")
    out += ["", "4. CHECKLIST",
            "  [ ] each model's TEXT matches its status label and numbers (a 'threshold' status means high auroc, low balanced accuracy)",
            "  [ ] 'reversed' only where category says reversed; a FLAT row must not be called reversed",
            "  [ ] 'weak' only where category says weak; 'much larger' only where ratio is clearly above 1",
            "  [ ] a row marked 'population change is small' (below 0.25 person-SDs; my cutoff) should not be called larger/reversed with any weight",
            "  [ ] 'reversed' is only a big statement if 'subj chg in SDs' is clearly nonzero (flat is not reversed; a ratio near 0 means flat whatever the sign)",
            "  [ ] no sentence states a cause; motion-artifact lines are the prompt's rule, not a finding",
            "  [ ] the first sentence about the models agrees with the rest of the paragraph (e.g. S5 'all three less accurate' vs two 'ok')",
            ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="reports/llm_explanations")
    parser.add_argument("--subjects", nargs="*", default=["S14", "S5", "S9"])
    args = parser.parse_args()
    for s in args.subjects:
        print(review(json.loads((Path(args.dir) / f"{s}.json").read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
