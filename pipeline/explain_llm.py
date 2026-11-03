"""
Grounded LLM descriptions of how the models behaved on a few held-out subjects.

What this is, and is not
------------------------
The LLM receives ONLY a JSON object of statistics computed by this repository:
  - the subject's feature comparison against the other subjects (explain.py), and
  - the per-subject diagnostics from evaluation/auroc_norm.py (AUROC, balanced
    accuracy at the default threshold, best-threshold balanced accuracy, mean
    predicted stress probability vs the true stress share, per normalization).
It never sees raw signals and never classifies anything. Its output is a
DESCRIPTION written for a reader, not evidence: no explanation here was tested,
and the output says so.

Safeguards built in
-------------------
1. The model must be named on the command line (no default), and the model that
   actually served the request (response.model) is saved next to the output.
   For 4.6-generation and later models a dateless ID such as claude-sonnet-5-5
   is a fixed snapshot per Anthropic's model-ID documentation; for older models
   use the full dated ID, never an alias.
2. Grounding check: every number in the output must be a number from the facts
   (exactly, or a rounding of one to the precision written; sign is ignored;
   percentages are compared as fractions; subject IDs such as S14 are ignored;
   0, 0.5 and 1.0 are allowed as reference points). Ungrounded numbers are listed. This checks that numbers exist in the data,
   NOT that the sentence around them is right (a wrong direction or a wrong
   comparison passes), so every output still needs the manual checklist below.
3. Causal / clinical wording is flagged for review (not blocked).
4. Prompt, facts, hashes, served model, token usage and the grounding report are
   saved with each output so the run can be audited.

Manual checklist for each saved description
-------------------------------------------
  [ ] each model's failure type matches the status label and numbers in the facts
  [ ] every stated direction (higher / lower, reversed / weak) matches the sign in the facts
  [ ] nothing is stated as a cause; hedged wording only
  [ ] no mention of diagnoses, medication or conditions
  [ ] it does not claim more than 15 subjects can support

Requires ANTHROPIC_API_KEY for the real call (not for --dry-run).

Usage:
    # Inspect the facts and prompt that would be sent (no API call, no cost):
    python pipeline/explain_llm.py --dry-run

    # Real call (a few cents for three subjects):
    python pipeline/explain_llm.py --model claude-sonnet-5-5
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explain import FEATURE_COLUMNS, build_subject_comparison  # noqa: E402

DEFAULT_SUBJECTS = ["S14", "S5", "S9"]
MODELS_IN_DIAGNOSTICS = ["random_forest", "gradient_boosting", "logistic_regression"]
RAW_NORM = "none"

# Judgment-call cutoffs that appear in evaluation/auroc_norm.py and train_eval.py.
STRUGGLE_CUTOFF = 0.7
RANK_FAIL_AUROC = 0.70
THRESHOLD_OK_AUROC = 0.85
THRESHOLD_GAP = 0.15

SYSTEM_PROMPT = """You are writing a short, cautious description of how wearable \
stress-detection models behaved on ONE held-out study participant. You receive \
only a JSON object of statistics computed by the study code. You never see raw \
signals and you never classify anything.

Meaning of the fields:
- auroc: how well the model's stress probabilities rank this participant's stress \
windows above their non-stress windows (0.5 is chance, 1.0 is perfect). It does not \
depend on where the decision threshold is.
- balanced_accuracy: accuracy averaged over the two classes at the model's default \
threshold (0.5 is chance).
- best_threshold_balanced_accuracy: the balanced accuracy if the threshold were \
chosen using this participant's own labels (an upper bound, not achievable in \
practice).
- mean_predicted_p_stress vs true_stress_share: if the mean predicted probability is \
far below the true stress share, the model's scores for this participant are shifted \
low (and the reverse if far above).
- status_label: "ok", "threshold" (ranks well but the cutoff is wrong), "ranking" \
(does not rank the classes) or "mixed", using the cutoffs in study_context.
- feature_comparison.stress_response_direction: for each feature, how this \
participant's stress-minus-non-stress change compares with the other participants \
(response_ratio 1.0 is typical, near 0 is flat, negative is reversed; weak_response \
true means the same direction but under a quarter of the typical size).

Write two short paragraphs of plain prose (no lists, no headings, at most 220 words \
in total):
Paragraph 1 - what the models did. For each of the three models give this \
participant's auroc and balanced_accuracy, and say whether the numbers point to \
poor ranking, to a shifted threshold (high auroc, low balanced_accuracy, and a mean \
predicted probability far from the true share), or to no problem. Use only the \
numbers.
Paragraph 2 - how this participant's data differ from the others. Report the \
features whose response is reversed (direction_matches_population false), weak \
(weak_response true) or far larger than typical (response_ratio well above 1), with \
their numbers. State that these are observations and that this analysis did not \
test whether they cause the models' behavior. End with one sentence on what these \
statistics cannot tell.

Rules:
- Use only numbers that appear in the JSON, written as given (you may round to fewer \
decimals). Do not compute new numbers: no differences, sums, ratios or percentages.
- Do not state a cause. Use wording such as "is consistent with" or "may".
- hrv_proxy_std_ms comes from a crude peak detector and its values are far larger \
than real heart-rate variability; mention it only to say it is unreliable.
- If accelerometer features are unusual, mention motion artifact as a data-quality \
possibility, not a physiological one.
- Do not mention diagnoses, medications, health conditions or anything not in the JSON.
- Refer to models by the names used in the JSON.
- Fewer than 15 participants and about 35 independent windows each: do not overstate."""

BANNER = ("DESCRIPTION, NOT EVIDENCE. Generated by a language model from the computed "
          "statistics listed in the JSON file next to this one. No explanation below was "
          "tested. Numbers were machine-checked against those statistics; the sentences "
          "around them were not.")

# Reference points the prompt itself explains (chance level, perfect score). They are
# allowed in the output without being in the facts; this is a deliberate, documented
# loophole of the grounding check.
ALLOWED_REFERENCE_NUMBERS = (0.0, 0.5, 1.0)

CAUSAL_OR_CLINICAL_PATTERNS = [
    r"\bbecause\b", r"\bcaus(?:e|es|ed|ing)\b", r"\bdue to\b", r"\bleads? to\b",
    r"\bresponsible for\b", r"\bexplains?\b", r"\bis why\b", r"\btherefore\b",
    r"\bresults? from\b", r"\bdiagnos\w*", r"\bmedicat\w*", r"\bdisease\w*",
    r"\bdisorder\w*", r"\billness\w*", r"\bpatholog\w*",
]


# Facts

def _clean(value):
    """CSV cell -> JSON-friendly number (NaN -> None)."""
    if pd.isna(value):
        return None
    return float(value)


def build_model_diagnostics(diag: pd.DataFrame, subject: str) -> dict:
    """Per model: raw-feature diagnostics, a population reference, and a
    normalization table, all read from reports/auroc_norm_per_subject.csv."""
    required = {"model", "norm", "subject", "prevalence", "auroc", "balanced_accuracy",
                "bal_acc_best_threshold", "mean_p_stress", "status"}
    missing = required - set(diag.columns)
    if missing:
        raise ValueError(f"Diagnostics file is missing columns: {sorted(missing)}")
    out = {}
    for model in MODELS_IN_DIAGNOSTICS:
        g = diag[diag["model"] == model]
        raw_all = g[g["norm"] == RAW_NORM]
        raw = raw_all[raw_all["subject"] == subject]
        if raw.empty:
            raise ValueError(f"No '{RAW_NORM}' diagnostics for subject {subject}, model {model}.")
        r = raw.iloc[0]
        entry = {
            "raw_features_default_threshold": {
                "auroc": _clean(r["auroc"]),
                "balanced_accuracy": _clean(r["balanced_accuracy"]),
                "best_threshold_balanced_accuracy": _clean(r["bal_acc_best_threshold"]),
                "mean_predicted_p_stress": _clean(r["mean_p_stress"]),
                "true_stress_share": _clean(r["prevalence"]),
                "status_label": str(r["status"]),
            },
            "all_subjects_reference": {
                "mean_auroc": round(float(raw_all["auroc"].mean()), 3),
                "mean_balanced_accuracy": round(float(raw_all["balanced_accuracy"].mean()), 3),
            },
            "by_normalization": {},
        }
        for _, row in g[g["subject"] == subject].iterrows():
            entry["by_normalization"][str(row["norm"])] = {
                "auroc": _clean(row["auroc"]),
                "balanced_accuracy": _clean(row["balanced_accuracy"]),
            }
        out[model] = entry
    return out


def build_facts(features: pd.DataFrame, diag: pd.DataFrame, subject: str) -> dict:
    target = features[features["subject"] == subject]
    if target.empty:
        raise ValueError(f"Subject {subject} not found in features.")
    return {
        "subject": subject,
        "study_context": {
            "n_subjects": int(features["subject"].nunique()),
            "window_seconds": 60,
            "approx_independent_windows_per_subject": 35,
            "n_windows_this_subject": int(len(target)),
            "struggling_balanced_accuracy_cutoff": STRUGGLE_CUTOFF,
            "ranking_status_auroc_below": RANK_FAIL_AUROC,
            "threshold_status_auroc_at_least": THRESHOLD_OK_AUROC,
            "threshold_status_gap_at_least": THRESHOLD_GAP,
        },
        "model_diagnostics": build_model_diagnostics(diag, subject),
        "feature_comparison": build_subject_comparison(features, subject, FEATURE_COLUMNS),
    }


# Grounding check

_SUBJECT_ID = re.compile(r"\bS\d{1,2}\b")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?%?")


def numeric_leaves(obj) -> list[float]:
    """All int/float values anywhere in a nested dict/list (bools excluded)."""
    found = []
    if isinstance(obj, bool) or obj is None:
        return found
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        for v in obj.values():
            found += numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += numeric_leaves(v)
    return found


def extract_numbers(text: str) -> list[dict]:
    """Numbers written in the text, as dicts with the token, value, decimals and
    percent flag. Subject IDs and leading list markers are removed first."""
    cleaned = _LIST_MARKER.sub("", _SUBJECT_ID.sub(" ", text))
    out, seen = [], set()
    for m in _NUMBER.finditer(cleaned):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        pct = tok.endswith("%")
        body = tok.rstrip("%").replace("−", "-")
        decimals = len(body.split(".")[1]) if "." in body else 0
        out.append({"token": tok, "value": float(body), "decimals": decimals, "percent": pct})
    return out


def _match(num: dict, fact_values: list[float]) -> str | None:
    """'exact', 'rounded' or None."""
    cited, decimals = abs(num["value"]), num["decimals"]
    if num["percent"]:
        cited, decimals = cited / 100.0, decimals + 2
    best = None
    for f in fact_values:
        f = abs(f)
        if abs(f - cited) < 1e-9:
            return "exact"
        if decimals > 0 and abs(f - cited) <= 0.5 * 10 ** (-decimals) + 1e-9:
            best = "rounded"
    return best


def check_grounding(text: str, facts: dict) -> dict:
    fact_values = numeric_leaves(facts) + list(ALLOWED_REFERENCE_NUMBERS)
    exact, rounded, ungrounded = [], [], []
    for num in extract_numbers(text):
        kind = _match(num, fact_values)
        (exact if kind == "exact" else rounded if kind == "rounded" else ungrounded).append(num["token"])
    return {
        "n_numbers": len(exact) + len(rounded) + len(ungrounded),
        "exact": exact, "rounded": rounded, "ungrounded": ungrounded,
        "passed": not ungrounded,
    }


def causal_language_flags(text: str) -> list[str]:
    flags = []
    for pat in CAUSAL_OR_CLINICAL_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            flags.append(m.group(0).lower())
    return sorted(set(flags))


# LLM call and saving

def _sha(obj) -> str:
    data = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def call_llm(facts: dict, model: str, client=None, max_tokens: int = 900) -> dict:
    """Send the facts and return the text plus the metadata needed to audit the
    call. `client` can be injected (tests use a fake)."""
    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use --dry-run to inspect the "
                               "facts and prompt without an API call.")
        import anthropic
        # Responses are under 1 KB, so ask for them uncompressed. This sidesteps
        # broken brotli/zstd decoder combinations in some Python environments
        # (seen as "APIConnectionError ... process() takes no keyword arguments").
        client = anthropic.Anthropic(api_key=api_key,
                                     default_headers={"Accept-Encoding": "identity"})

    kwargs = dict(model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
                  messages=[{"role": "user", "content": json.dumps(facts, indent=2)}])
    temperature = 0.0
    try:
        resp = client.messages.create(temperature=temperature, **kwargs)
    except Exception as exc:  # some models reject an explicit temperature
        if "temperature" not in str(exc).lower():
            raise
        temperature = None
        resp = client.messages.create(**kwargs)

    usage = getattr(resp, "usage", None)
    return {
        "text": "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip(),
        "model_requested": model,
        "model_served": getattr(resp, "model", None),
        "stop_reason": getattr(resp, "stop_reason", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def explain_subject(features: pd.DataFrame, diag: pd.DataFrame, subject: str,
                    model: str, client=None) -> dict:
    facts = build_facts(features, diag, subject)
    call = call_llm(facts, model, client=client)
    return {
        "subject": subject,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **call,
        "grounding": check_grounding(call["text"], facts),
        "causal_or_clinical_flags": causal_language_flags(call["text"]),
        "system_prompt_sha256_16": _sha(SYSTEM_PROMPT),
        "facts_sha256_16": _sha(facts),
        "system_prompt": SYSTEM_PROMPT,
        "facts": facts,
    }


def save_result(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject = result["subject"]
    jpath, mpath = out_dir / f"{subject}.json", out_dir / f"{subject}.md"
    jpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    g = result["grounding"]
    md = [f"# {subject}: model-written description", "", f"> {BANNER}", "",
          result["text"], "", "---",
          f"Model requested: {result['model_requested']}; served: {result['model_served']}; "
          f"temperature: {result['temperature']}; tokens in/out: "
          f"{result['input_tokens']}/{result['output_tokens']}; stop: {result['stop_reason']}.",
          f"Numbers checked: {g['n_numbers']} ({len(g['exact'])} exact, {len(g['rounded'])} rounded, "
          f"{len(g['ungrounded'])} not found in the facts"
          + (f": {', '.join(g['ungrounded'])}" if g["ungrounded"] else "") + ").",
          "Causal or clinical wording flagged for review: "
          + (", ".join(result["causal_or_clinical_flags"]) or "none") + ".",
          f"Prompt hash {result['system_prompt_sha256_16']}, facts hash {result['facts_sha256_16']}."]
    mpath.write_text("\n".join(md) + "\n", encoding="utf-8")
    return jpath, mpath


# CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--diagnostics", default="reports/auroc_norm_per_subject.csv")
    parser.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    parser.add_argument("--model", default=None,
                        help="exact model ID (required for a real call; no default on purpose)")
    parser.add_argument("--out-dir", default="reports/llm_explanations")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the facts and prompt; make no API call")
    args = parser.parse_args()

    if not args.dry_run and not args.model:
        parser.error("--model is required for a real call (use the exact ID from the Anthropic "
                     "models page; see the module docstring for pinning)")

    features = pd.read_parquet(args.features)
    diag = pd.read_csv(args.diagnostics)

    if args.dry_run:
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        for s in args.subjects:
            facts = build_facts(features, diag, s)
            text = json.dumps(facts, indent=2)
            print(f"\n=== FACTS for {s} ({len(text)} characters, "
                  f"{len(numeric_leaves(facts))} numbers) ===")
            print(text)
        print("\nDry run: no API call made.")
        return

    rows = []
    for s in args.subjects:
        print(f"Asking {args.model} about {s} ...", flush=True)
        result = explain_subject(features, diag, s, args.model)
        jpath, mpath = save_result(result, Path(args.out_dir))
        g = result["grounding"]
        rows.append({"subject": s, "served": result["model_served"], "numbers": g["n_numbers"],
                     "ungrounded": len(g["ungrounded"]),
                     "causal_flags": len(result["causal_or_clinical_flags"])})
        print(f"\n--- {s} ---\n{result['text']}\n")
        print(f"[numbers checked: {g['n_numbers']}; not found in facts: {g['ungrounded'] or 'none'}; "
              f"causal/clinical flags: {result['causal_or_clinical_flags'] or 'none'}]")
        print(f"[saved {jpath} and {mpath}]")
    print("\n=== Summary ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNext: read each .md against the checklist at the top of pipeline/explain_llm.py. "
          "The number check does not verify directions or comparisons.")


if __name__ == "__main__":
    main()
