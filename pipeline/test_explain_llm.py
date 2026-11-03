"""
Tests for explain_llm.py. No API calls: a fake client stands in for Anthropic.

What is and is not tested:
  - tested: facts are built from the right sources and are strict-JSON (no NaN);
    the grounding check accepts exact and correctly rounded numbers, percentages
    and reference points, and rejects invented numbers; causal wording is flagged;
    the call records requested vs served model; a rejected temperature setting is
    retried without it while other errors propagate; results are saved with the
    banner; --model is required for a real call.
  - NOT tested (cannot be, offline): that the real model follows the prompt, or
    that the real API accepts the arguments. Run it once with the real key.

Run directly: python pipeline/test_explain_llm.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explain import FEATURE_COLUMNS  # noqa: E402
import explain_llm as E  # noqa: E402
import review_llm as R  # noqa: E402


# Fixtures

def make_features(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for subj in ["S2", "S3", "S5", "S9", "S14"]:
        for label, shift in [("non_stress", 0.0), ("stress", 1.0)]:
            for _ in range(8):
                row = {"subject": subj, "label": label, "window_start_sec": len(rows) * 30}
                for col in FEATURE_COLUMNS:
                    row[col] = rng.normal(shift, 0.3) + 10
                rows.append(row)
    return pd.DataFrame(rows)


def make_diag():
    # Numbers copied from the real run for S14 (RF, GB) so the fixture is realistic.
    rows = []
    def add(model, norm, subj, auroc, bal, best, mean_p, prev, status):
        rows.append(dict(model=model, norm=norm, subject=subj, n_windows=40, prevalence=prev,
                         auroc=auroc, balanced_accuracy=bal, bal_acc_best_threshold=best,
                         threshold_gap=best - bal, mean_p_stress=mean_p, status=status, note=""))
    for model in E.MODELS_IN_DIAGNOSTICS:
        for subj in ["S2", "S3", "S5", "S9"]:
            add(model, "none", subj, 0.95, 0.85, 0.9, 0.3, 0.3, "ok")
    add("random_forest", "none", "S14", 0.982, 0.500, 0.949, 0.066, 0.300, "threshold")
    add("gradient_boosting", "none", "S14", 0.738, 0.524, 0.789, 0.079, 0.300, "mixed")
    add("logistic_regression", "none", "S14", 0.990, 0.930, 0.990, 0.310, 0.300, "ok")
    for model in E.MODELS_IN_DIAGNOSTICS:
        add(model, "full", "S14", 0.91, 0.77, 0.85, 0.28, 0.3, "ok")
        add(model, "early20", "S14", float("nan"), float("nan"), float("nan"), float("nan"), 0.3, "n/a")
    return pd.DataFrame(rows)


class FakeClient:
    """Mimics client.messages.create. `behaviors` is a list of callables or
    exceptions consumed in order; default returns `text`."""

    def __init__(self, text="ok", model="claude-fake-1", behaviors=None):
        self.calls = []
        self._text, self._model = text, model
        self._behaviors = list(behaviors or [])
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._behaviors:
            b = self._behaviors.pop(0)
            if isinstance(b, Exception):
                raise b
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            model=self._model, stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1200, output_tokens=310))


# Facts

def test_facts_have_model_diagnostics_and_comparison_and_are_strict_json():
    facts = E.build_facts(make_features(), make_diag(), "S14")
    json.dumps(facts, allow_nan=False)  # raises on NaN/inf
    assert facts["subject"] == "S14"
    assert set(facts["model_diagnostics"]) == set(E.MODELS_IN_DIAGNOSTICS)
    rf = facts["model_diagnostics"]["random_forest"]["raw_features_default_threshold"]
    assert rf["auroc"] == 0.982 and rf["balanced_accuracy"] == 0.5
    assert rf["mean_predicted_p_stress"] == 0.066 and rf["true_stress_share"] == 0.3
    assert rf["status_label"] == "threshold"
    # normalization table keeps NaN as None, not NaN
    early = facts["model_diagnostics"]["random_forest"]["by_normalization"]["early20"]
    assert early == {"auroc": None, "balanced_accuracy": None}
    assert "stress_response_direction" in facts["feature_comparison"]
    assert facts["study_context"]["n_subjects"] == 5
    print("test_facts_have_model_diagnostics_and_comparison_and_are_strict_json: OK")


def test_reference_means_cover_all_subjects_not_only_the_target():
    facts = E.build_facts(make_features(), make_diag(), "S14")
    ref = facts["model_diagnostics"]["random_forest"]["all_subjects_reference"]
    expected = round(float(np.mean([0.95] * 4 + [0.982])), 3)
    assert ref["mean_auroc"] == expected, ref
    print("test_reference_means_cover_all_subjects_not_only_the_target: OK")


def test_missing_subject_or_columns_raise():
    for bad in [lambda: E.build_facts(make_features(), make_diag(), "S99"),
                lambda: E.build_facts(make_features(), make_diag().drop(columns=["auroc"]), "S14"),
                lambda: E.build_facts(make_features(), make_diag()[lambda d: d["subject"] != "S9"], "S9")]:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    print("test_missing_subject_or_columns_raise: OK")


# Grounding

def facts_s14():
    return E.build_facts(make_features(), make_diag(), "S14")


def test_extract_numbers_ignores_subject_ids_and_list_markers():
    nums = E.extract_numbers("S14 and S5.\n1. First 0.98 then\n- second -0.63 and 30%.")
    toks = [n["token"] for n in nums]
    assert toks == ["0.98", "-0.63", "30%"], toks
    print("test_extract_numbers_ignores_subject_ids_and_list_markers: OK")


def test_grounding_accepts_exact_rounded_percent_and_reference_numbers():
    text = ("AUROC was 0.982 for random_forest (about 0.98), balanced accuracy 0.5, "
            "mean predicted 0.066 versus a true share of 30%, chance is 0.5 and perfect is 1.0.")
    g = E.check_grounding(text, facts_s14())
    assert g["passed"], g
    assert "0.982" in g["exact"] and "0.98" in g["rounded"] and "30%" in g["rounded"] + g["exact"]
    print("test_grounding_accepts_exact_rounded_percent_and_reference_numbers: OK")


def test_grounding_rejects_invented_numbers_and_wrong_rounding():
    g = E.check_grounding("AUROC was 0.97 and accuracy 0.62.", facts_s14())
    assert not g["passed"] and set(g["ungrounded"]) == {"0.97", "0.62"}, g
    # 0.75 is not a rounding of 0.738 at two decimals
    assert E.check_grounding("gb was 0.75", facts_s14())["ungrounded"] == ["0.75"]
    # integers must match exactly: 14 is not a fact (the subject ID is ignored separately)
    assert E.check_grounding("there were 14 windows", facts_s14())["ungrounded"] == ["14"]
    print("test_grounding_rejects_invented_numbers_and_wrong_rounding: OK")


def test_grounding_ignores_sign():
    # response change values can be negative in the facts; the sentence may say "fell by 0.63"
    facts = {"x": {"change": -0.63}}
    assert E.check_grounding("it fell by 0.63", facts)["passed"]
    assert E.check_grounding("it was −0.63", facts)["passed"]
    print("test_grounding_ignores_sign: OK")


def test_grounding_known_limit_a_wrong_claim_with_a_real_number_passes():
    """Documents what the check does NOT do, so nobody reads 'passed' as 'correct'."""
    wrong_but_grounded = "random_forest ranked S14 poorly with AUROC 0.982."
    assert E.check_grounding(wrong_but_grounded, facts_s14())["passed"]
    print("test_grounding_known_limit_a_wrong_claim_with_a_real_number_passes: OK")


def test_numeric_leaves_skips_bools_and_none():
    assert sorted(E.numeric_leaves({"a": 1, "b": [2.5, None, True], "c": {"d": "x", "e": 3}})) == [1.0, 2.5, 3.0]
    print("test_numeric_leaves_skips_bools_and_none: OK")


def test_causal_language_flags():
    flags = E.causal_language_flags("The failure is because of EDA, which explains it; diagnosed anxiety.")
    assert {"because", "explains", "diagnosed"} <= set(flags), flags
    assert E.causal_language_flags("This is consistent with a shifted threshold.") == []
    print("test_causal_language_flags: OK")


# Call and save

def test_call_records_requested_and_served_model_and_sends_facts_only():
    client = FakeClient(text="hello", model="claude-served-9")
    out = E.call_llm({"k": 1}, "claude-requested-1", client=client)
    assert out["model_requested"] == "claude-requested-1" and out["model_served"] == "claude-served-9"
    assert out["input_tokens"] == 1200 and out["output_tokens"] == 310 and out["temperature"] == 0.0
    sent = client.calls[0]
    assert sent["model"] == "claude-requested-1" and sent["system"] == E.SYSTEM_PROMPT
    assert json.loads(sent["messages"][0]["content"]) == {"k": 1}
    print("test_call_records_requested_and_served_model_and_sends_facts_only: OK")


def test_rejected_temperature_is_retried_without_it():
    client = FakeClient(behaviors=[RuntimeError("`temperature` is not supported for this model")])
    out = E.call_llm({"k": 1}, "m", client=client)
    assert out["temperature"] is None and len(client.calls) == 2
    assert "temperature" in client.calls[0] and "temperature" not in client.calls[1]
    print("test_rejected_temperature_is_retried_without_it: OK")


def test_other_api_errors_propagate():
    client = FakeClient(behaviors=[RuntimeError("rate limited")])
    try:
        E.call_llm({"k": 1}, "m", client=client)
    except RuntimeError as exc:
        assert "rate limited" in str(exc) and len(client.calls) == 1
    else:
        raise AssertionError("expected RuntimeError")
    print("test_other_api_errors_propagate: OK")


def test_missing_api_key_raises_clear_error():
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        try:
            E.call_llm({"k": 1}, "m")
        except RuntimeError as exc:
            assert "ANTHROPIC_API_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old
    print("test_missing_api_key_raises_clear_error: OK")


def test_explain_subject_end_to_end_and_save():
    text = ("random_forest has AUROC 0.982 but balanced accuracy 0.5, and its mean predicted "
            "probability 0.066 is far below the stress share 0.3; this is consistent with a shifted "
            "threshold. gradient_boosting AUROC 0.738 is lower.")
    client = FakeClient(text=text, model="claude-served-9")
    res = E.explain_subject(make_features(), make_diag(), "S14", "claude-requested-1", client=client)
    assert res["grounding"]["passed"] and res["causal_or_clinical_flags"] == []
    assert res["facts"]["subject"] == "S14" and res["system_prompt"] == E.SYSTEM_PROMPT
    with tempfile.TemporaryDirectory() as d:
        jpath, mpath = E.save_result(res, Path(d) / "nested" / "out")
        saved = json.loads(jpath.read_text(encoding="utf-8"))
        assert saved["model_served"] == "claude-served-9" and saved["facts_sha256_16"] == res["facts_sha256_16"]
        md = mpath.read_text(encoding="utf-8")
        assert "DESCRIPTION, NOT EVIDENCE" in md and text in md and "claude-served-9" in md
    print("test_explain_subject_end_to_end_and_save: OK")


def test_failed_grounding_is_reported_in_the_saved_markdown():
    client = FakeClient(text="random_forest had AUROC 0.61 because of sweat.")
    res = E.explain_subject(make_features(), make_diag(), "S14", "m", client=client)
    assert not res["grounding"]["passed"] and res["grounding"]["ungrounded"] == ["0.61"]
    assert "because" in res["causal_or_clinical_flags"]
    with tempfile.TemporaryDirectory() as d:
        _, mpath = E.save_result(res, Path(d))
        md = mpath.read_text(encoding="utf-8")
        assert "1 not found in the facts: 0.61" in md and "because" in md
    print("test_failed_grounding_is_reported_in_the_saved_markdown: OK")


def test_facts_hash_is_stable_and_changes_with_content():
    f = facts_s14()
    assert E._sha(f) == E._sha(json.loads(json.dumps(f)))
    g = json.loads(json.dumps(f))
    g["study_context"]["n_subjects"] = 99
    assert E._sha(f) != E._sha(g)
    print("test_facts_hash_is_stable_and_changes_with_content: OK")


def test_cli_requires_model_for_a_real_call():
    old = sys.argv
    sys.argv = ["explain_llm.py", "--features", "missing.parquet"]
    try:
        try:
            E.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit")
    finally:
        sys.argv = old
    print("test_cli_requires_model_for_a_real_call: OK")


def test_prompt_demands_no_new_numbers_and_no_causes():
    p = E.SYSTEM_PROMPT
    assert "Do not compute new numbers" in p and "Do not state a cause" in p
    assert "never classify" in p or "never see raw" in p
    print("test_prompt_demands_no_new_numbers_and_no_causes: OK")


# review_llm.py (the printout used for the hand review)

def test_review_recovers_between_person_sd_from_z_scores():
    feat = make_features()
    comp = E.build_facts(feat, make_diag(), "S14")["feature_comparison"]
    for f in FEATURE_COLUMNS:
        sd = R.between_subject_sd(comp, f)
        truths = [feat[(feat.subject != "S14") & (feat.label == lab)].groupby("subject")[f].mean().std()
                  for lab in ["non_stress", "stress"]]
        assert sd is None or any(abs(sd - t) / t < 0.03 for t in truths), (f, sd, truths)
    # a subject exactly at the others' mean has z ~ 0: no SD can be recovered
    flat = {"per_class": {"stress": {"x": {"subject_mean": 1.0, "others_mean": 1.0, "z_score": 0.0}}}}
    assert R.between_subject_sd(flat, "x") is None
    print("test_review_recovers_between_person_sd_from_z_scores: OK")


def test_review_flags_flat_reversed_and_small_population_effects():
    assert R.category({"response_ratio": -0.0, "direction_matches_population": False,
                       "weak_response": False}).startswith("FLAT")
    facts = E.build_facts(make_features(), make_diag(), "S14")
    d = facts["feature_comparison"]["stress_response_direction"]["eda_slope"]
    d["population_stress_minus_nonstress"] = 1e-6  # population barely changes
    res = {"subject": "S14", "model_served": "m", "temperature": None, "text": "eda_slope was reversed.",
           "grounding": {"n_numbers": 0, "ungrounded": []}, "facts": facts}
    out = R.review(res)
    assert "population change is small" in out and "TEXT: eda_slope was reversed." in out
    print("test_review_flags_flat_reversed_and_small_population_effects: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
