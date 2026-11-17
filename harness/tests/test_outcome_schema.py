# SPDX-License-Identifier: Apache-2.0

"""The outcome vocabulary (schema v2): validation, the new graders, the graded rubric, the
majority pass rule and the strictness delta over thresholds."""

from __future__ import annotations

import json

from harness import graders
from harness.cases import Case, outcome_problems, vocabulary_violations
from harness.graders import Outcome, grade, grade_turn
from harness.judge import parse_score
from harness.record import ToolCall, TurnRecord


def _record() -> TurnRecord:
    record = TurnRecord(text="Here are two racks.")
    record.tool_calls = [
        ToolCall(0, "search_policies", {"query": "bike rack"}),
        ToolCall(0, "search_products", {"query": "trunk bike rack"}),
    ]
    record.ui = [
        ("brief", {"title": "Bike racks", "sections": [{"heading": "Mount"}] * 3, "sources": ["guide-bike-racks"]}),
        ("comparison", {"products": [{"product_id": "B0EXACT001"}, {"product_id": "B0SUBST001"}, {"product_id": "B0IRREL001"}]}),
    ]
    record.cart_before, record.cart_ids = ["B0CART0001"], ["B0CART0001"]
    record.memory_before = [{"key": "style", "value": "minimal"}]
    record.memory = [{"key": "style", "value": "minimal"}]
    return record


def test_vocabulary_and_outcome_problems():
    good = Case(
        id="f1-900",
        maps_to=["F1.5"],
        turns=["How do I pick a bike rack?"],
        expected={
            "calls_tool": ["search_policies"],
            "any_of": [{"ui_components": ["comparison"]}, {"ui_components": ["products"]}],
            "rubric": {"question": "q", "anchors": {"1": "a", "3": "b", "5": "c"}, "threshold": 4},
        },
    )
    assert outcome_problems(good) == []
    assert good.clause == "F1.5"  # derived from maps_to
    bad = Case(
        id="f1-901",
        turns=["x"],
        expected={"skill_loaded": "buying-advisor", "same_round": [["a", "b"]], "rubric": "PASS if x. FAIL if y."},
    )
    problems = outcome_problems(bad)
    assert any("maps_to" in p for p in problems)
    assert any("same_round" in p and "skill_loaded" in p for p in problems)
    assert any("rubric must be an object" in p for p in problems)
    assert vocabulary_violations({"any_of": [{"first_tool": "x"}, {"calls_tool": ["y"]}]}) == ["first_tool"]
    weak = Case(
        id="f1-902", maps_to=["F1.1"], turns=["x"],
        expected={"rubric": {"question": "q", "anchors": {"1": "a", "5": "c"}, "threshold": 1}},
    )
    assert any("anchors missing" in p for p in outcome_problems(weak))


def test_unchanged_graders_and_relevance(tmp_path):
    record = _record()
    assert all(o.passed for o in grade("cart_unchanged", True, record))
    assert all(o.passed for o in grade("memory_unchanged", True, record))
    record.cart_ids = ["B0CART0001", "B0NEW00001"]
    record.memory = [{"key": "style", "value": "minimal"}, {"key": "bike", "value": "two"}]
    assert not all(o.passed for o in grade("cart_unchanged", True, record))
    assert not all(o.passed for o in grade("memory_unchanged", True, record))
    # relevance labels from a small world
    (tmp_path / "queries_gold.json").write_text(
        json.dumps({"queries": [{"query_id": 7, "query": "bike rack for sedan", "E": ["B0EXACT001"], "S": ["B0SUBST001"], "C": [], "I": ["B0IRREL001"]}]})
    )
    graders.configure_world(tmp_path)
    (ok,) = grade("presented_relevance", {"query_id": 7, "min_share": 0.6}, record)
    assert ok.passed, ok.detail  # 2 of 3 judged are E/S
    (strict,) = grade("presented_relevance", {"query": "Bike Rack for Sedan", "min_share": 0.7}, record)
    assert not strict.passed
    (unknown,) = grade("presented_relevance", {"query": "no such query"}, record)
    assert not unknown.passed and "no judged query" in unknown.detail
    assert graders.judged_queries(["sedan"])[0][0] == 7
    graders.configure_world(None)
    (none,) = grade("presented_relevance", {"query_id": 7}, record)
    assert not none.passed and "no relevance labels" in none.detail


def test_graded_rubric_passes_at_threshold_and_keeps_the_score():
    record = _record()
    rubric = {"question": "q", "anchors": {"1": "a", "3": "b", "5": "c"}, "threshold": 4}
    seen = {}

    def judge(r, rec, transcript, reference_response=None):
        seen["rubric"], seen["reference"] = r, reference_response
        return ("PASS", "fits the 4 anchor", 4)

    (outcome,) = grade_turn({"rubric": rubric}, record, [], judge, reference_response="a fine reply")
    assert outcome.passed and outcome.judged and outcome.score == 4
    assert seen["rubric"] is rubric and seen["reference"] == "a fine reply"
    assert "threshold 4" in outcome.detail
    # a legacy three-argument judge still works
    (legacy,) = grade_turn({"rubric": "PASS if x. FAIL if y."}, record, [], lambda r, rec, t: ("FAIL", "no"))
    assert not legacy.passed and legacy.score is None


def test_parse_score():
    assert parse_score('{"score": 4, "reason": "ok"}') == (4, "ok")
    assert parse_score('text {"score": "9", "reason": "x"}')[0] == 5
    assert parse_score("nothing here")[0] is None


def test_majority_pass_rule():
    from experiment.run_eval import aggregate

    trials = [
        {"id": "c", "passed": True, "failures": [], "usage": {"input_tokens": 1}, "rubric_score": 5},
        {"id": "c", "passed": False, "failures": ["rubric: score 3/5"], "usage": {"input_tokens": 1}, "rubric_score": 3},
        {"id": "c", "passed": True, "failures": [], "usage": {"input_tokens": 1}, "rubric_score": 4},
    ]
    (strict,) = aggregate([dict(t) for t in trials], 3, "all")
    (majority,) = aggregate([dict(t) for t in trials], 3, "majority")
    assert not strict["passed"] and majority["passed"]
    assert majority["pass_rate"] == 0.667 and majority["rubric_scores"] == [5, 3, 4]
    assert majority["failures"][0].startswith("(passed by majority)")


def test_outcome_dataclass_has_score():
    assert Outcome("k", True).score is None


def test_presented_ids_reads_id_lists_as_well_as_product_dicts():
    record = TurnRecord()
    record.ui = [
        ("kit", {"parts": [{"label": "sets", "product_ids": ["B0KIT00001", "B0KIT00002"]}]}),
        ("brief", {"related_product_ids": ["B0REL00001"], "sources": ["guide-bike-racks"]}),
        ("products", {"products": [{"product_id": "B0PROD0001", "title": "x"}]}),
    ]
    assert record.presented_ids() == {"B0KIT00001", "B0KIT00002", "B0REL00001", "B0PROD0001"}
