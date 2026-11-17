# SPDX-License-Identifier: Apache-2.0

from harness.graders import grade_turn
from harness.record import ToolCall, TurnRecord


def _record(tools, components):
    record = TurnRecord()
    record.tool_calls = [ToolCall(round=0, tool=t, input={}) for t in tools]
    record.ui = [(c, {}) for c in components]
    return record


def test_any_of_passes_when_one_alternative_passes():
    record = _record(["load_skill", "present_suggestions"], ["suggestions"])
    expected = {
        "any_of": [
            {"ui_components": ["brief"]},
            {"never_calls": ["present_brief"], "ui_components": ["suggestions"]},
        ]
    }
    outcomes = grade_turn(expected, record, [], None)
    assert outcomes and all(o.passed for o in outcomes)


def test_any_of_reports_closest_alternative_when_none_passes():
    record = _record(["present_products"], ["products"])
    expected = {
        "any_of": [
            {"ui_components": ["brief"]},
            {"ui_components": ["kit"], "calls_tool": ["add_to_cart"]},
        ]
    }
    outcomes = grade_turn(expected, record, [], None)
    assert not all(o.passed for o in outcomes)
    assert outcomes[-1].key == "any_of"
    # the closest alternative (one failure) is the one reported
    assert sum(not o.passed for o in outcomes) == 2
