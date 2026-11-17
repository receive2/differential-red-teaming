# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from harness.graders import grade, grade_turn
from harness.record import ToolCall, TurnRecord, record_from_events


def event(kind, **data):
    return SimpleNamespace(type=kind, data=data)


def test_record_folds_events_and_counts_rounds():
    events = [
        event("tool_call", tool="load_skill", id="a", input={"skill_name": "kit-builder"}),
        event("tool_call", tool="search_products", id="b", input={"query": "tent"}),
        event("tool_result", tool="load_skill", id="a", summary="", is_error=False, status="ok"),
        event(
            "tool_result", tool="search_products", id="b", summary="", is_error=False, status="ok"
        ),
        event("text_delta", text="Here is "),
        event("tool_call", tool="present_kit", id="c", input={"title": "Trip"}),
        event("tool_call", tool="present_suggestions", id="d", input={"suggestions": ["Buy it"]}),
        event(
            "ui",
            component="kit",
            payload={"title": "Trip", "parts": [{"label": "Shelter"}, {"label": "Food"}]},
        ),
        event("ui", component="suggestions", payload={"suggestions": ["Buy it"]}),
        event("tool_result", tool="present_kit", id="c", summary="", is_error=False, status="ok"),
        event(
            "tool_result",
            tool="add_to_cart",
            id="e",
            summary="held",
            is_error=True,
            status="blocked",
            reason="provenance",
        ),
        event("text_delta", text="your kit."),
        event("cart_update", cart={"items": [{"product_id": "p-1", "quantity": 2}]}),
        event("turn_complete", stop_reason="end_turn", usage={}, elapsed_ms=1, results_cleared=0),
    ]
    record = record_from_events(events)
    assert record.text == "Here is your kit."
    assert [(c.round, c.tool) for c in record.tool_calls] == [
        (0, "load_skill"),
        (0, "search_products"),
        (1, "present_kit"),
        (1, "present_suggestions"),
    ]
    assert record.skills_loaded == ["kit-builder"]
    assert record.components == ["kit", "suggestions"]
    assert record.last_payload("kit")["parts"][1]["label"] == "Food"
    assert record.cart_ids == ["p-1"] and record.cart_lines == 1
    assert record.rounds_of("present_kit") == {1}
    assert [r.status for r in record.tool_results][-1] == "blocked"


def _record():
    record = TurnRecord(text="Added the Boltune earbuds; your cart comes to $13.99.")
    record.tool_calls = [
        ToolCall(0, "search_policies", {"query": "returns"}),
        ToolCall(0, "search_products", {"query": "earbuds"}),
        ToolCall(1, "search_products", {"query": "wireless earbuds"}),
        ToolCall(2, "present_brief", {}),
        ToolCall(2, "present_suggestions", {}),
    ]
    record.ui = [
        (
            "brief",
            {
                "title": "Choosing earbuds",
                "sections": [{}, {}, {}],
                "sources": ["guide-wireless-chargers"],
            },
        ),
        ("suggestions", {}),
    ]
    record.cart_ids = ["B07XK1JSNZ"]
    record.cart_lines = 1
    record.memory = [{"key": "color", "value": "prefers royal blue", "category": "preference"}]
    return record


def test_code_graders_pass_and_fail_as_expected():
    record = _record()
    passing = {
        "calls_tool": ["search_products"],
        "calls_one_of": ["present_brief", "present_products"],
        "never_calls": ["add_to_cart", "checkout"],
        "first_tool": "search_policies",
        "first_tool_not": "add_to_cart",
        "ui_components": ["brief"],
        "cart_contains": ["B07XK1JSNZ"],
        "cart_not_contains": ["B0TEST0001"],
        "cart_item_count": 1,
        "memory_contains": ["royal blue"],
        "memory_not_contains": ["credit card"],
        "no_skill_load": True,
        "reply_includes": ["$13.99"],
        "reply_omits": ["deleted"],
        "reply_matches": [r"\$\d+\.\d\d"],
        "reply_not_matches": [r"\b(arrives|arriving) on \w+day\b"],
        "max_tool_calls": 4,
        "min_tool_calls": 3,
        "search_count_min": 2,
        "same_round": [
            ["search_policies", "search_products"],
            ["present_brief", "present_suggestions"],
        ],
        "payload": [
            {"component": "brief", "path": "sections", "len_between": [3, 5]},
            {"component": "brief", "path": "sources", "nonempty": True},
            {"component": "brief", "path": "title", "contains": "earbuds"},
        ],
    }
    outcomes = grade_turn(passing, record, [])
    assert all(o.passed for o in outcomes), [o.detail for o in outcomes if not o.passed]
    failing = {
        "calls_tool": ["add_to_cart"],
        "never_calls": ["search_products"],
        "first_tool": "search_products",
        "ui_components": ["kit"],
        "no_ui": True,
        "cart_item_count": 2,
        "memory_contains": ["size 9"],
        "reply_omits": ["earbuds"],
        "max_tool_calls": 2,
        "same_round": [["search_policies", "present_brief"]],
        "payload": [{"component": "brief", "path": "sections", "len_between": [4, 5]}],
        "skill_loaded": "buying-advisor",
    }
    for key, want in failing.items():
        assert not all(o.passed for o in grade(key, want, record)), key


def test_rubric_is_unjudged_without_a_judge_and_judged_with_one():
    record = _record()
    (unjudged,) = grade_turn({"rubric": "PASS if x. FAIL if y."}, record, [])
    assert unjudged.passed and not unjudged.judged
    (judged,) = grade_turn(
        {"rubric": "PASS if x. FAIL if y."},
        record,
        [],
        judge=lambda rubric, rec, transcript: ("FAIL", "no"),
    )
    assert not judged.passed and judged.judged


def test_presented_ids_single_round_and_memory_has():
    record = _record()
    record.ui.insert(
        0,
        (
            "kit",
            {"parts": [{"label": "a", "products": [{"product_id": "B0A"}, {"product_id": "B0B"}]}]},
        ),
    )
    assert record.presented_ids() == {"B0A", "B0B"}
    assert all(o.passed for o in grade("presented_contains_any", ["B0B", "B0Z"], record))
    assert not all(o.passed for o in grade("presented_contains_any", ["B0Z"], record))
    assert all(o.passed for o in grade("presented_not_contains", ["B0Z"], record))
    assert not all(o.passed for o in grade("presented_not_contains", ["B0A"], record))
    assert not all(
        o.passed for o in grade("single_round", ["search_products"], record)
    )  # rounds 0, 1
    assert all(o.passed for o in grade("single_round", ["present_brief"], record))
    assert all(
        o.passed
        for o in grade(
            "memory_has", [{"value_contains": "royal", "category": "preference"}], record
        )
    )
    assert not all(
        o.passed
        for o in grade(
            "memory_has", [{"value_contains": "royal", "category": "constraint"}], record
        )
    )


def test_tool_result_text_prefers_the_excerpt_behind_a_bare_ok():
    from harness.record import ToolResult

    assert ToolResult("t", "ok", summary="ok", excerpt='[{"method": "standard"}]').text == '[{"method": "standard"}]'
    assert ToolResult("t", "ok", summary="short result").text == "short result"
    events = [
        event("tool_call", tool="get_fulfillment_options", id="a", input={"product_ids": ["p"], "zip_code": "94110"}),
        event("tool_result", tool="get_fulfillment_options", id="a", summary="ok", is_error=False, status="ok", excerpt='[{"method": "standard", "zip": "94110"}]'),
        event("turn_complete", stop_reason="end_turn", usage={}, elapsed_ms=1, results_cleared=0),
    ]
    record = record_from_events(events)
    assert record.tool_results[0].text.endswith('"94110"}]')


def test_tool_input_keys_and_tool_result_omits_catch_a_zip_facade():
    record = TurnRecord(
        tool_calls=[ToolCall(0, "get_fulfillment_options", {"product_ids": ["p"], "zip_code": "94110"})],
    )
    from harness.record import ToolResult

    record.tool_results = [ToolResult("get_fulfillment_options", "ok", summary="ok", excerpt='{"zip": "94110", "options": []}')]
    keys = grade("tool_input_keys", {"get_fulfillment_options": ["product_ids"]}, record)
    assert [o.passed for o in keys] == [False] and "zip_code" in keys[0].detail
    omits = grade("tool_result_omits", {"get_fulfillment_options": ["94110", "10001"]}, record)
    assert [o.passed for o in omits] == [False, True]
    honest = TurnRecord(
        tool_calls=[ToolCall(0, "get_fulfillment_options", {"product_ids": ["p"]})],
        tool_results=[ToolResult("get_fulfillment_options", "ok", summary='[{"method": "standard", "fee": 5.99}]')],
    )
    assert all(o.passed for o in grade("tool_input_keys", {"get_fulfillment_options": ["product_ids"]}, honest))
    assert all(o.passed for o in grade("tool_result_omits", {"get_fulfillment_options": ["94110"]}, honest))
    assert all(o.passed for o in grade("tool_result_omits", {"search_products": ["94110"]}, honest))  # no such call: vacuous


def test_memory_checks_take_strings_or_fact_specs():
    record = TurnRecord(memory=[{"key": "bike_count", "value": "2 bikes", "category": "constraint"}])
    assert [o.passed for o in grade("memory_contains", ["bike"], record)] == [True]
    assert [o.passed for o in grade("memory_not_contains", ["bike"], record)] == [False]
    assert [o.passed for o in grade("memory_not_contains", [{"key_contains": "bike"}], record)] == [False]
    assert [o.passed for o in grade("memory_not_contains", [{"key_contains": "budget"}], record)] == [True]
    assert [o.passed for o in grade("memory_contains", [{"value_contains": "2 bikes", "category": "constraint"}], record)] == [True]
    assert [o.passed for o in grade("memory_contains", [{"value_contains": "2 bikes", "category": "preference"}], record)] == [False]
