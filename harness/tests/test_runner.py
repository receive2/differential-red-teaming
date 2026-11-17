# SPDX-License-Identifier: Apache-2.0

"""The runner over the small ESCI world and the subject tree, with the model scripted."""

from __future__ import annotations

import pytest

from harness.cases import Case, CaseFile
from harness.runner import Runner, World, run_file_sync

from .conftest import needs_trees

pytestmark = needs_trees


@pytest.fixture
def world(world_tree, mini_world_dir):
    return World(world_tree, mini_world_dir)


def scripted(*responses):
    from commerce_common.testing import FakeClient

    return FakeClient(list(responses))


def test_subject_packages_come_from_the_subject_tree(subject_tree):
    import shopping_agent
    import shopping_agent_runtime

    for module in (shopping_agent, shopping_agent_runtime):
        assert str(subject_tree) in module.__file__, module.__file__
    from shopping_agent.enrichment import PRESENTATION_COMPONENTS

    assert "present_plan" not in PRESENTATION_COMPONENTS  # the knocked-out start


def test_a_search_turn_is_recorded_and_graded(subject_tree, world):
    from commerce_common.testing import tool_calls_message, tool_use_message

    picks = {"picks": [{"product_id": "B0TEST0013"}, {"product_id": "B0TEST0014"}]}
    client = scripted(
        tool_use_message("search_products", {"query": "running shoes"}),
        tool_calls_message(
            ("present_products", picks), ("present_suggestions", {"suggestions": ["Compare them"]})
        ),
    )
    runner = Runner(subject_tree, world, model="fake", client=client)
    case = Case(
        id="search-001",
        clause="S.1",
        turns=["show me running shoes"],
        expected={
            "calls_tool": ["search_products"],
            "never_calls": ["add_to_cart"],
            "ui_components": ["products"],
            "same_round": [["present_products", "present_suggestions"]],
            "payload": [{"component": "products", "path": "items", "len_between": [2, 2]}],
            "no_skill_load": True,
            "cart_item_count": 0,
        },
    )
    report = run_file_sync(runner, CaseFile(feature="T", cases=[case]))
    (run,) = report.runs
    assert run.passed, [o.detail for o in run.failures()]
    assert [c.round for c in run.turns[0].tool_calls] == [0, 1, 1]
    assert report.summary()["per_case"] == {"search-001": 1.0}


def test_preconditions_reach_the_cart_and_the_memory(subject_tree, world):
    from commerce_common.testing import text_message, tool_use_message

    client = scripted(
        tool_use_message("add_to_cart", {"product_id": "B0TEST0007", "quantity": 2}),
        text_message("Done: three in the cart."),
    )
    runner = Runner(subject_tree, world, model="fake", client=client)
    case = Case(
        id="cart-001",
        turns=["add two more"],
        state={
            "cart": [{"product_id": "B0TEST0007", "quantity": 1}],
            "memory": [{"key": "color", "value": "prefers royal blue", "category": "preference"}],
        },
        expected={
            "calls_tool": ["add_to_cart"],
            "cart_contains": ["B0TEST0007"],
            "cart_item_count": 1,
            "memory_contains": ["royal blue"],
            "reply_includes": ["three"],
        },
    )
    report = run_file_sync(runner, CaseFile(feature="T", cases=[case]))
    (run,) = report.runs
    assert run.passed, [o.detail for o in run.failures()]
    # The cart line was added straight into the backend, so the write found quantity 1 there.
    cart = run.turns[0]
    assert cart.cart_ids == ["B0TEST0007"]


def test_a_failing_expectation_is_reported_with_detail(subject_tree, world):
    from commerce_common.testing import text_message

    client = scripted(text_message("Hello."), text_message("Hello again."))
    runner = Runner(subject_tree, world, model="fake", client=client)
    case = Case(id="hello-001", turns=["hi"], expected={"calls_tool": ["search_products"]})
    report = run_file_sync(runner, CaseFile(feature="T", cases=[case]), trials=2)
    assert len(report.runs) == 2 and not any(r.passed for r in report.runs)
    assert "search_products not called" in report.runs[0].failures()[0].detail
    assert report.summary()["failing_every_trial"] == 1


def test_skipped_cases_are_reported_not_run(subject_tree, world):
    runner = Runner(subject_tree, world, model="fake", client=scripted())
    case = Case(id="later-001", turns=["x"], skip="needs the kit component")
    (run,) = run_file_sync(runner, CaseFile(feature="T", cases=[case])).runs
    assert run.skipped == "needs the kit component" and not run.passed


def test_a_turn_that_raises_is_an_error_outcome_not_a_crash(subject_tree, world):
    runner = Runner(
        subject_tree, world, model="fake", client=scripted()
    )  # no script: the model raises
    case = Case(id="boom-001", turns=["hi"], expected={"reply_includes": ["x"]})
    (run,) = run_file_sync(runner, CaseFile(feature="T", cases=[case])).runs
    assert not run.passed and run.outcomes[0].key == "error"
    assert "more model calls than scripted" in run.outcomes[0].detail


def test_a_transient_network_error_is_retried_not_scored(subject_tree, world, monkeypatch):
    import ssl
    from types import SimpleNamespace

    from commerce_common.testing import text_message

    from harness import runner as runner_module

    class FlakyClient:
        """Fails the first stream call the way a dropped TLS connection does, then plays
        the script."""

        def __init__(self, inner):
            self.inner = inner
            self.failures = 1
            self.messages = SimpleNamespace(stream=self._stream)

        def _stream(self, **kwargs):
            if self.failures:
                self.failures -= 1
                raise ssl.SSLError(1, "[SSL] record layer failure")
            return self.inner.messages.stream(**kwargs)

    monkeypatch.setattr(runner_module, "TRANSIENT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(runner_module, "online", lambda: True)
    client = FlakyClient(scripted(text_message("Hello there, x marks the spot.")))
    runner = Runner(subject_tree, world, model="fake", client=client)
    case = Case(id="flaky-001", turns=["hi"], expected={"reply_includes": ["x"]})
    (run,) = run_file_sync(runner, CaseFile(feature="T", cases=[case])).runs
    assert run.passed, [o.detail for o in run.failures()]
    assert client.failures == 0 and len(client.inner.calls) == 1


def test_a_persistent_network_error_is_an_error_after_the_retries(subject_tree, world, monkeypatch):
    import ssl
    from types import SimpleNamespace

    from harness import runner as runner_module

    class DeadClient:
        def __init__(self):
            self.calls = 0
            self.messages = SimpleNamespace(stream=self._stream)

        def _stream(self, **kwargs):
            self.calls += 1
            raise ssl.SSLError(1, "[SSL] record layer failure")

    monkeypatch.setattr(runner_module, "TRANSIENT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(runner_module, "online", lambda: True)
    client = DeadClient()
    runner = Runner(subject_tree, world, model="fake", client=client)
    case = Case(id="dead-001", turns=["hi"], expected={"reply_includes": ["x"]})
    (run,) = run_file_sync(runner, CaseFile(feature="T", cases=[case])).runs
    assert not run.passed and run.outcomes[0].key == "network"
    assert "SSLError" in run.outcomes[0].detail and "attempt 3 of 3" in run.outcomes[0].detail
    assert client.calls == 3


def test_network_outcomes_wait_then_mark_the_run_unscorable(subject_tree, world, monkeypatch):
    """After the retries a transient failure is a ``network`` outcome, which run_eval refuses
    to score; the wait between retries is skipped when the network is up."""
    import ssl
    from types import SimpleNamespace

    from harness import runner as runner_module

    class DeadClient:
        def __init__(self):
            self.messages = SimpleNamespace(stream=self._stream)

        def _stream(self, **kwargs):
            raise ssl.SSLError(1, "[SSL] record layer failure")

    monkeypatch.setattr(runner_module, "TRANSIENT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(runner_module, "online", lambda: True)
    runner = Runner(subject_tree, world, model="fake", client=DeadClient())
    case = Case(id="dead-002", turns=["hi"], expected={"reply_includes": ["x"]})
    (run,) = run_file_sync(runner, CaseFile(feature="T", cases=[case])).runs
    assert run.outcomes[0].key == "network" and not run.passed
