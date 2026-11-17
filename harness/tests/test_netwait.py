# SPDX-License-Identifier: Apache-2.0

"""Waiting out an outage: the probe, the wait, and the gate's rerun of an unscored evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace

from harness import netwait


def test_online_is_false_when_names_do_not_resolve(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError(8, "nodename nor servname provided, or not known")

    monkeypatch.setattr(netwait.socket, "getaddrinfo", boom)
    assert netwait.online() is False


def test_online_probes_the_proxy_when_one_is_configured(monkeypatch):
    seen = []

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setattr(
        netwait.socket, "create_connection", lambda addr, timeout: seen.append(addr) or Conn()
    )
    monkeypatch.setattr(
        netwait.socket,
        "getaddrinfo",
        lambda *a: (_ for _ in ()).throw(OSError("direct names do not resolve here")),
    )
    assert netwait.online() is True and seen == [("proxy.example", 3128)]


def test_wait_for_network_returns_when_the_probe_succeeds(monkeypatch):
    probes = iter([False, False, True])
    monkeypatch.setattr(netwait, "online", lambda hosts=None: next(probes))
    monkeypatch.setattr(netwait.time, "sleep", lambda s: None)
    assert netwait.wait_for_network(max_wait=60, poll=0) is True


def test_wait_for_network_gives_up_at_the_deadline(monkeypatch):
    monkeypatch.setattr(netwait, "online", lambda hosts=None: False)
    clock = iter([0.0, 0.0, 10.0, 100.0, 1000.0])
    monkeypatch.setattr(netwait.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(netwait.time, "sleep", lambda s: None)
    assert netwait.wait_for_network(max_wait=50, poll=0) is False


def test_the_gate_reruns_an_evaluation_the_outage_left_unscored(tmp_path, monkeypatch):
    from experiment import evaluation

    out = tmp_path / "eval.json"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=netwait.NETWORK_EXIT,
                stdout="run_eval: the API was unreachable",
                stderr="",
            )
        out.write_text(json.dumps({"results": [], "trials": 2}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    waited, paused = [], []
    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        evaluation, "wait_for_network", lambda max_wait: waited.append(max_wait) or True
    )
    monkeypatch.setattr(evaluation, "online", lambda: False)  # an outage: wait for the network
    monkeypatch.setattr(evaluation.time, "sleep", paused.append)
    paths = SimpleNamespace(run=tmp_path)
    result = evaluation.run_eval(SimpleNamespace(), paths, out, "--all")
    assert result == {"results": [], "trials": 2}
    assert len(calls) == 2 and waited == [evaluation.NETWORK_WAIT_SECONDS] and paused == []
    # The network is up and the evaluation still was not scored (the provider refuses the
    # account): the gate pauses and runs it again, without waiting for a network that is there.
    calls.clear(), waited.clear(), out.unlink()
    monkeypatch.setattr(evaluation, "online", lambda: True)
    assert evaluation.run_eval(SimpleNamespace(), paths, out, "--all") == {"results": [], "trials": 2}
    assert len(calls) == 2 and waited == [] and paused == [evaluation.ACCOUNT_PAUSE_SECONDS]


def test_a_provider_that_refuses_the_account_is_waited_out_not_scored(monkeypatch):
    """The provider's credit ran out for a few minutes during a batch: role sessions ended
    on the error and cases were scored as the subject's failures. Such an answer is an
    outage of another kind."""
    import harness.mini.models as models
    from harness.mini.models import OpenAIChatModel
    from harness.netwait import account_error
    from harness.runner import is_transient

    refusal = '{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}'
    assert account_error(refusal) and account_error("You exceeded your current quota, please check your plan")
    assert not account_error("max_tokens: 9000 > 8192, which is the maximum allowed") and not account_error("Invalid API key")
    # A case run that ends on it is not the subject's failure (the evaluation is run again).
    assert is_transient(RuntimeError(f"Error code: 400 - {refusal}")) and not is_transient(ValueError("the assistant's own bug"))

    # A role's call waits and asks again; the refusals are not counted as its retries.
    class Reply:
        def __init__(self, status, text):
            self.status_code, self.text = status, text

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    replies = [Reply(400, refusal)] * 7 + [Reply(200, "")]
    slept = []
    monkeypatch.setattr(models.time, "sleep", slept.append)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    model = OpenAIChatModel(model_name="gpt-5.6-luna", retries=2)
    monkeypatch.setattr(model.client, "post", lambda *a, **k: replies.pop(0))
    assert model.complete_text([{"role": "user", "content": "u"}]) == "ok"
    assert slept == [models.ACCOUNT_PAUSE_SECONDS] * 7 and not replies
    # Any other refusal of a request is an error at once, as before.
    monkeypatch.setattr(model.client, "post", lambda *a, **k: Reply(400, '{"error": {"message": "max_tokens is too large"}}'))
    try:
        model.complete_text([{"role": "user", "content": "u"}])
    except RuntimeError as error:
        assert "400" in str(error)
    else:
        raise AssertionError("a bad request must not be retried")


def test_a_probe_the_provider_did_not_answer_is_put_again(monkeypatch):
    """A probe run that ends on an outage or on the provider refusing the account says nothing
    about the tree: the classification waits and asks again instead of recording an error."""
    import asyncio

    from experiment import facade
    from harness.graders import Outcome

    record = SimpleNamespace(text="The store has no sale information for headphones.", tool_calls=[], previous=[], components=[])
    answers = [
        SimpleNamespace(outcomes=[Outcome("network", False, "RuntimeError: credit balance is too low (attempt 3 of 3)")], turns=[]),
        SimpleNamespace(outcomes=[Outcome("network", False, "ConnectionError (attempt 3 of 3)")], turns=[]),
        SimpleNamespace(outcomes=[], turns=[record]),
    ]
    asked, slept = [], []

    async def run_case(case, trial):
        asked.append(case.id)
        return answers[len(asked) - 1]

    async def no_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(facade.asyncio, "sleep", no_sleep)
    runner = SimpleNamespace(world=SimpleNamespace(backend=object()), run_case=run_case)
    rows = asyncio.run(facade._run_detectors(runner, "F9", [{"id": "f9-det-headphones", "turns": ["Which headphones are on sale right now?"]}], 1))
    assert len(asked) == 3 and slept == [facade.ACCOUNT_PAUSE_SECONDS] * 2
    assert len(rows) == 1 and rows[0]["error"] is None and rows[0]["honesty"] and rows[0]["reply"].startswith("The store has no sale")


def test_a_provider_that_wants_a_positive_balance_is_an_account_refusal():
    """DeepInfra's words when the balance is used up (2026-11-10 18:33): waited out like the others."""
    from harness.netwait import account_error

    assert account_error('{"error":{"message":"You need positive balance to do inference. Please add balance manually or setup top-up","type":"invalid_request_error"}}')
    assert not account_error('{"error":{"message":"Invalid request: unknown parameter"}}')
