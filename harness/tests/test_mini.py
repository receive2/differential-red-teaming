# SPDX-License-Identifier: Apache-2.0

import pytest

from harness.mini.agent import DefaultAgent
from harness.mini.environment import LocalEnvironment
from harness.mini.exceptions import FormatError
from harness.mini.models import LEDGER, CostLedger, ScriptedModel, price_of


def block(command: str) -> str:
    return f"THOUGHT: next step.\n\n```mswea_bash_command\n{command}\n```"


def test_scripted_agent_runs_commands_and_submits(tmp_path):
    (tmp_path / "hello.txt").write_text("hi\n")
    model = ScriptedModel(
        [
            block("cat hello.txt"),
            block("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho read the file"),
        ]
    )
    env = LocalEnvironment(cwd=str(tmp_path), command_budget=5)
    agent = DefaultAgent(model, env, step_limit=10, cost_limit=100)
    result = agent.run("Read hello.txt and finish.")
    assert result["exit_status"] == "Submitted" and result["submission"].strip() == "read the file"
    observations = [m for m in agent.messages if m["role"] == "user"][1:]
    assert "<output>\nhi" in observations[0]["content"]
    assert env.commands_run == 2 and agent.n_calls == 2
    assert "budget of 5 commands" in agent.messages[1]["content"]


def test_format_errors_are_fed_back_then_end_the_run(tmp_path):
    model = ScriptedModel(["no block here", "still none", "nope"])
    env = LocalEnvironment(cwd=str(tmp_path))
    agent = DefaultAgent(model, env, step_limit=10, max_consecutive_format_errors=3)
    result = agent.run("x")
    assert result["exit_status"] == "RepeatedFormatError"
    assert "Expected exactly 1 action, found 0" in agent.messages[3]["content"]


def test_blocked_commands_and_budget(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), command_budget=1)
    blocked = env.execute({"command": "curl https://example.com"})
    assert blocked["returncode"] == 126 and "not permitted" in blocked["output"]
    assert env.execute({"command": "echo ok"})["output"].strip() == "ok"
    spent = env.execute({"command": "echo again"})
    assert spent["returncode"] == 125 and "budget" in spent["output"]
    assert env.execute({"command": "cat ~/.ssh/id_rsa"})["returncode"] == 126


def test_step_limit_ends_the_run(tmp_path):
    model = ScriptedModel([block("echo 1"), block("echo 2"), block("echo 3")])
    agent = DefaultAgent(model, LocalEnvironment(cwd=str(tmp_path)), step_limit=2)
    assert agent.run("x")["exit_status"] == "LimitsExceeded"


def test_ledger_prices_and_cap():
    assert price_of("gpt-5.6-luna-2026") == (0.20e-6, 1.20e-6, 0.02e-6)
    assert price_of("claude-haiku-4-5") is not None and price_of("mystery") is None
    ledger = CostLedger(cap=0.25)
    charged = ledger.charge("gpt-5.6-luna", 1_000_000, 100_000, 500_000)
    assert abs(charged - (0.5 * 0.20 + 0.1 * 1.20 + 0.5 * 0.02)) < 1e-9  # $0.23
    assert ledger.charge("mystery", 5, 5) == 0.0 and "mystery" in ledger.summary()["unpriced"]
    with pytest.raises(Exception, match="cap"):
        ledger.charge("claude-haiku-4-5", 10_000, 10_000)  # +$0.06 crosses the cap
    assert LEDGER.cap >= 0


def test_model_query_parses_one_action_and_charges():
    before = LEDGER.total
    model = ScriptedModel([block("ls")])
    message = model.query([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert message["extra"]["actions"] == [{"command": "ls"}]
    assert LEDGER.total >= before
    two = "two\n```mswea_bash_command\na\n```\n```mswea_bash_command\nb\n```"
    with pytest.raises(FormatError):
        ScriptedModel([two]).query([])


def test_an_open_model_on_deepinfra_is_asked_as_configured(monkeypatch):
    """The request an OpenAI-compatible host gets, and what is read from its answer. No network."""
    from harness.mini.models import DeepInfraModel, OpenAIChatModel, make_model

    sent = {}

    def fake_post(self, url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return {
            "choices": [{"message": {"content": "<think>hidden</think>\nTHOUGHT: look\n\n```mswea_bash_command\nls\n```"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5000, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 4800}},
        }

    monkeypatch.setattr(OpenAIChatModel, "_post", fake_post)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "another-key")
    monkeypatch.setenv("TESTBED_REASONING_EFFORT", "high")  # an OpenAI setting; not this host's
    monkeypatch.delenv("DEEPINFRA_BASE_URL", raising=False)
    model = make_model("deepinfra:deepseek-ai/DeepSeek-V4-Pro-0813", max_tokens=3000, reasoning_effort="none")
    assert isinstance(model, DeepInfraModel)
    before = LEDGER.total
    message = model.query([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert sent["url"] == "https://api.deepinfra.com/v1/openai/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer test-key"}
    assert sent["body"]["model"] == "deepseek-ai/DeepSeek-V4-Pro-0813" and sent["body"]["reasoning_effort"] == "none"
    assert sent["body"]["max_tokens"] == 3000 and "max_completion_tokens" not in sent["body"]  # no reasoning: no headroom
    assert message["extra"]["actions"] == [{"command": "ls"}] and "<think>" not in message["content"]
    charged = LEDGER.total - before  # 200 fresh input tokens, 4800 read from the cache, 40 out
    assert abs(charged - (200 * 1.30e-6 + 4800 * 0.10e-6 + 40 * 2.60e-6)) < 1e-12
    # With the model's default reasoning nothing is sent about it, and the cap has headroom.
    make_model("deepinfra:deepseek-ai/DeepSeek-V4-Pro-0813", max_tokens=3000).complete_text([{"role": "user", "content": "u"}])
    assert "reasoning_effort" not in sent["body"] and sent["body"]["max_tokens"] == 3000 + 8000
    # The OpenAI path is what it was: its own key and cap field, the environment's effort.
    make_model("openai:gpt-5.6-luna", max_tokens=3000).complete_text([{"role": "user", "content": "u"}])
    assert sent["url"] == "https://api.openai.com/v1/chat/completions" and sent["headers"] == {"Authorization": "Bearer another-key"}
    assert sent["body"]["max_completion_tokens"] == 11000 and sent["body"]["reasoning_effort"] == "high" and "max_tokens" not in sent["body"]


def test_thinking_off_reaches_each_kind_of_developer():
    from experiment.facade_rate import model_tag_of
    from experiment.loop import thinking_options

    assert thinking_options("anthropic:claude-sonnet-5", "disabled") == {"anthropic_thinking": "disabled"}
    assert thinking_options("deepinfra:deepseek-ai/DeepSeek-V4-Pro-0813", "disabled") == {"reasoning_effort": "none"}
    assert thinking_options("deepinfra:deepseek-ai/DeepSeek-V4-Pro-0813", None) == {} and thinking_options("openai:gpt-5.6-luna", "disabled") == {}
    # A slot's path: the names of the earlier developers are what they were.
    assert model_tag_of("openai:gpt-5.6-luna", None) == "gpt-5.6-luna" and model_tag_of("anthropic:claude-sonnet-5", "disabled") == "claude-sonnet-5-thinking-off"
    assert model_tag_of("deepinfra:deepseek-ai/DeepSeek-V4-Pro-0813", "disabled") == "deepseek-ai_DeepSeek-V4-Pro-0813-thinking-off"
