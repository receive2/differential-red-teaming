# SPDX-License-Identifier: Apache-2.0

"""Model adapters for the vendored agent, written on the HTTP APIs with httpx: no litellm,
no provider SDK. ``TextbasedModel`` holds the shared protocol (one ```mswea_bash_command
block per reply, observations rendered from a template, cost from a price table and a
global ledger with a hard cap); ``OpenAIChatModel`` and ``AnthropicModel`` do the calls;
``ScriptedModel`` plays back replies for tests."""

from __future__ import annotations

import logging
import os
import re
import ssl
import sys
import threading
import time
from typing import Any

import httpx
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from ..netwait import (
    ACCOUNT_PAUSE_SECONDS,
    ACCOUNT_WAIT_SECONDS,
    account_error,
    online,
    wait_for_network,
)
from .exceptions import FormatError
from .templates import FORMAT_ERROR_TEMPLATE, OBSERVATION_TEMPLATE

logger = logging.getLogger("mini.models")

# Dollars per token: (input, output, cached input). A model not listed costs 0 and is
# flagged in the ledger. The Anthropic rows are the pricing page as read on 2026-11-11
# (https://platform.claude.com/docs/en/about-claude/pricing): Sonnet 5 stayed at $2/$10 (the
# rise to $3/$15 was cancelled; the table before this date had 3/15 and so overstated every
# Sonnet 5 cost by half), Haiku 5.5 is $0.10/$0.50 for prompts up to 100,000 tokens (five
# times that above; no request of this study comes near), and the table before this date
# gave Haiku 5.5 the Haiku 4.5 price. Luna from OpenAI's pricing page and the DeepInfra rows
# from deepinfra.com/pricing, as a research pass read them on 2026-11-11.
PRICES: dict[str, tuple[float, float, float]] = {
    "luna": (0.20e-6, 1.20e-6, 0.02e-6),
    "claude-haiku-4-5": (1.0e-6, 5.0e-6, 0.10e-6),
    "claude-sonnet-5-5": (2.0e-6, 10.0e-6, 0.10e-6),
    "claude-sonnet-5": (2.0e-6, 10.0e-6, 0.20e-6),
    "claude-haiku-5-5": (0.10e-6, 0.50e-6, 0.01e-6),
    "DeepSeek-V4-Pro": (1.30e-6, 2.60e-6, 0.10e-6),
    "DeepSeek-V4.1-Flash": (0.098e-6, 0.294e-6, 0.003e-6),  # a discount with no end date; list 0.20/0.60
}


def price_of(model: str) -> tuple[float, float, float] | None:
    for key, price in PRICES.items():
        if key in model:
            return price
    return None


class BudgetExceeded(RuntimeError):
    """The ledger's hard cap was reached; every further query raises until it is reset."""


class CostLedger:
    """Dollars spent per model across the whole process, with a hard cap."""

    def __init__(self, cap: float = 0.0) -> None:
        self.cap = cap
        self.spent: dict[str, float] = {}
        self.tokens: dict[str, dict[str, int]] = {}
        self.unpriced: set[str] = set()
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        return sum(self.spent.values())

    def charge(self, model: str, prompt: int, completion: int, cached: int = 0, cache_write: int = 0) -> float:
        """``prompt`` counts every input token; ``cached`` of them were read from the cache
        and ``cache_write`` of them were written to it (billed a quarter above the input price)."""
        price = price_of(model)
        if price is None:
            self.unpriced.add(model)
            cost = 0.0
        else:
            cost = (prompt - cached) * price[0] + completion * price[1] + cached * price[2] + cache_write * price[0] * 0.25
        with self._lock:
            self.spent[model] = self.spent.get(model, 0.0) + cost
            bucket = self.tokens.setdefault(
                model, {"prompt": 0, "completion": 0, "cached": 0, "calls": 0}
            )
            bucket["prompt"] += prompt
            bucket["completion"] += completion
            bucket["cached"] += cached
            bucket["calls"] += 1
            if self.cap and self.total > self.cap:
                raise BudgetExceeded(f"spent ${self.total:.2f}, cap ${self.cap:.2f}")
        return cost

    def summary(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total, 4),
            "by_model": {m: round(c, 4) for m, c in self.spent.items()},
            "tokens": self.tokens,
            "unpriced": sorted(self.unpriced),
        }


LEDGER = CostLedger(cap=float(os.environ.get("TESTBED_COST_CAP_USD", "0") or 0))


def _ssl_context() -> ssl.SSLContext | bool:
    """The system trust store when pip's vendored truststore is at hand (a corporate proxy
    may re-sign TLS), else the default bundle."""
    try:
        from pip._vendor import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return True


class TextbasedModelConfig(BaseModel):
    model_name: str
    max_tokens: int = 4000
    temperature: float | None = None
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    format_error_template: str = FORMAT_ERROR_TEMPLATE
    observation_template: str = OBSERVATION_TEMPLATE
    timeout: float = 180.0
    retries: int = 4
    # Anthropic only: "disabled" turns extended thinking off (a Claude 5 model thinks by
    # default and may spend the whole output budget on it); None leaves the model's default.
    anthropic_thinking: str | None = None
    # OpenAI-compatible endpoints: the reasoning effort sent with each request ("none" turns
    # a reasoning model's hidden reasoning off); None leaves the model's default.
    reasoning_effort: str | None = None
    # Anthropic only: mark the system prompt and the newest message as cache breakpoints. A
    # role's conversation only grows, so each call reads the earlier turns from the cache
    # (a tenth of the input price) and writes its new tail (a quarter more). The model sees
    # the same text either way.
    anthropic_cache: bool = True


class TextbasedModel:
    def __init__(self, *, config_class: type = TextbasedModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.client = httpx.Client(verify=_ssl_context(), timeout=self.config.timeout)

    # -- provider hook -------------------------------------------------------------------
    def _complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int], str | None]:
        """``(text, usage, finish_reason)``; usage has prompt, completion, cached."""
        raise NotImplementedError

    def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        attempt = 0
        account_deadline = time.monotonic() + ACCOUNT_WAIT_SECONDS
        while attempt <= self.config.retries:
            try:
                response = self.client.post(url, headers=headers, json=body)
            except httpx.TransportError as error:
                # A network outage is waited out (up to two hours) and is not one of the retries.
                if not online():
                    logger.warning("request failed (%s); the network is down, waiting", error)
                    began = time.monotonic()
                    back = wait_for_network()
                    self.waited_seconds = getattr(self, "waited_seconds", 0.0) + (time.monotonic() - began)
                    if back:
                        continue
                if attempt == self.config.retries:
                    raise
                logger.warning("request failed (%s); retrying in %.0fs", error, delay)
            except httpx.HTTPError as error:
                if attempt == self.config.retries:
                    raise
                logger.warning("request failed (%s); retrying in %.0fs", error, delay)
            else:
                if response.status_code < 400:
                    return response.json()
                if account_error(response.text) and time.monotonic() < account_deadline:
                    # The provider refuses the account (its credit ran out). Not one of the
                    # retries: the call waits until the account is served again.
                    logger.warning("status %s, the provider refuses the account (%s); waiting %.0fs", response.status_code, response.text[:120], ACCOUNT_PAUSE_SECONDS)
                    print(f"models: the provider refuses the account ({response.text[:100]}); waiting", file=sys.stderr, flush=True)
                    time.sleep(ACCOUNT_PAUSE_SECONDS)
                    # time spent waiting for the provider is not the session's (the agent's wall clock leaves it out)
                    self.waited_seconds = getattr(self, "waited_seconds", 0.0) + ACCOUNT_PAUSE_SECONDS
                    continue
                retryable = response.status_code in (408, 409, 429, 500, 502, 503, 504)
                if retryable and attempt < self.config.retries:
                    logger.warning("status %s; retrying in %.0fs", response.status_code, delay)
                else:
                    raise RuntimeError(f"{url} -> {response.status_code}: {response.text[:300]}")
            attempt += 1
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    # -- protocol ---------------------------------------------------------------------------
    @staticmethod
    def _strip(messages: list[dict]) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("system", "user", "assistant")
        ]

    def query(self, messages: list[dict], **kwargs) -> dict:
        text, usage, finish_reason = self._complete(self._strip(messages))
        cost = LEDGER.charge(
            self.config.model_name,
            usage.get("prompt", 0),
            usage.get("completion", 0),
            usage.get("cached", 0),
            usage.get("cache_write", 0),
        )
        extra = {
            "cost": cost,
            "usage": usage,
            "timestamp": time.time(),
            "finish_reason": finish_reason,
        }
        actions = [a.strip() for a in re.findall(self.config.action_regex, text, re.DOTALL)]
        if len(actions) != 1:
            error = f"Expected exactly 1 action, found {len(actions)}."
            content = Template(self.config.format_error_template, undefined=StrictUndefined).render(
                actions=actions, error=error, finish_reason=finish_reason
            )
            raise FormatError(
                {
                    "role": "user",
                    "content": content,
                    "extra": {
                        "interrupt_type": "FormatError",
                        "n_actions": len(actions),
                        "model_response": text,
                        **extra,
                    },
                }
            )
        return {
            "role": "assistant",
            "content": text,
            "extra": {"actions": [{"command": actions[0]}], **extra},
        }

    def complete_text(self, messages: list[dict]) -> str:
        """One plain completion (no action protocol), charged to the ledger: for a role that
        answers with a document rather than a shell command."""
        text, usage, _ = self._complete(self._strip(messages))
        LEDGER.charge(
            self.config.model_name,
            usage.get("prompt", 0),
            usage.get("completion", 0),
            usage.get("cached", 0),
            usage.get("cache_write", 0),
        )
        return text

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        rendered = []
        for output in outputs:
            content = Template(self.config.observation_template, undefined=StrictUndefined).render(
                output=output, **(template_vars or {})
            )
            rendered.append(
                {
                    "role": "user",
                    "content": content,
                    "extra": {
                        "raw_output": output.get("output", ""),
                        "returncode": output.get("returncode"),
                        "timestamp": time.time(),
                        **output.get("extra", {}),
                    },
                }
            )
        return rendered

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


REASONING_HEADROOM = 8000


class OpenAIChatModel(TextbasedModel):
    """Chat Completions over HTTP. ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL``)."""

    BASE_ENV, DEFAULT_BASE, KEY_ENV = "OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY"
    LIMIT_FIELD = "max_completion_tokens"

    def _effort(self) -> str | None:
        return self.config.reasoning_effort or os.environ.get("TESTBED_REASONING_EFFORT") or None

    def _complete(self, messages):
        base = os.environ.get(self.BASE_ENV, self.DEFAULT_BASE).rstrip("/")
        effort = self._effort()
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            # Reasoning models count their reasoning against this cap, so the visible reply
            # gets the configured budget only with headroom on top.
            self.LIMIT_FIELD: self.config.max_tokens + (0 if effort == "none" else REASONING_HEADROOM),
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if effort:
            body["reasoning_effort"] = effort
        headers = {"Authorization": f"Bearer {os.environ[self.KEY_ENV]}"}
        data = self._post(f"{base}/chat/completions", headers, body)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        return (
            self._visible(choice["message"].get("content") or ""),
            {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "cached": cached,
            },
            choice.get("finish_reason"),
        )

    @staticmethod
    def _visible(content: str) -> str:
        return content


class DeepInfraModel(OpenAIChatModel):
    """DeepInfra's OpenAI-compatible Chat Completions (open-weight models it hosts).
    ``DEEPINFRA_API_KEY`` (and optionally ``DEEPINFRA_BASE_URL``). The reasoning effort is the
    configured one only: ``none`` is the model's non-thinking mode."""

    BASE_ENV, DEFAULT_BASE, KEY_ENV = "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"
    LIMIT_FIELD = "max_tokens"

    def _effort(self) -> str | None:
        return self.config.reasoning_effort

    @staticmethod
    def _visible(content: str) -> str:
        # A reasoning trace that comes inline is not part of the reply (the harness keeps the
        # reply only, as for every model).
        return re.sub(r"^\s*<think>.*?</think>\s*", "", content, count=1, flags=re.S)


class AnthropicModel(TextbasedModel):
    """Messages API over HTTP. ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_BASE_URL``)."""

    def _complete(self, messages):
        base = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if self.config.anthropic_cache and turns:
            turns = [dict(m) for m in turns]
            turns[-1]["content"] = [
                {"type": "text", "text": turns[-1]["content"], "cache_control": {"type": "ephemeral"}}
            ]
        # A Claude 5 model thinks before it answers unless told not to, and the thinking counts
        # against max_tokens: without headroom a long thought leaves an empty reply. The same
        # headroom the OpenAI adapter gives its reasoning models.
        thinking_on = self.config.anthropic_thinking != "disabled"
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens + (REASONING_HEADROOM if thinking_on else 0),
            "messages": turns,
        }
        if system:
            body["system"] = (
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if self.config.anthropic_cache
                else system
            )
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.anthropic_thinking == "disabled":
            body["thinking"] = {"type": "disabled"}
        headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
        data = self._post(f"{base.rstrip('/')}/v1/messages", headers, body)
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        usage = data.get("usage", {})
        cached = usage.get("cache_read_input_tokens", 0) or 0
        written = usage.get("cache_creation_input_tokens", 0) or 0
        return (
            text,
            {
                "prompt": usage.get("input_tokens", 0) + cached + written,
                "completion": usage.get("output_tokens", 0),
                "cached": cached,
                "cache_write": written,
            },
            data.get("stop_reason"),
        )


class ScriptedModel(TextbasedModel):
    """Plays back scripted replies (tests and dry runs); a reply may be a callable of the messages."""

    def __init__(self, replies: list[Any], model_name: str = "scripted", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def _complete(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("more model calls than scripted replies")
        reply = self.replies.pop(0)
        text = reply(messages) if callable(reply) else reply
        return text, {"prompt": 10, "completion": 5, "cached": 0}, "stop"


def make_model(spec: str, **kwargs) -> TextbasedModel:
    """``provider:model`` -> adapter; ``openai:gpt-5.6-luna``, ``anthropic:claude-haiku-4-5``,
    ``deepinfra:deepseek-ai/DeepSeek-V4-Pro-0813``."""
    provider, _, name = spec.partition(":")
    if provider == "openai":
        return OpenAIChatModel(model_name=name, **kwargs)
    if provider == "anthropic":
        return AnthropicModel(model_name=name, **kwargs)
    if provider == "deepinfra":
        return DeepInfraModel(model_name=name, **kwargs)
    raise ValueError(f"unknown provider in {spec!r}")
