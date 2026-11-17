# SPDX-License-Identifier: Apache-2.0

"""What one turn did, read off the event stream and the session: the reply text, the tool
calls with the round each sat in, the tool results and gates, the components and their
payloads, the cart and the memory before and after the turn, and the skills loaded."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    round: int
    tool: str
    input: dict[str, Any]
    id: str = ""


@dataclass
class ToolResult:
    tool: str
    status: str  # ok | error | blocked
    reason: str | None = None
    summary: str = ""
    excerpt: str | None = None  # the result's text when it was too long for the summary

    @property
    def text(self) -> str:
        """What the model was given, as far as the event stream carries it: the summary
        holds the whole result when it is short, otherwise ``ok`` and an excerpt."""
        if self.excerpt and (not self.summary or self.summary == "ok"):
            return self.excerpt
        return self.summary or ""


@dataclass
class TurnRecord:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    ui: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # (component, payload)
    cart_ids: list[str] = field(default_factory=list)
    cart_lines: int = 0
    memory: list[dict[str, Any]] = field(default_factory=list)  # facts after the turn
    cart_before: list[str] = field(default_factory=list)  # the cart's ids when the turn began
    memory_before: list[dict[str, Any]] = field(default_factory=list)  # facts when it began
    stop_reason: str | None = None
    error: str | None = None
    previous: list[TurnRecord] = field(default_factory=list)  # earlier turns of the case
    usage: dict[str, int] = field(default_factory=dict)  # the turn's model usage totals

    @property
    def tool_names(self) -> list[str]:
        return [call.tool for call in self.tool_calls]

    @property
    def components(self) -> list[str]:
        return [component for component, _ in self.ui]

    @property
    def skills_loaded(self) -> list[str]:
        return [
            str(c.input.get("skill_name", "")) for c in self.tool_calls if c.tool == "load_skill"
        ]

    def last_payload(self, component: str) -> dict[str, Any] | None:
        for name, payload in reversed(self.ui):
            if name == component:
                return payload
        return None

    def rounds_of(self, tool: str) -> set[int]:
        return {call.round for call in self.tool_calls if call.tool == tool}

    def presented_ids(self) -> set[str]:
        """Every product id carried by a component this turn, however the payload nests it: a
        ``product_id`` value, or a string under an ``ids`` key (``product_ids``,
        ``related_product_ids``) when the payload lists ids without expanding them."""
        found: set[str] = set()

        def walk(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                pid = value.get("product_id")
                if isinstance(pid, str):
                    found.add(pid)
                for name, inner in value.items():
                    walk(inner, str(name))
            elif isinstance(value, list):
                for inner in value:
                    walk(inner, key)
            elif isinstance(value, str) and key.endswith("ids") and value.strip():
                found.add(value.strip())

        for _component, payload in self.ui:
            walk(payload)
        return found


def record_from_events(events: list[Any]) -> TurnRecord:
    """Fold a turn's events. A round boundary is a ``tool_call`` arriving after a
    ``tool_result`` in the current round (the loop yields a round's calls, then its
    results, then the next round's calls)."""
    record = TurnRecord()
    round_index = 0
    results_seen_this_round = False
    text_parts: list[str] = []
    for event in events:
        kind, data = event.type, event.data
        if kind == "text_delta":
            text_parts.append(data.get("text", ""))
        elif kind == "tool_call":
            if results_seen_this_round:
                round_index += 1
                results_seen_this_round = False
            record.tool_calls.append(
                ToolCall(
                    round_index, data["tool"], dict(data.get("input") or {}), data.get("id", "")
                )
            )
        elif kind == "tool_result":
            results_seen_this_round = True
            record.tool_results.append(
                ToolResult(
                    data["tool"],
                    data.get("status", "ok"),
                    data.get("reason"),
                    data.get("summary", ""),
                    data.get("excerpt"),
                )
            )
        elif kind == "ui":
            record.ui.append((data["component"], dict(data.get("payload") or {})))
        elif kind == "cart_update":
            cart = data.get("cart") or {}
            items = cart.get("items") or []
            record.cart_ids = [str(item.get("product_id")) for item in items]
            record.cart_lines = len(items)
        elif kind == "turn_complete":
            record.stop_reason = data.get("stop_reason")
            record.usage = {
                k: int(v)
                for k, v in (data.get("usage") or {}).items()
                if isinstance(v, int | float)
            }
        elif kind == "error":
            record.error = data.get("message")
    record.text = "".join(text_parts)
    return record
