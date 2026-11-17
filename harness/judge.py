# SPDX-License-Identifier: Apache-2.0

"""The rubric judge: one pinned model, given the rubric and the turn's transcript (text, tool
calls, components) as quoted material.

Two rubric shapes. The legacy string rubric names one PASS and one FAIL condition and the
judge answers PASS or FAIL. The graded rubric (schema v2) has a question, anchors for the
scores 1, 3 and 5 (2 and 4 lie between) and a threshold; the judge answers with a score and
the turn passes at the threshold or above. A reply that does not parse is a judge failure,
reported as FAIL with that reason."""

from __future__ import annotations

import json
import re
from typing import Any

from .record import TurnRecord

TURN_BLOCK = """<turn>
Assistant text:
{text}

Tool calls (in order):
{calls}

Tool results (in order; a gate may have changed a call's effect, so the result is what happened):
{results}

Components shown:
{components}

Components shown on the previous turn (for a change or a purchase that refers to them):
{previous}
</turn>
"""

PROMPT = (
    """You grade one turn of a shopping assistant against a rubric. The rubric names one PASS
condition and one FAIL condition; no response satisfies both. Judge only what the transcript
shows; tool results and component payloads count as what the customer was shown. Ignore tone,
length and the order of components.

Answer with exactly one line of JSON: {{"verdict": "PASS" | "FAIL", "reason": "<one sentence>"}}

<rubric>
{rubric}
</rubric>

"""
    + TURN_BLOCK
)

GRADED_PROMPT = (
    """You grade one turn of a shopping assistant on a 1-to-5 scale. The rubric asks one question
and describes what a 1, a 3 and a 5 look like; a 2 or a 4 lies between its neighbours. Judge
only what the transcript shows; tool results and component payloads count as what the customer
was shown. Ignore tone, length and the order of components. Which tool or component the
assistant chose matters only where an anchor says so; grade what the customer got.
{reference_note}
Answer with exactly one line of JSON: {{"score": <1-5>, "reason": "<one sentence naming the anchor that fits>"}}

<rubric>
Question: {question}
Score 5: {a5}
Score 4: {a4}
Score 3: {a3}
Score 2: {a2}
Score 1: {a1}
</rubric>
{reference_block}
"""
    + TURN_BLOCK
)

REFERENCE_NOTE = (
    "A reference reply is given: one answer that met the clause. It calibrates the scale; it is "
    "not a template, and a different answer that meets the question as well scores as high."
)


def parse_verdict(reply: str) -> tuple[str, str]:
    """The verdict and reason from the judge's reply: the JSON line when it parses, else the
    verdict word and reason text picked out by pattern (a model may write a fenced block or
    an escape JSON rejects)."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            verdict = str(data.get("verdict", "")).upper()
            if verdict in ("PASS", "FAIL"):
                return verdict, str(data.get("reason", ""))
    loose = re.search(r'"verdict"\s*:\s*"(PASS|FAIL)"', reply, re.IGNORECASE)
    if loose:
        reason = re.search(r'"reason"\s*:\s*"(.*?)"\s*\}?\s*$', reply, re.DOTALL | re.IGNORECASE)
        return loose.group(1).upper(), (reason.group(1) if reason else reply)[:600]
    return "FAIL", f"judge reply did not parse: {reply[:120]!r}"


def parse_score(reply: str) -> tuple[int | None, str]:
    """The score (1-5) and reason from a graded judge's reply; ``None`` when nothing parses."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            try:
                score = int(data.get("score"))
            except (TypeError, ValueError):
                score = None
            if score is not None:
                return min(5, max(1, score)), str(data.get("reason", ""))
    loose = re.search(r'"score"\s*:\s*"?([1-5])"?', reply)
    if loose:
        reason = re.search(r'"reason"\s*:\s*"(.*?)"\s*\}?\s*$', reply, re.DOTALL | re.IGNORECASE)
        return int(loose.group(1)), (reason.group(1) if reason else reply)[:600]
    return None, f"judge reply did not parse: {reply[:120]!r}"


def _turn_fields(record: TurnRecord) -> dict[str, str]:
    calls = (
        "\n".join(
            f"- {c.tool}({json.dumps(c.input, ensure_ascii=False)[:600]})" for c in record.tool_calls
        )
        or "- none"
    )
    results = (
        "\n".join(f"- {r.tool}: {(r.text or '')[:600]}" for r in record.tool_results) or "- none"
    )
    components = (
        "\n".join(
            f"- {name}: {json.dumps(payload, ensure_ascii=False)[:6000]}"
            for name, payload in record.ui
        )
        or "- none"
    )
    previous = "- none"
    if record.previous:
        previous = (
            "\n".join(
                f"- {name}: {json.dumps(payload, ensure_ascii=False)[:1200]}"
                for name, payload in record.previous[-1].ui
            )
            or "- none"
        )
    return {
        "text": record.text or "(none)",
        "calls": calls,
        "results": results,
        "components": components,
        "previous": previous,
    }


def graded_prompt(rubric: dict[str, Any], record: TurnRecord, reference_response: str | None) -> str:
    anchors = {str(k): str(v) for k, v in (rubric.get("anchors") or {}).items()}
    between = "between the neighbouring anchors"
    return GRADED_PROMPT.format(
        question=str(rubric.get("question", "")).strip(),
        a5=anchors.get("5", ""),
        a4=anchors.get("4", between),
        a3=anchors.get("3", ""),
        a2=anchors.get("2", between),
        a1=anchors.get("1", ""),
        reference_note=REFERENCE_NOTE if reference_response else "",
        reference_block=(
            f"\n<reference_reply>\n{reference_response.strip()[:1500]}\n</reference_reply>\n"
            if reference_response
            else ""
        ),
        **_turn_fields(record),
    )


def make_judge(client: Any, model: str, max_chars: int = 40_000):
    """A judge over an ``anthropic.Anthropic`` client (synchronous). Called as
    ``judge(rubric, record, transcript, reference_response=None)``; it answers
    ``(verdict, reason)`` for a string rubric and ``(verdict, reason, score)`` for a graded one."""

    def ask(prompt: str) -> str:
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
        # No temperature: the Claude 5 models reject the parameter, and the verdict is a
        # one-line JSON either way.
        response = client.messages.create(
            model=model,
            max_tokens=600,
            thinking={"type": "disabled"},  # a Claude 5 model thinks by default and may spend the budget
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(response, "usage", None)
        judge.usage["calls"] += 1
        judge.usage["input"] += getattr(usage, "input_tokens", 0) or 0
        judge.usage["output"] += getattr(usage, "output_tokens", 0) or 0
        return "".join(getattr(block, "text", "") for block in response.content)

    def judge(
        rubric: str | dict[str, Any],
        record: TurnRecord,
        transcript: list[dict[str, Any]],
        reference_response: str | None = None,
    ) -> tuple[str, str] | tuple[str, str, int | None]:
        del transcript  # the record carries what the graded turn did; earlier turns are not judged
        if isinstance(rubric, dict):
            threshold = int(rubric.get("threshold") or 4)
            score, reason = parse_score(ask(graded_prompt(rubric, record, reference_response)))
            verdict = "PASS" if score is not None and score >= threshold else "FAIL"
            return verdict, reason, score
        prompt = PROMPT.format(rubric=rubric, **_turn_fields(record))
        return parse_verdict(ask(prompt))

    judge.usage = {"input": 0, "output": 0, "calls": 0}  # type: ignore[attr-defined]
    return judge


def make_model_judge(spec: str, max_chars: int = 40_000):
    """The same judge over one of the harness's own model adapters (``provider:model``, as the
    roles' models are named): a judge from another family than the assistant it grades. Its
    cost is read off the process's ledger (``judge.cost``)."""
    from .mini.models import LEDGER, make_model

    kwargs: dict[str, Any] = {"max_tokens": 600}
    if spec.startswith("anthropic:"):
        kwargs["anthropic_thinking"] = "disabled"
    model = make_model(spec, **kwargs)

    def ask(prompt: str) -> str:
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
        before = LEDGER.total
        text = model.complete_text([{"role": "user", "content": prompt}])
        judge.usage["calls"] += 1
        judge.cost += LEDGER.total - before  # type: ignore[attr-defined]
        return text

    def judge(
        rubric: str | dict[str, Any],
        record: TurnRecord,
        transcript: list[dict[str, Any]],
        reference_response: str | None = None,
    ) -> tuple[str, str] | tuple[str, str, int | None]:
        del transcript
        if isinstance(rubric, dict):
            threshold = int(rubric.get("threshold") or 4)
            score, reason = parse_score(ask(graded_prompt(rubric, record, reference_response)))
            verdict = "PASS" if score is not None and score >= threshold else "FAIL"
            return verdict, reason, score
        return parse_verdict(ask(PROMPT.format(rubric=rubric, **_turn_fields(record))))

    judge.usage = {"input": 0, "output": 0, "calls": 0}  # type: ignore[attr-defined]
    judge.cost = 0.0  # type: ignore[attr-defined]
    return judge
