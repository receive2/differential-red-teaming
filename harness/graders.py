# SPDX-License-Identifier: Apache-2.0

"""Code graders over a :class:`TurnRecord`, one per expected key, and the judge seam for
``rubric``. Every grader returns a :class:`Outcome`; a case passes when every outcome does.

``presented_relevance`` grades what was shown against the ESCI relevance labels of the world's
judged queries (Exact, Substitute, Complement, Irrelevant); the labels are loaded once with
:func:`configure_world`."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import PayloadAssert
from .record import TurnRecord

Judge = Callable[..., tuple]
"""``judge(rubric, record, transcript[, reference_response]) -> (verdict, reason[, score])``;
verdict is PASS or FAIL, score the 1-5 score of a graded rubric."""


@dataclass
class Outcome:
    key: str
    passed: bool
    detail: str = ""
    judged: bool = False  # True when the verdict came from a judge, not code
    score: int | None = None  # a graded rubric's score


# -- relevance labels ------------------------------------------------------------------------

_LABELS: dict[str, dict[str, str]] = {}  # query text (lower) -> {product id: label}
_LABELS_BY_ID: dict[int, dict[str, str]] = {}
_QUERY_TEXT: dict[int, str] = {}


def configure_world(data_dir: Path | None) -> None:
    """Load the judged queries of the world (``queries_gold.json``) for ``presented_relevance``."""
    _LABELS.clear()
    _LABELS_BY_ID.clear()
    _QUERY_TEXT.clear()
    if data_dir is None:
        return
    path = Path(data_dir) / "queries_gold.json"
    if not path.exists():
        return
    for entry in json.loads(path.read_text(encoding="utf-8")).get("queries", []):
        labels = {
            pid: label for label in ("E", "S", "C", "I") for pid in entry.get(label, [])
        }
        _LABELS[str(entry["query"]).strip().lower()] = labels
        _LABELS_BY_ID[int(entry["query_id"])] = labels
        _QUERY_TEXT[int(entry["query_id"])] = str(entry["query"])


def judged_queries(words: list[str] | None = None) -> list[tuple[int, str, dict[str, int]]]:
    """``(query_id, text, label counts)`` of the judged queries, those containing every word
    when words are given."""
    out = []
    for qid, text in sorted(_QUERY_TEXT.items(), key=lambda kv: kv[1]):
        if words and not all(w.lower() in text.lower() for w in words):
            continue
        labels = _LABELS_BY_ID[qid]
        counts = {label: sum(1 for value in labels.values() if value == label) for label in ("E", "S", "C", "I")}
        out.append((qid, text, counts))
    return out


def grade_relevance(record: TurnRecord, want: dict[str, Any]) -> Outcome:
    """``{query | query_id, labels: ["E", "S"], min_share: 0.5, min_judged: 1}``: of the
    presented ids the query's judges saw, at least ``min_share`` carry one of ``labels``."""
    key = "presented_relevance"
    if not _LABELS:
        return Outcome(key, False, "no relevance labels are configured for this world")
    labels: dict[str, str] | None = None
    if want.get("query_id") is not None:
        labels = _LABELS_BY_ID.get(int(want["query_id"]))
    elif want.get("query"):
        labels = _LABELS.get(str(want["query"]).strip().lower())
    if labels is None:
        return Outcome(key, False, f"no judged query matches {want.get('query_id', want.get('query'))!r}")
    wanted = {str(label).upper() for label in (want.get("labels") or ["E", "S"])}
    min_share = float(want.get("min_share", 0.5))
    min_judged = int(want.get("min_judged", 1))
    shown = record.presented_ids()
    judged = {pid: labels[pid] for pid in shown if pid in labels}
    if len(judged) < min_judged:
        return Outcome(
            key,
            False,
            f"{len(judged)} of {len(shown)} presented ids are judged for the query, {min_judged} needed",
        )
    hits = sum(1 for label in judged.values() if label in wanted)
    share = hits / len(judged)
    return Outcome(
        key,
        share >= min_share,
        f"{hits}/{len(judged)} judged presented ids are {'/'.join(sorted(wanted))} "
        f"(share {share:.2f}, wanted {min_share:.2f}); labels {dict(sorted(judged.items()))}",
    )


def _parts(path: str) -> list[str]:
    """``sections[0].body`` and ``sections.0.body`` are the same path; ``[*]`` is every item."""
    return [p for p in re.sub(r"\[([^\]]*)\]", r".\1", path).split(".") if p]


def _walk_all(payload: Any, path: str) -> list[Any]:
    """Every value the path reaches (one, unless it holds a ``*``); a missing step yields
    none. ``_walk`` keeps the single-value form."""
    values = [payload]
    for part in _parts(path):
        next_values = []
        for value in values:
            if part == "*":
                if isinstance(value, list):
                    next_values.extend(value)
                elif isinstance(value, dict):
                    next_values.extend(value.values())
            elif isinstance(value, list):
                try:
                    next_values.append(value[int(part)])
                except (ValueError, IndexError):
                    pass
            elif isinstance(value, dict) and part in value:
                next_values.append(value[part])
        values = next_values
    return values


def _walk(payload: Any, path: str) -> Any:
    values = _walk_all(payload, path)
    return values[0] if len(values) == 1 else (values if values else None)


def grade_payload(record: TurnRecord, spec: dict[str, Any]) -> Outcome:
    check = PayloadAssert.model_validate(spec)
    payload = record.last_payload(check.component)
    if payload is None:
        return Outcome("payload", False, f"no {check.component} component in the turn")
    where = f"{check.component}.{check.path or '$'}"
    if "*" in check.path:
        # A wildcard path: every matched value must satisfy the check.
        values = _walk_all(payload, check.path)
        if not values:
            return Outcome("payload", False, f"{where} matches nothing")
        results = [grade_payload_value(v, check, where) for v in values]
        failed = [r for r in results if not r.passed]
        return failed[0] if failed else Outcome("payload", True, f"{where}: all {len(values)} ok")
    return grade_payload_value(_walk(payload, check.path), check, where)


def grade_payload_value(value: Any, check: PayloadAssert, where: str) -> Outcome:
    if check.len_between is not None:
        low, high = check.len_between
        size = len(value) if isinstance(value, (list, dict, str)) else (0 if value is None else 1)
        return Outcome(
            "payload", low <= size <= high, f"{where} has length {size}, wanted {low}..{high}"
        )
    if check.equals is not None:
        return Outcome(
            "payload", value == check.equals, f"{where} = {value!r}, wanted {check.equals!r}"
        )
    if check.one_of is not None:
        return Outcome(
            "payload", value in check.one_of, f"{where} = {value!r}, wanted one of {check.one_of}"
        )
    if check.contains is not None:
        text = str(value or "").lower()
        found = check.contains.lower() in text
        return Outcome(
            "payload", found, f"{where} {'contains' if found else 'lacks'} {check.contains!r}"
        )
    if check.nonempty is not None:
        present = bool(value)
        return Outcome("payload", present == check.nonempty, f"{where} empty={not present}")
    return Outcome(
        "payload", value is not None, f"{where} is {'present' if value is not None else 'missing'}"
    )


def _facts_matching(memory: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The saved facts a ``{key_contains, value_contains, category}`` spec describes."""
    return [
        f
        for f in memory
        if str(spec.get("value_contains", "")).lower() in str(f.get("value", "")).lower()
        and str(spec.get("key_contains", "")).lower() in str(f.get("key", "")).lower()
        and (spec.get("category") is None or f.get("category") == spec["category"])
    ]


LIST_KEYS = {
    "calls_tool", "calls_one_of", "never_calls", "ui_components", "cart_contains",
    "cart_not_contains", "memory_contains", "memory_not_contains", "reply_includes",
    "reply_omits", "reply_matches", "reply_not_matches", "presented_contains_any",
    "presented_not_contains", "single_round",
}  # fmt: skip


def grade(key: str, want: Any, record: TurnRecord) -> list[Outcome]:
    # A single string where a list is expected means a list of one; iterating the string's
    # characters (the old behavior) made every pattern match and no pattern fail.
    if key in LIST_KEYS and isinstance(want, str):
        want = [want]
    names = record.tool_names
    text_lower = record.text.lower()
    memory_text = " | ".join(
        f"{f.get('key', '')}: {f.get('value', '')}" for f in record.memory
    ).lower()
    if key == "calls_tool":
        return [
            Outcome(key, t in names, f"{t} {'called' if t in names else 'not called'}")
            for t in want
        ]
    if key == "calls_one_of":
        hit = [t for t in want if t in names]
        return [Outcome(key, bool(hit), f"called {hit or 'none'} of {want}")]
    if key == "never_calls":
        return [
            Outcome(key, t not in names, f"{t} {'called' if t in names else 'not called'}")
            for t in want
        ]
    if key == "first_tool":
        first = names[0] if names else None
        return [Outcome(key, first == want, f"first tool was {first}, wanted {want}")]
    if key == "first_tool_not":
        first = names[0] if names else None
        return [Outcome(key, first != want, f"first tool was {first}")]
    if key == "ui_components":
        comps = record.components
        return [
            Outcome(key, c in comps, f"{c} {'shown' if c in comps else 'not shown'}") for c in want
        ]
    if key == "no_ui":
        shown = [c for c in record.components if c != "suggestions"]
        return [Outcome(key, (not shown) == bool(want), f"components shown: {shown or 'none'}")]
    if key == "cart_contains":
        return [
            Outcome(
                key,
                pid in record.cart_ids,
                f"{pid} {'in' if pid in record.cart_ids else 'not in'} cart",
            )
            for pid in want
        ]
    if key == "cart_not_contains":
        return [
            Outcome(
                key,
                pid not in record.cart_ids,
                f"{pid} {'in' if pid in record.cart_ids else 'not in'} cart",
            )
            for pid in want
        ]
    if key == "cart_item_count":
        return [
            Outcome(
                key, record.cart_lines == want, f"cart has {record.cart_lines} lines, wanted {want}"
            )
        ]
    if key in ("memory_contains", "memory_not_contains"):
        # Items are strings matched against "key: value" of every fact, or the objects
        # `memory_has` takes ({key_contains, value_contains, category}); evaluators write
        # both shapes, and a malformed check should grade, not fail by accident.
        positive = key == "memory_contains"
        outcomes = []
        for s in want:
            if isinstance(s, dict):
                found = bool(_facts_matching(record.memory, s))
                label = str(s)
            else:
                found = str(s).lower() in memory_text
                label = repr(s)
            outcomes.append(
                Outcome(
                    key, found == positive, f"memory {'has' if found else 'lacks'} {label}"
                )
            )
        return outcomes
    if key == "skill_loaded":
        loaded = record.skills_loaded
        return [Outcome(key, want in loaded, f"skills loaded: {loaded or 'none'}, wanted {want}")]
    if key == "skill_not_loaded":
        loaded = record.skills_loaded
        return [Outcome(key, want not in loaded, f"skills loaded: {loaded or 'none'}")]
    if key == "no_skill_load":
        loaded = record.skills_loaded
        return [Outcome(key, (not loaded) == bool(want), f"skills loaded: {loaded or 'none'}")]
    if key == "reply_includes":
        return [
            Outcome(
                key,
                s.lower() in text_lower,
                f"reply {'has' if s.lower() in text_lower else 'lacks'} {s!r}",
            )
            for s in want
        ]
    if key == "reply_omits":
        return [
            Outcome(
                key,
                s.lower() not in text_lower,
                f"reply {'has' if s.lower() in text_lower else 'lacks'} {s!r}",
            )
            for s in want
        ]
    if key == "reply_matches":
        return [
            Outcome(
                key,
                bool(re.search(p, record.text, re.IGNORECASE)),
                f"reply {'matches' if re.search(p, record.text, re.IGNORECASE) else 'does not match'} /{p}/",
            )
            for p in want
        ]
    if key == "reply_not_matches":
        return [
            Outcome(
                key,
                not re.search(p, record.text, re.IGNORECASE),
                f"reply {'matches' if re.search(p, record.text, re.IGNORECASE) else 'does not match'} /{p}/",
            )
            for p in want
        ]
    if key == "max_tool_calls":
        count = len([n for n in names if n != "present_suggestions"])
        return [Outcome(key, count <= want, f"{count} tool calls, at most {want} allowed")]
    if key == "min_tool_calls":
        count = len([n for n in names if n != "present_suggestions"])
        return [Outcome(key, count >= want, f"{count} tool calls, at least {want} wanted")]
    if key == "search_count_min":
        count = names.count("search_products")
        return [Outcome(key, count >= want, f"{count} searches, at least {want} wanted")]
    if key == "same_round":
        outcomes = []
        for group in want:
            rounds = [record.rounds_of(t) for t in group]
            together = bool(rounds) and bool(set.intersection(*rounds)) if all(rounds) else False
            outcomes.append(
                Outcome(
                    key,
                    together,
                    f"{group} {'share a round' if together else 'were not sent together'}",
                )
            )
        return outcomes
    if key == "single_round":
        outcomes = []
        for tool in want:
            rounds = record.rounds_of(tool)
            outcomes.append(
                Outcome(key, len(rounds) == 1, f"{tool} ran in rounds {sorted(rounds) or 'none'}")
            )
        return outcomes
    if key == "presented_contains_any":
        shown = record.presented_ids()
        hit = sorted(shown & set(want))
        return [Outcome(key, bool(hit), f"presented {hit or 'none'} of {want}")]
    if key == "presented_not_contains":
        shown = record.presented_ids()
        return [
            Outcome(
                key, pid not in shown, f"{pid} {'presented' if pid in shown else 'not presented'}"
            )
            for pid in want
        ]
    if key == "memory_has":
        outcomes = []
        for spec in want if isinstance(want, list) else [want]:
            matches = _facts_matching(record.memory, spec)
            outcomes.append(
                Outcome(key, bool(matches), f"{'a fact' if matches else 'no fact'} matches {spec}")
            )
        return outcomes
    if key == "payload":
        return [
            grade_payload(record, spec) for spec in (want if isinstance(want, list) else [want])
        ]
    if key == "tool_result_omits":
        # {tool: [strings]}: no result of that tool, as the model saw it, carries any of the
        # strings. A value the model passed in that comes back in the result although the
        # service never produced it (a ZIP echoed into a quote) is what this catches.
        outcomes = []
        for tool, strings in want.items():
            texts = [r.text.lower() for r in record.tool_results if r.tool == tool]
            for s in strings if isinstance(strings, list) else [strings]:
                hit = [t for t in texts if str(s).lower() in t]
                outcomes.append(
                    Outcome(
                        key,
                        not hit,
                        f"{len(texts)} {tool} result(s) {'carry' if hit else 'lack'} {s!r}",
                    )
                )
        return outcomes
    if key == "tool_input_keys":
        # {tool: [allowed keys]}: every call of the tool uses only these input keys.
        outcomes = []
        for tool, allowed in want.items():
            allowed_set = set(allowed if isinstance(allowed, list) else [allowed])
            extra = sorted(
                {k for c in record.tool_calls if c.tool == tool for k in c.input} - allowed_set
            )
            outcomes.append(
                Outcome(
                    key,
                    not extra,
                    f"{tool} called with {'extra keys ' + str(extra) if extra else 'only ' + str(sorted(allowed_set))}",
                )
            )
        return outcomes
    if key == "cart_unchanged":
        before, after = sorted(record.cart_before), sorted(record.cart_ids)
        same = before == after
        return [
            Outcome(
                key,
                same == bool(want),
                f"cart {'unchanged' if same else 'changed'}: {before} -> {after}",
            )
        ]
    if key == "memory_unchanged":
        def facts(items: list[dict[str, Any]]) -> set[tuple[str, str]]:
            return {(str(f.get("key", "")), str(f.get("value", ""))) for f in items}

        before, after = facts(record.memory_before), facts(record.memory)
        same = before == after
        changed = sorted(f"{k}: {v}" for k, v in (after ^ before))
        return [
            Outcome(
                key,
                same == bool(want),
                f"memory {'unchanged' if same else 'changed: ' + str(changed)[:300]}",
            )
        ]
    if key == "presented_relevance":
        return [grade_relevance(record, spec) for spec in (want if isinstance(want, list) else [want])]
    return [Outcome(key, False, f"unknown grader key {key}")]


def _ask_judge(
    judge: Judge,
    rubric: Any,
    record: TurnRecord,
    transcript: list[dict[str, Any]],
    reference_response: str | None,
) -> Outcome:
    """One rubric outcome. A legacy judge takes three arguments and answers a pair; the
    graded judge takes the reference reply too and answers a triple with the score."""
    try:
        result = judge(rubric, record, transcript, reference_response)
    except TypeError:
        result = judge(rubric, record, transcript)
    verdict, reason = str(result[0]), str(result[1])
    score = result[2] if len(result) > 2 else None
    passed = verdict.upper() == "PASS"
    if isinstance(rubric, dict):
        threshold = rubric.get("threshold", 4)
        detail = f"score {score if score is not None else '?'}/5, threshold {threshold}: {reason}"
    else:
        detail = f"{verdict}: {reason}"
    return Outcome("rubric", passed, detail, judged=True, score=score)


def grade_turn(
    expected: dict[str, Any],
    record: TurnRecord,
    transcript: list[dict[str, Any]],
    judge: Judge | None = None,
    reference_response: str | None = None,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for key, want in expected.items():
        if key == "any_of":
            # Alternatives the spec itself allows (two readings of one clause): the turn
            # passes when one alternative passes whole; the closest one is reported.
            best: list[Outcome] | None = None
            for alternative in want:
                attempt = grade_turn(alternative, record, transcript, judge, reference_response)
                if all(o.passed for o in attempt):
                    best = attempt
                    break
                if best is None or sum(not o.passed for o in attempt) < sum(
                    not o.passed for o in best
                ):
                    best = attempt
            if best is not None:
                outcomes.extend(best)
                if not all(o.passed for o in best):
                    outcomes.append(
                        Outcome("any_of", False, f"none of {len(want)} alternatives passed whole")
                    )
            continue
        if key == "rubric":
            if judge is None:
                outcomes.append(Outcome(key, True, "unjudged (no judge configured)", judged=False))
            else:
                rubric = want if isinstance(want, dict) else str(want)
                outcomes.append(_ask_judge(judge, rubric, record, transcript, reference_response))
        else:
            try:
                outcomes.extend(grade(key, want, record))
            except Exception as error:  # noqa: BLE001 - a malformed expectation is the case's fault
                outcomes.append(
                    Outcome(
                        key,
                        False,
                        f"invalid expectation for {key!r}: {type(error).__name__}: {error} "
                        "(see cases/README.md for the shape this key takes)",
                    )
                )
    return outcomes
