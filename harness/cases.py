# SPDX-License-Identifier: Apache-2.0

"""The case file: the shape ``commerce-evals`` describes, plus the keys this harness adds for
clause-level gold. ``expected`` applies to the last turn unless ``turn_expected`` gives one
block per turn.

Two vocabularies. The first (schema v1) grades anything the trace shows, mechanics included:
which skill loaded, in which round a call sat, what words the reply used. The outcome
vocabulary (schema v2) grades what the customer or the backend sees: the tools that were
consulted (unordered), the components and their shapes, the cart and the memory after the
turn, the relevance of what was presented, and a graded rubric with anchors and a threshold.
A v2 case names the clauses it maps to (``maps_to``); a check on mechanics makes the file
invalid, and an invalid file counts for nothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

CODE_GRADER_KEYS = (
    "calls_tool", "calls_one_of", "never_calls", "first_tool", "first_tool_not",
    "ui_components", "no_ui", "cart_contains", "cart_item_count", "cart_not_contains",
    "memory_contains", "memory_not_contains", "skill_loaded", "skill_not_loaded",
    "no_skill_load", "reply_includes", "reply_omits", "reply_matches", "reply_not_matches",
    "max_tool_calls", "min_tool_calls", "same_round", "single_round", "payload",
    "search_count_min", "presented_contains_any", "presented_not_contains", "memory_has",
    "any_of", "tool_result_omits", "tool_input_keys",
    "cart_unchanged", "memory_unchanged", "presented_relevance",
)  # fmt: skip
JUDGE_KEYS = ("rubric",)

# Schema v2: outcomes only. Not in it: skill_loaded and its kin (which flow ran is the
# assistant's business; the customer sees the brief or the kit), first_tool / same_round /
# single_round (call order and rounds), reply_* (wording is the rubric's to judge),
# min_tool_calls / search_count_min (how much work it took).
OUTCOME_KEYS = (
    "calls_tool", "calls_one_of", "never_calls", "max_tool_calls",
    "ui_components", "no_ui", "payload",
    "presented_contains_any", "presented_not_contains", "presented_relevance",
    "cart_contains", "cart_not_contains", "cart_item_count", "cart_unchanged",
    "memory_contains", "memory_not_contains", "memory_has", "memory_unchanged",
    "tool_input_keys", "tool_result_omits",
    "any_of", "rubric",
)  # fmt: skip
# Schema v2 of a system whose evaluator and developer are kept apart (``v2-apart``): the
# evaluator writes before the developer has changed anything and never sees the change, and
# the developer never sees the cases. A case cannot then fix what only the change will name:
# the names of a tool's inputs, the inner shape of a component.
APART_ONLY = ("payload", "tool_input_keys")
APART_KEYS = tuple(key for key in OUTCOME_KEYS if key not in APART_ONLY)
RUBRIC_ANCHORS = ("1", "3", "5")


class CartLine(BaseModel):
    product_id: str
    quantity: int = 1


class MemorySeed(BaseModel):
    key: str
    value: str
    # The backend's memory categories (commerce_common.types.MemoryCategory). A seed with any
    # other makes the case file invalid: its author is told, and it counts for nothing.
    category: Literal["preference", "constraint", "context"] = "preference"
    updated_at: str | None = None  # ISO time; older facts fall out of the context block's cap


class CaseState(BaseModel):
    seen_products: list[str] = Field(default_factory=list)
    cart: list[CartLine] = Field(default_factory=list)
    memory: list[MemorySeed] = Field(default_factory=list)


class PayloadAssert(BaseModel):
    """One check on the last ``ui`` payload of ``component``: the value at ``path`` (dotted
    keys and integer indexes) has a length in ``len_between``, equals ``equals``, is in
    ``one_of``, or is non-empty."""

    component: str
    path: str = ""
    len_between: tuple[int, int] | None = None
    equals: Any = None
    one_of: list[Any] | None = None
    nonempty: bool | None = None
    contains: str | None = None


class GradedRubric(BaseModel):
    """A 1-to-5 rubric: ``question`` says what is judged, ``anchors`` describe at least the
    1, 3 and 5 scores (2 and 4 lie between), and the turn passes at ``threshold`` or above."""

    question: str
    anchors: dict[str, str]
    threshold: int = 4

    @model_validator(mode="after")
    def _anchors_present(self) -> GradedRubric:
        missing = [a for a in RUBRIC_ANCHORS if not str(self.anchors.get(a, "")).strip()]
        if missing:
            raise ValueError(f"rubric anchors missing for scores {missing}")
        if not 2 <= self.threshold <= 5:
            raise ValueError("rubric threshold must be between 2 and 5")
        return self


class Case(BaseModel):
    id: str
    clause: str | None = None  # the spec clause this case pins, e.g. "F2.3"
    maps_to: list[str] = Field(default_factory=list)  # every clause the case pins (v2)
    priority: str = "medium"
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    skip: str | None = None
    state: CaseState = Field(default_factory=CaseState)
    turns: list[str]
    expected: dict[str, Any] = Field(default_factory=dict)
    turn_expected: list[dict[str, Any]] | None = None  # per turn, overrides expected
    now: str | None = None  # ISO local time for the session clock; the run's anchor when absent
    user_id: str = "demo-user"
    notes: str | None = None
    # One reply that met the clauses, shown to the judge as a calibration point, never as a
    # template (v2, optional).
    reference_response: str | None = None

    @model_validator(mode="after")
    def _clause_from_maps_to(self) -> Case:
        if self.clause is None and self.maps_to:
            self.clause = self.maps_to[0]
        if not self.maps_to and self.clause:
            self.maps_to = [self.clause]
        return self

    def expected_for(self, turn_index: int, last: bool) -> dict[str, Any]:
        if self.turn_expected is not None and turn_index < len(self.turn_expected):
            return self.turn_expected[turn_index]
        return self.expected if last else {}

    def blocks(self) -> list[dict[str, Any]]:
        return [self.expected, *(self.turn_expected or [])]


class CaseFile(BaseModel):
    feature: str
    title: str = ""
    schema_version: str = "v1"  # "v2": the outcome vocabulary
    cases: list[Case]


def load_cases(path: Path) -> CaseFile:
    return CaseFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_cases(path: Path, cases: CaseFile) -> None:
    path.write_text(cases.model_dump_json(indent=1, exclude_none=True) + "\n", encoding="utf-8")


# -- the outcome vocabulary ------------------------------------------------------------------


def vocabulary_violations(block: dict[str, Any], allowed: tuple[str, ...] = OUTCOME_KEYS) -> list[str]:
    """Keys of an ``expected`` block (alternatives of ``any_of`` included) outside ``allowed``."""
    bad: list[str] = []
    for key, value in (block or {}).items():
        if key == "any_of":
            for alternative in value if isinstance(value, list) else []:
                bad += vocabulary_violations(alternative, allowed)
        elif key not in allowed:
            bad.append(key)
    return bad


def rubric_problems(block: dict[str, Any]) -> list[str]:
    """A v2 rubric is a graded one: ``{question, anchors: {1, 3, 5}, threshold}``."""
    problems: list[str] = []
    for key, value in (block or {}).items():
        if key == "any_of":
            for alternative in value if isinstance(value, list) else []:
                problems += rubric_problems(alternative)
        elif key == "rubric":
            if not isinstance(value, dict):
                problems.append("rubric must be an object {question, anchors, threshold}")
                continue
            try:
                GradedRubric.model_validate(value)
            except ValidationError as error:
                messages = [e.get("msg", "").removeprefix("Value error, ") for e in error.errors()]
                problems.append("rubric: " + "; ".join(m[:160] for m in messages))
    return problems


def outcome_problems(case: Case, allowed: tuple[str, ...] = OUTCOME_KEYS) -> list[str]:
    """Why a case is not a valid outcome-schema (v2) case; empty when it is. ``allowed``:
    ``APART_KEYS`` for a system whose evaluator and developer are kept apart."""
    problems: list[str] = []
    if not case.maps_to:
        problems.append("maps_to is missing: name the spec clause(s) the case pins")
    for block in case.blocks():
        bad = vocabulary_violations(block, allowed)
        apart = sorted({key for key in bad if key in OUTCOME_KEYS})
        mechanics = sorted({key for key in bad if key not in OUTCOME_KEYS})
        if mechanics:
            problems.append(
                f"keys outside the outcome vocabulary: {mechanics} "
                "(the case grades how the assistant is built, not what the customer sees)"
            )
        if apart:
            problems.append(
                f"keys this team's cases cannot use: {apart} (the developer does not see your cases and you do not see "
                "its change: a case cannot fix the names of a tool's inputs or the inner shape of a component)"
            )
        problems += rubric_problems(block)
    if not any(block for block in case.blocks()):
        problems.append("no expectations")
    return problems
