# SPDX-License-Identifier: Apache-2.0

"""The visible case set as V keeps it: one JSON file per case under ``cases/<feature>/``,
read and validated by the harness schema; a snapshot/diff for the edit log. Two schemas: v1
grades anything the trace shows; v2, the outcome vocabulary, grades what the customer or the
backend sees and rejects a file that checks mechanics (see ``harness/cases.py``)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from harness.cases import APART_KEYS, OUTCOME_KEYS, Case, CaseFile, outcome_problems

CASES_README_V1 = """# Evaluation cases

One JSON file per case, in the directory of the feature it belongs to (`F1/`, `F2/`, ...).
A file holds one object:

```json
{
  "id": "f1-001-short-name",
  "clause": "F1.4",
  "turns": ["the customer's message", "a follow-up, when the behavior spans turns"],
  "state": {"seen_products": ["B0..."], "cart": [{"product_id": "B0...", "quantity": 1}],
            "memory": [{"key": "k", "value": "v", "category": "constraint"}]},
  "expected": {"calls_tool": ["search_products"], "ui_components": ["brief"],
               "rubric": "PASS if ... FAIL if ..."},
  "notes": "what the case pins and the catalog fact that decides it"
}
```

`expected` applies to the last turn; `turn_expected` (a list, one object per turn) grades
each turn instead. Grader keys (all code graders unless noted):

| Key | Meaning |
|---|---|
| `calls_tool`, `calls_one_of`, `never_calls`, `first_tool`, `first_tool_not` | tool names the turn called, in order |
| `same_round` (list of lists), `single_round` (tool names) | calls sent in one model round |
| `ui_components`, `no_ui` | component names shown (`products`, `comparison`, `brief`, `kit`, `order_status`, `checkout`, `suggestions`) |
| `payload` (list of `{component, path, len_between | equals | one_of | nonempty | contains}`) | checks on the last payload of a component |
| `presented_contains_any`, `presented_not_contains` | product ids any component carried |
| `cart_contains`, `cart_not_contains`, `cart_item_count` | the cart after the turn |
| `memory_contains`, `memory_not_contains` (strings, or `{key_contains, value_contains}` objects), `memory_has` (`{value_contains, key_contains, category}`) | saved facts after the turn |
| `skill_loaded`, `skill_not_loaded`, `no_skill_load` | the flow loaded with `load_skill` |
| `reply_includes`, `reply_omits`, `reply_matches`, `reply_not_matches` (regex) | the assistant's text |
| `max_tool_calls`, `min_tool_calls`, `search_count_min` | call counts |
| `tool_input_keys` (`{tool: [keys]}`), `tool_result_omits` (`{tool: [strings]}`) | the input keys a tool was called with; strings no result of a tool may carry |
| `rubric` | one PASS and one FAIL condition, graded by a pinned judge (the only non-code key) |

Ids in `state` and `expected` are real product ids from the store; `run_eval` prints the
search results a turn saw, which is the way to find them. The customer is `demo-user`
(profile: Priya, ACME Subscription member, Maple Heights); the session clock is fixed.
"""

CASES_README_V2 = """# Evaluation cases

One JSON file per case, in the directory of the feature it belongs to (`F1/`, `F2/`, ...).
A case grades what the customer or the store sees on a turn: the tools consulted, the
components and their shapes, the cart and the memory afterwards, the relevance of what was
shown, and the quality of the answer on a 1-to-5 rubric. It does not grade how the assistant
is built (which skill loaded, in which order the calls went, which words the reply used);
a file that does is invalid and counts for nothing until fixed.

```json
{
  "id": "f1-001-short-name",
  "maps_to": ["F1.5", "F1.4"],
  "turns": ["the customer's message", "a follow-up, when the behavior spans turns"],
  "state": {"seen_products": ["B0..."], "cart": [{"product_id": "B0...", "quantity": 1}],
            "memory": [{"key": "k", "value": "v", "category": "constraint"}]},
  "expected": {
    "calls_tool": ["search_policies", "search_products"],
    "ui_components": ["brief"],
    "payload": [{"component": "brief", "path": "sections", "len_between": [3, 5]}],
    "rubric": {
      "question": "Do the brief's criteria come from the store's guide and fit the request?",
      "anchors": {"5": "every criterion is one the guide names, phrased for this customer",
                  "3": "the criteria are sound but one is not in the guide or the product records",
                  "1": "the criteria are generic, or the turn asks for facts instead of teaching"},
      "threshold": 4
    }
  },
  "reference_response": "optional: one reply that met the clauses, as a calibration point for the judge",
  "notes": "what the case pins and the catalog fact that decides it"
}
```

`maps_to` names the spec clause(s) the case pins; every case needs at least one. `expected`
applies to the last turn; `turn_expected` (a list, one object per turn) grades each turn
instead. `any_of` (a list of alternative `expected` blocks) passes when one alternative passes
whole, for a clause the spec leaves two ways to meet.

| Key | Meaning |
|---|---|
| `calls_tool` | every listed tool was called on the turn (any order, any round) |
| `calls_one_of` | at least one of the listed tools was called |
| `never_calls` | none of the listed tools was called |
| `max_tool_calls` | at most this many tool calls on the turn |
| `ui_components`, `no_ui` | component names shown (`products`, `comparison`, `brief`, `kit`, `order_status`, `checkout`, `suggestions`) |
| `payload` (list of `{component, path, len_between | equals | one_of | nonempty | contains}`) | the shape of a component's last payload |
| `presented_contains_any`, `presented_not_contains` | product ids any component carried |
| `presented_relevance` (`{query | query_id, labels: ["E","S"], min_share: 0.5, min_judged: 1}`) | of the presented ids the store's judges rated for that query, at least `min_share` are Exact/Substitute (`run_eval --list-queries <words>` lists the judged queries) |
| `cart_contains`, `cart_not_contains`, `cart_item_count`, `cart_unchanged` | the cart after the turn (`cart_unchanged: true`: the same as before it) |
| `memory_contains`, `memory_not_contains` (strings, or `{key_contains, value_contains}` objects), `memory_has`, `memory_unchanged` | saved facts after the turn |
| `tool_input_keys` (`{tool: [keys]}`), `tool_result_omits` (`{tool: [strings]}`) | the input keys a tool was called with; strings no result of a tool may carry |
| `rubric` (`{question, anchors: {"1","3","5" (2 and 4 optional)}, threshold}`) | graded 1-5 by a pinned judge from the transcript; passes at `threshold` or above (the only non-code key) |

A good rubric asks one question about the customer's outcome and gives anchors a stranger
could apply; it does not name tools or wording. Ids in `state` and `expected` are real product
ids from the store; `run_eval` prints the search results a turn saw, which is the way to find
them. The customer is `demo-user` (profile: Priya, ACME Subscription member, Maple Heights);
the session clock is fixed.
"""

CASES_README = CASES_README_V1  # the name earlier code imports


FIND_IDS = "`run_eval` prints the search results a turn saw, which is the way to find them."
FIND_IDS_BLIND = "`run_eval --list-products <words>` lists products of the catalog, which is the way to find them."


# What the note on the case format says in a tree whose evaluator is kept apart from the
# developer (``v2-apart``): the two keys that fix what only the developer's change will name
# are not in the vocabulary. (old text, new text) pairs; each old text occurs once in the note.
APART_EDITS = (
    ("the tools consulted, the\ncomponents and their shapes, the cart", "the tools consulted, the\ncomponents shown, the cart"),
    ('    "payload": [{"component": "brief", "path": "sections", "len_between": [3, 5]}],\n', ""),
    ("| `payload` (list of `{component, path, len_between | equals | one_of | nonempty | contains}`) | the shape of a component's last payload |\n", ""),
    (
        "| `tool_input_keys` (`{tool: [keys]}`), `tool_result_omits` (`{tool: [strings]}`) | the input keys a tool was called with; strings no result of a tool may carry |",
        "| `tool_result_omits` (`{tool: [strings]}`) | strings no result of a tool may carry |",
    ),
    (
        "A good rubric asks one question",
        "On this team the developer does not see your cases and you do not see its change. A case\n"
        "therefore cannot fix what only the change will name: the names of a tool's inputs\n"
        "(`tool_input_keys`) and the inner shape of a component (`payload`) are not in the vocabulary\n"
        "here, and a file that uses them is invalid. Say which tools must be consulted, what the\n"
        "customer must be shown or told, and what must be true of the cart and the memory afterwards.\n\n"
        "A good rubric asks one question",
    ),
)


def readme_for(schema: str, blind: bool = False, apart: bool = False) -> str:
    """The note on the case format an evaluation tree carries. ``blind``: the evaluator of
    the run is never shown the changed assistant, so real product ids come from the catalog.
    ``apart``: the cases are held to the cut-apart vocabulary (``v2-apart``)."""
    text = CASES_README_V2 if schema.startswith("v2") else CASES_README_V1
    if blind:
        pattern = r"\s+".join(re.escape(word) for word in FIND_IDS.split())
        text, n = re.subn(pattern, lambda _m: FIND_IDS_BLIND, text, count=1)
        if n != 1:
            raise RuntimeError("the note on the case format no longer says where product ids come from")
    if apart:
        for old, new in APART_EDITS:
            if text.count(old) != 1:
                raise RuntimeError(f"the note on the case format no longer has {old[:40]!r}")
            text = text.replace(old, new)
    return text


def _parse(path: Path, schema: str) -> list[Case]:
    """The cases of one file; raises with the reason when the file is not a valid case."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:  # a whole file, tolerated
        parsed = [Case.model_validate(c) for c in data["cases"]]
    else:
        parsed = [Case.model_validate(data)]
    if schema.startswith("v2"):
        for case in parsed:
            problems = outcome_problems(case, APART_KEYS if schema == "v2-apart" else OUTCOME_KEYS)
            if problems:
                raise ValueError(f"{case.id}: " + "; ".join(problems))
    return parsed


def load_case_dir(root: Path, feature: str, schema: str = "v1") -> CaseFile:
    """The feature's cases. A file that does not parse as a case (or, under schema v2, grades
    mechanics) is skipped with a note on stderr: it counts for nothing (``invalid_files``
    tells the evaluator), and it must not stop the gate."""
    cases: list[Case] = []
    directory = root / feature
    # The safety set is the harness's own and keeps its shape under either schema.
    file_schema = schema if feature.startswith("F") else "v1"
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                parsed = _parse(path, file_schema)
            except Exception as error:  # noqa: BLE001 - the file is the evaluator's problem
                print(
                    f"cases: skipping {path.relative_to(root)}: {type(error).__name__}: {str(error)[:200]}",
                    file=sys.stderr,
                )
                continue
            cases.extend(parsed)
    return CaseFile(feature=feature, schema_version=schema, cases=cases)


def load_all(root: Path, features: list[str], schema: str = "v1") -> dict[str, CaseFile]:
    return {feature: load_case_dir(root, feature, schema) for feature in features}


def invalid_files(root: Path, schema: str = "v1") -> dict[str, str]:
    """Files that do not parse as a case, or grade mechanics under schema v2: V is told, and
    they count for nothing."""
    bad: dict[str, str] = {}
    for path in sorted(root.glob("F*/*.json")):
        try:
            _parse(path, schema)
        except Exception as error:  # noqa: BLE001
            bad[str(path.relative_to(root))] = str(error)[:300]
    return bad


def snapshot(root: Path) -> dict[str, dict[str, Any]]:
    """``{relative path: {"sha": ..., "case": <dict or None>}}`` for every case file."""
    result = {}
    for path in sorted(root.glob("F*/*.json")):
        text = path.read_text(encoding="utf-8")
        try:
            case = json.loads(text)
        except json.JSONDecodeError:
            case = None
        result[str(path.relative_to(root))] = {
            "sha": hashlib.sha1(text.encode()).hexdigest(),
            "case": case,
        }
    return result


def diff(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    return {
        "added": sorted(k for k in after if k not in before),
        "deleted": sorted(k for k in before if k not in after),
        "modified": sorted(k for k in after if k in before and after[k]["sha"] != before[k]["sha"]),
    }
