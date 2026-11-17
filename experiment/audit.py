# SPDX-License-Identifier: Apache-2.0

"""Does a case follow from the spec? A check, outside every role's view.

For a case, a model is shown the clauses the case maps to (with the spec's scope, interfaces
and parameters) and the case's checks and rubric, and says of each check whether the clauses
entail it, whether it is stricter than they are (it demands a tool, an order, a wording, a
component or a number the clauses do not), whether it is looser (a violation would pass), or
unrelated; the case's verdict is the worst of its checks. The loop runs it on every case V adds
or rewrites when ``audit_model`` is set; the command runs it after the fact over a run's edit
log, or over a gold file.

    python -m experiment.audit --run runs/v2-coupled-s11 [--classes over-strict-or-wrong-add] [--json out]
    python -m experiment.audit --gold hidden/F1/gold.json [--json out]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from harness.mini.models import LEDGER, make_model

from .config import ROOT
from .report import load_events

VERDICTS = ("entailed", "stricter", "looser", "unrelated")
RANK = {"entailed": 0, "looser": 1, "stricter": 2, "unrelated": 3}

PROMPT = """You audit one evaluation case against the specification clauses it claims to pin. The
specification is the only authority; you are not told how the assistant is built, and you must
not assume one way of meeting a clause is the only way.

The specification's Checking section says which clauses it grades by code and which by rubric;
a code check the Checking section calls for (a read made on the turn, a component and its
shape, the cart or the memory unchanged) is entailed, not stricter.

For each check in the case (every key of `expected` or `turn_expected`, each `any_of`
alternative as a whole, and the rubric) say whether:
- "entailed": a compliant assistant must satisfy it, given the clauses, the interfaces and the parameters;
- "stricter": it demands something the clauses do not require (a particular tool or component where another would meet the clause, an order of calls, exact wording, a count or a limit tighter than the spec's, a fact the catalog may not support);
- "looser": it would let a violation of the clauses pass (the rubric accepts what a clause forbids, the threshold is below what the anchors call compliant, a MUST is left unchecked);
- "unrelated": it checks something no mapped clause speaks to.

Then give the case one verdict: "stricter" if any check is stricter, else "unrelated" if any is unrelated, else "looser" if any is looser, else "entailed".

Answer with one JSON object and nothing else:
{{"verdict": "...", "checks": [{{"check": "<key or 'rubric'>", "verdict": "...", "why": "<one sentence>"}}], "reason": "<one sentence on the case>"}}

<specification feature="{feature}">
{spec}
</specification>

<case>
{case}
</case>
"""


KEPT_SECTIONS = ("scope", "interfaces", "parameters", "checking", "changes to earlier features")


def clause_text(spec: str, clauses: list[str]) -> str:
    """The mapped clauses in full, with the spec's scope, interfaces, parameters, checking and
    changes sections, so the auditor reads what the authors of the case read (the checking
    section says which clauses the spec itself grades by code)."""
    lines = spec.splitlines()
    keep: list[str] = []
    section = ""
    wanted = set(clauses)
    clause_re = re.compile(r"^\d+\.\s+\*\*(F\d\.\d+)\*\*")
    current_clause_kept = False
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip().lower()
            if section in KEPT_SECTIONS:
                keep.append(line)
            current_clause_kept = False
            continue
        if section == "behavior":
            match = clause_re.match(line)
            if match:
                current_clause_kept = match.group(1) in wanted
            if current_clause_kept:
                keep.append(line)
        elif section in KEPT_SECTIONS:
            keep.append(line)
    return "\n".join(keep).strip()


def spec_for(feature: str, spec_dir: Path = ROOT / "specs") -> str:
    path = next(iter(sorted(spec_dir.glob(f"{feature}-*.md"))), None)
    return path.read_text(encoding="utf-8") if path else ""


def case_feature(case: dict[str, Any], fallback: str | None = None) -> str | None:
    for clause in case.get("maps_to") or ([case["clause"]] if case.get("clause") else []):
        match = re.match(r"(F\d)\.", str(clause))
        if match:
            return match.group(1)
    match = re.match(r"(f\d)-", str(case.get("id", "")))
    return match.group(1).upper() if match else fallback


def parse_audit(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    data: dict[str, Any] = {}
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = {}
    verdict = str(data.get("verdict", "")).lower()
    if verdict not in VERDICTS:
        checks = [str(c.get("verdict", "")).lower() for c in data.get("checks", []) if isinstance(c, dict)]
        known = [c for c in checks if c in VERDICTS]
        verdict = max(known, key=lambda v: RANK[v]) if known else "unparsed"
    return {
        "verdict": verdict,
        "checks": [c for c in data.get("checks", []) if isinstance(c, dict)][:20],
        "reason": str(data.get("reason", ""))[:400],
    }


def audit_case(case: dict[str, Any], model: Any, spec_dir: Path = ROOT / "specs") -> dict[str, Any]:
    feature = case_feature(case)
    clauses = [str(c) for c in (case.get("maps_to") or ([case["clause"]] if case.get("clause") else []))]
    spec = spec_for(feature, spec_dir) if feature else ""
    if not spec or not clauses:
        return {"verdict": "unrelated", "checks": [], "reason": "the case names no clause of a known spec"}
    shown = {
        k: case.get(k)
        for k in ("id", "maps_to", "turns", "state", "expected", "turn_expected", "notes")
        if case.get(k) not in (None, [], {})
    }
    prompt = PROMPT.format(
        feature=feature,
        spec=clause_text(spec, clauses),
        case=json.dumps(shown, indent=1, ensure_ascii=False)[:9000],
    )
    text = model.complete_text([{"role": "user", "content": prompt}])
    return {**parse_audit(text), "clauses": clauses, "feature": feature}


def audit_edits(model_spec: str, edits: list[dict[str, Any]], after: dict[str, dict[str, Any]]) -> None:
    """Attach an ``audit`` to every added or modified edit record (in place)."""
    model = make_model(model_spec, max_tokens=2500, anthropic_thinking="disabled")
    for edit in edits:
        if edit.get("kind") not in ("added", "modified"):
            continue
        case = (after.get(edit["path"]) or {}).get("case")
        if not isinstance(case, dict):
            continue
        try:
            edit["audit"] = audit_case(case, model)
        except Exception as error:  # noqa: BLE001 - a checker must not stop the round
            edit["audit"] = {"verdict": "error", "reason": f"{type(error).__name__}: {error}"[:300]}


def audit_run(run: Path, model: Any, classes: set[str] | None, limit: int | None, spec_dir: Path = ROOT / "specs") -> list[dict[str, Any]]:
    """Audit the cases V added or rewrote in a run (from the edit log and the after-snapshots)."""
    events = load_events(run)
    cfg = next(e for e in events if e["kind"] == "start")["config"]
    out: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] != "adjudication":
            continue
        snap_path = run / "log/cases" / f"{event['round']:02d}-after.json"
        if not snap_path.exists():
            continue
        after = json.loads(snap_path.read_text(encoding="utf-8"))
        for edit in event.get("edits", []):
            if edit.get("kind") not in ("added", "modified"):
                continue
            if classes and edit.get("class") not in classes:
                continue
            case = (after.get(edit["path"]) or {}).get("case")
            if not isinstance(case, dict):
                continue
            result = audit_case(case, model, spec_dir)
            out.append(
                {
                    "run": cfg["name"],
                    "mode": cfg["mode"],
                    "round": event["round"],
                    "path": edit["path"],
                    "id": case.get("id"),
                    "class": edit.get("class"),
                    **result,
                }
            )
            if limit and len(out) >= limit:
                return out
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.audit")
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--gold", type=Path, action="append", default=[])
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--model", default="anthropic:claude-sonnet-5")
    parser.add_argument("--limit", type=int, default=None, help="at most this many cases per run")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--spec-dir", type=Path, default=ROOT / "specs", help="the specs the cases were written against (a run from an earlier spec revision needs that revision)")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=True)
    LEDGER.cap = 0
    model = make_model(args.model, max_tokens=2500, anthropic_thinking="disabled")
    results: list[dict[str, Any]] = []
    for gold in args.gold:
        data = json.loads(gold.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            result = audit_case(case, model, args.spec_dir)
            results.append({"gold": str(gold), "id": case.get("id"), "class": "gold", **result})
            print(f"{case.get('id')}: {result['verdict']} - {result['reason'][:160]}")
    for run in args.run:
        for row in audit_run(run, model, set(args.classes) if args.classes else None, args.limit, args.spec_dir):
            results.append(row)
            print(f"{row['run']} r{row['round']} {row['id']} [{row['class']}]: {row['verdict']} - {row['reason'][:140]}")
    by = Counter((r.get("mode", "gold"), r["class"], r["verdict"]) for r in results)
    print("\nverdicts by mode, class:")
    for (mode, cls, verdict), n in sorted(by.items()):
        print(f"  {mode} {cls} {verdict}: {n}")
    print(f"audit cost ${LEDGER.total:.3f}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
