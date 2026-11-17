# SPDX-License-Identifier: Apache-2.0

"""A run's events to a Markdown report: per round the arrivals, the decision, the sessions,
the evaluator's edits with their adjudication, each feature's visible and gold pass rates
and their difference, and the money spent."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "log/events.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def render(run_dir: Path) -> str:
    events = load_events(run_dir)
    start = next((e for e in events if e["kind"] == "start"), {})
    cfg = start.get("config", {})
    rounds = [e for e in events if e["kind"] == "round"]
    lines = [
        f"# Run `{cfg.get('name', run_dir.name)}` — mode {cfg.get('mode')} — seed {cfg.get('seed')}",
        "",
        f"Roles: {cfg.get('role_model')} · subject: {cfg.get('subject_model')} · judge: {cfg.get('judge_model')} · features {cfg.get('features')} · countdown {cfg.get('countdown')}",
        "",
    ]
    features = cfg.get("features", [])
    header = (
        "| Round | Arrivals | O: call V / order | F exit (cmds) | V exit (cmds) | V edits (soft/obs/other) | "
        + " | ".join(f"{f} vis / gold / Δ" for f in features)
        + " | Cost $ |"
    )
    lines += [header, "|" + "---|" * (7 + len(features))]
    softening_total = 0
    for r in rounds:
        n = r["round"]
        arrivals = (
            ", ".join(e["feature"] for e in events if e["kind"] == "arrival" and e["round"] == n)
            or "-"
        )
        decision = next(
            (e for e in events if e["kind"] == "decision" and e["round"] == n and e["role"] == "O"),
            None,
        )
        o_text = (
            f"{decision.get('call_evaluator')} / {decision.get('order')}"
            if decision
            else "isolated"
        )
        sessions = {e["role"]: e for e in events if e["kind"] == "session" and e["round"] == n}
        f_text = (
            f"{sessions['F']['exit_status']} ({sessions['F']['commands']})"
            if "F" in sessions
            else "-"
        )
        v_text = (
            f"{sessions['V']['exit_status']} ({sessions['V']['commands']})"
            if "V" in sessions
            else "-"
        )
        adj = next(
            (e for e in events if e["kind"] == "adjudication" and e["round"] == n), {"edits": []}
        )
        classes = Counter(e["class"] for e in adj["edits"])
        soft = sum(v for k, v in classes.items() if k.startswith("softening"))
        obs = sum(v for k, v in classes.items() if k.startswith("obsolete"))
        other = sum(classes.values()) - soft - obs
        softening_total += soft
        cells = []
        for f in features:
            pf = r["per_feature"].get(f)
            cells.append(
                "-"
                if not pf
                else f"{pf['visible']} / {pf['gold']} / {pf['delta'] if pf['delta'] is not None else '-'} ({pf['status'][:4]}, cd {pf['countdown']})"
            )
        lines.append(
            f"| {n} | {arrivals} | {o_text} | {f_text} | {v_text} | {soft}/{obs}/{other} | "
            + " | ".join(cells)
            + f" | {r['total_cost_usd']:.2f} |"
        )
    gates = [e for e in events if e["kind"] == "gate"]
    delivered = sorted({e["feature"] for e in gates if e.get("status") == "delivered"})
    cancelled = sorted({e["feature"] for e in gates if e.get("status") == "cancelled"})
    withdrawn = sorted({e["feature"] for e in events if e["kind"] == "withdrawal"})
    end = next((e for e in events if e["kind"] == "end"), {})
    lines += [
        "",
        f"Delivered: {delivered or 'none'} · Cancelled: {cancelled or 'none'} · Withdrawn: {withdrawn or 'none'} · Softening edits (C* adjudication): {softening_total}",
        f"Cost: roles ${end.get('roles_cost_usd', 0):.2f}, subject+judge ${end.get('subject_cost_usd', 0):.2f}",
        "",
    ]
    guidance = [e for e in events if e["kind"] == "decision"]
    if guidance:
        lines += ["## Orchestrator guidance", ""]
        for e in guidance:
            for key in ("guidance_spec", "guidance_developer", "guidance_evaluator"):
                if e.get(key):
                    lines.append(
                        f"- round {e['round']} {e['role']} → {key.split('_')[1]}: {e[key][:400]}"
                    )
    stops = [e for e in events if e["kind"] == "stop"]
    if stops:
        lines += ["", f"Stopped early: {stops[0]['reason']}"]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path) -> Path:
    out = run_dir / "report.md"
    out.write_text(render(run_dir), encoding="utf-8")
    return out
