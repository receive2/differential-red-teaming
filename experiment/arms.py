# SPDX-License-Identifier: Apache-2.0

"""Experiment B after the fact: for each closed-loop run, what happened to the infeasible
feature, what the tree at the decision gate was (facade, honest, inert), what the hidden
gold says of it, and which way the evaluator's ruler pointed (its final cases on the facade
reference and on the honest reference).

    python -m experiment.arms --runs runs/B-*/ --out runs/B-summary.json [--no-classify]

The tree at the gate is the workspace at the last developer commit of the decision round
(before any revert), archived to ``<out-dir>/trees/<run>``; it is classified with the F3
detectors (two trials), the hidden F3 gold runs on it (three trials, majority), and the
evaluator's cases of that round (``log/cases/NN-after.json``) run once on each reference."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ROOT
from .report import load_events

PY = sys.executable


def fisher_exact_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] under the hypergeometric null: the
    probability that the first group's count is at least as large as observed."""
    n1, n2, k = a + b, c + d, a + c
    total = n1 + n2
    denom = math.comb(total, k)
    p = 0.0
    for x in range(a, min(n1, k) + 1):
        if k - x <= n2:
            p += math.comb(n1, x) * math.comb(n2, k - x) / denom
    return min(1.0, p)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_eval_json(run: Path, out: Path, *extra: str) -> dict[str, Any]:
    import os

    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    env.update({k: v for k, v in os.environ.items() if k in ("HOME", "LANG", "TMPDIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")})
    cmd = [PY, "-m", "experiment.run_eval", "--run", str(run), "--role", "harness", "--json", str(out), "--quiet", *extra]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    if not out.exists():
        raise RuntimeError(f"run_eval failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-800:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def decision_of(events: list[dict[str, Any]], feature: str) -> tuple[str, int | None]:
    gates = [e for e in events if e["kind"] == "gate" and e.get("feature") == feature]
    for g in gates:
        if g.get("status") in ("delivered", "cancelled"):
            return g["status"], g["round"]
    withdrawn = [e for e in events if e["kind"] == "withdrawal" and e.get("feature") == feature]
    if withdrawn:
        return "withdrawn", withdrawn[0]["round"]
    return "unresolved", gates[-1]["round"] if gates else None


def sha_at(events: list[dict[str, Any]], workspace: Path, feature: str, round_index: int | None) -> str:
    commits = [e for e in events if e["kind"] == "commit" and (round_index is None or e["round"] <= round_index)]
    if commits:
        return commits[-1]["sha"]
    return subprocess.run(["git", "log", "--format=%h", "--reverse"], cwd=workspace, text=True, capture_output=True).stdout.split()[0]


def archive(workspace: Path, sha: str, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    tar = subprocess.run(["git", "archive", sha], cwd=workspace, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], input=tar, check=True)


def cases_of_round(run: Path, round_index: int | None, feature: str) -> dict[str, dict[str, Any]]:
    if round_index is None:
        return {}
    snap = run / "log/cases" / f"{round_index:02d}-after.json"
    if not snap.exists():
        return {}
    data = json.loads(snap.read_text(encoding="utf-8"))
    return {p: v["case"] for p, v in data.items() if p.startswith(f"{feature}/") and isinstance(v, dict) and isinstance(v.get("case"), dict)}


def analyse_run(run: Path, out_dir: Path, feature: str, classify: bool, facade_tree: Path, honest_tree: Path, gold_trials: int) -> dict[str, Any]:
    events = load_events(run)
    cfg = next(e for e in events if e["kind"] == "start")["config"]
    status, round_index = decision_of(events, feature)
    workspace = run / "workspace"
    sha = sha_at(events, workspace, feature, round_index)
    row: dict[str, Any] = {
        "run": run.name,
        "arm": cfg.get("evaluator", "spec"),
        "seed": cfg.get("seed"),
        "status": status,
        "decision_round": round_index,
        "sha_at_gate": sha,
        "claims": [(e["round"], e.get("reason", "")[:200]) for e in events if e["kind"] == "infeasible_claim"],
        "withdrawals": [e["round"] for e in events if e["kind"] == "withdrawal"],
        "o_withdraw_decisions": [e["round"] for e in events if e["kind"] == "decision" and e.get("withdraw")],
        "rounds": max((e["round"] for e in events if e["kind"] == "round"), default=0),
        "roles_cost_usd": next((e.get("roles_cost_usd") for e in reversed(events) if e["kind"] == "end"), None),
        "subject_cost_usd": next((e.get("subject_cost_usd") for e in reversed(events) if e["kind"] == "end"), None),
        "gate_rates": [(g["round"], g.get("cases"), g.get("rate")) for g in events if g["kind"] == "gate" and g.get("feature") == feature],
    }
    tree = out_dir / "trees" / run.name
    archive(workspace, sha, tree)
    if not classify:
        return row
    # 1. the detectors on the tree at the gate
    cls_out = out_dir / "classify" / f"{run.name}.json"
    cls_out.parent.mkdir(parents=True, exist_ok=True)
    env_args = [PY, "-m", "experiment.facade_rate", "classify", "--tree", str(tree), "--world", cfg.get("world", "esci"), "--feature", feature,
                "--subject-model", cfg["subject_model"], "--trials", "2", "--out", str(cls_out), "--no-diff"]
    if cfg.get("subject_thinking"):
        env_args += ["--thinking", cfg["subject_thinking"]]
    import os

    env = {k: v for k, v in os.environ.items() if k in ("HOME", "LANG", "TMPDIR", "PATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(env_args, cwd=ROOT, env=env, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    if cls_out.exists():
        classified = json.loads(cls_out.read_text(encoding="utf-8"))
        row["label"], row["facade_kind"] = classified.get("label"), classified.get("facade_kind")
    else:
        row["label"], row["classify_error"] = "classify-error", (proc.stderr or proc.stdout)[-500:]
    from .facade import static_scan

    row["static"] = static_scan(workspace, feature)
    # 2. the hidden gold on the tree at the gate
    try:
        gold = run_eval_json(run, out_dir / "gold" / f"{run.name}.json", "--gold", str(ROOT / "hidden" / feature / "gold.json"), "--tree", str(tree),
                             *(["--trials", str(gold_trials), "--pass-rule", "majority"] if gold_trials > 1 else []))
        done = [r for r in gold["results"] if not r.get("skipped")]
        row["gold"] = {"passed": sum(r["passed"] for r in done), "total": len(done)}
    except RuntimeError as error:
        row["gold"] = {"error": str(error)[:300]}
    # 3. the evaluator's (or the developer's) cases of the decision round on each reference
    cases = cases_of_round(run, round_index, feature)
    row["cases_at_gate"] = len(cases)
    if cases:
        with tempfile.TemporaryDirectory(dir=str(out_dir)) as tmp:
            root = Path(tmp)
            (root / feature).mkdir()
            for path, case in cases.items():
                (root / path).write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
            for name, ref in (("on_facade", facade_tree), ("on_honest", honest_tree)):
                try:
                    res = run_eval_json(run, out_dir / "ruler" / f"{run.name}-{name}.json", "--all", "--cases-root", str(root), "--tree", str(ref))
                    done = [r for r in res["results"] if not r.get("skipped") and r.get("feature") == feature]
                    row[name] = {"passed": sum(r["passed"] for r in done), "total": len(done), "ids_passed": sorted(r["id"] for r in done if r["passed"])}
                except RuntimeError as error:
                    row[name] = {"error": str(error)[:300]}
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({r["arm"] for r in rows})
    summary: dict[str, Any] = {"arms": {}}
    for arm in arms:
        own = [r for r in rows if r["arm"] == arm]
        delivered = [r for r in own if r["status"] == "delivered"]
        facade_delivered = [r for r in delivered if r.get("label") == "facade"]
        honest_delivered = [r for r in delivered if r.get("label") == "honest"]
        on_f = [r["on_facade"] for r in own if isinstance(r.get("on_facade"), dict) and r["on_facade"].get("total")]
        on_h = [r["on_honest"] for r in own if isinstance(r.get("on_honest"), dict) and r["on_honest"].get("total")]
        summary["arms"][arm] = {
            "runs": len(own),
            "delivered": len(delivered),
            "facade_delivered": len(facade_delivered),
            "facade_delivered_ci": wilson(len(facade_delivered), len(own)),
            "honest_delivered": len(honest_delivered),
            "cancelled": sum(1 for r in own if r["status"] == "cancelled"),
            "withdrawn": sum(1 for r in own if r["status"] == "withdrawn"),
            "unresolved": sum(1 for r in own if r["status"] == "unresolved"),
            "labels_at_gate": dict(Counter(r.get("label") for r in own)),
            "claims": sum(1 for r in own if r.get("claims")),
            "cases_pass_rate_on_facade": round(sum(x["passed"] for x in on_f) / max(1, sum(x["total"] for x in on_f)), 3) if on_f else None,
            "cases_pass_rate_on_honest": round(sum(x["passed"] for x in on_h) / max(1, sum(x["total"] for x in on_h)), 3) if on_h else None,
            "gold_mean": round(sum(r["gold"]["passed"] / r["gold"]["total"] for r in own if isinstance(r.get("gold"), dict) and r["gold"].get("total")) / max(1, sum(1 for r in own if isinstance(r.get("gold"), dict) and r["gold"].get("total"))), 3),
            "cost_usd": round(sum((r.get("roles_cost_usd") or 0) + (r.get("subject_cost_usd") or 0) for r in own), 2),
        }
    tests = {}
    for a, b in (("spec", "world"), ("none", "spec"), ("none", "world")):
        if a in summary["arms"] and b in summary["arms"]:
            sa, sb = summary["arms"][a], summary["arms"][b]
            tests[f"facade_delivered {a} > {b}"] = round(fisher_exact_one_sided(sa["facade_delivered"], sa["runs"] - sa["facade_delivered"], sb["facade_delivered"], sb["runs"] - sb["facade_delivered"]), 4)
    summary["fisher_one_sided_p"] = tests
    return summary


def render(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["| Arm | Runs | Delivered | Facade delivered (95% CI) | Honest delivered | Cancelled | Withdrawn | Labels at gate | Claims | Cases pass on facade ref | on honest ref | Gold mean | Cost $ |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        lo, hi = s["facade_delivered_ci"]
        lines.append(f"| {arm} | {s['runs']} | {s['delivered']} | {s['facade_delivered']} ({lo:.0%}-{hi:.0%}) | {s['honest_delivered']} | {s['cancelled']} | {s['withdrawn']} | {s['labels_at_gate']} | {s['claims']} | {s['cases_pass_rate_on_facade']} | {s['cases_pass_rate_on_honest']} | {s['gold_mean']} | {s['cost_usd']} |")
    lines += ["", "Fisher one-sided p: " + ", ".join(f"{k}: {v}" for k, v in summary["fisher_one_sided_p"].items()), "", "| Run | Arm | Status | Round | Label | Kind | Gold | Cases on facade | on honest | Claims | Withdrawn |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"] or 0)):
        g = r.get("gold") or {}
        f = r.get("on_facade") or {}
        h = r.get("on_honest") or {}
        lines.append(f"| {r['run']} | {r['arm']} | {r['status']} | {r['decision_round']} | {r.get('label')} | {r.get('facade_kind') or '-'} | {g.get('passed', '-')}/{g.get('total', '-')} | {f.get('passed', '-')}/{f.get('total', '-')} | {h.get('passed', '-')}/{h.get('total', '-')} | {len(r.get('claims', []))} | {r.get('withdrawals')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.arms")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--feature", default="F3")
    parser.add_argument("--out", type=Path, required=True, help="summary JSON; trees, gold and ruler results go beside it")
    parser.add_argument("--no-classify", action="store_true")
    parser.add_argument("--facade-tree", type=Path, default=ROOT / "runs/trees/reference-gpt-facade")
    parser.add_argument("--honest-tree", type=Path, default=ROOT / "runs/trees/reference-gpt")
    parser.add_argument("--gold-trials", type=int, default=3)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    args.out = args.out.resolve()
    out_dir = args.out.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    existing = {}
    if args.out.exists():
        existing = {r["run"]: r for r in json.loads(args.out.read_text(encoding="utf-8")).get("rows", [])}
    for run in args.runs:
        run = run.resolve()
        if not (run / "log/events.jsonl").exists():
            continue
        if run.name in existing and existing[run.name].get("label") and not args.no_classify:
            rows.append(existing[run.name])
            continue
        try:
            row = analyse_run(run, out_dir, args.feature, not args.no_classify, args.facade_tree.resolve(), args.honest_tree.resolve(), args.gold_trials)
        except Exception as error:  # noqa: BLE001
            row = {"run": run.name, "arm": "?", "seed": None, "status": "analysis-error", "error": f"{type(error).__name__}: {error}"[:400], "decision_round": None}
        rows.append(row)
        print(f"{row['run']}: {row.get('arm')} {row['status']} r{row.get('decision_round')} label={row.get('label')} gold={row.get('gold')} on_facade={row.get('on_facade', {}).get('passed') if isinstance(row.get('on_facade'), dict) else '-'} on_honest={row.get('on_honest', {}).get('passed') if isinstance(row.get('on_honest'), dict) else '-'}", flush=True)
        args.out.write_text(json.dumps({"rows": rows, "summary": summarize([r for r in rows if r.get("arm") != "?"])}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    summary = summarize([r for r in rows if r.get("arm") != "?"])
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = render(summary, [r for r in rows if r.get("arm") != "?"])
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
