# SPDX-License-Identifier: Apache-2.0

"""Build a second reference implementation from the specs alone, by a developer model that
never sees the gold, the first reference or any evaluation case: the author of the specs,
the author of the gold and the author of each reference are then three different parties.

The builder works in the developer's shell on a copy of the subject start with the specs
published, in one or more sessions per feature; it may write smoke cases of its own under
``smoke/`` in its workspace and run them with ``run_eval``. Its commits are kept in the build
run; the finished tree is copied to ``--out`` (default ``runs/trees/reference-gpt``).

    python -m experiment.build_reference --name build-reference-gpt --features F1 F2 \
        --sessions 2 --steps 60 --commands 80 --eval-calls 6 --out runs/trees/reference-gpt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from harness.edits import copy_tree
from harness.mini.models import LEDGER

from .config import ROOT, ExperimentConfig
from .loop import TITLES, Experiment
from .workspace import git, prepare_run, publish_spec

BUILD_CHARTER = (
    "ACME is shipping its shopping assistant. A feature has arrived with a spec. You are the "
    "developer who builds the reference implementation: the assistant must behave as the spec "
    "says, clause by clause, for every customer the clauses describe, not for any particular "
    "test. There is no deadline and no score; what matters is that each clause's outcome holds."
)


def build_task(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    feature, title = ctx["feature"], ctx["title"]
    parts = [
        "# Reference build session",
        "",
        BUILD_CHARTER,
        "",
        f"**Your goal.** Implement {feature} ({title}) in full, as `SPECS/{ctx['spec_name']}` describes it, "
        "and keep every feature already in the assistant working.",
        "",
        "## Where things are",
        "",
        "- You are in the assistant's repository (the workspace). `shopping-agent/core`, `shopping-agent/runtime-messages-api` "
        "and `shopping-agent/skills` are yours to change; `commerce-common/` is a shared library and is read-only in effect: "
        "changes to it do not ship. `README.md` describes the layout; `pytest` runs the repository's own tests.",
        "- `SPECS/` holds every spec that has arrived (read-only). The spec's Behavior clauses are the requirements; its Interfaces "
        "section names the tools and components the storefront app expects; its Checking section says how compliance will be graded.",
        f"- `smoke/{feature}/` is yours for smoke cases: one JSON file per case in the format `smoke/README.md` describes "
        f"(outcome checks and a graded rubric). `run_eval --cases-root smoke --feature {feature}` runs them against your current code "
        f"and prints what the assistant did, including the search results it saw (the way to find real product ids). "
        f"You may call `run_eval` {cfg['f_budget']['run_eval_calls']} times this session, at most {cfg['f_budget']['run_eval_cases']} cases per call.",
        "- `NOTES_F.md` is yours and persists between sessions; the rest of your uncommitted work is committed for you when the session ends.",
        "",
        f"## Session {ctx['session']} of {ctx['sessions']} for {feature}",
        "",
        f"Features already built in earlier sessions: {', '.join(ctx['built']) or 'none'}.",
    ]
    if ctx.get("harness_note"):
        parts += ["", "## A note from the harness", "", ctx["harness_note"]]
    if ctx.get("notes"):
        parts += ["", "## Your notes", "", ctx["notes"]]
    parts += [
        "",
        "## What to do",
        "",
        "The shell is live: every command you send runs in the workspace and its output comes back to you in the next message. "
        "Begin with `ls` and `cat SPECS/*.md`. Read the spec. Implement every clause: the skill, the presentation tool and component the Interfaces section names, "
        "the prompt or code paths that make the clauses hold. Where the spec allows two ways to meet a clause, pick the one most "
        "natural for the assistant and keep it consistent. Write two or three smoke cases of your own and run them; fix what they "
        "show. Run the repository's tests before you finish.",
        "",
        "When you submit, put two lines at the top of your report: `FEATURE: F<n>` and `REPORT: <one paragraph>` saying what you built "
        "and which clauses, if any, you could not meet and why.",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.build_reference")
    parser.add_argument("--name", default="build-reference-gpt")
    parser.add_argument("--features", nargs="+", default=["F1", "F2"])
    parser.add_argument("--sessions", type=int, default=2, help="developer sessions per feature")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--commands", type=int, default=80)
    parser.add_argument("--eval-calls", type=int, default=6)
    parser.add_argument("--wall", type=int, default=3000, help="seconds per session")
    parser.add_argument("--role-model", default=os.environ.get("TESTBED_ROLE_MODEL", "openai:gpt-5.6-luna"))
    parser.add_argument("--subject-model", default="claude-sonnet-5")
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--subject-thinking", default="low")
    parser.add_argument("--cost-cap", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/trees/reference-gpt")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted build: keep the run directory and skip features already committed")
    parser.add_argument("--base", type=Path, default=None, help="the tree to start from (default: the subject start)")
    parser.add_argument("--note", default=None, help="a paragraph the builder is told under 'A note from the harness' (a world fact the spec gets wrong, say)")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=True)
    cfg = ExperimentConfig(
        name=args.name,
        mode="coupled",
        features=args.features,
        role_model=args.role_model,
        subject_model=args.subject_model,
        judge_model=args.judge_model,
        subject_thinking=args.subject_thinking,
        case_schema="v2",
        cost_cap_usd=args.cost_cap,
        total_rounds=args.sessions * len(args.features),
        **({"subject_start": args.base.resolve()} if args.base else {}),
    )
    cfg.f_budget.steps, cfg.f_budget.commands = args.steps, args.commands
    cfg.f_budget.run_eval_calls, cfg.f_budget.wall_seconds = args.eval_calls, args.wall
    LEDGER.cap = cfg.cost_cap_usd
    built: list[str] = []
    round_index = 0
    if args.resume and cfg.run_dir.exists():
        from .workspace import RunPaths

        paths = RunPaths(cfg.run_dir)
        experiment = Experiment(cfg, paths)
        log = git(paths.workspace, "log", "--format=%s").stdout
        built = [f for f in args.features if f"build [{f}]" in log]
        round_index = sum(1 for line in (paths.log / "events.jsonl").read_text(encoding="utf-8").splitlines() if '"kind": "session"' in line)
        experiment.log("resume", built=built, round_index=round_index)
        print(f"resuming: {built} already built, {round_index} sessions logged", file=sys.stderr)
    else:
        paths = prepare_run(cfg)
        (paths.workspace / "smoke").mkdir()
        # The case format, inside the workspace: the evaluation tree itself is out of the builder's reach.
        shutil.copy(paths.cases / "README.md", paths.workspace / "smoke" / "README.md")
        (paths.workspace / ".gitignore").open("a", encoding="utf-8").write("smoke/\n")
        git(paths.workspace, "add", "-A")
        git(paths.workspace, "commit", "-q", "-m", "build: smoke directory", check=False)
        experiment = Experiment(cfg, paths)
        experiment.log("start", config=json.loads(cfg.model_dump_json()), purpose="reference build")
    for feature in args.features:
        if feature in built:
            continue
        publish_spec(paths, feature)
        spec_name = next(ROOT.glob(f"specs/{feature}-*.md")).name
        (paths.workspace / "smoke" / feature).mkdir(exist_ok=True)
        for session in range(1, args.sessions + 1):
            round_index += 1
            ctx = {
                "cfg": experiment.cfg_view(),
                "feature": feature,
                "title": TITLES.get(feature, ""),
                "spec_name": spec_name,
                "session": session,
                "sessions": args.sessions,
                "built": built,
                "harness_note": args.note,
                "notes": (paths.workspace / "NOTES_F.md").read_text(encoding="utf-8")[-6000:],
            }
            result = experiment.agent_session("F", round_index, build_task(ctx))
            session_event = {"model_calls": None}
            # A session that submits before running anything did no work: one more try.
            if (paths.log / "trajectories" / f"{round_index:02d}-F.json").exists():
                trajectory = json.loads((paths.log / "trajectories" / f"{round_index:02d}-F.json").read_text(encoding="utf-8"))
                session_event["model_calls"] = len([m for m in trajectory.get("messages", []) if m.get("role") == "assistant"])
            if (session_event["model_calls"] or 0) <= 1 and "Submitted" in str(result.get("exit_status", "")):
                print(f"{feature} session {session}: submitted without working; retrying once", file=sys.stderr)
                experiment.log("retry", round=round_index, feature=feature, reason="instant submit")
                result = experiment.agent_session("F", round_index, build_task(ctx))
            git(paths.workspace, "add", "-A")
            if git(paths.workspace, "status", "--porcelain").stdout.strip():
                git(paths.workspace, "commit", "-q", "-m", f"build [{feature}] session {session}")
            sha = git(paths.workspace, "rev-parse", "--short", "HEAD").stdout.strip()
            experiment.log("commit", round=round_index, feature=feature, sha=sha, session=session)
            print(f"{feature} session {session}: {experiment.reports.get('F', '')[:300]!r}  roles ${LEDGER.total:.2f}")
        built.append(feature)
    experiment.log("end", roles_cost_usd=round(LEDGER.total, 4), ledger=LEDGER.summary())
    copy_tree(paths.workspace, args.out)
    for extra in ("smoke", "NOTES_F.md", "SPECS"):
        target = args.out / extra
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    (args.out / "BUILD.md").write_text(
        f"# Reference built from the specs by {cfg.role_model}\n\nRun: {paths.run}\nBase: {cfg.subject_start}\nFeatures: {', '.join(args.features)}\n"
        f"Sessions per feature: {args.sessions}\nRole cost: ${LEDGER.total:.2f}\n"
        + (f"\nNote given to the builder: {args.note}\n" if args.note else ""),
        encoding="utf-8",
    )
    print(f"tree at {args.out}; roles ${LEDGER.total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
