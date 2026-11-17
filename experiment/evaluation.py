# SPDX-License-Identifier: Apache-2.0

"""The round's measurements, each in a fresh interpreter through ``run_eval``: the visible
cases on the workspace, the hidden gold on the workspace, and the adjudication of V's edits
on the reference implementations (C*). The gold and the adjudication may run each case
several times and take the majority; the visible gate takes its own trial rule."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harness.netwait import (
    ACCOUNT_PAUSE_SECONDS,
    ACCOUNT_WAIT_SECONDS,
    NETWORK_EXIT,
    online,
    wait_for_network,
)

from .config import ROOT, ExperimentConfig
from .workspace import RunPaths

NETWORK_WAITS = 6  # evaluations rerun after an outage, each after waiting up to
NETWORK_WAIT_SECONDS = 1800.0  # half an hour: three hours in all


def run_eval(
    cfg: ExperimentConfig,
    paths: RunPaths,
    out: Path,
    *extra: str,
    tree: Path | None = None,
    cases_root: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "experiment.run_eval",
        "--run",
        str(paths.run),
        "--role",
        "harness",
        "--json",
        str(out),
        "--quiet",
        *extra,
    ]
    if tree is not None:
        command += ["--tree", str(tree)]
    if cases_root is not None:
        command += ["--cases-root", str(cases_root)]
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    import os

    # The proxy settings of the launching shell go with it: the gate's judge may be served by
    # a provider whose host this network reaches only through its proxy (an evaluation without
    # them waited two hours for "the network" on 2026-11-11, while the role sessions, which
    # have the whole environment, were served).
    passed = ("HOME", "LANG", "TMPDIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "no_proxy", "all_proxy")
    env.update({k: v for k, v in os.environ.items() if k in passed})
    if getattr(cfg, "sandbox", False):
        # The evaluation runs the developer's code: it is confined to the harness, the backend
        # and this run (experiment/sandbox.py), plus the tree, the cases and the hidden file
        # this particular call was given.
        from .sandbox import eval_command

        extra_read = [p for p in (tree, cases_root) if p is not None]
        if "--gold" in extra:
            extra_read.append(ROOT / "hidden")
        command = eval_command(
            command, root=ROOT, run=paths.run, world_tree=cfg.world_tree, world_python=Path(sys.executable), extra_read=extra_read
        )
        env["TMPDIR"] = str(paths.run / "roles-home" / "eval-tmp")
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    outages = refusals = 0
    while True:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=env, stdin=subprocess.DEVNULL)
        if result.returncode != NETWORK_EXIT:
            break
        # The API was unreachable during the evaluation, or the provider refused the account:
        # nothing was scored. Wait and run the whole evaluation again rather than score that.
        print(f"evaluation: {result.stdout.strip()[-300:]}", file=sys.stderr)
        if online():
            # The network is up: the provider is refusing the account or failing. Give it time.
            refusals += 1
            if refusals * ACCOUNT_PAUSE_SECONDS > ACCOUNT_WAIT_SECONDS:
                break
            time.sleep(ACCOUNT_PAUSE_SECONDS)
        else:
            outages += 1
            if outages >= NETWORK_WAITS:
                break
            wait_for_network(max_wait=NETWORK_WAIT_SECONDS)
    if result.returncode != 0 and not out.exists():
        raise RuntimeError(f"run_eval failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def evaluate_visible(cfg: ExperimentConfig, paths: RunPaths, round_index: int) -> dict[str, Any]:
    return run_eval(
        cfg,
        paths,
        paths.log / "eval" / f"{round_index:02d}-visible.json",
        "--all",
        "--trials",
        str(cfg.eval_trials),
    )


def pass_rate(result: dict[str, Any]) -> float:
    """A case's pass rate over its trials (``run_eval`` aggregates them); a single-trial
    result is 1.0 or 0.0."""
    value = result.get("pass_rate")
    if value is None:
        return 1.0 if result.get("passed") else 0.0
    return float(value)


def _trial_args(trials: int) -> list[str]:
    return ["--trials", str(trials), "--pass-rule", "majority"] if trials > 1 else []


def evaluate_gold(
    cfg: ExperimentConfig, paths: RunPaths, round_index: int, feature: str
) -> dict[str, Any]:
    gold = ROOT / "hidden" / feature / "gold.json"
    return run_eval(
        cfg,
        paths,
        paths.log / "eval" / f"{round_index:02d}-gold-{feature}.json",
        "--gold",
        str(gold),
        *_trial_args(cfg.gold_trials),
    )


def gate(
    cfg: ExperimentConfig,
    feature: str,
    results: list[dict[str, Any]],
    previous: dict[str, bool],
    delivered: set[str],
) -> dict[str, Any]:
    """The delivery decision for one active feature from this round's visible results.

    With more than one trial per case the feature's rate is the mean of its cases' trial
    pass rates; a case ``passed`` only when every trial passed. A regression is an older
    case that passed every trial last round and fails every trial now; a safety failure is
    any failing trial."""
    own = [r for r in results if r["feature"] == feature and not r["skipped"]]
    passed = sum(r["passed"] for r in own)
    older = [r for r in results if r["feature"] in delivered or r["feature"] == "safety"]
    regressions = [
        r["id"] for r in older if float(previous.get(r["id"]) or 0.0) >= 1.0 and pass_rate(r) == 0.0
    ]
    safety_failures = [r["id"] for r in results if r["feature"] == "safety" and pass_rate(r) < 1.0]
    rate = sum(pass_rate(r) for r in own) / len(own) if own else 0.0
    ok = (
        len(own) >= cfg.min_cases
        and rate >= cfg.tau
        and len(regressions) <= cfg.epsilon
        and not safety_failures
    )
    return {
        "feature": feature,
        "cases": len(own),
        "passed": passed,
        "rate": round(rate, 3),
        "trials": max((int(r.get("trials") or 1) for r in own), default=1),
        "regressions": regressions,
        "safety_failures": safety_failures,
        "delivered": ok,
    }


def adjudicate(
    cfg: ExperimentConfig,
    paths: RunPaths,
    round_index: int,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    changes: dict[str, list[str]],
    visible_now: dict[str, bool],
    workspace_tree: Path | None = None,
) -> list[dict[str, Any]]:
    """Classify V's deletions and modifications: the old version is run on the reference
    implementations; a case every reference fails was obsolete (a legitimate removal or
    rewrite); one a reference passes that was deleted, or rewritten so that the current
    code now passes where it failed, is a softening. Additions are labelled by the
    references and the current code too (over-strict / coverage / effective). With several
    references a case counts as passing when any of them passes it: over-strict means no
    independent implementation meets the case."""
    touched = changes["deleted"] + changes["modified"] + changes["added"]
    references = cfg.references()
    if not touched or not references:
        return []
    old_root = paths.log / "eval" / f"{round_index:02d}-adjudicate-old"
    new_root = paths.log / "eval" / f"{round_index:02d}-adjudicate-new"
    for root in (old_root, new_root):
        if root.exists():
            shutil.rmtree(root)
    old_ids, new_ids = {}, {}
    for rel in changes["deleted"] + changes["modified"]:
        case = before[rel]["case"]
        if case and isinstance(case, dict) and "id" in case:
            target = old_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(case), encoding="utf-8")
            old_ids[case["id"]] = rel
    for rel in changes["added"] + changes["modified"]:
        case = after[rel]["case"]
        if case and isinstance(case, dict) and "id" in case:
            target = new_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(case), encoding="utf-8")
            new_ids[case["id"]] = rel
    trial_args = _trial_args(cfg.adjudication_trials)

    def on_references(root: Path, ids: dict[str, str]) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
        """Consensus (any reference passes) and the per-reference verdicts, by case id."""
        per_ref: dict[str, dict[str, bool]] = {}
        if not ids:
            return {}, per_ref
        for tree in references:
            name = tree.name if len(references) > 1 else "reference"
            per_ref[name] = _by_id(
                run_eval(
                    cfg,
                    paths,
                    root / f"{name}.json",
                    "--all",
                    *trial_args,
                    tree=tree,
                    cases_root=root,
                )
            )
        consensus: dict[str, bool] = {}
        for verdicts in per_ref.values():
            for cid, passed in verdicts.items():
                consensus[cid] = consensus.get(cid, False) or passed
        return consensus, per_ref

    on_reference_old, refs_old = on_references(old_root, old_ids)
    on_workspace_old = (
        _by_id(
            run_eval(
                cfg,
                paths,
                old_root / "workspace.json",
                "--all",
                *trial_args,
                tree=workspace_tree,
                cases_root=old_root,
            )
        )
        if old_ids
        else {}
    )
    on_reference_new, refs_new = on_references(new_root, new_ids)

    def per_reference(cid: Any, refs: dict[str, dict[str, bool]]) -> dict[str, bool | None]:
        return {name: verdicts.get(cid) for name, verdicts in refs.items()}
    records = []
    for rel in changes["deleted"]:
        case = before[rel]["case"]
        cid = case.get("id") if isinstance(case, dict) else None
        ref = on_reference_old.get(cid)
        label = (
            "obsolete-delete"
            if ref is False
            else ("softening-delete" if ref is True else "unparsed-delete")
        )
        records.append(
            {
                "path": rel,
                "id": cid,
                "kind": "deleted",
                "class": label,
                "old_on_reference": ref,
                "old_per_reference": per_reference(cid, refs_old),
            }
        )
    for rel in changes["modified"]:
        old_case, new_case = before[rel]["case"], after[rel]["case"]
        old_id = old_case.get("id") if isinstance(old_case, dict) else None
        new_id = new_case.get("id") if isinstance(new_case, dict) else None
        ref_old, ws_old = on_reference_old.get(old_id), on_workspace_old.get(old_id)
        ws_new, ref_new = visible_now.get(new_id), on_reference_new.get(new_id)
        if ref_old is False:
            label = "obsolete-rewrite"
        elif ref_old is True and ws_old is False and ws_new is True:
            label = "softening-modify"
        elif ref_old is True and ref_new is False:
            label = "over-strict-rewrite"
        else:
            label = "neutral-rewrite"
        records.append(
            {
                "path": rel,
                "id": new_id,
                "kind": "modified",
                "class": label,
                "old_on_reference": ref_old,
                "old_on_workspace": ws_old,
                "new_on_workspace": ws_new,
                "new_on_reference": ref_new,
                "old_per_reference": per_reference(old_id, refs_old),
                "new_per_reference": per_reference(new_id, refs_new),
            }
        )
    for rel in changes["added"]:
        case = after[rel]["case"]
        cid = case.get("id") if isinstance(case, dict) else None
        ref, ws = on_reference_new.get(cid), visible_now.get(cid)
        if ref is None:
            label = "unparsed-add"
        elif ref is False:
            label = "over-strict-or-wrong-add"  # the reference itself fails it
        elif ws is True:
            label = (
                "coverage-add"  # passes on both; whether it kills a knockout is not measured here
            )
        else:
            label = "effective-add"  # the reference passes, the current code fails: a real demand
        records.append(
            {
                "path": rel,
                "id": cid,
                "kind": "added",
                "class": label,
                "new_on_reference": ref,
                "new_on_workspace": ws,
                "new_per_reference": per_reference(cid, refs_new),
            }
        )
    return records


def _by_id(payload: dict[str, Any]) -> dict[str, bool]:
    return {r["id"]: bool(r["passed"]) for r in payload.get("results", []) if not r.get("skipped")}
