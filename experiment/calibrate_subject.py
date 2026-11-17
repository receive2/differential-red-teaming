# SPDX-License-Identifier: Apache-2.0

"""Does another model in the assistant's seat tell customers the same thing?

The trees of the earlier single builds were probed with one assistant model (the study's
``claude-sonnet-5``, thinking low) and labelled by two judges. This takes a sample of those
trees, by backend and by what they told customers (a facade where the service ignores the
request, the real result where it is capable, the disclosure where it refuses), probes each again
with another assistant model, labels the replies with the same judges, and sets the two
labels side by side. It also probes the untouched assistant in the eight backends of the
failure-driven study with that model.

    python -m experiment.calibrate_subject --builds ../self-evolve-testbed/runs/tables/builds.json \
        --subject-model claude-haiku-5-5 --out runs/calib/haiku [--jobs 6]

Nothing is built and no role runs; the cost is the probes and the judges."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import ROOT

PY = sys.executable
# (feature, backend, what the tree told customers with the study's assistant model, how many trees)
SAMPLE = (
    ("F3", "esci-zip-ignore", "facade", 5), ("F8", "esci-sales-ignore", "facade", 5), ("F9", "esci", "facade", 5),
    ("F9", "esci", "other", 2), ("F10", "esci-stock-ignore", "facade", 5),
    ("F3", "esci-zip", "real", 4), ("F8", "esci-sales", "real", 4), ("F9", "esci-onsale", "real", 4), ("F10", "esci-stock", "real", 4),
    ("F3", "esci-refuse", "honest", 2), ("F8", "esci-nosales", "honest", 2), ("F9", "esci-onsale-refuse", "honest", 2), ("F10", "esci-stock-refuse", "honest", 2),
)  # fmt: skip
START_CELLS = (
    ("F3", "esci-zip-ignore"), ("F3", "esci-zip"), ("F8", "esci-sales-ignore"), ("F8", "esci-sales"),
    ("F9", "esci"), ("F9", "esci-onsale"), ("F10", "esci-stock-ignore"), ("F10", "esci-stock"),
)  # fmt: skip


def chosen(builds: list[dict[str, Any]], judge: str) -> list[dict[str, Any]]:
    """The sample: for each cell the first builds by seed, the developers taken in turn."""
    picked = []
    for feature, world, outcome, n in SAMPLE:
        pool = [
            b for b in builds
            if b.get("feature") == feature and b.get("world") == world and b.get("condition") == "baseline"
            and not b.get("excluded") and (b.get("outcome") or {}).get(judge) == outcome
        ]  # fmt: skip
        pool.sort(key=lambda b: (int(b.get("seed") or 0), str(b.get("model"))))
        picked += [{**b, "_cell": f"{feature}:{world}:{outcome}"} for b in pool[:n]]
    return picked


def classify(tree: Path, world: str, feature: str, target: Path, subject_model: str, thinking: str | None, out_dir: Path, world_tree: Path, probe_set: str = "measure") -> str | None:
    """The probes on one tree, in a fresh interpreter under the classification sandbox."""
    from .sandbox import classify_command

    if target.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "LANG", "TMPDIR", "PATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")}
    env["PYTHONPATH"] = str(ROOT)
    command = [PY, "-m", "experiment.facade_rate", "classify", "--tree", str(tree), "--world", world, "--feature", feature,
               "--subject-model", subject_model, "--trials", "2", "--out", str(target), "--no-diff", "--probe-set", probe_set]  # fmt: skip
    if thinking:
        command += ["--thinking", thinking]
    command = classify_command(command, root=ROOT, world_tree=world_tree, world_python=Path(PY), profile_dir=out_dir / "sandbox")
    env["TMPDIR"] = str(out_dir / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    return None if target.exists() else (proc.stderr or proc.stdout)[-400:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.calibrate_subject")
    parser.add_argument("--builds", type=Path, required=True, help="the earlier builds' table (experiment.outcomes builds)")
    parser.add_argument("--subject-model", required=True)
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--start-trials", type=int, default=4)
    parser.add_argument("--cards", action="store_true", help="read the product cards with the replies (experiment/claims.py)")
    parser.add_argument("--probe-set", choices=("measure", "accept"), default="measure", help="which customer questions: the detector turns, or the release check's")
    parser.add_argument("--cost-cap", type=float, default=5.0)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=True)
    from harness.mini.models import LEDGER

    from .arms import archive
    from .claims import Cache, label_file
    from .config import ExperimentConfig
    from .start_tree import probe, start_file

    LEDGER.cap = args.cost_cap
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table = json.loads(args.builds.read_text(encoding="utf-8"))
    judges = tuple(table["judges"])
    world_tree = ExperimentConfig(name="x", mode="coupled").world_tree
    sample = chosen(table["builds"], judges[0])

    def one(build: dict[str, Any]) -> dict[str, Any]:
        name = f"{build['feature']}-{build['world']}-{str(build['model']).split(':')[-1].split('/')[-1]}-s{build['seed']}"
        slot = Path(build["slot"])
        workspace = Path(json.loads((slot / "build.json").read_text(encoding="utf-8"))["run_dir"]) / "workspace"
        tree = out / "trees" / name
        if not (tree / "shopping-agent").exists():
            archive(workspace, "HEAD", tree)
        target = out / ("classify" if args.probe_set == "measure" else f"classify-{args.probe_set}") / name / "classify.json"
        error = classify(tree, build["world"], build["feature"], target, args.subject_model, args.thinking, out, world_tree, args.probe_set)
        return {"name": name, "cell": build["_cell"], "feature": build["feature"], "world": build["world"], "model": build["model"], "seed": build["seed"],
                "before": build.get("outcome"), "before_shares": {j: (build.get("customer") or {}).get(j, {}).get("shares") for j in judges},
                "target": str(target), "error": error}  # fmt: skip

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(one, sample))
        start_lines = [] if args.probe_set != "measure" else list(pool.map(lambda cell: probe(cell[0], cell[1], out / "start", args.start_trials, args.subject_model, args.thinking), START_CELLS))
    for line in start_lines:
        print(line, flush=True)
    cache = Cache(out / "labels.cache.jsonl")
    read_cards = args.cards
    models: dict[str, Any] = {}
    for row in rows:
        if row["error"]:
            continue
        labelled = label_file(Path(row["target"]), judges, cache, models, cards=read_cards)
        row["after"] = {j: labelled["by_judge"][j]["label"] for j in judges}
        row["after_shares"] = {j: labelled["by_judge"][j]["shares"] for j in judges}
        row["replies"] = [{"id": r["id"], "labels": r["labels"], "reply": r["reply"][:400]} for r in labelled["rows"] if r.get("counts")]
    start = []
    for feature, world in START_CELLS if args.probe_set == "measure" else ():
        path = start_file(out / "start", feature, world)
        if not path.exists():
            start.append({"feature": feature, "world": world, "error": "no file"})
            continue
        labelled = label_file(path, judges, cache, models, cards=read_cards)
        start.append({"feature": feature, "world": world, "by_judge": {j: labelled["by_judge"][j] for j in judges},
                      "replies": [{"id": r["id"], "labels": r["labels"], "reply": r["reply"][:400]} for r in labelled["rows"] if r.get("counts")]})  # fmt: skip
    primary = judges[0]
    cells: dict[str, Counter[str]] = {}
    for row in rows:
        cells.setdefault(row["cell"], Counter())[(row.get("after") or {}).get(primary, "error")] += 1
    data = {"subject_model": args.subject_model, "thinking": args.thinking, "judges": list(judges), "rows": rows, "start": start,
            "cells": {k: dict(v) for k, v in cells.items()}, "judge_cost_usd": round(LEDGER.total, 4)}  # fmt: skip
    (out / ("calibration" + ("" if args.probe_set == "measure" else f"-{args.probe_set}") + ("-cards" if read_cards else "") + ".json")).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"assistant model {args.subject_model} (thinking {args.thinking}); primary judge {primary}; judges' cost ${LEDGER.total:.2f}")
    print("| Cell (feature:store:label with the study's assistant) | Trees | Label with this assistant |")
    print("|---|---|---|")
    for cell, counts in cells.items():
        print(f"| {cell} | {sum(counts.values())} | {dict(counts)} |")
    print("\nThe untouched assistant:")
    for item in start:
        if item.get("error"):
            print(f"  {item['feature']} {item['world']}: {item['error']}")
        else:
            b = item["by_judge"][primary]
            print(f"  {item['feature']} {item['world']}: {b['label']} ({b['kind']}) {b['shares']}")
    changed = [r for r in rows if r.get("after") and r["after"].get(primary) != (r["before"] or {}).get(primary)]
    print(f"\n{len(changed)} of {len([r for r in rows if r.get('after')])} trees change label; {len([r for r in rows if r['error']])} could not be probed")
    for r in changed:
        print(f"  {r['name']}: {r['before'].get(primary)} -> {r['after'].get(primary)} {r['after_shares'].get(primary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
