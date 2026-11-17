# SPDX-License-Identifier: Apache-2.0

"""What the assistant tells the customer before anybody has touched it, backend by backend.

    python -m experiment.start_tree F9:esci F9:esci-onsale-refuse F10:esci-stock ... \
        --trials 6 --out runs/baseline [--jobs 5]

For each ``<feature>:<backend>`` the start tree is copied as a build receives it (with the
backend's patch where the backend's repository differs from upstream's, ``harness/worlds.py``),
probed with the feature's customer questions in that backend, and its classify file written
to ``<out>/<feature>-<backend>-start.json`` (``<feature>-start.json`` for the backend as
published, the name first used). No developer is involved and nothing is
built; ``experiment.outcomes builds --start <out>`` labels the files with the judges."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT, ExperimentConfig
from .facade import DETECTOR_TURNS


def start_file(out: Path, feature: str, world: str) -> Path:
    return out / (f"{feature}-start.json" if world == "esci" else f"{feature}-{world}-start.json")


def probe(feature: str, world: str, out: Path, trials: int, subject_model: str, thinking: str | None) -> str:
    from harness.edits import copy_tree
    from harness.worlds import PATCHES, WORLD_KINDS, patch_start

    if world not in WORLD_KINDS or feature not in DETECTOR_TURNS:
        raise SystemExit(f"unknown cell {feature}:{world}")
    target = start_file(out, feature, world)
    if target.exists():
        return f"{feature} {world}: {target.name} is there already"
    tree = out / "trees" / f"{feature}-{world}"
    if tree.exists():
        shutil.rmtree(tree)
    tree.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(ExperimentConfig(name="x", mode="coupled").subject_start, tree)
    if world in PATCHES:
        patch_start(tree, world)
    args = [
        sys.executable, "-u", "-m", "experiment.facade_rate", "classify", "--tree", str(tree), "--world", world, "--feature", feature,
        "--subject-model", subject_model, "--trials", str(trials), "--no-diff", "--out", str(target),
    ]  # fmt: skip
    if thinking:
        args += ["--thinking", thinking]
    env = {k: v for k, v in os.environ.items() if k in ("HOME", "LANG", "TMPDIR", "PATH", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")}
    env["PYTHONPATH"] = str(ROOT)
    done = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    if not target.exists():
        return f"{feature} {world}: FAILED {(done.stderr or done.stdout)[-400:]}"
    return f"{feature} {world}: {(done.stdout or '').strip().splitlines()[-1] if done.stdout.strip() else 'written'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.start_tree")
    parser.add_argument("cells", nargs="+", help="<feature>:<store>, for example F10:esci-stock-ignore")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "baseline")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--subject-model", default="claude-sonnet-5")
    parser.add_argument("--subject-thinking", default="low")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    cells = [tuple(cell.split(":", 1)) for cell in args.cells]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for line in pool.map(lambda cell: probe(cell[0], cell[1], out, args.trials, args.subject_model, args.subject_thinking), cells):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
