# SPDX-License-Identifier: Apache-2.0

"""The commands of the experiment on the second backend:
the design of ``experiment/jobs.py`` over the nine conditions of the WANDS catalog, one
``experiment.run`` command a run, dealt over lanes for ``experiment/lane.sh``.

    python -m experiment.jobs_wands --out paper/jobs --lanes 28     # the experiment: W-<lane>.jobs, W-runs.tsv
    python -m experiment.jobs_wands --out paper/jobs --smoke 1      # its smoke runs (seed 20): WS.jobs, WS-runs.tsv
    python -m experiment.jobs_wands --out paper/jobs --smoke 2      # the second set: WS2.jobs, WS2-runs.tsv

The arrangements, the role models, the seeds, the two arms with further seeds, the flags of a
run and the way the runs are dealt over the lanes are ``experiment/jobs.py``'s own, taken from
it: a command here differs from one there in the run's name, the topic, the backend and the
checkout the backend is taken from. A run is named ``W-<number>``, the numbers dealt in an
order drawn with a fixed seed; ``W-runs.tsv`` says what each run is.

The design: 9 conditions x 7 arrangements x 2 role models x 8 seeds (31 to 38), and 16
further seeds (39 to 54) for ``pair`` and ``compare`` in the four conditions whose backend is
capable: 1,008 + 256 = 1,264 runs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from . import jobs
from .jobs import (
    ARRANGEMENTS,
    FURTHER_ARMS,
    FURTHER_SEEDS,
    MINUTES,
    ROLE_MODELS,
    SEEDS,
    SMOKE_SEED,
    deal,
    table,
)

# (topic, backend, the backend's kind): ign takes the request and does not use it, non has no
# input for it, cap is capable of it.
CONDITIONS = (
    ("F21", "wands-fit-ignore", "ign"), ("F21", "wands-fit", "cap"),
    ("F22", "wands-date-ignore", "ign"), ("F22", "wands", "non"), ("F22", "wands-date", "cap"),
    ("F23", "wands-new-ignore", "ign"), ("F23", "wands-new", "cap"),
    ("F24", "wands-stock-ignore", "ign"), ("F24", "wands-stock", "cap"),
)  # fmt: skip
ORDER_SEED = 20261115
WORLD_TREE = "runs/world-wands-f4b4055"  # the backend checkout at f4b4055 with its catalog built, copied once for the experiment
# The smoke runs: every arrangement, both role models, every topic, every kind of backend; the
# checker in a backend of each kind.
SMOKE = (
    ("F21", "wands-fit-ignore", "pair", "lu"), ("F24", "wands-stock", "pair", "ds"), ("F23", "wands-new-ignore", "isolated", "lu"),
    ("F22", "wands-date", "isolated", "ds"), ("F24", "wands-stock-ignore", "solo", "ds"), ("F22", "wands-date-ignore", "monitor", "lu"),
    ("F22", "wands", "redteam", "ds"), ("F21", "wands-fit-ignore", "compare", "lu"), ("F23", "wands-new", "compare", "ds"),
    ("F21", "wands-fit", "sealed", "lu"), ("F23", "wands-new-ignore", "sealed", "ds"), ("F22", "wands", "compare", "lu"),
)  # fmt: skip
# A second set: in the first the one run of the checker in a backend that ignores the request
# never came to a release (its gate did not pass in five rounds), so the checker is tried in
# each of the four.
SMOKE_2 = (
    ("F22", "wands-date-ignore", "compare", "lu"), ("F23", "wands-new-ignore", "compare", "ds"),
    ("F24", "wands-stock-ignore", "compare", "lu"), ("F21", "wands-fit-ignore", "compare", "ds"),
)  # fmt: skip
SMOKE_SETS = {1: (SMOKE, 1, "WS"), 2: (SMOKE_2, 13, "WS2")}  # the set, its first number, its files' stem


def command(name: str, topic: str, world: str, arrangement: str, family: str, seed: int) -> str:
    """The command of ``experiment/jobs.py`` for such a run, with this backend's checkout."""
    line = jobs.command(name, topic, world, arrangement, family, seed)
    ending = f" --world-tree {jobs.WORLD_TREE}"
    if not line.endswith(ending):
        raise RuntimeError("experiment/jobs.py no longer ends a command with the backend's checkout")
    return line[: -len(ending)] + f" --world-tree {WORLD_TREE}"


def design() -> list[dict[str, object]]:
    """Every run of the experiment, named: the full table with eight seeds, and the further
    seeds of ``pair`` and ``compare`` where the backend is capable."""
    runs = [
        {"topic": topic, "world": world, "kind": kind, "arrangement": arrangement, "family": family, "seed": seed}
        for topic, world, kind in CONDITIONS
        for arrangement in ARRANGEMENTS
        for family in ROLE_MODELS
        for seed in (*SEEDS, *(FURTHER_SEEDS if kind == "cap" and arrangement in FURTHER_ARMS else ()))
    ]
    rng = random.Random(ORDER_SEED)
    rng.shuffle(runs)
    for number, run in enumerate(runs, 1):
        run["name"] = f"W-{number:04d}"
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.jobs_wands")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=28)
    parser.add_argument("--smoke", type=int, choices=tuple(SMOKE_SETS), default=None, help="a set of smoke runs (seed 20, counted nowhere) instead of the experiment")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        kinds = {(topic, world): kind for topic, world, kind in CONDITIONS}
        chosen, first, stem = SMOKE_SETS[args.smoke]
        runs = [
            {"name": f"WS-{n:02d}", "topic": topic, "world": world, "kind": kinds[(topic, world)], "arrangement": arrangement, "family": family, "seed": SMOKE_SEED}
            for n, (topic, world, arrangement, family) in enumerate(chosen, first)
        ]
        (args.out / f"{stem}.jobs").write_text("".join(command(r["name"], r["topic"], r["world"], r["arrangement"], r["family"], r["seed"]) + "\n" for r in runs), encoding="utf-8")
        (args.out / f"{stem}-runs.tsv").write_text(table(runs), encoding="utf-8")
        print(f"{len(runs)} smoke runs: {args.out / (stem + '.jobs')}")
        return 0
    runs = design()
    dealt = deal(runs, args.lanes)
    for lane, own in enumerate(dealt, 1):
        (args.out / f"W-{lane:02d}.jobs").write_text(
            "".join(command(r["name"], r["topic"], r["world"], r["arrangement"], r["family"], r["seed"]) + "\n" for r in own), encoding="utf-8"
        )
    (args.out / "W-runs.tsv").write_text(table(runs), encoding="utf-8")
    hours = [sum(MINUTES[(r["arrangement"], r["family"])] for r in own) / 60 for own in dealt]
    print(f"{len(runs)} runs over {args.lanes} lanes: {min(len(own) for own in dealt)} to {max(len(own) for own in dealt)} a lane, {min(hours):.1f} to {max(hours):.1f} expected hours a lane")
    return 0


if __name__ == "__main__":
    sys.exit(main())
