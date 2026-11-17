# SPDX-License-Identifier: Apache-2.0

"""The commands of the experiment on the first backend (revision 4 of the protocol):
one ``experiment.run`` command a run, dealt over lanes for ``experiment/lane.sh``.

    python -m experiment.jobs --out paper/jobs --lanes 28     # the experiment: R-<lane>.jobs, R-runs.tsv
    python -m experiment.jobs --out paper/jobs --smoke 1      # its smoke runs (seed 20): S.jobs, S-runs.tsv
    python -m experiment.jobs --out paper/jobs --smoke 2      # the second set, after a mend: S2.jobs, S2-runs.tsv
    python -m experiment.jobs --out paper/jobs --smoke 3      # the third and the fourth, after another: S3.jobs, S4.jobs, ...

A run is named ``R-<number>``, the numbers dealt in an order drawn with a fixed seed. A role
sees the path of its run's directory, and such a name tells it nothing of the condition or of
the arrangement. ``R-runs.tsv`` says
what each run is: its lane, topic, backend, the backend's kind, arrangement, role model, seed.

The design: 8 conditions x 7 arrangements x 2 role models x 8 seeds (31 to 38), and 16
further seeds (39 to 54) for ``pair`` and ``compare`` in the three conditions whose backend is
capable, so that the difference of their real fixes is estimated more closely: 896 + 192 =
1,088 runs. No command carries a step
limit: the limits are experiment/config.py's."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# (topic, backend, the backend's kind): ign takes the request and does not use it, non has no
# input for it, cap is capable of it.
CONDITIONS = (
    ("F3", "esci-zip-ignore", "ign"), ("F3", "esci", "non"), ("F3", "esci-zip", "cap"),
    ("F8", "esci-sales-ignore", "ign"), ("F8", "esci-sales", "cap"),
    ("F9", "esci", "ign"),
    ("F10", "esci-stock-ignore", "ign"), ("F10", "esci-stock", "cap"),
)  # fmt: skip
ARRANGEMENTS = ("solo", "pair", "isolated", "monitor", "redteam", "compare", "sealed")
ROLE_MODELS = {"lu": "openai:gpt-5.6-luna", "ds": "deepinfra:deepseek-ai/DeepSeek-V4.1-Flash-1017"}
SEEDS = tuple(range(31, 39))
FURTHER_ARMS, FURTHER_SEEDS = ("pair", "compare"), tuple(range(39, 55))
ORDER_SEED = 20261114
WORLD_TREE = "runs/world-esci-e57457b"  # the backend checkout at e57457b, copied once for the experiment
FLAGS = (
    "--source failures --rounds 5 --countdown 5 --trials 2 --schema v2 --gold-rounds never --no-adjudication "
    "--end-when-resolved --subject-model claude-haiku-5-5 --subject-thinking low --judge-model openai:gpt-5.6-luna "
    "--service-results --f-eval-calls 2 --v-eval-calls 4 --cost-cap 4"
)
# Minutes a run is expected to take, for dealing the lanes evenly: the means of an earlier
# experiment (2026-11-12) by arrangement and role model, and three minutes for the spec session.
MINUTES = {
    ("solo", "lu"): 10, ("solo", "ds"): 12, ("pair", "lu"): 14, ("pair", "ds"): 16, ("isolated", "lu"): 18, ("isolated", "ds"): 25,
    ("monitor", "lu"): 20, ("monitor", "ds"): 29, ("redteam", "lu"): 21, ("redteam", "ds"): 24, ("compare", "lu"): 32, ("compare", "ds"): 33,
    ("sealed", "lu"): 24, ("sealed", "ds"): 27,
}  # fmt: skip
# The smoke runs: every arrangement, both role models, every topic, every kind of backend.
SMOKE = (
    ("F3", "esci-zip-ignore", "pair", "lu"), ("F10", "esci-stock", "pair", "ds"), ("F8", "esci-sales-ignore", "isolated", "lu"),
    ("F3", "esci-zip", "isolated", "ds"), ("F10", "esci-stock-ignore", "solo", "ds"), ("F10", "esci-stock-ignore", "monitor", "lu"),
    ("F3", "esci", "redteam", "ds"), ("F3", "esci-zip-ignore", "compare", "lu"), ("F8", "esci-sales", "compare", "ds"),
    ("F10", "esci-stock", "sealed", "lu"), ("F9", "esci", "sealed", "ds"), ("F9", "esci", "compare", "lu"),
)  # fmt: skip
# A second set, made after the first showed where the developer's spec has to live and that
# the oracle check took the backend from the default checkout.
SMOKE_2 = (
    ("F10", "esci-stock-ignore", "pair", "lu"), ("F3", "esci-zip-ignore", "isolated", "lu"),
    ("F10", "esci-stock", "compare", "ds"), ("F8", "esci-sales-ignore", "monitor", "ds"),
    ("F10", "esci-stock", "sealed", "lu"), ("F3", "esci-zip-ignore", "sealed", "ds"),
)  # fmt: skip
# A third, made after the first two showed the orchestrator telling the developer to leave the
# assistant alone in the round of its spec: it had taken the round's three sessions for three rounds.
SMOKE_3 = (
    ("F10", "esci-stock", "pair", "lu"), ("F3", "esci-zip-ignore", "pair", "ds"),
    ("F10", "esci-stock", "compare", "lu"), ("F8", "esci-sales-ignore", "compare", "ds"),
    ("F9", "esci", "sealed", "lu"), ("F8", "esci-sales", "redteam", "lu"),
    ("F3", "esci-zip", "monitor", "lu"), ("F10", "esci-stock-ignore", "isolated", "ds"),
)  # fmt: skip
# A fourth: in the third, two developers of eight still took their session for the change for
# the one for the spec, from guidance written for both; the orchestrator now guides the two apart.
SMOKE_4 = (
    ("F3", "esci-zip-ignore", "pair", "ds"), ("F8", "esci-sales", "redteam", "lu"),
    ("F8", "esci-sales-ignore", "pair", "lu"), ("F3", "esci-zip", "compare", "lu"),
    ("F10", "esci-stock", "compare", "ds"), ("F8", "esci-sales-ignore", "sealed", "ds"),
    ("F3", "esci", "monitor", "ds"), ("F9", "esci", "isolated", "lu"),
)  # fmt: skip
SMOKE_SETS = {1: (SMOKE, 1, "S"), 2: (SMOKE_2, 21, "S2"), 3: (SMOKE_3, 31, "S3"), 4: (SMOKE_4, 41, "S4")}  # the set, its first number, its files' stem
SMOKE_SEED = 20


def command(name: str, topic: str, world: str, arrangement: str, family: str, seed: int) -> str:
    return (
        f"experiment.run --name {name} --arrangement {arrangement} --seed {seed} --features {topic} --arrivals {topic}=1 "
        f"--world {world} --role-model {ROLE_MODELS[family]} {FLAGS} --world-tree {WORLD_TREE}"
    )


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
        run["name"] = f"R-{number:04d}"
    return runs


def deal(runs: list[dict[str, object]], lanes: int) -> list[list[dict[str, object]]]:
    """The runs over the lanes: the longest first, each to the lane with the least to do, then
    every lane's own order drawn anew, so that no condition sits at one end of the night."""
    dealt: list[list[dict[str, object]]] = [[] for _ in range(lanes)]
    load = [0] * lanes
    for run in sorted(runs, key=lambda r: (-MINUTES[(r["arrangement"], r["family"])], r["name"])):
        lane = load.index(min(load))
        dealt[lane].append(run)
        load[lane] += MINUTES[(run["arrangement"], run["family"])]
    rng = random.Random(ORDER_SEED + 1)
    for lane, own in enumerate(dealt, 1):
        rng.shuffle(own)
        for run in own:
            run["lane"] = lane
    return dealt


def table(runs: list[dict[str, object]]) -> str:
    columns = ("name", "lane", "topic", "world", "kind", "arrangement", "family", "seed")
    return "\n".join(["\t".join(columns), *("\t".join(str(run.get(c, "")) for c in columns) for run in sorted(runs, key=lambda r: str(r["name"])))]) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.jobs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=28)
    parser.add_argument("--smoke", type=int, choices=tuple(SMOKE_SETS), default=None, help="a set of smoke runs (seed 20, counted nowhere) instead of the experiment")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        kinds = {(topic, world): kind for topic, world, kind in CONDITIONS}
        chosen, first, stem = SMOKE_SETS[args.smoke]
        runs = [
            {"name": f"S-{n:02d}", "topic": topic, "world": world, "kind": kinds[(topic, world)], "arrangement": arrangement, "family": family, "seed": SMOKE_SEED}
            for n, (topic, world, arrangement, family) in enumerate(chosen, first)
        ]
        (args.out / f"{stem}.jobs").write_text("".join(command(r["name"], r["topic"], r["world"], r["arrangement"], r["family"], r["seed"]) + "\n" for r in runs), encoding="utf-8")
        (args.out / f"{stem}-runs.tsv").write_text(table(runs), encoding="utf-8")
        print(f"{len(runs)} smoke runs: {args.out / (stem + '.jobs')}")
        return 0
    runs = design()
    dealt = deal(runs, args.lanes)
    for lane, own in enumerate(dealt, 1):
        (args.out / f"R-{lane:02d}.jobs").write_text(
            "".join(command(r["name"], r["topic"], r["world"], r["arrangement"], r["family"], r["seed"]) + "\n" for r in own), encoding="utf-8"
        )
    (args.out / "R-runs.tsv").write_text(table(runs), encoding="utf-8")
    hours = [sum(MINUTES[(r["arrangement"], r["family"])] for r in own) / 60 for own in dealt]
    print(f"{len(runs)} runs over {args.lanes} lanes: {min(len(own) for own in dealt)} to {max(len(own) for own in dealt)} a lane, {min(hours):.1f} to {max(hours):.1f} expected hours a lane")
    return 0


if __name__ == "__main__":
    sys.exit(main())
