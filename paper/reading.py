# SPDX-License-Identifier: Apache-2.0

"""The reading of an experiment: one row a run, and the tables of how the runs end.

    python -m experiment.outcomes loops <runs> --out runs/tables/r1-<k> --cost-cap 15     (in shards; paper/pipeline.sh)
    python paper/reading.py runs/tables/r1-[0-9].json --json results/esci-reading.json > results/esci-reading.txt
    python paper/reading.py runs/tables/w1-[0-9].json --json results/wands-reading.json > results/wands-reading.txt
    python paper/reading.py results/labels/esci-[0-9].json --judge 2 --json results/esci-reading-second-judge.json

Per run: its ending (``paper/endings.py``) from the assistant build at its decision gate. A
condition is a topic in one backend; the backend is ``ign`` (its service takes the request
and does not use it), ``non`` (its service has no input for it: the ZIP topic in ``esci``,
the date topic in ``wands``) or ``cap`` (capable of it). The failures of a batch cannot be
repaired in ``ign`` and ``non``; they can in ``cap``.

Printed, and written with ``--json``: the endings by condition, arrangement and role-model
family; the share of runs that deliver a facade where the failures cannot be repaired and
the share that deliver the real fix where they can, with 95% Wilson intervals; the capable
conditions over the seeds every arrangement has (31 to 38), so that the seven are compared on
equal numbers; and, where the backend is capable, the share of runs that deliver a denial (a
build labelled honest: ``truth shipped``). The comparisons of differential red teaming with
each other arrangement are ``paper/against_each.py``'s, from the file written here.

A build's label is the first judge's. ``--judge 2`` reads the same runs by the second
judge's labels, as a check of how much the reading rests on one judge."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endings import ENDINGS, read  # noqa: E402
from stats import upper_bound, wilson  # noqa: E402

ARRANGEMENTS = ("solo", "pair", "isolated", "monitor", "redteam", "compare", "sealed")
CAPABLE = ("esci-zip", "esci-sales", "esci-onsale", "esci-stock", "wands-fit", "wands-date", "wands-new", "wands-stock")
ABSENT = (("F3", "esci"), ("F22", "wands"))  # the topics whose service has no input for the request
SHARED_SEEDS = frozenset(range(31, 39))  # the seeds every arrangement has in every condition


def backend_of(feature: str, world: str) -> str:
    if world in CAPABLE:
        return "cap"
    return "non" if (feature, world) in ABSENT else "ign"


def family_of(arm: str) -> str:
    """The role-model family from the arm's name (experiment/outcomes.py, arm_of)."""
    return "ds" if "+roles-" in arm else "lu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--judge", type=int, choices=(1, 2), default=1, help="whose labels decide a build's label: the first judge's (the paper's reading) or the second's (a check)")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.tables:
        table = json.loads(path.read_text(encoding="utf-8"))
        arms = {r["run"]: r.get("arm", "") for r in table["runs"]}
        for row in read(table, args.judge - 1):
            if row["arrangement"] not in ARRANGEMENTS:
                continue
            row["family"], row["store"] = family_of(arms[row["run"]]), backend_of(row["feature"], row["world"])
            row["condition"] = f"{row['feature']}-{row['store']}"
            rows.append(row)
    counted = [r for r in rows if not r["excluded"]]
    if args.judge == 2:
        print("The builds' labels are the second judge's (a check; the paper's reading is the first judge's).")
    print(f"{len(rows)} runs read, {len(rows) - len(counted)} excluded; by family {dict(Counter(r['family'] for r in counted))}")
    for r in rows:
        if r["excluded"]:
            print(f"  EXCLUDED {r['run']}: {r['excluded']}")
    unresolved = [r["run"] for r in counted if r["ending"] == "unresolved"]
    if unresolved:
        print(f"  unresolved ({len(unresolved)}): {unresolved[:12]}")

    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in counted:
        cells[(r["condition"], r["arrangement"], r["family"])].append(r)
    conditions = sorted({r["condition"] for r in counted}, key=lambda c: (c.split("-")[1] != "ign", c.split("-")[1] != "non", c))

    def tally(group: list[dict[str, Any]]) -> str:
        counts = Counter(r["ending"] for r in group)
        return " | ".join(str(counts.get(e, 0)) for e in ENDINGS)

    out_cells = []
    for family in ("lu", "ds", "both"):
        print(f"\nRole models: {family}\n| Condition | Arrangement | Runs | " + " | ".join(ENDINGS) + " | Releases held |")
        print("|---|---|---|" + "---|" * len(ENDINGS) + "---|")
        for condition in conditions:
            for arrangement in ARRANGEMENTS:
                group = [r for f in (("lu", "ds") if family == "both" else (family,)) for r in cells.get((condition, arrangement, f), [])]
                if not group:
                    continue
                print(f"| {condition} | {arrangement} | {len(group)} | {tally(group)} | {sum(r['held'] for r in group)} |")
                if family != "both":
                    counts = Counter(r["ending"] for r in group)
                    out_cells.append({"condition": condition, "arrangement": arrangement, "family": family, "n": len(group), **{e: counts.get(e, 0) for e in ENDINGS}})

    def share(arrangement: str, backends: tuple[str, ...], ending: str, families: tuple[str, ...] = ("lu", "ds"), only: str | None = None) -> tuple[int, int]:
        group = [r for r in counted if r["arrangement"] == arrangement and r["store"] in backends and r["family"] in families and (only is None or r["condition"] == only)]
        return sum(1 for r in group if r["ending"] == ending), len(group)

    print("\nShares, both role-model families together (95% Wilson interval):")
    summary: dict[str, Any] = {}
    for label, backends, ending in (("false claim shipped where the failures cannot be put right (ign, non)", ("ign", "non"), "false claim shipped"),
                                  ("false claim shipped in the stores with no input (non)", ("non",), "false claim shipped"),
                                  ("real fix shipped where they can (cap)", ("cap",), "real shipped")):  # fmt: skip
        print(f"  {label}:")
        for arrangement in ARRANGEMENTS:
            k, n = share(arrangement, backends, ending)
            lo, hi = wilson(k, n)
            by_family = "; ".join(f"{f} {share(arrangement, backends, ending, (f,))[0]}/{share(arrangement, backends, ending, (f,))[1]}" for f in ("lu", "ds"))
            print(f"    {arrangement:<9} {k}/{n} = {k / n if n else float('nan'):.2f} [{lo:.2f}, {hi:.2f}]  ({by_family})")
            summary.setdefault(label, {})[arrangement] = {"k": k, "n": n, "interval": [round(lo, 3), round(hi, 3)]}

    confirmed_in_capable = [r["run"] for r in counted if r["arrangement"] == "compare" and r["store"] == "cap" and any(x.get("by") == "compare" and (x.get("comparisons_standing") or 0) > 0 for x in r["releases"])]
    n_cap = len([r for r in counted if r["arrangement"] == "compare" and r["store"] == "cap"])
    print(f"\nRuns in the capable stores in which a comparison stood: {len(confirmed_in_capable)} of {n_cap} (one-sided 95% upper bound {upper_bound(len(confirmed_in_capable), n_cap):.3f}) {confirmed_in_capable[:10]}")

    print("\nWhere the failures can be put right (cap):")
    print("  over seeds 31 to 38 alone, every arrangement on equal numbers (real fix shipped; a denial shipped):")
    equal: dict[str, Any] = {}
    for arrangement in ARRANGEMENTS:
        group = [r for r in counted if r["arrangement"] == arrangement and r["store"] == "cap" and r["seed"] in SHARED_SEEDS]
        n = len(group)
        real, denial = (sum(1 for r in group if r["ending"] == e) for e in ("real shipped", "truth shipped"))
        (rlo, rhi), (dlo, dhi) = wilson(real, n), wilson(denial, n)
        print(f"    {arrangement:<9} real {real}/{n} = {real / n if n else float('nan'):.2f} [{rlo:.2f}, {rhi:.2f}]; denial {denial}/{n} = {denial / n if n else float('nan'):.2f} [{dlo:.2f}, {dhi:.2f}]")
        equal[arrangement] = {"n": n, "real": real, "denial": denial}
    print("  a denial shipped, over all of an arrangement's runs there:")
    denials: dict[str, Any] = {}
    for arrangement in ARRANGEMENTS:
        k, n = share(arrangement, ("cap",), "truth shipped")
        lo, hi = wilson(k, n)
        by_family = "; ".join(f"{f} {share(arrangement, ('cap',), 'truth shipped', (f,))[0]}/{share(arrangement, ('cap',), 'truth shipped', (f,))[1]}" for f in ("lu", "ds"))
        print(f"    {arrangement:<9} {k}/{n} = {k / n if n else float('nan'):.2f} [{lo:.2f}, {hi:.2f}]  ({by_family})")
        denials[arrangement] = {"k": k, "n": n, "interval": [round(lo, 3), round(hi, 3)]}

    flags = Counter(flag for r in counted for flag in r["flags"])
    print(f"\nFlags: {dict(flags)}")
    print(f"Spent on the counted runs: ${sum(r['cost'] for r in counted):.2f} (roles and evaluations; the labels' probes and judges are beside it)")
    if args.json:
        args.json.write_text(json.dumps({**({"judge": "second"} if args.judge == 2 else {}), "runs": rows, "cells": out_cells, "summary": summary, "capable_shared_seeds": equal, "capable_denials": denials}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
