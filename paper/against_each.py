# SPDX-License-Identifier: Apache-2.0

"""Differential red teaming against each of the other architectures, from a reading.

    python paper/against_each.py results/esci-reading.json [--json results/esci-against-each.json]
    python paper/against_each.py results/esci-reading.json results/wands-reading.json     # both backends together

The readings are the files ``paper/reading.py`` writes with ``--json``; with more than one,
the backends are taken together (the strata are backend by condition by family). Two things
are printed, with the statistics of ``paper/stats.py``:

    false claims   where the failures cannot be put right, the share of runs that ship a
                   false claim under ``compare`` against each other arrangement: the
                   Fisher exact test on the pooled counts (two-sided), Holm over the
                   five; beside it the same comparison stratified by backend, condition
                   and family (Cochran-Mantel-Haenszel, two-sided, continuity correction)
    real fixes     where they can, the share of runs that ship the real fix under ``compare``
                   against each other arrangement: the difference and its 95% Newcombe
                   interval, over every run and over the seeds every arrangement has

``sealed`` is the reference: its false claims are shown without a test. A share is printed
as the tables of the paper print it: one decimal, rounded half up (41/80 is 51.3%)."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reading import SHARED_SEEDS  # noqa: E402
from stats import cmh, fisher, holm, newcombe  # noqa: E402

OTHERS = ("solo", "pair", "isolated", "monitor", "redteam")


def share(k: int, n: int) -> str:
    """100 x k / n with one decimal, rounded half up (paper/endings_table.py does the same)."""
    return f"{(Decimal(100 * k) / Decimal(n)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python paper/against_each.py")
    parser.add_argument(
        "readings", nargs="*", type=Path, default=[Path("results/esci-reading.json")]
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = [
        {**run, "backend": backend}
        for backend, path in enumerate(args.readings, 1)
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]
        if not run.get("excluded")
    ]
    out: dict[str, dict] = {
        "readings": [str(p) for p in args.readings],
        "false claims": {},
        "real fixes": {},
        "real fixes, shared seeds": {},
    }

    def count(
        arrangement: str, capable: bool, ending: str, where=lambda run: True
    ) -> tuple[int, int]:
        got = [
            run
            for run in runs
            if run["arrangement"] == arrangement
            and (run["store"] == "cap") == capable
            and where(run)
        ]
        return sum(run["ending"] == ending for run in got), len(got)

    strata_keys = sorted(
        {(run["backend"], run["condition"], run["family"]) for run in runs if run["store"] != "cap"}
    )

    def strata(first: str, second: str) -> list[tuple[int, int, int, int]]:
        return [
            (
                *count(
                    first,
                    False,
                    "false claim shipped",
                    lambda run, key=key: (run["backend"], run["condition"], run["family"]) == key,
                ),
                *count(
                    second,
                    False,
                    "false claim shipped",
                    lambda run, key=key: (run["backend"], run["condition"], run["family"]) == key,
                ),
            )
            for key in strata_keys
        ]

    print("Readings: " + ", ".join(str(p) for p in args.readings))
    print(
        "\nFalse claim shipped where the failures cannot be put right: compare against each (Fisher exact on the pooled counts, Holm over the five; the same comparison stratified by condition and family beside it)"
    )
    k0, n0 = count("compare", False, "false claim shipped")
    tests = {
        other: {
            **fisher(k0, n0, *count(other, False, "false claim shipped")),
            "stratified": cmh(strata("compare", other)),
        }
        for other in OTHERS
    }
    adjusted = holm({other: t["p"] for other, t in tests.items()})
    for other in OTHERS:
        k, n = count(other, False, "false claim shipped")
        t = tests[other]
        print(
            f"  compare {k0}/{n0} ({share(k0, n0)}) against {other:<8} {k}/{n} ({share(k, n)}): Fisher p = {t['p']:.2g}, Holm-corrected p = {adjusted[other]:.2g}; stratified CMH {t['stratified']['statistic']}, p = {t['stratified']['p']:.2g}"
        )
        out["false claims"][other] = {
            "compare": [k0, n0],
            "other": [k, n],
            "p": t["p"],
            "holm": adjusted[other],
            "stratified": t["stratified"],
        }
    k, n = count("sealed", False, "false claim shipped")
    print(f"  sealed (reference) {k}/{n} ({share(k, n)})")
    for title, key, where in (
        ("every run", "real fixes", lambda run: True),
        (
            "the seeds every arrangement has",
            "real fixes, shared seeds",
            lambda run: run["seed"] in SHARED_SEEDS,
        ),
    ):
        print(
            f"\nReal fix shipped where the failures can be put right, {title}: compare against each (difference, 95% Newcombe interval)"
        )
        k0, n0 = count("compare", True, "real shipped", where)
        for other in (*OTHERS, "sealed"):
            k, n = count(other, True, "real shipped", where)
            d, lo, hi = newcombe(k0, n0, k, n)
            print(
                f"  compare {k0}/{n0} ({share(k0, n0)}) against {other:<8} {k}/{n} ({share(k, n)}): difference {d:+.3f}, interval [{lo:+.3f}, {hi:+.3f}]"
            )
            out[key][other] = {
                "compare": [k0, n0],
                "other": [k, n],
                "difference": d,
                "interval": [lo, hi],
            }
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
