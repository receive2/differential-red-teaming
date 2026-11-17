# SPDX-License-Identifier: Apache-2.0

"""The statistics of the paper's scripts: intervals of a share and of a difference of shares,
Fisher's exact test, the Cochran-Mantel-Haenszel test over strata, Holm's correction and an
exact upper bound. Standard library only."""

from __future__ import annotations

import math
from typing import Any


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """The difference of two shares (first minus second) with its 95% Newcombe interval."""
    if not n1 or not n2:
        return float("nan"), -1.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any]:
    """Fisher's exact test on the 2 by 2 table ``k1`` of ``n1`` against ``k2`` of ``n2``: the
    two-sided p-value (under the table's margins, the probability of every table as likely as
    the one seen or less), computed in integers, and the difference of shares (first minus
    second)."""
    total, events = n1 + n2, k1 + k2

    def ways(a: int) -> int:
        return math.comb(n1, a) * math.comb(n2, events - a)

    seen = ways(k1)
    as_likely_or_less = sum(w for a in range(max(0, events - n2), min(n1, events) + 1) if (w := ways(a)) <= seen)
    p = min(1.0, as_likely_or_less / math.comb(total, events)) if total else 1.0
    difference = (k1 / n1 if n1 else float("nan")) - (k2 / n2 if n2 else float("nan"))
    return {"first": [k1, n1], "second": [k2, n2], "difference": difference, "p": p}


def cmh(strata: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    """Cochran-Mantel-Haenszel over 2 by 2 strata ``(events1, n1, events2, n2)``: the
    statistic with the continuity correction, its two-sided p-value (one degree of freedom),
    and the Mantel-Haenszel difference of shares (first minus second). Strata that hold one
    arm only, or in which every run or no run has the event, say nothing and carry no
    weight in the statistic."""
    observed = expected = variance = 0.0
    num = den = 0.0
    used = 0
    for a, n1, c, n2 in strata:
        total = n1 + n2
        if n1 == 0 or n2 == 0:
            continue
        num += (a * n2 - c * n1) / total
        den += n1 * n2 / total
        m1, m2 = a + c, total - a - c
        if total < 2 or m1 == 0 or m2 == 0:
            continue
        used += 1
        observed += a
        expected += n1 * m1 / total
        variance += n1 * n2 * m1 * m2 / (total * total * (total - 1))
    if variance == 0:
        return {"statistic": 0.0, "p": 1.0, "difference": num / den if den else float("nan"), "strata": used}
    statistic = max(0.0, abs(observed - expected) - 0.5) ** 2 / variance
    return {"statistic": round(statistic, 3), "p": math.erfc(math.sqrt(statistic / 2)), "difference": num / den, "strata": used}


def holm(ps: dict[str, float]) -> dict[str, float]:
    ordered = sorted(ps, key=lambda k: ps[k])
    out, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * ps[key]))
        out[key] = running
    return out


def upper_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """The exact one-sided upper bound of a share when k of n are seen (bisection on the binomial tail)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return hi
