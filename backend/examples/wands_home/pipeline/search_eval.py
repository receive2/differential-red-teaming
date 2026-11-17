# SPDX-License-Identifier: Apache-2.0

"""How well the backend's search surfaces what WANDS says it should. For each of the
catalog's queries: recall@k of the Exact products, recall@k of Exact or Partial, the share of
Irrelevant products in the top k, and the rank of the first Exact hit. Run after a build
and before the catalog is used in an experiment, so a weak search is tuned here rather
than measured later as agent behavior."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shopping_agent import ShoppingSessionContext, StorefrontBackend


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


async def evaluate(
    backend: StorefrontBackend, gold: list[dict[str, Any]], k: int = 8
) -> dict[str, Any]:
    session = ShoppingSessionContext(session_id="search-eval", user_id="demo-user")
    rows = []
    for entry in gold:
        results = await backend.search_products(session, entry["query"], limit=k)
        top = [product.product_id for product in results]
        exact, relevant = set(entry["E"]), set(entry["E"]) | set(entry["S"])
        irrelevant = set(entry["I"])
        first_exact = next((rank for rank, pid in enumerate(top, 1) if pid in exact), None)
        rows.append(
            {
                "query_id": entry["query_id"],
                "query": entry["query"],
                "returned": len(top),
                "recall_exact": len(set(top) & exact) / min(len(exact), k) if exact else None,
                "recall_relevant": (
                    len(set(top) & relevant) / min(len(relevant), k) if relevant else None
                ),
                "irrelevant_share": len(set(top) & irrelevant) / len(top) if top else 0.0,
                "first_exact_rank": first_exact,
            }
        )
    return {
        "k": k,
        "queries": len(rows),
        "empty_results": sum(row["returned"] == 0 for row in rows),
        "recall_exact": _mean(row["recall_exact"] for row in rows),
        "recall_relevant": _mean(row["recall_relevant"] for row in rows),
        "irrelevant_share": _mean(row["irrelevant_share"] for row in rows),
        "mrr_exact": _mean(
            1.0 / row["first_exact_rank"] if row["first_exact_rank"] else 0.0 for row in rows
        ),
        "worst": sorted(
            rows, key=lambda row: row["recall_exact"] if row["recall_exact"] is not None else 1.0
        )[:10],
    }


def run(data_dir: Path, k: int = 8) -> dict[str, Any]:
    from wands_home.api.wands_backend import WandsStorefront

    gold = json.loads((data_dir / "queries_gold.json").read_text(encoding="utf-8"))["queries"]
    return asyncio.run(evaluate(WandsStorefront(data_dir), gold, k))


def format_report(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:5.1f}%"

    lines = [
        f"search quality over {report['queries']} queries, k={report['k']}",
        f"  recall@k exact            {pct(report['recall_exact'])}",
        f"  recall@k exact+partial    {pct(report['recall_relevant'])}",
        f"  irrelevant share in top k {pct(report['irrelevant_share'])}",
        f"  MRR of first exact hit    {pct(report['mrr_exact'])}",
        f"  queries with no results   {report['empty_results']}",
        "  weakest queries (recall@k exact):",
    ]
    for row in report["worst"]:
        lines.append(
            f"    {pct(row['recall_exact'])}  {row['query']!r}  returned={row['returned']}"
        )
    return "\n".join(lines)
