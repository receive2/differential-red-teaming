# SPDX-License-Identifier: Apache-2.0

"""Which WANDS queries, and so which products, make up the catalog.

The catalog is chosen by query, as the ESCI example's is: a seeded sample of the real search
queries that have enough judged products, with every product a human judged for each. The
queries and the product names are screened with the ESCI example's content rules. This module
is deterministic for a ``SelectionParams``.
"""

from __future__ import annotations

import random
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from esci_retail.pipeline.select import exclusion_reason

LABELS = ("Exact", "Partial", "Irrelevant")


@dataclass(frozen=True)
class SelectionParams:
    """How many queries, the seed, and the thresholds a query must meet to be drawn."""

    n_queries: int = 50
    seed: int = 20261114
    min_exact: int = 2
    min_irrelevant: int = 1
    min_judged: int = 8
    max_judged: int = 300

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def without_diacritics(text: str) -> str:
    """``text`` with accents and compatibility forms folded, so "décor" reads as "decor"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def unusable(text: str) -> str | None:
    """Why a query or a product name stays out of the catalog, or None when it may be used:
    the ESCI example's rules (empty, mostly non-ASCII, Spanish, excluded content) over the
    text with its diacritics folded, since WANDS writes "décor" in ordinary English names."""
    return exclusion_reason(without_diacritics(text))


def first_judgments(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The rows with one judgment per (query, product) pair, and how many were dropped.
    WANDS judges some pairs more than once, a few with two different labels; the first
    judgment in the file's order is kept."""
    seen: set[tuple[int, str]] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        pair = (int(row["query_id"]), str(row["product_id"]))
        if pair in seen:
            dropped += 1
            continue
        seen.add(pair)
        kept.append(
            {
                "query_id": pair[0],
                "query": str(row["query"]),
                "product_id": pair[1],
                "label": str(row["label"]),
            }
        )
    return kept, dropped


@dataclass
class Selection:
    params: SelectionParams
    queries: dict[int, str]  # query_id -> text
    judgments: list[dict[str, Any]]
    excluded: Counter[str] = field(default_factory=Counter)
    eligible: int = 0  # queries that met the thresholds before sampling

    @property
    def product_ids(self) -> set[str]:
        return {row["product_id"] for row in self.judgments}

    def by_query(self) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.judgments:
            grouped[row["query_id"]].append(row)
        return grouped

    def summary(self) -> dict[str, Any]:
        labels = Counter(row["label"] for row in self.judgments)
        return {
            "params": self.params.as_dict(),
            "eligible_queries": self.eligible,
            "counts": {
                "queries": len(self.queries),
                "judgments": len(self.judgments),
                "products": len(self.product_ids),
                "labels": {label: labels[label] for label in LABELS},
            },
            "excluded": dict(sorted(self.excluded.items())),
        }


def _meets_thresholds(rows: list[dict[str, Any]], params: SelectionParams) -> bool:
    counts = Counter(row["label"] for row in rows)
    return (
        counts["Exact"] >= params.min_exact
        and counts["Irrelevant"] >= params.min_irrelevant
        and params.min_judged <= len(rows) <= params.max_judged
    )


def select_queries(rows: Iterable[Mapping[str, Any]], params: SelectionParams) -> Selection:
    """Sample ``params.n_queries`` of the queries that are usable and meet the thresholds,
    with every judgment they carry. The same rows and params give the same selection."""
    judgments, repeated = first_judgments(rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    text: dict[int, str] = {}
    for row in judgments:
        grouped[row["query_id"]].append(row)
        text[row["query_id"]] = row["query"]

    excluded: Counter[str] = Counter()
    if repeated:
        excluded["judgment_repeated"] = repeated
    eligible: list[int] = []
    for query_id in sorted(grouped):
        reason = unusable(text[query_id])
        if reason is not None:
            excluded[f"query_{reason}"] += 1
        elif not _meets_thresholds(grouped[query_id], params):
            excluded["query_thresholds"] += 1
        else:
            eligible.append(query_id)

    drawn = random.Random(params.seed).sample(eligible, min(params.n_queries, len(eligible)))
    chosen = sorted(drawn)
    kept = [row for query_id in chosen for row in grouped[query_id]]
    return Selection(params, {q: text[q] for q in chosen}, kept, excluded, len(eligible))


def finalize(selection: Selection, products: Mapping[str, Mapping[str, Any]]) -> Selection:
    """Drop the products the product table lacks or whose name is unusable, then re-apply
    the thresholds, since a query may have lost the judgments that qualified it."""
    excluded = Counter(selection.excluded)
    usable: set[str] = set()
    for product_id in sorted(selection.product_ids):
        row = products.get(product_id)
        if row is None:
            excluded["product_missing"] += 1
            continue
        reason = unusable(str(row.get("product_name") or ""))
        if reason is not None:
            excluded[f"product_{reason}"] += 1
            continue
        usable.add(product_id)

    kept: list[dict[str, Any]] = []
    queries: dict[int, str] = {}
    for query_id, rows in sorted(selection.by_query().items()):
        rows = [row for row in rows if row["product_id"] in usable]
        if not _meets_thresholds(rows, selection.params):
            excluded["query_thresholds_after_products"] += 1
            continue
        kept.extend(rows)
        queries[query_id] = selection.queries[query_id]
    return Selection(selection.params, queries, kept, excluded, selection.eligible)
