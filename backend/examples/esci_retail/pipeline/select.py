# SPDX-License-Identifier: Apache-2.0

"""Which ESCI queries, and so which products, make up the store.

The catalog is chosen by query, not by product: a sample of real shopping queries from the
Shopping Queries Dataset together with every product a human judged for them, so each
query in the store has labelled Exact, Substitute, Complement and Irrelevant results. Rows
arrive as plain dicts (``extract`` reads them from parquet, the tests from JSONL fixtures),
so this module is standard library only and deterministic for a ``SelectionParams``.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

LABELS = ("E", "S", "C", "I")


@dataclass(frozen=True)
class SelectionParams:
    """One locale, the ``small`` (hard-query) or ``large`` version, how many queries, the
    seed, and the thresholds a query must meet to be drawn."""

    locale: str = "us"
    version: str = "small"
    n_queries: int = 300
    seed: int = 20261106
    min_exact: int = 2
    min_irrelevant: int = 1
    min_judged: int = 8
    max_judged: int = 40

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# The ESCI queries are real search logs. A general store's assistant is not evaluated on
# these: with them in, the agent's refusals would measure content policy, not the flows.
# Bare words that also name ordinary goods (gun, weed, lube, nude) are deliberately absent.
_EXCLUDED_TERMS = (
    "sex", "sexual", "dildo", "dildos", "vibrator", "vibrators", "vibrating", "strap on",
    "strap-on", "anal", "penis", "vagina", "bondage", "fetish", "condom", "condoms", "porn",
    "erotic", "masturbator", "masturbation", "cock ring", "xxx", "adult toy", "adult toys",
    "handgun", "handguns", "shotgun", "shotguns", "rifle", "rifles", "pistol", "pistols",
    "revolver", "ammo", "ammunition", "firearm", "firearms", "holster", "holsters",
    "crossbow", "taser", "stun gun", "pepper spray", "brass knuckles", "switchblade",
    "machete", "bb gun", "bb guns", "airsoft",
    "cigarette", "cigarettes", "cigar", "cigars", "tobacco", "vape", "vapes", "vaping",
    "e-cig", "ecig", "hookah", "bong", "bongs", "marijuana", "cannabis", "cbd", "thc",
    "kratom", "nicotine", "dab rig", "weed grinder", "rolling papers",
)  # fmt: skip
_EXCLUDED = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in _EXCLUDED_TERMS) + r")\b", re.IGNORECASE
)
# Two hits mark a query as not English; one hit alone is too often an English word.
_SPANISH_HINTS = frozenset(
    (
        "para", "con", "sin", "del", "de", "una", "unas", "unos", "los", "las", "el", "hombre",
        "hombres", "mujer", "mujeres", "niño", "niños", "niña", "niñas", "bebé", "bebe", "zapatos",
        "ropa", "juguetes", "cocina", "regalo", "regalos", "perro", "perros", "gato", "gatos",
        "grande", "pequeño", "negro", "blanco", "azul", "rojo", "y",
    )
)  # fmt: skip
_WORD = re.compile(r"[a-záéíóúñü]+")


def exclusion_reason(text: str) -> str | None:
    """Why a query or a listing title stays out of the store, or None when it may be used."""
    if not text or not text.strip():
        return "empty"
    if sum(ch.isascii() for ch in text) / len(text) < 0.9:
        return "non_ascii"
    if sum(word in _SPANISH_HINTS for word in _WORD.findall(text.lower())) >= 2:
        return "non_english"
    if _EXCLUDED.search(text):
        return "excluded_content"
    return None


def filter_rows(rows: Iterable[Mapping[str, Any]], params: SelectionParams) -> list[dict[str, Any]]:
    """The judgments of ``params.locale`` in the chosen version, reduced to the keys the
    pipeline reads. Rows that lack the locale or version columns are taken as pre-filtered."""
    flag = f"{params.version}_version"
    kept = []
    for row in rows:
        if row.get("product_locale", params.locale) != params.locale:
            continue
        if flag in row and int(row[flag] or 0) != 1:
            continue
        kept.append(
            {
                "query_id": int(row["query_id"]),
                "query": str(row["query"]),
                "product_id": str(row["product_id"]),
                "esci_label": str(row["esci_label"]),
                "split": str(row.get("split") or ""),
            }
        )
    return kept


@dataclass
class Selection:
    params: SelectionParams
    queries: dict[int, str]  # query_id -> text
    judgments: list[dict[str, Any]]  # query_id, query, product_id, esci_label, split
    excluded: Counter[str] = field(default_factory=Counter)
    eligible: int = 0  # queries that met the thresholds before sampling

    @property
    def product_ids(self) -> set[str]:
        return {row["product_id"] for row in self.judgments}

    def by_query(self) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.judgments:
            grouped[row["query_id"]].append(row)
        return dict(grouped)

    def gold(self) -> list[dict[str, Any]]:
        """One entry per query with the product ids under each label: the shape of
        ``queries_gold.json``."""
        entries = []
        for query_id, rows in sorted(self.by_query().items()):
            entry: dict[str, Any] = {"query_id": query_id, "query": self.queries[query_id]}
            for label in LABELS:
                entry[label] = sorted(r["product_id"] for r in rows if r["esci_label"] == label)
            entries.append(entry)
        return entries

    def summary(self) -> dict[str, Any]:
        labels = Counter(row["esci_label"] for row in self.judgments)
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
    counts = Counter(row["esci_label"] for row in rows)
    return (
        counts["E"] >= params.min_exact
        and counts["I"] >= params.min_irrelevant
        and params.min_judged <= len(rows) <= params.max_judged
    )


def select_queries(rows: Iterable[Mapping[str, Any]], params: SelectionParams) -> Selection:
    """Sample ``params.n_queries`` of the queries that are usable and meet the thresholds,
    with every judgment they carry. The same rows and params give the same selection."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    text: dict[int, str] = {}
    for row in filter_rows(rows, params):
        grouped[row["query_id"]].append(row)
        text[row["query_id"]] = row["query"]
    excluded: Counter[str] = Counter()
    eligible = []
    for query_id in sorted(grouped):
        reason = exclusion_reason(text[query_id])
        if reason is not None:
            excluded[f"query_{reason}"] += 1
        elif not _meets_thresholds(grouped[query_id], params):
            excluded["query_thresholds"] += 1
        else:
            eligible.append(query_id)
    drawn = random.Random(params.seed).sample(eligible, min(params.n_queries, len(eligible)))
    chosen = sorted(drawn)
    judgments = [row for query_id in chosen for row in grouped[query_id]]
    return Selection(params, {q: text[q] for q in chosen}, judgments, excluded, len(eligible))


def finalize(selection: Selection, products: Mapping[str, Mapping[str, Any]]) -> Selection:
    """Drop the listings the products table lacks or that are unusable (no title, a
    non-ASCII title, excluded content in the title or bullets), then re-apply the
    thresholds, since a query may have lost the judgments that qualified it."""
    excluded = Counter(selection.excluded)
    usable: set[str] = set()
    for product_id in sorted(selection.product_ids):
        row = products.get(product_id)
        if row is None:
            excluded["product_missing"] += 1
            continue
        reason = exclusion_reason(str(row.get("product_title") or ""))
        if reason is None and _EXCLUDED.search(str(row.get("product_bullet_point") or "")):
            reason = "excluded_content"
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
