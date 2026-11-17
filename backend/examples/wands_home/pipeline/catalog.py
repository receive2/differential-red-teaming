# SPDX-License-Identifier: Apache-2.0

"""Stage two: the raw WANDS subset to the backend's catalog. Standard library only.

WANDS gives a product's name, its class, its path in the category tree, a description, a list
of features and its real rating and review count. A record takes these cleaned (``text.py``)
and otherwise as they are: the class is the category, three features are the attributes a
search filters on (color, material, style), and a fixed list of features makes the
specifications. What WANDS lacks is synthesized: the price, from a band for the product's
class or department, and the stock, both from a hash of the seed and the product id, so a
product's figures never depend on which other products were drawn.

Two features stay out of every record: a product's overall width and its weight (WANDS states
no unit; the source lists inches and pounds). They go into columns of their own, which the
backend reads and does not serve; the specifications leave out every value that states a
dimension, and a name or a description loses the statements of a length it makes
(``text.py``). The same columns hold the words of every class and category a product is
filed under, for the category filter.

The outputs are ``catalog.sqlite`` (one JSON record per product plus two FTS5 indexes: stemmed
words for ranking, trigrams for misspellings), ``queries_gold.json`` (the WANDS labels of the
catalog's queries under the letters the ESCI example's gold file uses), ``orders.json`` (demo
order history over in-stock products) and ``manifest.json``. A build is a function of the raw
subset, the seed and the anchor date: no output holds the time it was built, so a second
build of the same subset gives the same tracked files.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from .extract import read_jsonl
from .select import LABELS, without_diacritics
from .text import (
    capitalized,
    clean_description,
    clean_name,
    first_sentence,
    repair,
    shorten,
    title_case,
)

GENERAL = "home-goods"
CURRENCY = "USD"
SEED = 20261114
# The day the demo orders are dated from, and the day of the session clock the harness that
# runs the assistant over this catalog sets: a Monday.
ANCHOR = date(2026, 11, 16)
# From this weight in pounds a product ships as freight (``api/wands_backend.py``).
FREIGHT_FROM_LB = 50.0

# The letter each WANDS label is filed under in the gold file and the judgments table.
GOLD_LETTERS = dict(zip(LABELS, ("E", "S", "I"), strict=True))
GOLD_KEYS = ("E", "S", "C", "I")  # the ESCI example's four; WANDS has no complement label

# Median price in dollars and the spread of its logarithm, by the slug of a product's class.
CLASS_BANDS: dict[str, tuple[float, float]] = {
    "accent-chairs": (260, 0.5),
    "accent-chests-cabinets": (280, 0.5),
    "accent-pillows": (35, 0.45),
    "adjustable-beds": (1100, 0.45),
    "area-rugs": (180, 0.7),
    "armoires-wardrobes": (430, 0.5),
    "bedding-sets": (95, 0.5),
    "beds": (480, 0.5),
    "benches": (190, 0.5),
    "bike-and-sport-racks": (90, 0.6),
    "chandeliers": (280, 0.6),
    "coffee-cocktail-tables": (260, 0.5),
    "cooking-utensils": (22, 0.5),
    "curtain-hardware-accessories": (35, 0.5),
    "curtains-drapes": (40, 0.5),
    "daybeds-guest-beds": (420, 0.5),
    "deck-boxes": (160, 0.5),
    "decorative-objects": (45, 0.6),
    "desks": (320, 0.55),
    "dining-chairs": (190, 0.5),
    "dining-table-sets": (560, 0.5),
    "dining-tables": (480, 0.5),
    "dressers-chests": (430, 0.5),
    "end-tables": (140, 0.5),
    "fencing-accessories": (120, 0.6),
    "flush-mount-lighting": (120, 0.55),
    "folding-tables": (110, 0.5),
    "food-storage-dispensers": (30, 0.5),
    "fountains": (150, 0.7),
    "headboards": (220, 0.5),
    "interior-doors": (330, 0.5),
    "kids-beds": (520, 0.5),
    "landscape-lighting": (45, 0.6),
    "molding-millwork": (30, 0.6),
    "mugs-teacups": (24, 0.45),
    "office-chairs": (210, 0.5),
    "ottomans": (150, 0.55),
    "outdoor-conversation-sets": (980, 0.5),
    "outdoor-wall-lights": (85, 0.55),
    "patio-dining-sets": (650, 0.55),
    "patio-sofas": (900, 0.5),
    "patio-tables": (190, 0.55),
    "range-hoods": (380, 0.55),
    "refrigerators": (1600, 0.5),
    "sectionals": (1400, 0.45),
    "shelving-racks": (110, 0.55),
    "shoe-storage": (95, 0.5),
    "shower-faucets-systems": (260, 0.6),
    "shower-heads": (70, 0.6),
    "sideboards-buffets": (520, 0.5),
    "slow-cookers": (60, 0.5),
    "sofa-console-tables": (230, 0.5),
    "vanity-bases": (600, 0.5),
    "wall-art": (70, 0.6),
    "wall-paneling": (75, 0.6),
    "wall-sconces": (95, 0.55),
}
# For a class without a band of its own: by the slug of the department, the first step of
# the product's category path.
DEPARTMENT_BANDS: dict[str, tuple[float, float]] = {
    "appliances": (420, 0.8),
    "baby-kids": (160, 0.8),
    "bed-bath": (80, 0.7),
    "commercial-business-furniture": (320, 0.7),
    "decor-pillows": (55, 0.7),
    "furniture": (300, 0.7),
    "holiday-decor": (40, 0.6),
    "home-improvement": (120, 0.8),
    "kitchen-tabletop": (40, 0.7),
    "lighting": (110, 0.7),
    "outdoor": (240, 0.8),
    "rugs": (170, 0.7),
    "school-furniture-and-supplies": (130, 0.7),
    "storage-organization": (85, 0.7),
}
GENERAL_BAND = (90.0, 0.9)
PRICE_CAP = 5999.0

# The attributes of a record and the features each is read from, the first that has a value.
ATTRIBUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("color", ("color", "primarycolor")),
    ("material", ("material", "primarymaterial", "framematerial")),
    ("style", ("style", "dsprimaryproductstyle")),
)
# The specifications in the order a record lists them: the name, the features it is read from
# (the first that has a value), and for a feature that is a number the unit written after it.
# A product with two values for a number has none to show: the source describes several sizes.
SPECS: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("Product Type", ("producttype",), None),
    ("Color", ("color", "primarycolor"), None),
    ("Material", ("material", "primarymaterial"), None),
    ("Frame Material", ("framematerial",), None),
    ("Top Material", ("topmaterial",), None),
    ("Upholstery Material", ("upholsterymaterial",), None),
    ("Finish", ("finish",), None),
    ("Style", ("style", "dsprimaryproductstyle"), None),
    ("Shape", ("shape",), None),
    ("Pattern", ("pattern",), None),
    ("Number of Drawers", ("numberofdrawers",), ""),
    ("Number of Shelves", ("numberofshelves",), ""),
    ("Number of Doors", ("numberofdoors",), ""),
    ("Number of Lights", ("numberoflights",), ""),
    ("Seating Capacity", ("seatingcapacity",), ""),
    ("Weight Capacity", ("weightcapacity",), " lb"),
    ("Mattress Size", ("mattresssize",), None),
    ("Capacity", ("capacityquarts",), " qt"),
    ("Power Source", ("powersource",), None),
    ("Bulb Type", ("bulbtype",), None),
    ("Location", ("location",), None),
    ("Assembly Required", ("adultassemblyrequired",), None),
    ("Level of Assembly", ("levelofassembly",), None),
    ("Country of Origin", ("countryoforigin",), None),
    ("Warranty Length", ("warrantylength",), None),
    ("Pieces Included", ("piecesincluded",), None),
)
MAX_SPECS = 12
# Specifications whose values are names or initialisms ("United States", "LED").
TITLED_SPECS = frozenset({"Bulb Type", "Country of Origin"})
WIDTH_FEATURE = "overallwidth-sidetoside"
WEIGHT_FEATURE = "overallproductweight"

# Values that say nothing; "none" says something of the level of assembly alone.
_SAYS_NOTHING = frozenset({"does not apply", "n/a", "na", "not applicable", "unknown", "other"})
_NONE_MEANS = frozenset({"levelofassembly"})
_NOT_SLUG = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")
# A length: a number with an inch or foot mark, a unit of length, or the side it measures
# ("32 w", "18 deep"), and two numbers with an "x" between them.
_DIMENSION = re.compile(
    r"\d\s*(?:\"|'(?![a-z])|-?\s*(?:in|inch|inches|cm|mm|ft|foot|feet)\b"
    r"|\s*(?:w|h|d|l|wide|width|high|height|deep|depth|long|length|tall)\b)"
    r"|\d\s*x\s*\d",
    re.IGNORECASE,
)


def slug(name: str) -> str:
    """A class or department name as a slug: "Accent Chests / Cabinets" is
    ``accent-chests-cabinets``, and an accented letter is its plain one."""
    return _NOT_SLUG.sub("-", without_diacritics(name).lower()).strip("-")


def states_dimension(value: str) -> bool:
    return _DIMENSION.search(value) is not None


def _number(value: str) -> float | None:
    return float(value) if _NUMBER.fullmatch(value) else None


def _plain(number: float) -> str:
    return str(int(number)) if number == int(number) else f"{number:g}"


_repaired = lru_cache(maxsize=None)(repair)  # a feature's value recurs across the catalog


def features_of(raw: str | None) -> dict[str, tuple[str, ...]]:
    """A product's features by key, each with its distinct values in the source's order,
    repaired and still lowercase. WANDS writes ``key:value|key:value`` and repeats a key for
    each of its values."""
    found: dict[str, dict[str, None]] = defaultdict(dict)
    for part in (raw or "").split("|"):
        key, colon, value = part.partition(":")
        key, value = key.strip().lower(), _repaired(value.strip())
        if colon and key and value:
            found[key][value] = None
    return {key: tuple(values) for key, values in found.items()}


@dataclass(frozen=True)
class Listing:
    """One raw WANDS product, read once: its name and description cleaned, the classes and
    the category path it is filed under, and its features."""

    product_id: str
    title: str
    description: str
    classes: tuple[str, ...]
    path: tuple[str, ...]
    features: dict[str, tuple[str, ...]]
    rating: float | None
    review_count: int | None


def _figure(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def read_listing(raw: Mapping[str, Any]) -> Listing:
    rating, reviews = _figure(raw.get("average_rating")), _figure(raw.get("review_count"))
    return Listing(
        product_id=str(raw["product_id"]),
        title=clean_name(raw.get("product_name")),
        description=clean_description(raw.get("product_description")),
        classes=tuple(
            name.strip() for name in str(raw.get("product_class") or "").split("|") if name.strip()
        ),
        path=tuple(
            step.strip()
            for step in str(raw.get("category hierarchy") or "").split("/")
            if step.strip()
        ),
        features=features_of(raw.get("product_features")),
        rating=None if rating is None else round(min(5.0, max(0.0, rating)), 1),
        review_count=None if reviews is None else int(reviews),
    )


def category_of(listing: Listing) -> str:
    """The slug of the product's first class; of the last step of its category path when
    WANDS gives it no class."""
    for name in (*listing.classes[:1], *listing.path[-1:]):
        if found := slug(name):
            return found
    return GENERAL


def department_of(listing: Listing) -> str | None:
    return slug(listing.path[0]) if listing.path else None


def category_words(listing: Listing) -> str:
    """The words of every class and every step of the category path the product is filed
    under, each once: what a category filter is matched against besides the slug."""
    text = without_diacritics(" ".join((*listing.classes, *listing.path))).lower()
    return " ".join(dict.fromkeys(_WORD.findall(text)))


def _told(features: Mapping[str, Sequence[str]], keys: Iterable[str]) -> tuple[str, ...]:
    """The values of the first of ``keys`` that has any that say something and state no
    dimension."""
    for key in keys:
        values = tuple(
            value
            for value in features.get(key, ())
            if value not in _SAYS_NOTHING
            and (value != "none" or key in _NONE_MEANS)
            and not states_dimension(value)
        )
        if values:
            return values
    return ()


def attribute_value(values: Sequence[str], most: int = 3, room: int = 60) -> str:
    """The first of ``values`` and as many of the next as fit, up to ``most`` in ``room``
    characters, separated by commas."""
    kept = [shorten(values[0], room, ellipsis="")]
    for value in values[1:most]:
        if len(", ".join([*kept, value])) > room:
            break
        kept.append(value)
    return ", ".join(kept)


def attributes_of(features: Mapping[str, Sequence[str]]) -> dict[str, str]:
    attributes = {}
    for name, keys in ATTRIBUTES:
        if values := _told(features, keys):
            attributes[name] = attribute_value(values)
    return attributes


def specs_of(features: Mapping[str, Sequence[str]]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for name, keys, unit in SPECS:
        if len(specs) == MAX_SPECS:
            break
        values = _told(features, keys)
        if not values:
            continue
        if unit is None:
            value = shorten(", ".join(values), 120)
            specs[name] = title_case(value) if name in TITLED_SPECS else capitalized(value)
        elif len(values) == 1 and (number := _number(values[0])) is not None:
            specs[name] = _plain(number) + unit
    return specs


def width_of(features: Mapping[str, Sequence[str]]) -> float | None:
    """The product's overall width, side to side, when the source states exactly one."""
    widths = {number for value in features.get(WIDTH_FEATURE, ()) if (number := _number(value))}
    return widths.pop() if len(widths) == 1 else None


def weight_of(features: Mapping[str, Sequence[str]]) -> float | None:
    """The product's weight: the largest the source states."""
    weights = [number for value in features.get(WEIGHT_FEATURE, ()) if (number := _number(value))]
    return max(weights, default=None)


def _unit(seed: int, product_id: str, name: str) -> float:
    """A uniform draw in (0, 1) fixed by the seed, the product and the figure's name."""
    digest = hashlib.sha256(f"{seed}|{name}|{product_id}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


def _normal(seed: int, product_id: str, name: str) -> float:
    u1, u2 = _unit(seed, product_id, f"{name}:1"), _unit(seed, product_id, f"{name}:2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def synthesize(seed: int, product_id: str, category: str, department: str | None) -> dict[str, Any]:
    """Price and stock for one product, from the seed and its id alone: the same product
    gets the same figures in any build with the same seed."""
    median, sigma = (
        CLASS_BANDS.get(category) or DEPARTMENT_BANDS.get(department or "") or GENERAL_BAND
    )
    price = median * math.exp(sigma * _normal(seed, product_id, "price"))
    price = min(PRICE_CAP, max(1.99, price))
    return {
        "price": math.floor(price) + 0.99 if price >= 5 else round(price, 2),
        "in_stock": _unit(seed, product_id, "stock") < 0.92,
    }


def build_record(listing: Listing, seed: int) -> dict[str, Any]:
    """A ``ProductDetails``-shaped record from one WANDS product."""
    category = category_of(listing)
    figures = synthesize(seed, listing.product_id, category, department_of(listing))
    description = listing.description
    return {
        "product_id": listing.product_id,
        "title": listing.title,
        "brand": None,
        "price": figures["price"],
        "currency": CURRENCY,
        "rating": listing.rating,
        "review_count": listing.review_count,
        "category": category,
        "labels": [],
        "attributes": attributes_of(listing.features),
        "in_stock": figures["in_stock"],
        "short_description": (
            shorten(first_sentence(description), 160)
            if description
            else shorten(listing.title, 160, ellipsis="")
        ),
        "long_description": shorten(description, 800) or None,
        "specs": specs_of(listing.features),
    }


def hidden_of(listing: Listing) -> dict[str, Any]:
    """What the catalog keeps of a product outside its record."""
    return {
        "category_words": category_words(listing),
        "width_in": width_of(listing.features),
        "weight_lb": weight_of(listing.features),
    }


def index_text(listing: Listing, record: Mapping[str, Any]) -> tuple[str, ...]:
    """The row of ``products_fts``: id, title, the classes and categories, the values of the
    attributes and specifications, the description."""
    shown = dict.fromkeys((*record["attributes"].values(), *record["specs"].values()))
    return (
        record["product_id"],
        record["title"],
        " ".join((*listing.classes, *listing.path)),
        " ".join(shown),
        listing.description[:2000],
    )


def gold_entries(judgments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per query with the product ids under each label's letter."""
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[int(row["query_id"])].append(row)
    entries = []
    for query_id, rows in sorted(grouped.items()):
        entry: dict[str, Any] = {"query_id": query_id, "query": rows[0]["query"]}
        for key in GOLD_KEYS:
            entry[key] = sorted(r["product_id"] for r in rows if GOLD_LETTERS[r["label"]] == key)
        entries.append(entry)
    return entries


SCHEMA = """
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    record TEXT NOT NULL,
    category TEXT NOT NULL,
    category_words TEXT NOT NULL,
    color TEXT,
    price REAL NOT NULL,
    rating REAL,
    in_stock INTEGER NOT NULL,
    width_in REAL,
    weight_lb REAL
);
CREATE VIRTUAL TABLE products_fts USING fts5(
    product_id UNINDEXED, title, category, features, description,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE products_trigram USING fts5(
    product_id UNINDEXED, title, tokenize = 'trigram'
);
CREATE TABLE queries (query_id INTEGER PRIMARY KEY, query TEXT NOT NULL);
CREATE TABLE judgments (
    query_id INTEGER NOT NULL, product_id TEXT NOT NULL, label TEXT NOT NULL, split TEXT,
    PRIMARY KEY (query_id, product_id)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def write_sqlite(
    path: Path,
    records: list[dict[str, Any]],
    hidden: list[dict[str, Any]],
    index_rows: list[tuple[str, ...]],
    judgments: list[dict[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    try:
        db.executescript(SCHEMA)
        db.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["product_id"],
                    json.dumps(r, ensure_ascii=False),
                    r["category"],
                    h["category_words"],
                    r["attributes"].get("color"),
                    r["price"],
                    r["rating"],
                    int(r["in_stock"]),
                    h["width_in"],
                    h["weight_lb"],
                )
                for r, h in zip(records, hidden, strict=True)
            ],
        )
        db.executemany(
            "INSERT INTO products_fts (product_id, title, category, features, description) "
            "VALUES (?, ?, ?, ?, ?)",
            index_rows,
        )
        db.executemany(
            "INSERT INTO products_trigram (product_id, title) VALUES (?, ?)",
            [row[:2] for row in index_rows],
        )
        queries = {int(j["query_id"]): j["query"] for j in judgments}
        db.executemany("INSERT INTO queries VALUES (?, ?)", sorted(queries.items()))
        db.executemany(
            "INSERT OR IGNORE INTO judgments VALUES (?, ?, ?, ?)",
            [
                (int(j["query_id"]), j["product_id"], GOLD_LETTERS[j["label"]], None)
                for j in judgments
            ],
        )
        db.executemany(
            "INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()]
        )
        db.execute("INSERT INTO products_fts(products_fts) VALUES ('optimize')")
        db.commit()
    finally:
        db.close()


def demo_orders(
    records: list[dict[str, Any]], weights: Mapping[str, float | None], seed: int, anchor: date
) -> dict[str, Any]:
    """Order history for the two demo profiles over in-stock products, dated from ``anchor``.
    The backend serves the dates as they are written here: its clock is the session's."""
    sellable = sorted((r for r in records if r["in_stock"]), key=lambda r: r["product_id"])
    rng = random.Random(f"orders:{seed}")
    shuffled = rng.sample(sellable, len(sellable))
    picks = (shuffled * 7)[:7] if shuffled else []
    # The delivered order carries one product that shipped as freight and one that shipped
    # as a parcel, so the two return terms (freight and everything else) both have a line
    # to apply to.
    freight = [r for r in shuffled if (weights.get(r["product_id"]) or 0) >= FREIGHT_FROM_LB]
    parcels = [r for r in shuffled if 0 < (weights.get(r["product_id"]) or 0) < FREIGHT_FROM_LB]
    if len(picks) == 7 and freight and parcels:
        picks[3], picks[4] = freight[0], parcels[0]
        used = {picks[3]["product_id"], picks[4]["product_id"]}
        rest = [r for r in parcels if r["product_id"] not in used]
        for index in (0, 1, 2, 5, 6):
            picks[index] = rest.pop(0) if rest else picks[index]

    def line(record: Mapping[str, Any], quantity: int = 1) -> dict[str, Any]:
        return {
            "product_id": record["product_id"],
            "title": record["title"],
            "quantity": quantity,
            "price": record["price"],
        }

    def placed(days_ago: int, hour: int, minute: int) -> str:
        return f"{(anchor - timedelta(days=days_ago)).isoformat()}T{hour:02d}:{minute:02d}:00Z"

    def on(days: int) -> str:
        return (anchor + timedelta(days=days)).isoformat()

    plan = [
        ("AH-78214", "demo-user", "shipped", placed(8, 14, 32), [(0, 1), (1, 1)], on(1)),
        (
            "AH-77903", "demo-user", "delayed", placed(14, 9, 5), [(2, 1)],
            f"{on(3)} (updated after a carrier delay; the original {on(-3)} estimate was missed)",
        ),
        ("AH-77410", "demo-user", "delivered", placed(23, 19, 47), [(3, 1), (4, 2)], on(-19)),
        ("AH-79236", "demo-user", "shipped", placed(2, 21, 15), [(5, 1)], on(6)),
        ("AH-76988", "demo-user-2", "processing", placed(3, 8, 12), [(6, 2)], on(4)),
    ]  # fmt: skip
    orders = []
    for order_id, user_id, status, placed_at, lines, estimate in plan:
        if not picks:
            break
        items = [line(picks[index], quantity) for index, quantity in lines]
        orders.append(
            {
                "order_id": order_id,
                "user_id": user_id,
                "status": status,
                "placed_at": placed_at,
                "items": items,
                "total": round(sum(item["price"] * item["quantity"] for item in items), 2),
                "estimated_delivery": estimate,
                "tracking_url": f"https://track.example.com/{order_id}",
            }
        )
    return {
        "_comment": (
            "Generated by wands_home.pipeline from in-stock products. The dates are written "
            f"for {anchor.isoformat()}, the day the manifest names, and are served as written."
        ),
        "orders": orders,
    }


def records_digest(records: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> str:
    """One hash over every record and what the catalog keeps beside it."""
    digest = hashlib.sha256()
    for record, beside in zip(records, hidden, strict=True):
        line = json.dumps([record, beside], ensure_ascii=False, sort_keys=True)
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def build_catalog(
    raw_dir: Path, out_dir: Path, seed: int = SEED, anchor: date = ANCHOR
) -> dict[str, Any]:
    """``data/raw/`` to the catalog database, the gold file, the order fixture and the
    manifest under ``out_dir``. Returns the manifest."""
    raw_products = sorted(read_jsonl(raw_dir / "products.jsonl"), key=lambda r: r["product_id"])
    judgments = read_jsonl(raw_dir / "examples.jsonl")
    selection_file = raw_dir / "selection.json"
    selection = (
        json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.exists() else {}
    )
    listings = [read_listing(raw) for raw in raw_products]
    records = [build_record(listing, seed) for listing in listings]
    hidden = [hidden_of(listing) for listing in listings]
    index_rows = [
        index_text(listing, record) for listing, record in zip(listings, records, strict=True)
    ]
    weights = {r["product_id"]: h["weight_lb"] for r, h in zip(records, hidden, strict=True)}
    departments = Counter(department_of(listing) or GENERAL for listing in listings)
    manifest = {
        "seed": seed,
        "orders_anchored_to": anchor.isoformat(),
        "selection": {k: v for k, v in selection.items() if k != "sources"},
        "sources": selection.get("sources", {}),
        "counts": {
            "products": len(records),
            "in_stock": sum(r["in_stock"] for r in records),
            "queries": len({j["query_id"] for j in judgments}),
            "judgments": len(judgments),
            "categories": len({r["category"] for r in records}),
            "with_width": sum(h["width_in"] is not None for h in hidden),
            "with_weight": sum(h["weight_lb"] is not None for h in hidden),
            "freight": sum((h["weight_lb"] or 0) >= FREIGHT_FROM_LB for h in hidden),
            "departments": dict(sorted(departments.items())),
        },
        "records_sha256": records_digest(records, hidden),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_sqlite(out_dir / "catalog.sqlite", records, hidden, index_rows, judgments, manifest)
    gold = {"seed": seed, "params": selection.get("params"), "queries": gold_entries(judgments)}
    (out_dir / "queries_gold.json").write_text(
        json.dumps(gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "orders.json").write_text(
        json.dumps(demo_orders(records, weights, seed, anchor), indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
