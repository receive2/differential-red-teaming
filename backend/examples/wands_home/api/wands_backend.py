# SPDX-License-Identifier: Apache-2.0

"""The WANDS example's ``StorefrontBackend`` over ``data/catalog.sqlite``: FTS5 search
(stemmed words ranked by BM25; trigrams when no word matches), per-session carts, the built
order fixtures and the store's policies. It follows ``examples/esci_retail/api/esci_backend.py``
and differs where the catalog does:

- a product's width and weight are columns of the catalog that no record carries
  (``width_of``, ``weight_of``);
- the category filter reads every class and category a product is filed under;
- a product of ``FREIGHT_FROM_LB`` pounds or more ships by freight, and an order with one
  has that option alone;
- the clock is the session's (``now_of``): the time the host gives the session, and for a
  session without one the morning of the day the orders are dated from. The backend reads
  no clock of its own, and the orders are served with the dates they were written with."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

from demo_common.storefront_fixtures import (
    SessionCarts,
    example_data_dir,
    find_order,
    load_orders,
    load_policies,
    load_users,
    newest_orders,
    orders_for,
    preferences_of,
    rank_products,
    search_help,
    summary_of,
    unavailable_detail,
    within_price_and_rating,
)
from shopping_agent import (
    Cart,
    FulfillmentOption,
    Order,
    Policy,
    Product,
    ProductDetails,
    SearchFilters,
    ShoppingSessionContext,
    StorefrontBackend,
    Unavailable,
    UserPreferences,
)

CATALOG_FILE = "catalog.sqlite"

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "my", "me", "i",
        "is", "it", "that", "this", "some", "any", "need", "want", "looking", "buy", "please",
        "can", "you", "get", "find", "show",
    )
)  # fmt: skip
_MAX_TERMS = 12
_CANDIDATES = 300
# bm25() takes one weight per column of the table, in order; the id column is unindexed.
_WORD_WEIGHTS = (0.0, 10.0, 2.0, 2.0, 1.0)  # id, title, category, features, description
_TRIGRAM_WEIGHTS = (0.0, 3.0)  # id, title
# A trigram candidate stays only when some query term shares this share of its trigrams
# with the listing's title; below it the "match" is a few letters in common.
_TRIGRAM_MIN_SIMILARITY = 0.6

# The terms of the shipping entry in policies.json, which is what the agent quotes.
FREE_SHIPPING_OVER = 49
FREIGHT_FROM_LB = 50.0  # the pipeline counts the catalog's freight products by the same figure
STANDARD_SHIPPING = FulfillmentOption(
    method="delivery", eta="3-5 business days (standard)", fee=5.99
)
EXPRESS_SHIPPING = FulfillmentOption(method="delivery", eta="2 business days (express)", fee=9.99)
FREIGHT_SHIPPING = FulfillmentOption(
    method="shipping", eta="1-3 weeks (freight, delivery by appointment)", fee=49.0
)
_STORE_OPENS, _STORE_CLOSES = 9, 21
_CLOCK_HOUR = 10  # of the day the orders are dated from, for a session without a clock


def default_data_dir() -> Path:
    """``data/`` beside ``api/``, or ``WANDS_HOME_DATA_DIR`` when set (the tests build a
    small catalog in a temporary directory and point the app at it)."""
    return Path(os.environ.get("WANDS_HOME_DATA_DIR") or example_data_dir(__file__))


DATA_DIR = default_data_dir()


def search_terms(query: str) -> list[str]:
    """The query's words without function words, at most ``_MAX_TERMS``."""
    return [term for term in _WORD.findall(query.lower()) if term not in _STOPWORDS][:_MAX_TERMS]


def word_match(terms: Iterable[str]) -> str:
    """An FTS5 expression matching any term; terms long enough also match as prefixes."""
    return " OR ".join(f"{term}*" if len(term) >= 4 else term for term in terms)


def trigrams(text: str) -> set[str]:
    text = text.lower()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def trigram_match(terms: Iterable[str]) -> str:
    """An FTS5 expression over the trigram index: any trigram of any term of four or more
    letters, so a misspelt word still shares most of its trigrams with the right one."""
    grams = sorted({gram for term in terms if len(term) >= 4 for gram in trigrams(term)})
    return " OR ".join(f'"{gram}"' for gram in grams[:60])


def trigram_similarity(terms: Iterable[str], text: str) -> float:
    """The best share, over the terms, of a term's trigrams that ``text`` contains."""
    text_grams = trigrams(text)
    shares = [
        len(grams & text_grams) / len(grams)
        for term in terms
        if len(term) >= 4 and (grams := trigrams(term))
    ]
    return max(shares, default=0.0)


def same_word(wanted: str, word: str) -> bool:
    """Two words of a category that differ by a plural at most: "desk" and "desks",
    "bench" and "benches", "rug" and "rugs"."""
    short, long = sorted((wanted, word), key=len)
    if not long.startswith(short):
        return False
    return short == long or (len(short) >= 3 and len(long) <= len(short) + 2)


class WandsStorefront(StorefrontBackend):
    def __init__(self, data_dir: Path | None = None) -> None:
        data_dir = data_dir or default_data_dir()
        catalog = data_dir / CATALOG_FILE
        if not catalog.exists():
            raise FileNotFoundError(
                f"{catalog} is missing. Build it first: PYTHONPATH=examples python -m "
                "wands_home.pipeline all --source-dir <WANDS dataset dir> "
                "(examples/wands_home/README.md)."
            )
        self._db = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True, check_same_thread=False)
        self.products: dict[str, ProductDetails] = {}
        self._filed_under: dict[str, frozenset[str]] = {}
        self._width: dict[str, float] = {}
        self._weight: dict[str, float] = {}
        for record, words, width, weight in self._db.execute(
            "SELECT record, category_words, width_in, weight_lb FROM products ORDER BY product_id"
        ):
            product = ProductDetails.model_validate(json.loads(record))
            self.products[product.product_id] = product
            self._filed_under[product.product_id] = frozenset(
                (*_WORD.findall(product.category or ""), *words.split())
            )
            if width is not None:
                self._width[product.product_id] = width
            if weight is not None:
                self._weight[product.product_id] = weight
        self.variants: dict[str, ProductDetails] = {}  # WANDS products carry no option families
        self.store_name = "ACME Home"
        (anchor,) = self._db.execute(
            "SELECT value FROM meta WHERE key = 'orders_anchored_to'"
        ).fetchone()
        self._day = datetime.fromisoformat(json.loads(anchor)).replace(hour=_CLOCK_HOUR)
        self._users = load_users(data_dir)
        self._orders = load_orders(data_dir)
        self._policies = load_policies(data_dir)
        self._carts = SessionCarts()

    def now_of(self, session: ShoppingSessionContext) -> datetime:
        """The session's local time; for a session without a clock, the morning of the day
        the catalog's orders are dated from. The backend reads no other clock."""
        now = session.local_now()
        return self._day if now is None else now.replace(tzinfo=None)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def product(self, product_id: str) -> ProductDetails | None:
        """A listing by id; the ids are upper-case, so a lower-cased id still resolves."""
        return self.products.get(product_id) or self.products.get(product_id.strip().upper())

    def listing_of(self, product_id: str) -> ProductDetails | None:
        return self.product(product_id)

    def width_of(self, product_id: str) -> float | None:
        """The product's overall width in inches, where the source states one."""
        return self._width.get(product_id)

    def weight_of(self, product_id: str) -> float | None:
        """The product's weight in pounds, where the source states one."""
        return self._weight.get(product_id)

    def _scores(self, terms: list[str]) -> dict[str, float]:
        """Word matches ranked by BM25 as positive scores; when no word matches at all, the
        trigram index stands in so a misspelt query still finds its product, scored by how
        much of a term's trigrams the listing shares and cut below a similarity floor."""
        placeholders = ", ".join("?" for _ in _WORD_WEIGHTS)
        rows = self._db.execute(
            f"SELECT product_id, bm25(products_fts, {placeholders}) FROM products_fts "
            "WHERE products_fts MATCH ? ORDER BY 2 LIMIT ?",
            (*_WORD_WEIGHTS, word_match(terms), _CANDIDATES),
        ).fetchall()
        if not rows and (expression := trigram_match(terms)):
            placeholders = ", ".join("?" for _ in _TRIGRAM_WEIGHTS)
            rows = self._db.execute(
                f"SELECT product_id, bm25(products_trigram, {placeholders}) FROM products_trigram "
                "WHERE products_trigram MATCH ? ORDER BY 2 LIMIT ?",
                (*_TRIGRAM_WEIGHTS, expression, _CANDIDATES),
            ).fetchall()
            similar = {}
            for product_id, _score in rows:
                product = self.products.get(product_id)
                if product is None:
                    continue
                share = trigram_similarity(terms, product.title)
                if share >= _TRIGRAM_MIN_SIMILARITY:
                    similar[product_id] = share
            return similar
        return {product_id: -score for product_id, score in rows}

    def _soft_filter(self, product: ProductDetails, filters: SearchFilters) -> bool:
        """Category and attribute filters as the model guesses them: every word of the
        wanted category must be a word of a class or a category the product is filed under,
        give or take a plural; an attribute matches the product's own value or, lacking
        one, its text."""
        if filters.category:
            filed_under = self._filed_under.get(product.product_id, frozenset())
            for wanted in _WORD.findall(filters.category.lower()):
                if not any(same_word(wanted, word) for word in filed_under):
                    return False
        for key, value in filters.attributes.items():
            wanted_value = str(value).lower().strip()
            if not wanted_value:
                continue
            actual = product.attributes.get(key.lower())
            haystack = (
                actual.lower()
                if actual is not None
                else " ".join(f"{k}={v}" for k, v in product.attributes.items()).lower()
                + " "
                + product.title.lower()
            )
            if wanted_value not in haystack:
                return False
        return True

    async def search_products(
        self,
        session: ShoppingSessionContext,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 8,
    ) -> list[Product]:
        del session
        terms = search_terms(query)
        if not terms:
            return []
        scores = self._scores(terms)
        candidates = [self.products[pid] for pid in scores if pid in self.products]
        ranked = rank_products(
            candidates,
            query,
            filters,
            limit,
            score=lambda product, _terms: scores[product.product_id],
            hard_filter=within_price_and_rating,
            soft_filter=self._soft_filter,
        )
        return [summary_of(product) for product in ranked]

    async def get_product_details(
        self, session: ShoppingSessionContext, product_id: str
    ) -> ProductDetails | None:
        del session
        return self.product(product_id)

    # ------------------------------------------------------------------
    # Cart
    # ------------------------------------------------------------------

    async def get_cart(self, session: ShoppingSessionContext) -> Cart:
        return self._carts.cart(session.session_id)

    async def add_to_cart(
        self, session: ShoppingSessionContext, product_id: str, quantity: int
    ) -> Cart:
        product = self.product(product_id)
        if product is None:
            # The executor's provenance gate holds this before it reaches a backend.
            raise KeyError(product_id)
        if not product.in_stock:
            raise Unavailable(unavailable_detail(product, None))
        existing = self._carts.lines(session.session_id).get(product.product_id)
        quantity += existing.quantity if existing else 0
        return self._carts.put(session.session_id, product, quantity)

    async def update_cart_item(
        self, session: ShoppingSessionContext, product_id: str, quantity: int
    ) -> Cart:
        return self._carts.set_quantity(session.session_id, product_id, quantity)

    async def remove_from_cart(self, session: ShoppingSessionContext, product_id: str) -> Cart:
        return self._carts.remove(session.session_id, product_id)

    def reset_session(self, session_id: str) -> None:
        self._carts.reset(session_id)

    # ------------------------------------------------------------------
    # Customer, orders, help content, fulfillment
    # ------------------------------------------------------------------

    async def get_preferences(self, session: ShoppingSessionContext) -> UserPreferences:
        return preferences_of(self._users, session.user_id)

    async def get_orders(self, session: ShoppingSessionContext, limit: int = 5) -> list[Order]:
        return orders_for(self._orders, session.user_id, limit)

    async def get_order(self, session: ShoppingSessionContext, order_id: str) -> Order | None:
        return find_order(self._orders, session.user_id, order_id)

    def recent_orders(self, limit: int = 6) -> list[Order]:
        return newest_orders(self._orders, limit)

    async def search_policies(self, session: ShoppingSessionContext, query: str) -> list[Policy]:
        del session
        return search_help(self._policies, query)

    def ships_freight(self, product_id: str) -> bool:
        return self._weight.get(product_id, 0.0) >= FREIGHT_FROM_LB

    @staticmethod
    def _pickup_eta(now: datetime) -> str:
        """Two hours of preparation from now (or from opening), promised as the top of an
        hour inside showroom hours, otherwise tomorrow morning."""
        opens = now.replace(hour=_STORE_OPENS, minute=0, second=0, microsecond=0)
        ready = max(now, opens) + timedelta(hours=2)
        if ready.minute or ready.second or ready.microsecond:
            ready = ready.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        if ready.date() != now.date() or ready.hour > _STORE_CLOSES:
            return "tomorrow morning"
        hour12 = ready.hour % 12 or 12
        return f"today by {hour12} {'AM' if ready.hour < 12 else 'PM'}"

    async def get_fulfillment_options(
        self, session: ShoppingSessionContext, product_ids: list[str]
    ) -> list[FulfillmentOption]:
        prefs = await self.get_preferences(session)
        location = prefs.default_location or "your area"
        quoted = [product for pid in product_ids if (product := self.product(pid))]
        if any(self.ships_freight(product.product_id) for product in quoted):
            return [FREIGHT_SHIPPING]
        standard = STANDARD_SHIPPING
        if sum(product.price for product in quoted) > FREE_SHIPPING_OVER:
            standard = standard.model_copy(update={"fee": 0.0})
        return [
            standard,
            EXPRESS_SHIPPING,
            FulfillmentOption(
                method="pickup",
                eta=self._pickup_eta(self.now_of(session)),
                fee=0.0,
                location=f"{self.store_name} {location}",
            ),
        ]
