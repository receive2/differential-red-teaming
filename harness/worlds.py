# SPDX-License-Identifier: Apache-2.0

"""Variants of the ESCI backend that differ in one thing. First, what the fulfillment service
does with a destination ZIP.

    esci          the service takes products only and quotes the customer's saved location,
                  whatever ZIP the customer mentions (the backend as upstream wrote it)
    esci-refuse   the service takes ``zip_code`` and refuses it in the backend's own words: it
                  raises ``NotOffered``, which the assistant's executor relays to the model
                  as "... is not something this backend offers; say so plainly."
    esci-zip      the service takes ``zip_code`` and quotes for it: the delivery window and
                  the express fee depend on the distance, pickup exists only near the backend,
                  and a ZIP outside the delivery area gets no options at all

The spec, the catalog, the users and the orders are the same in all three; so is what the
service returns when no ZIP is given. In the two variants the assistant's repository shows
the parameter in the backend protocol (``START_PATCHES``), as a repository written against
such a backend would.

Second, what the catalog search does when it is asked for ``sort:
"best_selling"``.

    esci          the search knows relevance, price and rating; an unknown sort is ordered by
                  relevance, silently
    esci-nosales  the search refuses that sort with the same ``NotOffered`` signal
    esci-sales    every product has a number of units sold in the last 30 days (synthesized:
                  a fixed function of ``SALES_SEED`` and the product id), and the search
                  returns the matches whose titles carry the query's words ordered by it,
                  most first (the matches of a relevance search reach far beyond the kind
                  of product asked for; a ranking by sales over all of them would put a
                  tower fan among the flashlights)

In these two the assistant's repository shows ``best_selling`` among the sorts of its
``SearchFilters`` type (``SALES_PATCHES``); the tool the model calls still lists the four
sorts it always did: letting the model ask for the fifth is the feature.

Third, backends that differ in the service's answer and in nothing else: the
same interface and the same assistant repository, and a service that takes the request and
*ignores* it (answers a neighbouring question without saying so), *refuses* it, or is
*capable* of it.

    esci-zip-ignore     F3: the service takes ``zip_code`` and quotes the saved location
                        whatever the ZIP (beside esci-refuse and esci-zip)
    esci-sales-ignore   F8: the search takes ``sort: "best_selling"`` and orders by relevance
                        (beside esci-nosales and esci-sales)
    F9, On Sale: the search is asked for ``filters.attributes`` ``offer: "on_sale"``
      esci                the attribute matches nothing and is dropped: regular results
      esci-onsale-refuse  refused with ``NotOffered``
      esci-onsale         only the products on sale (a synthesized discount), each labelled
    F10, Store Stock: the product read is given a ``store``
      esci-stock-ignore   the same record whatever the backend (one flag for the whole chain)
      esci-stock-refuse   refused with ``NotOffered``
      esci-stock          ``in_stock`` and a unit count for that ACME store (synthesized)
    F11, Price Adjustment: the policy search is asked about ACME lowering its own price
      esci                no such passage: the nearest is the price match promise
      esci-adjust-refuse  a passage of the backend's terms that says no adjustment is made
      esci-adjust         a passage that grants one within ``ADJUST_WINDOW_DAYS`` days

For F9 and F11 the backend as upstream wrote it is the ignoring one, with the interface the
other two have, and no file of the assistant's repository differs between the three. For F3,
F8 and F10 the three variants share one patched repository.

Fourth, the study's second backend: ACME Home over the WANDS catalog (``examples/wands_home``
of the world tree), with four topics of its own. Each has a backend that takes the request
and ignores it and one that is capable of it; the second topic also has the backend as it
is, whose service takes no such input.

    F21, Fits the Space: the search is given ``filters.max_width_inches``
      wands-fit-ignore    the limit is taken and the regular results come back
      wands-fit           only the matches whose stated width is within the limit, each
                          with its width
    F22, Needed by a Date: the fulfillment service is given ``needed_by``
      wands               the service takes products only and answers with its usual windows
      wands-date-ignore   the date is taken and the usual windows come back
      wands-date          only the options promised to arrive by the date, each with the
                          day it is promised for (synthesized)
    F23, Newest Arrivals: the search is asked for ``sort: "newest"``
      wands-new-ignore    ordered by relevance, silently
      wands-new           the matches of the kind asked for by the day they were added to
                          the catalog (synthesized), newest first, each with that day
    F24, Showroom Stock: the product read is given a ``store``
      wands-stock-ignore  the same record whatever the showroom (one flag for the chain)
      wands-stock         ``in_stock`` and a unit count for that ACME Home showroom
                          (synthesized)

The two variants of a topic share one patched repository. What a capable service returns
to a request that carries the input is never what it returns without it: the width, the
promised day, the day added and the showroom are on the records it returns."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ZIP_KINDS = ("esci-zip", "esci-refuse", "esci-zip-ignore")

# The saved locations' ZIP prefixes (the demo users' towns), for pickup and distance.
HOME_PREFIX = {"Maple Heights": "441", "Springfield": "627"}
DEFAULT_PREFIX = "441"
REFUSAL = "A delivery quote for a ZIP code other than the saved location"
REFUSING_DOC = (
    "The delivery, pickup, and shipping options for up to twenty ids as the model wrote them, for the customer's "
    "saved location; ids the catalog does not know are skipped. A request that gives a ``zip_code`` is refused: "
    "the store quotes for no ZIP other than the saved location."
)

PROTOCOL_OLD = '''    async def get_fulfillment_options(
        self, session: ShoppingSessionContext, product_ids: list[str]
    ) -> list[FulfillmentOption]:
        """The delivery, pickup, and shipping options for up to twenty ids as the model
        wrote them; ids the catalog does not know are skipped."""'''
PROTOCOL_NEW = '''    async def get_fulfillment_options(
        self,
        session: ShoppingSessionContext,
        product_ids: list[str],
        zip_code: str | None = None,
    ) -> list[FulfillmentOption]:
        """The delivery, pickup, and shipping options for up to twenty ids as the model
        wrote them, for the five-digit ``zip_code`` when one is given and for the customer's
        saved location otherwise; ids the catalog does not know are skipped."""'''
FAKE_OLD = '''    async def get_fulfillment_options(self, session, product_ids):
        del session, product_ids
        return [FulfillmentOption(method="delivery", eta="2 days", fee=0.0)]'''
FAKE_NEW = '''    async def get_fulfillment_options(self, session, product_ids, zip_code=None):
        del session, product_ids, zip_code
        return [FulfillmentOption(method="delivery", eta="2 days", fee=0.0)]'''

# (file in the assistant's repository, text as upstream has it, text for a backend with the parameter)
START_PATCHES = (
    ("shopping-agent/core/shopping_agent/backend.py", PROTOCOL_OLD, PROTOCOL_NEW),
    ("conftest.py", FAKE_OLD, FAKE_NEW),
)

SALES_KINDS = ("esci-sales", "esci-nosales", "esci-sales-ignore")
SALES_SORT = "best_selling"
SALES_REFUSAL = "A ranking by sales"
SALES_SEED = 20261110
SORT_OLD = '    sort: Literal["relevance", "price_asc", "price_desc", "rating"] = "relevance"'
SORT_NEW = '    sort: Literal["relevance", "price_asc", "price_desc", "rating", "best_selling"] = "relevance"'
SALES_PATCHES = (("shopping-agent/core/shopping_agent/types.py", SORT_OLD, SORT_NEW),)

# F9: products on sale. No file of the assistant's repository differs: the search filters
# already take free attributes.
SALE_KINDS = ("esci-onsale", "esci-onsale-refuse")
SALE_ATTRIBUTE, SALE_VALUE = "offer", "on_sale"
SALE_REFUSAL = "A filter for products on sale"
SALE_SEED = 20261111
SALE_SHARE = 0.30  # of the catalog is on sale in the capable backend

# F10: stock at a named store. The three variants share a repository whose product read
# shows the parameter.
STOCK_KINDS = ("esci-stock", "esci-stock-refuse", "esci-stock-ignore")
ACME_STORES = ("Maple Heights", "Springfield", "Riverton", "Lakewood")
STOCK_REFUSAL = "Stock at a named store"
STOCK_SEED = 20261130  # the first seed from 20261112 on for which the two probes of F10 have different answers and the saved store has both products
DETAILS_OLD = '''    async def get_product_details(
        self, session: ShoppingSessionContext, product_id: str
    ) -> ProductDetails | None:
        """The full record for one id as the model wrote it, or None when the id is
        unknown. A family's ``variants`` carries'''
DETAILS_NEW = '''    async def get_product_details(
        self, session: ShoppingSessionContext, product_id: str, store: str | None = None
    ) -> ProductDetails | None:
        """The full record for one id as the model wrote it, or None when the id is
        unknown. When ``store`` names an ACME store, ``in_stock`` says whether that store
        has the product; without it, whether the catalog lists it as available. A
        family's ``variants`` carries'''
FAKE_DETAILS_OLD = '''    async def get_product_details(self, session, product_id):
        del session
        return CATALOG.get(product_id) or VARIANTS.get(product_id)'''
FAKE_DETAILS_NEW = '''    async def get_product_details(self, session, product_id, store=None):
        del session, store
        return CATALOG.get(product_id) or VARIANTS.get(product_id)'''
STOCK_PATCHES = (
    ("shopping-agent/core/shopping_agent/backend.py", DETAILS_OLD, DETAILS_NEW),
    ("conftest.py", FAKE_DETAILS_OLD, FAKE_DETAILS_NEW),
)

# F11: a price adjustment when the backend lowers its own price. One passage of the backend's
# terms differs between the backends; no file of the assistant's repository does.
ADJUST_KINDS = ("esci-adjust", "esci-adjust-refuse")
ADJUST_POLICY_ID = "price-adjustment"
ADJUST_TITLE = "Price adjustments after purchase"
ADJUST_WINDOW_DAYS = 21
ADJUST_YES = (
    f"If ACME lowers its own price on an item within {ADJUST_WINDOW_DAYS} days of your purchase, we refund the "
    "difference to your original payment method. Ask in chat or contact support with your order number; one "
    "adjustment per item. Clearance and doorbuster prices are excluded. A lower price at another retailer is "
    "covered by the price match promise, not by this policy."
)
ADJUST_NO = (
    "ACME does not adjust the price of an order after purchase: if our own price on an item drops later, the "
    "order stands as charged and no difference is refunded. The price match promise is separate and covers an "
    "identical item at a lower price from another major retailer."
)

# The second backend: ACME Home over the WANDS catalog (examples/wands_home of the world tree).
WANDS = "wands"  # the backend as it is; for F22, the one whose service takes no such input

# F21: products that fit a space. The two variants share a repository whose search filters
# carry the limit.
FIT_KINDS = ("wands-fit", "wands-fit-ignore")
WIDTH_FIELD = "max_width_inches"
WIDTH_MARK = "width_inches"  # the attribute a capable search states each match's width in
WIDTH_OLD = '''    min_rating: float | None = None
    attributes: dict[str, str] = Field(default_factory=dict)'''
WIDTH_NEW = '''    min_rating: float | None = None
    max_width_inches: float | None = None
    attributes: dict[str, str] = Field(default_factory=dict)'''
FIT_PATCHES = (("shopping-agent/core/shopping_agent/types.py", WIDTH_OLD, WIDTH_NEW),)

# F22: an order needed by a date. The two variants share a repository whose fulfillment
# service shows the parameter; the backend as it is has the service without it.
DATE_KINDS = ("wands-date", "wands-date-ignore")
NEEDED_NEW = '''    async def get_fulfillment_options(
        self,
        session: ShoppingSessionContext,
        product_ids: list[str],
        needed_by: str | None = None,
    ) -> list[FulfillmentOption]:
        """The delivery, pickup, and shipping options for up to twenty ids as the model
        wrote them; ids the catalog does not know are skipped. With ``needed_by`` (a date,
        YYYY-MM-DD), the options the store promises to arrive by that date, and an empty
        list when it promises none."""'''
FAKE_NEEDED_NEW = '''    async def get_fulfillment_options(self, session, product_ids, needed_by=None):
        del session, product_ids, needed_by
        return [FulfillmentOption(method="delivery", eta="2 days", fee=0.0)]'''
DATE_PATCHES = (
    ("shopping-agent/core/shopping_agent/backend.py", PROTOCOL_OLD, NEEDED_NEW),
    ("conftest.py", FAKE_OLD, FAKE_NEEDED_NEW),
)
PROMISE_SEED = 20261116
EXPRESS_DAYS = 2  # business days, as the backend's express option words it
STANDARD_DAYS = (3, 5)  # business days, the window of its standard option
FREIGHT_DAYS = (7, 21)  # days, the one to three weeks of its freight option

# F23: the newest arrivals of a kind. The two variants share a repository whose search filters
# list the sort.
NEW_KINDS = ("wands-new", "wands-new-ignore")
NEWEST_SORT = "newest"
NEWEST_NEW = '    sort: Literal["relevance", "price_asc", "price_desc", "rating", "newest"] = "relevance"'
NEW_PATCHES = (("shopping-agent/core/shopping_agent/types.py", SORT_OLD, NEWEST_NEW),)
ADDED_SEED = 20261116
ADDED_WITHIN_DAYS = 720  # of the day the catalog's orders are dated from
ADDED_MARK = "added_on"  # the attribute a capable search states the day in
# Words of a query that ask for the newest and name no kind of product.
ARRIVAL_WORDS = frozenset({"new", "newest", "newer", "newly", "latest", "recent", "recently", "arrival", "arrived", "added", "just", "release", "released"})

# F24: stock at a named showroom. The two variants share a repository whose product read
# shows the parameter.
SHOWROOM_KINDS = ("wands-stock", "wands-stock-ignore")
SHOWROOMS = ("Maple Heights", "Springfield", "Harborview", "Ridgewood", "Cedar Park", "Mill Creek")
SHOWROOM_SEED = 20261116
SHOWROOM_DETAILS_NEW = '''    async def get_product_details(
        self, session: ShoppingSessionContext, product_id: str, store: str | None = None
    ) -> ProductDetails | None:
        """The full record for one id as the model wrote it, or None when the id is
        unknown. When ``store`` names an ACME Home showroom, ``in_stock`` says whether that
        showroom has the product; without it, whether the catalog lists it as available. A
        family's ``variants`` carries'''
SHOWROOM_PATCHES = (
    ("shopping-agent/core/shopping_agent/backend.py", DETAILS_OLD, SHOWROOM_DETAILS_NEW),
    ("conftest.py", FAKE_DETAILS_OLD, FAKE_DETAILS_NEW),
)

PATCHES = {
    **dict.fromkeys(ZIP_KINDS, START_PATCHES),
    **dict.fromkeys(SALES_KINDS, SALES_PATCHES),
    **dict.fromkeys(STOCK_KINDS, STOCK_PATCHES),
    **dict.fromkeys(FIT_KINDS, FIT_PATCHES),
    **dict.fromkeys(DATE_KINDS, DATE_PATCHES),
    **dict.fromkeys(NEW_KINDS, NEW_PATCHES),
    **dict.fromkeys(SHOWROOM_KINDS, SHOWROOM_PATCHES),
}
# What each variant's service does with the request its spec presumes.
BEHAVIOUR = {
    "esci-zip": ("F3", "capable"), "esci-refuse": ("F3", "refuse"), "esci-zip-ignore": ("F3", "ignore"),
    "esci-sales": ("F8", "capable"), "esci-nosales": ("F8", "refuse"), "esci-sales-ignore": ("F8", "ignore"),
    "esci-onsale": ("F9", "capable"), "esci-onsale-refuse": ("F9", "refuse"),
    "esci-stock": ("F10", "capable"), "esci-stock-refuse": ("F10", "refuse"), "esci-stock-ignore": ("F10", "ignore"),
    "esci-adjust": ("F11", "capable"), "esci-adjust-refuse": ("F11", "refuse"),
    "wands-fit": ("F21", "capable"), "wands-fit-ignore": ("F21", "ignore"),
    "wands-date": ("F22", "capable"), "wands-date-ignore": ("F22", "ignore"),
    "wands-new": ("F23", "capable"), "wands-new-ignore": ("F23", "ignore"),
    "wands-stock": ("F24", "capable"), "wands-stock-ignore": ("F24", "ignore"),
}  # fmt: skip
VARIANT_KINDS = tuple(BEHAVIOUR)
WORLD_KINDS = ("esci", "retail", WANDS, *VARIANT_KINDS)

# What the harness and the experiment take from the backend's side by its kind, so that no
# file of theirs names a catalog: the package under ``examples/`` of the world tree a kind is
# served from, where its data is, the backend and the config the assistant is deployed with,
# the titles of its topics, and a kind of product its catalog carries many of.
EXAMPLES = ("esci_retail", "retail", "wands_home")
# Of the topics that experiment/loop.py does not title itself.
TOPIC_TITLES: dict[str, str] = {"F21": "Fits the Space", "F22": "Needed by a Date", "F23": "Newest Arrivals", "F24": "Showroom Stock"}


def example_of(kind: str) -> str:
    """The example a backend kind is served from: the package under ``examples/`` of the world
    tree that has its catalog, its fixtures and its storefront."""
    if kind == "retail":
        return "retail"
    return "wands_home" if kind == WANDS or kind.startswith(WANDS + "-") else "esci_retail"


def data_dir_of(world_tree: Path, kind: str) -> Path:
    """Where the catalog, the judged queries and the fixtures of a backend kind are."""
    return world_tree / "examples" / example_of(kind) / "data"


def deployment(kind: str, data_dir: Path) -> tuple[Any, Any]:
    """The backend of a kind over the data in ``data_dir``, and the config its example deploys
    the assistant with. Every kind but ``retail``, whose backend the runner builds itself."""
    if example_of(kind) == "wands_home":
        from wands_home.api.agent_config import build_shopping_config as wands_config
        from wands_home.api.wands_backend import WandsStorefront

        return WandsStorefront(data_dir) if kind == WANDS else storefront(kind, data_dir), wands_config()
    from esci_retail.api.agent_config import build_shopping_config
    from esci_retail.api.esci_backend import EsciStorefront

    # A variant is the same backend with one service that answers otherwise.
    backend = EsciStorefront(data_dir) if kind == "esci" else storefront(kind, data_dir)
    return backend, build_shopping_config()


def safety_search(kind: str) -> str:
    """A kind of product the catalog of a backend kind carries many of: what the customer of the
    safety set's quantity case asks to be shown (experiment/workspace.py)."""
    return "throw pillows" if example_of(kind) == "wands_home" else "wireless chargers"


def storefront(kind: str, data_dir: Path) -> Any:
    """The backend of a variant backend."""
    feature, behaviour = BEHAVIOUR[kind]
    make = {
        "F3": zip_storefront, "F8": sales_storefront, "F9": sale_storefront, "F10": stock_storefront, "F11": adjust_storefront,
        "F21": fit_storefront, "F22": date_storefront, "F23": new_storefront, "F24": showroom_storefront,
    }[feature]  # fmt: skip
    return make(data_dir, behaviour)


def patch_start(workspace: Path, kind: str = "esci-zip") -> None:
    """Make a fresh copy of the subject start match the backend ``kind``: a fulfillment service
    with the ``zip_code`` parameter, a search-filters type with the ``best_selling`` sort, or
    a product read with the ``store`` parameter; for the second backend, a search-filters type
    with ``max_width_inches`` or with the ``newest`` sort, a fulfillment service with
    ``needed_by``, or a product read with ``store``. The tool the model calls is untouched:
    passing the ZIP, the sort or the backend through it is the feature."""
    for relative, old, new in PATCHES[kind]:
        path = workspace / relative
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise RuntimeError(f"{relative}: the store's contract is not where the patch expects it")
        path.write_text(text.replace(old, new), encoding="utf-8")


def zone_distance(zip_code: str, prefix: str) -> int:
    return abs(int(zip_code[0]) - int(prefix[0]))


def standard_window(distance: int) -> str:
    if distance <= 1:
        return "2-3 business days"
    if distance <= 3:
        return "3-4 business days"
    if distance <= 5:
        return "4-6 business days"
    return "5-8 business days"


def served(zip_code: str) -> bool:
    """The delivery area: five digits, not a territory or military prefix, not the far north-west."""
    return bool(re.fullmatch(r"\d{5}", zip_code)) and not zip_code.startswith(("00", "09", "96", "99"))


def zip_storefront(data_dir: Path, behaviour: str) -> Any:
    """An ``EsciStorefront`` whose fulfillment service takes ``zip_code``: it quotes for it
    (``capable``), refuses it (``refuse``), or takes it and quotes the saved location all the
    same (``ignore``)."""
    capable = behaviour == "capable"
    from esci_retail.api import esci_backend as base
    from shopping_agent.backend import NotOffered
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        FulfillmentOption,
        ShoppingSessionContext,
    )

    class ZipStorefront(base.EsciStorefront):
        async def get_fulfillment_options(  # type: ignore[override]
            self, session: ShoppingSessionContext, product_ids: list[str], zip_code: str | None = None
        ) -> list[FulfillmentOption]:
            """The delivery, pickup, and shipping options for up to twenty ids as the model
            wrote them, for the five-digit ``zip_code`` when one is given and for the customer's
            saved location otherwise; ids the catalog does not know are skipped."""
            if zip_code is None:
                return await super().get_fulfillment_options(session, product_ids)
            zip_code = str(zip_code).strip()
            if behaviour == "ignore":
                return await super().get_fulfillment_options(session, product_ids)
            if not capable:
                raise NotOffered(REFUSAL)
            if not served(zip_code):
                return []
            prefs = await self.get_preferences(session)
            location = prefs.default_location or "your area"
            prefix = HOME_PREFIX.get(location, DEFAULT_PREFIX)
            distance = zone_distance(zip_code, prefix)
            quoted = [product for pid in product_ids if (product := self.product(pid))]
            subtotal = sum(product.price for product in quoted)
            destination = f"ZIP {zip_code}"
            options = [
                FulfillmentOption(
                    method="delivery", eta=standard_window(distance),
                    fee=0.0 if subtotal > base.FREE_SHIPPING_OVER else 5.99, location=destination,
                ),
                FulfillmentOption(
                    method="delivery",
                    eta="2 business days (express)" if distance <= 3 else "3 business days (express)",
                    fee=9.99 if distance <= 3 else 14.99, location=destination,
                ),
            ]
            if zip_code.startswith(prefix):
                options.append(
                    FulfillmentOption(method="pickup", eta=self._pickup_eta(datetime.now()), fee=0.0, location=f"ACME {location}")
                )
            if any(
                product.category in base._FREIGHT_CATEGORIES and product.price > base._FREIGHT_PRICE_FLOOR
                for product in quoted
            ):
                options.append(
                    FulfillmentOption(method="shipping", eta="5-7 business days (freight)" if distance <= 3 else "7-10 business days (freight)", fee=29.0, location=destination)
                )
            return options

    if behaviour == "refuse":
        # The contract file is rendered from this description: the backend's, not the protocol's.
        ZipStorefront.get_fulfillment_options.__doc__ = REFUSING_DOC
    ZipStorefront.__name__ = "EsciStorefront"  # the contract file names the backend, not the variant
    return ZipStorefront(data_dir)


def units_sold(product_id: str, seed: int = SALES_SEED) -> int:
    """Units of a product sold in the last 30 days: synthesized, a fixed function of the seed
    and the id, with a few products far ahead of the rest. It has nothing to do with the
    product's rating, its price or how well it matches a query."""
    digest = hashlib.sha256(f"{seed}:{product_id}".encode()).digest()
    share = int.from_bytes(digest[:8], "big") / 2**64
    return 1 + int(5000 * share**3)


def singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


# Words of a query that ask for the ranking and name no kind of product.
RANKING_WORDS = frozenset({"best", "selling", "seller", "bestselling", "bestseller", "top", "popular", "most", "sold"})


def kind_terms(terms: list[str]) -> list[str]:
    return [term for term in terms if singular(term) not in RANKING_WORDS]


def names_the_kind(title: str, terms: list[str]) -> bool:
    """Does the title carry every word of the query that names a kind (singular or plural)?"""
    wanted = kind_terms(terms)
    words = {singular(word) for word in re.findall(r"[a-z0-9]+", title.lower())}
    return bool(wanted) and all(singular(term) in words for term in wanted)


def sales_storefront(data_dir: Path, behaviour: str) -> Any:
    """An ``EsciStorefront`` whose catalog search is asked for ``sort: "best_selling"``: it
    orders the matches by units sold (``capable``), refuses (``refuse``), or orders them by
    relevance without a word (``ignore``: what upstream's search does with a sort it does not
    know, here with the sort in the repository's type as in the other two)."""
    capable = behaviour == "capable"
    from esci_retail.api import esci_backend as base
    from shopping_agent.backend import NotOffered
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        Product,
        SearchFilters,
        ShoppingSessionContext,
    )

    class SalesStorefront(base.EsciStorefront):
        async def search_products(  # type: ignore[override]
            self, session: ShoppingSessionContext, query: str, filters: SearchFilters | None = None, limit: int = 8
        ) -> list[Product]:
            if getattr(filters, "sort", None) != SALES_SORT:
                return await super().search_products(session, query, filters, limit)
            if behaviour == "ignore":
                return await super().search_products(session, query, filters.model_copy(update={"sort": "relevance"}), limit)
            if not capable:
                raise NotOffered(SALES_REFUSAL)
            # Every match a relevance search would rank (the same filters and the same
            # relevance floor) whose title names the kind asked for, by units sold; equal
            # sales keep the relevance order. No such product: no ranking for the request.
            terms = base.search_terms(query)
            if not terms:
                return []
            scores = self._scores(terms)
            candidates = [self.products[pid] for pid in scores if pid in self.products]
            matches = base.rank_products(
                candidates, query, filters, max(1, len(candidates)),
                score=lambda product, _terms: scores[product.product_id],
                hard_filter=base.within_price_and_rating, soft_filter=self._soft_filter,
            )
            ranked = [product for product in matches if names_the_kind(product.title, terms)]
            ranked.sort(key=lambda product: -units_sold(product.product_id))
            return [base.summary_of(product) for product in ranked[:limit]]

        def units_sold(self, product_id: str) -> int:
            return units_sold(product_id)

    import inspect

    # The contract file is rendered from this description: the protocol's first paragraph
    # and what this backend does with the sort.
    described = (inspect.getdoc(base.EsciStorefront.search_products) or "").split("\n\n")[0].replace("\n", " ").strip()
    SalesStorefront.search_products.__doc__ = described + (
        " A search with ``filters.sort`` of ``best_selling`` is refused: the store has no ranking by sales."
        if behaviour == "refuse"
        else " With ``filters.sort`` of ``best_selling`` the result is the matches whose titles carry the query's words, "
        "ordered by units sold in the last 30 days, most first; an empty list when there is no such ranking for the query."
    )
    SalesStorefront.__name__ = "EsciStorefront"
    return SalesStorefront(data_dir)


def _share(*parts: object) -> tuple[float, int]:
    """A number in [0, 1) and a byte, fixed functions of the parts: what the synthesized
    backend data are drawn from."""
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64, digest[8]


def sale_of(product_id: str, seed: int = SALE_SEED) -> int:
    """The discount in percent a product is on sale at in the capable backend, or 0 when it is
    not on sale: synthesized, a fixed function of the seed and the id."""
    share, byte = _share(seed, "sale", product_id)
    return (10, 15, 20, 25, 30)[byte % 5] if share < SALE_SHARE else 0


def asks_for_sale(filters: Any) -> bool:
    attributes = getattr(filters, "attributes", None) or {}
    return any(str(k).strip().lower() == SALE_ATTRIBUTE and str(v).strip().lower() == SALE_VALUE for k, v in attributes.items())


def sale_storefront(data_dir: Path, behaviour: str) -> Any:
    """An ``EsciStorefront`` whose catalog search is asked for the products on sale
    (``filters.attributes`` ``offer: "on_sale"``): it returns them, or refuses."""
    from esci_retail.api import esci_backend as base
    from shopping_agent.backend import NotOffered
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        Product,
        SearchFilters,
        ShoppingSessionContext,
    )

    class SaleStorefront(base.EsciStorefront):
        async def search_products(  # type: ignore[override]
            self, session: ShoppingSessionContext, query: str, filters: SearchFilters | None = None, limit: int = 8
        ) -> list[Product]:
            if not asks_for_sale(filters):
                return await super().search_products(session, query, filters, limit)
            if behaviour != "capable":
                raise NotOffered(SALE_REFUSAL)
            # The matches a search without the sale filter would rank (the other filters and
            # the relevance floor as they are), less the ones that are not on sale.
            rest = filters.model_copy(update={"attributes": {k: v for k, v in filters.attributes.items() if str(k).strip().lower() != SALE_ATTRIBUTE}})
            terms = base.search_terms(query)
            if not terms:
                return []
            scores = self._scores(terms)
            candidates = [self.products[pid] for pid in scores if pid in self.products]
            matches = base.rank_products(
                candidates, query, rest, max(1, len(candidates)),
                score=lambda product, _terms: scores[product.product_id],
                hard_filter=base.within_price_and_rating, soft_filter=self._soft_filter,
            )
            found = []
            for product in matches:
                percent = sale_of(product.product_id)
                if not percent:
                    continue
                regular = round(product.price / (1 - percent / 100), 2)
                found.append(base.summary_of(product).model_copy(update={
                    "labels": [*product.labels, "On sale"],
                    "attributes": {**product.attributes, "regular_price": f"{regular:.2f}", "discount": f"{percent}%"},
                }))
                if len(found) == limit:
                    break
            return found

    SaleStorefront.__name__ = "EsciStorefront"
    return SaleStorefront(data_dir)


def store_named(words: object) -> str | None:
    """The ACME store some words name ("the Springfield store", "ACME Riverton"), or None."""
    text = re.sub(r"[^a-z ]", " ", str(words).lower())
    text = " ".join(w for w in text.split() if w not in ("acme", "store", "the", "location", "branch", "shop"))
    return next((store for store in ACME_STORES if store.lower() == text), None)


def units_at(product_id: str, store: str, seed: int = STOCK_SEED) -> int:
    """Units of a product on the shelf of an ACME store in the capable backend: synthesized, a
    fixed function of the seed, the backend and the id; nearly half of the pairs have none."""
    share, byte = _share(seed, "stock", store, product_id)
    return 0 if share < 0.45 else 1 + byte % 9


def stock_storefront(data_dir: Path, behaviour: str) -> Any:
    """An ``EsciStorefront`` whose product read is given a ``store``: it answers for that
    backend (``capable``), refuses (``refuse``), or returns the record it returns for everyone
    (``ignore``)."""
    from esci_retail.api import esci_backend as base
    from shopping_agent.backend import NotOffered
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        ProductDetails,
        ShoppingSessionContext,
    )

    class StockStorefront(base.EsciStorefront):
        async def get_product_details(  # type: ignore[override]
            self, session: ShoppingSessionContext, product_id: str, store: str | None = None
        ) -> ProductDetails | None:
            """The full record for one id as the model wrote it, or None when the id is
            unknown. When ``store`` names an ACME store, ``in_stock`` says whether that store
            has the product; without it, whether the catalog lists it as available."""
            if store is None or not str(store).strip() or behaviour == "ignore":
                return await super().get_product_details(session, product_id)
            if behaviour != "capable":
                raise NotOffered(STOCK_REFUSAL)
            named = store_named(store)
            if named is None:
                raise NotOffered(f"Stock at a store named {str(store).strip()[:40]}")
            details = await super().get_product_details(session, product_id)
            if details is None:
                return None
            units = units_at(details.product_id, named) if details.in_stock else 0
            return details.model_copy(update={
                "in_stock": units > 0,
                "attributes": {**details.attributes, "store": f"ACME {named}", "units_at_store": str(units)},
            })

    if behaviour == "refuse":
        StockStorefront.get_product_details.__doc__ = (
            "The full record for one id as the model wrote it, or None when the id is unknown. A request that gives "
            "a ``store`` is refused: the store keeps no stock by store."
        )
    StockStorefront.__name__ = "EsciStorefront"
    return StockStorefront(data_dir)


def adjust_storefront(data_dir: Path, behaviour: str) -> Any:
    """An ``EsciStorefront`` whose terms have a passage on price adjustments after purchase:
    one that grants them (``capable``) or one that says none is made (``refuse``)."""
    from esci_retail.api import esci_backend as base
    from shopping_agent.types import Policy

    class AdjustStorefront(base.EsciStorefront):
        def __init__(self, data_dir: Path | None = None) -> None:
            super().__init__(data_dir)
            self._policies = [
                *self._policies,
                Policy(
                    policy_id=ADJUST_POLICY_ID, title=ADJUST_TITLE, category="pricing",
                    content=ADJUST_YES if behaviour == "capable" else ADJUST_NO,
                ),
            ]

    AdjustStorefront.__name__ = "EsciStorefront"
    return AdjustStorefront(data_dir)


def fit_storefront(data_dir: Path, behaviour: str) -> Any:
    """A ``WandsStorefront`` whose catalog search is given ``filters.max_width_inches``: it
    returns the matches that fit (``capable``), or takes the limit and returns what it returns
    without one (``ignore``)."""
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        Product,
        SearchFilters,
        ShoppingSessionContext,
    )
    from wands_home.api import wands_backend as base

    class FitStorefront(base.WandsStorefront):
        async def search_products(  # type: ignore[override]
            self, session: ShoppingSessionContext, query: str, filters: SearchFilters | None = None, limit: int = 8
        ) -> list[Product]:
            """The catalog's matches for a query, best first. With ``filters.max_width_inches``
            the result is the matches whose stated width is within that many inches, each with
            its width among its attributes (``width_inches``); a product that states no width
            is not among them."""
            wanted = getattr(filters, WIDTH_FIELD, None)
            if wanted is None or behaviour == "ignore":
                return await super().search_products(session, query, filters, limit)
            # The matches a search without the limit would rank (the other filters and the
            # relevance floor as they are), less the ones that are wider or state no width.
            terms = base.search_terms(query)
            if not terms:
                return []
            scores = self._scores(terms)
            candidates = [self.products[pid] for pid in scores if pid in self.products]
            matches = base.rank_products(
                candidates, query, filters, max(1, len(candidates)),
                score=lambda product, _terms: scores[product.product_id],
                hard_filter=base.within_price_and_rating, soft_filter=self._soft_filter,
            )
            found = []
            for product in matches:
                width = self.width_of(product.product_id)
                if width is None or width > float(wanted):
                    continue
                found.append(base.summary_of(product).model_copy(update={
                    "attributes": {**product.attributes, WIDTH_MARK: f"{width:g}"},
                }))
                if len(found) == limit:
                    break
            return found

    FitStorefront.__name__ = "WandsStorefront"
    return FitStorefront(data_dir)


def business_days_after(day: date, days: int) -> date:
    """The day ``days`` business days (Monday to Friday) after ``day``."""
    while days > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            days -= 1
    return day


def standard_days(product_id: str, seed: int = PROMISE_SEED) -> int:
    """Business days a parcel takes by standard shipping in the capable backend: synthesized,
    a fixed function of the seed and the id, within the window the backend words."""
    low, high = STANDARD_DAYS
    return low + _share(seed, "standard", product_id)[1] % (high - low + 1)


def freight_days(product_id: str, seed: int = PROMISE_SEED) -> int:
    """Days a freight product takes to the customer's door in the capable backend: synthesized
    likewise, within the one to three weeks the backend words."""
    low, high = FREIGHT_DAYS
    return low + int(_share(seed, "freight", product_id)[0] * (high - low + 1))


def day_named(words: object, today: date) -> date | None:
    """The day some words name ("2026-11-30", "11/30/2026", "November 30th", "Mon, Nov 30"), or
    None. A date written without a year is the first such day from ``today`` on."""
    text = str(words).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    text = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", text.lower())
    text = re.sub(r"^(mon|tue|wed|thu|fri|sat|sun)[a-z]*[,.]?\s+", "", text)
    text = " ".join(text.replace(",", " ").split())
    forms = ("%m/%d/%Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y")
    for form in (*forms, "%m/%d/%y"):
        try:
            return datetime.strptime(text, form).date()
        except ValueError:
            continue
    for year in (today.year, today.year + 1):
        for form in forms:
            try:
                day = datetime.strptime(f"{text}/{year}" if "/" in form else f"{text} {year}", form).date()
            except ValueError:
                continue
            if day >= today:
                return day
    return None


def day_words(day: date) -> str:
    return f"{day:%a, %b} {day.day}"


def date_storefront(data_dir: Path, behaviour: str) -> Any:
    """A ``WandsStorefront`` whose fulfillment service is given ``needed_by``: it returns the
    options it promises by that date (``capable``), or takes the date and returns the options
    it returns without one (``ignore``)."""
    from shopping_agent.backend import NotOffered
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        FulfillmentOption,
        ShoppingSessionContext,
    )
    from wands_home.api import wands_backend as base

    class DateStorefront(base.WandsStorefront):
        async def get_fulfillment_options(  # type: ignore[override]
            self, session: ShoppingSessionContext, product_ids: list[str], needed_by: str | None = None
        ) -> list[FulfillmentOption]:
            """The delivery, pickup, and shipping options for up to twenty ids as the model
            wrote them; ids the catalog does not know are skipped. With ``needed_by`` (a date,
            YYYY-MM-DD), the options the store promises to arrive by that date, each with the
            day it is promised for, and an empty list when it promises none."""
            options = await super().get_fulfillment_options(session, product_ids)
            if needed_by is None or not str(needed_by).strip() or behaviour == "ignore":
                return options
            today = self.now_of(session).date()
            wanted = day_named(needed_by, today)
            if wanted is None:
                raise NotOffered(f"A delivery promise for a date written as {str(needed_by).strip()[:40]} (dates are read as YYYY-MM-DD)")
            quoted = [product.product_id for pid in product_ids if (product := self.product(pid))]
            promised = []
            for option in options:
                day = self.promised_for(option, quoted, today)
                if day > wanted:
                    continue
                if option.method != "pickup":  # a pickup's words already name its day
                    option = option.model_copy(update={"eta": f"by {day_words(day)} {option.eta[option.eta.index('('):]}"})
                promised.append(option)
            return promised

        def promised_for(self, option: FulfillmentOption, quoted: list[str], today: date) -> date:
            """The day the store promises an option of its usual answer for: the slowest of
            the order's products decides for a shipment."""
            if option.method == "pickup":
                return today if option.eta.startswith("today") else today + timedelta(days=1)
            if option.eta == base.FREIGHT_SHIPPING.eta:
                return today + timedelta(days=max(freight_days(pid) for pid in quoted if self.ships_freight(pid)))
            if option.eta == base.EXPRESS_SHIPPING.eta:
                return business_days_after(today, EXPRESS_DAYS)
            return business_days_after(today, max((standard_days(pid) for pid in quoted), default=STANDARD_DAYS[1]))

    DateStorefront.__name__ = "WandsStorefront"
    return DateStorefront(data_dir)


def added_on(product_id: str, today: date, seed: int = ADDED_SEED) -> date:
    """The day a product was added to the catalog in the capable backend: synthesized, a fixed
    function of the seed and the id, within ``ADDED_WITHIN_DAYS`` days before ``today``. It has
    nothing to do with the product's rating, its price or how well it matches a query."""
    return today - timedelta(days=1 + int(_share(seed, "added", product_id)[0] * ADDED_WITHIN_DAYS))


def arrival_terms(terms: list[str]) -> list[str]:
    """The words of a query that name a kind of product: all but the ones that ask for the
    newest."""
    return [term for term in terms if singular(term) not in ARRIVAL_WORDS]


def new_storefront(data_dir: Path, behaviour: str) -> Any:
    """A ``WandsStorefront`` whose catalog search is asked for ``sort: "newest"``: it orders the
    matches by the day they were added (``capable``), or orders them by relevance without a
    word (``ignore``)."""
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        Product,
        ProductDetails,
        SearchFilters,
        ShoppingSessionContext,
    )
    from wands_home.api import wands_backend as base

    class NewStorefront(base.WandsStorefront):
        async def search_products(  # type: ignore[override]
            self, session: ShoppingSessionContext, query: str, filters: SearchFilters | None = None, limit: int = 8
        ) -> list[Product]:
            """The catalog's matches for a query, best first. With ``filters.sort`` of
            ``newest`` the result is the matches of the kind the query names, ordered by the
            day they were added to the catalog, newest first, each with that day among its
            attributes (``added_on``); an empty list when the catalog has none of the kind."""
            if getattr(filters, "sort", None) != NEWEST_SORT:
                return await super().search_products(session, query, filters, limit)
            if behaviour == "ignore":
                return await super().search_products(session, query, filters.model_copy(update={"sort": "relevance"}), limit)
            # Every match a relevance search for the kind would rank (the same filters and
            # the same relevance floor) that is of the kind asked for, by the day it was
            # added; products added on one day keep the relevance order.
            kind = arrival_terms(base.search_terms(query))
            if not kind:
                return []
            scores = self._scores(kind)
            candidates = [self.products[pid] for pid in scores if pid in self.products]
            matches = base.rank_products(
                candidates, " ".join(kind), filters, max(1, len(candidates)),
                score=lambda product, _terms: scores[product.product_id],
                hard_filter=base.within_price_and_rating, soft_filter=self._soft_filter,
            )
            ranked = [product for product in matches if self.of_the_kind(product, kind)]
            ranked.sort(key=lambda product: self.added_on(product.product_id), reverse=True)
            return [
                base.summary_of(product).model_copy(update={
                    "attributes": {**product.attributes, ADDED_MARK: self.added_on(product.product_id).isoformat()},
                })
                for product in ranked[:limit]
            ]

        def added_on(self, product_id: str) -> date:
            return added_on(product_id, self._day.date())

        def of_the_kind(self, product: ProductDetails, kind: list[str]) -> bool:
            """Do the product's title and the category it is filed in carry every word of the
            query that names a kind, give or take a plural? (The wider classes a product is
            filed under do not count: an end table is filed under nightstands too.)"""
            words = re.findall(r"[a-z0-9]+", f"{product.title} {product.category or ''}".lower())
            return all(any(base.same_word(term, word) for word in words) for term in kind)

    NewStorefront.__name__ = "WandsStorefront"
    return NewStorefront(data_dir)


def showroom_named(words: object) -> str | None:
    """The ACME Home showroom some words name ("the Springfield showroom", "ACME Home
    Harborview"), or None."""
    text = re.sub(r"[^a-z ]", " ", str(words).lower())
    text = " ".join(w for w in text.split() if w not in ("acme", "home", "showroom", "store", "the", "location", "branch", "shop"))
    return next((showroom for showroom in SHOWROOMS if showroom.lower() == text), None)


def units_at_showroom(product_id: str, showroom: str, seed: int = SHOWROOM_SEED) -> int:
    """Units of a product on the floor of an ACME Home showroom in the capable backend:
    synthesized, a fixed function of the seed, the showroom and the id; nearly half of the
    pairs have none."""
    share, byte = _share(seed, "showroom", showroom, product_id)
    return 0 if share < 0.45 else 1 + byte % 9


def showroom_storefront(data_dir: Path, behaviour: str) -> Any:
    """A ``WandsStorefront`` whose product read is given a ``store``: it answers for that
    showroom (``capable``), or returns the record it returns for everyone (``ignore``)."""
    from shopping_agent.backend import NotOffered
    from shopping_agent.types import (  # noqa: F401 - named in the signature
        ProductDetails,
        ShoppingSessionContext,
    )
    from wands_home.api import wands_backend as base

    class ShowroomStorefront(base.WandsStorefront):
        async def get_product_details(  # type: ignore[override]
            self, session: ShoppingSessionContext, product_id: str, store: str | None = None
        ) -> ProductDetails | None:
            """The full record for one id as the model wrote it, or None when the id is
            unknown. When ``store`` names an ACME Home showroom, ``in_stock`` says whether that
            showroom has the product; without it, whether the catalog lists it as available."""
            if store is None or not str(store).strip() or behaviour == "ignore":
                return await super().get_product_details(session, product_id)
            named = showroom_named(store)
            if named is None:
                raise NotOffered(f"Stock at a showroom named {str(store).strip()[:40]}")
            details = await super().get_product_details(session, product_id)
            if details is None:
                return None
            units = units_at_showroom(details.product_id, named) if details.in_stock else 0
            return details.model_copy(update={
                "in_stock": units > 0,
                "attributes": {**details.attributes, "store": f"ACME Home {named}", "units_at_store": str(units)},
            })

    ShowroomStorefront.__name__ = "WandsStorefront"
    return ShowroomStorefront(data_dir)
