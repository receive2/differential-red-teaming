# SPDX-License-Identifier: Apache-2.0

import json
import re
from datetime import UTC, datetime

import pytest

from shopping_agent import (
    Product,
    ProductDetails,
    SearchFilters,
    ShoppingSessionContext,
    Unavailable,
)
from wands_home.api.agent_config import PRODUCT_ID_PATTERN, build_shopping_config
from wands_home.api.wands_backend import (
    EXPRESS_SHIPPING,
    FREE_SHIPPING_OVER,
    FREIGHT_FROM_LB,
    FREIGHT_SHIPPING,
    STANDARD_SHIPPING,
    WandsStorefront,
    same_word,
    search_terms,
    trigram_match,
    trigram_similarity,
    word_match,
)
from wands_home.pipeline import catalog


def at(hour: int, minute: int = 0) -> ShoppingSessionContext:
    """The demo profile's session with its clock at that time of the anchor day."""
    return ShoppingSessionContext(
        session_id="s-clock", user_id="demo-user", now=datetime(2026, 11, 16, hour, minute)
    )


def test_catalog_loads_from_the_database(backend):
    assert len(backend.products) == 33 and backend.store_name == "ACME Home"
    assert all(re.fullmatch(PRODUCT_ID_PATTERN.strip(r"\b"), pid) for pid in backend.products)
    desk = backend.products["B0WF000001"]
    assert desk.title == "Glenbrook L-Shape Desk" and desk.category == "desks"
    assert desk.brand is None and (desk.rating, desk.review_count) == (4.5, 210)
    assert desk.attributes == {
        "color": "brown",
        "material": "manufactured wood",
        "style": "modern & contemporary",
    }
    assert desk.specs["Number of Drawers"] == "2"
    assert desk.specs["Country of Origin"] == "United States"
    assert desk.short_description == (
        "This l-shaped desk fits a corner and gives you two work surfaces."
    )
    assert "float in a room" in (desk.long_description or "")
    assert backend.product("b0wf000001") is desk  # ids typed by hand resolve
    assert backend.variants == {}
    # A product the source gives neither a rating nor a description.
    standing = backend.products["B0WF000004"]
    assert standing.rating is None and standing.long_description is None
    assert standing.short_description == standing.title


def test_width_and_weight_are_the_backends_and_no_record_carries_them(backend):
    assert backend.width_of("B0WF000001") == 59.1 and backend.weight_of("B0WF000001") == 85.0
    assert backend.width_of("B0WF000013") is None  # a rug listed in two sizes
    assert backend.width_of("B0NOPE0000") is None and backend.weight_of("B0NOPE0000") is None
    for product in backend.products.values():
        names = {name.lower() for name in (*product.specs, *product.attributes)}
        assert not names & {"width", "overall width", "weight", "overall product weight"}
    shown = json.loads(backend.products["B0WF000001"].model_dump_json())
    assert "59.1" not in json.dumps(shown) and "85" not in json.dumps(shown)
    assert FREIGHT_FROM_LB == catalog.FREIGHT_FROM_LB
    assert backend.ships_freight("B0WF000001") and not backend.ships_freight("B0WF000003")
    assert not backend.ships_freight("B0NOPE0000")
    assert sum(backend.ships_freight(pid) for pid in backend.products) == 11


def test_query_terms_and_match_expressions():
    assert search_terms("I need a desk for the 2 of us, please") == ["desk", "2", "us"]
    assert word_match(["desk", "ab"]) == "desk* OR ab"
    assert trigram_match(["desk", "ab"]) == '"des" OR "esk"'
    assert trigram_match(["ab"]) == ""


def test_category_words_match_give_or_take_a_plural():
    for wanted, word in [("desk", "desks"), ("benches", "bench"), ("rug", "rugs"), ("art", "arts")]:
        assert same_word(wanted, word) and same_word(word, wanted)
    assert same_word("lighting", "lighting")
    for wanted, word in [
        ("chair", "chaise"),
        ("bed", "bedding"),
        ("tv", "tvs"),
        ("out", "outdoor"),
    ]:
        assert not same_word(wanted, word)


async def test_search_ranks_an_exact_product_first(backend, session):
    for query, exact in [
        ("l shaped desk", {"B0WF000001", "B0WF000002"}),
        ("mid century accent chair", {"B0WF000007", "B0WF000008"}),
        ("blue area rug", {"B0WF000013", "B0WF000014"}),
        ("sectional sofa with chaise", {"B0WF000019", "B0WF000020"}),
        ("led desk lamp", {"B0WF000025", "B0WF000026"}),
        ("outdoor dining set", {"B0WF000030", "B0WF000031"}),
    ]:
        results = await backend.search_products(session, query)
        assert results and results[0].product_id in exact, query
        assert all(type(result) is Product for result in results)  # summaries, not records
    assert await backend.search_products(session, "zzzqqq") == []
    assert await backend.search_products(session, "the of") == []
    assert len(await backend.search_products(session, "desk", limit=3)) == 3


async def test_a_misspelt_query_falls_back_to_trigrams(backend, session):
    results = await backend.search_products(session, "sectonal")
    assert "B0WF000020" in {p.product_id for p in results}
    assert trigram_similarity(["sectonal"], "Tessa 3-Piece Modular Sectional") > 0.6
    # A word that merely shares a few letters with some titles is not a match.
    assert trigram_similarity(["cafetera"], "Upholstered Dining Chair (Set of 2)") < 0.6
    assert await backend.search_products(session, "cafetera") == []


async def test_search_filters_and_sort(backend, session):
    async def search(query: str, **filters) -> list[Product]:
        return await backend.search_products(session, query, SearchFilters(**filters), limit=20)

    cheap = await search("desk", max_price=100)
    assert cheap and all(p.price <= 100 for p in cheap)
    by_price = await search("lamp", sort="price_asc")
    assert len(by_price) >= 3 and [p.price for p in by_price] == sorted(p.price for p in by_price)
    blue = await search("rug", attributes={"color": "blue"})
    assert [p.product_id for p in blue] == ["B0WF000013"]
    styled = await search("chair", attributes={"style": "mid-century"})
    assert styled and all("mid-century" in p.attributes["style"] for p in styled)
    # A key no product carries is read against the attributes' text and the title.
    named = await search("chair", attributes={"brand": "ottis"})
    assert [p.product_id for p in named] == ["B0WF000008"]


async def test_the_category_filter_reads_every_class_and_category(backend, session):
    async def categories(query: str, category: str) -> set[str]:
        found = await backend.search_products(
            session, query, SearchFilters(category=category), limit=20
        )
        return {p.category for p in found}

    assert await categories("rug", "area-rugs") == {"area-rugs"}
    assert await categories("rug", "area rug") == {"area-rugs"}
    assert await categories("rug", "rugs") == {"area-rugs", "rug-pads"}  # the department
    assert await categories("chair", "accent chairs") == {"accent-chairs"}
    # A recliner is filed under a category of chairs; a kids' desk has "Desks" as a second class.
    assert await categories("chair", "chair") == {"accent-chairs", "dining-chairs", "recliners"}
    assert "kids-desks" in await categories("desk", "desks")
    assert await categories("lamp", "lighting") == {"desk-lamps", "floor-lamps", "table-lamps"}
    # A category that would leave nothing is dropped, as the other examples' backends do.
    everything = await backend.search_products(session, "desk", limit=20)
    unmatched = await backend.search_products(
        session, "desk", SearchFilters(category="sofa"), limit=20
    )
    assert unmatched == everything and len(everything) >= 5


async def test_out_of_stock_products_are_searchable_but_not_sellable(backend, session):
    desk = backend.products["B0WF000003"]
    desk.in_stock = False
    results = await backend.search_products(session, "writing desk")
    assert any(p.product_id == "B0WF000003" and p.in_stock is False for p in results)
    with pytest.raises(Unavailable):
        await backend.add_to_cart(session, "B0WF000003", 1)
    with pytest.raises(KeyError):
        await backend.add_to_cart(session, "B0NOPE0000", 1)


async def test_cart_flow(backend, session, other_session):
    cart = await backend.add_to_cart(session, "B0WF000025", 1)
    cart = await backend.add_to_cart(session, "B0WF000025", 2)
    assert cart.items[0].quantity == 3 and cart.item_count == 3
    cart = await backend.update_cart_item(session, "B0WF000025", 1)
    assert cart.items[0].quantity == 1
    assert (await backend.get_cart(other_session)).items == []  # carts are per session
    cart = await backend.remove_from_cart(session, "B0WF000025")
    assert cart.items == []
    backend.reset_session(session.session_id)


async def test_orders_reference_catalog_products_and_are_served_as_written(
    backend, session, other_session
):
    orders = await backend.get_orders(session)
    assert len(orders) == 4 and orders[0].placed_at >= orders[-1].placed_at
    for order in orders:
        for item in order.items:
            assert item.product_id in backend.products and item.variant_of is None
    delivered = await backend.get_order(session, "ah-77410")
    assert delivered.status == "delivered" and delivered.estimated_delivery == "2026-10-28"
    assert sorted(backend.ships_freight(item.product_id) for item in delivered.items) == [
        False,
        True,
    ]
    assert await backend.get_order(other_session, "AH-77410") is None  # not theirs
    assert len(await backend.get_orders(other_session)) == 1
    assert len(backend.recent_orders(limit=3)) == 3
    # The dates are the fixture's whatever the session's clock says.
    later = ShoppingSessionContext(
        session_id="s-later", user_id="demo-user", now=datetime(2027, 3, 1, 9, 0)
    )
    assert await backend.get_orders(later) == orders


async def test_policies_answer_and_state_the_backends_terms(backend, session):
    returns = await backend.search_policies(session, "how do refunds and returns work")
    assert returns and returns[0].policy_id == "returns"
    guides = [p for p in backend._policies if p.policy_id.startswith("guide-")]
    assert len(guides) == 5 and all(p.category == "guide" for p in guides)
    rugs = await backend.search_policies(session, "what size rug for a living room")
    assert rugs and rugs[0].policy_id == "guide-area-rugs"
    freight = await backend.search_policies(session, "how long does freight delivery take")
    assert "shipping" in {p.policy_id for p in freight}
    # What the agent quotes from the shipping entry is what get_fulfillment_options returns.
    shipping = next(p for p in backend._policies if p.policy_id == "shipping").content
    for stated in (
        f"Standard shipping is ${STANDARD_SHIPPING.fee}, free over ${FREE_SHIPPING_OVER}",
        "arrives in 3-5 business days",
        f"Express arrives in 2 business days for ${EXPRESS_SHIPPING.fee}",
        f"Items of {FREIGHT_FROM_LB:g} lb or more ship from our warehouse by freight",
        f"${FREIGHT_SHIPPING.fee:g} per order, delivered in 1 to 3 weeks",
        "9 AM to 9 PM",
    ):
        assert stated in shipping, stated
    assert STANDARD_SHIPPING.eta.startswith("3-5 business days")
    assert EXPRESS_SHIPPING.eta.startswith("2 business days")
    assert FREIGHT_SHIPPING.eta.startswith("1-3 weeks")


async def test_fulfillment_for_parcels_and_for_freight(backend, session, other_session):
    lamp, ceramic_lamp, sectional = "B0WF000025", "B0WF000017", "B0WF000019"
    assert backend.products[lamp].price <= FREE_SHIPPING_OVER
    standard, express, pickup = await backend.get_fulfillment_options(session, [lamp])
    assert (standard, express) == (STANDARD_SHIPPING, EXPRESS_SHIPPING)
    assert (pickup.method, pickup.fee) == ("pickup", 0.0)
    assert pickup.location == "ACME Home Maple Heights"  # the profile's saved showroom
    # Standard shipping is free once the quoted products come to more than the threshold.
    assert backend.products[lamp].price + backend.products[ceramic_lamp].price > FREE_SHIPPING_OVER
    free, *rest = await backend.get_fulfillment_options(session, [ceramic_lamp, lamp])
    assert free == STANDARD_SHIPPING.model_copy(update={"fee": 0.0}) and rest == [express, pickup]
    # A freight product ships by freight, and so does everything quoted with it.
    assert backend.ships_freight(sectional)
    assert await backend.get_fulfillment_options(session, [sectional]) == [FREIGHT_SHIPPING]
    assert await backend.get_fulfillment_options(session, [lamp, sectional]) == [FREIGHT_SHIPPING]
    # Nothing to quote: the parcel options at the full fee.
    for ids in ([], ["B0NOPE0000"]):
        assert await backend.get_fulfillment_options(session, ids) == [standard, express, pickup]
    theirs = await backend.get_fulfillment_options(other_session, [lamp])
    assert theirs[-1].location == "ACME Home Springfield"


async def test_the_clock_is_the_sessions(backend, session):
    async def pickup_eta(context: ShoppingSessionContext) -> str:
        return (await backend.get_fulfillment_options(context, ["B0WF000025"]))[-1].eta

    for context, promised in [
        (at(7, 30), "today by 11 AM"),  # two hours from opening at 9
        (at(10), "today by 12 PM"),
        (at(18, 59), "today by 9 PM"),
        (at(19, 1), "tomorrow morning"),  # ready after closing at 9 PM
        (at(22), "tomorrow morning"),
    ]:
        assert await pickup_eta(context) == promised, context.now
    # A session without a clock is at ten in the morning of the day the orders are dated from.
    assert backend.now_of(session) == datetime(2026, 11, 16, 10, 0)
    assert await pickup_eta(session) == "today by 12 PM"
    aware = ShoppingSessionContext(
        session_id="s-utc", user_id="demo-user", now=datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    )
    assert backend.now_of(aware) == datetime(2026, 11, 20, 15, 0)


def test_config_recognizes_the_ids_and_keeps_the_defaults():
    config = build_shopping_config()
    assert config.brand_name == "ACME Home" and len(config.product_id_patterns) == 3
    assert re.search(PRODUCT_ID_PATTERN, "is B0WF000001 in stock?")
    assert re.search(config.product_id_patterns[0], "what about AR-1201?")
    assert not re.search(PRODUCT_ID_PATTERN, "GLENBROOK1 is not an id")


def test_a_missing_catalog_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="wands_home.pipeline"):
        WandsStorefront(tmp_path)


def test_records_are_product_details(backend):
    assert all(type(product) is ProductDetails for product in backend.products.values())
