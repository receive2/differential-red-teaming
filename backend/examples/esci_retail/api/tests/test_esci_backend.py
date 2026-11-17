# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from esci_retail.api.agent_config import ASIN_PATTERN, build_shopping_config
from esci_retail.api.esci_backend import (
    FREE_SHIPPING_OVER,
    STANDARD_SHIPPING,
    EsciStorefront,
    search_terms,
    trigram_match,
    trigram_similarity,
    word_match,
)
from shopping_agent import SearchFilters, Unavailable


def test_catalog_loads_from_the_database(backend):
    assert len(backend.products) == 33 and backend.store_name == "ACME"
    assert all(re.fullmatch(ASIN_PATTERN.strip(r"\b"), pid) for pid in backend.products)
    tent = backend.products["B0TEST0001"]
    assert tent.brand == "Northwind" and tent.category == "sports-outdoors"
    assert tent.attributes["color"] == "green" and tent.specs["Capacity"] == "2 people"
    assert "Two-person tent" in (tent.long_description or "")
    assert backend.product("b0test0001") is tent  # ids typed by hand resolve
    assert backend.variants == {}


def test_query_terms_and_match_expressions():
    assert search_terms("I need a tent for the 2 of us, please") == ["tent", "2", "us"]
    assert word_match(["tent", "ab"]) == "tent* OR ab"
    assert trigram_match(["tent", "ab"]) == '"ent" OR "ten"'
    assert trigram_match(["ab"]) == ""


async def test_search_ranks_the_exact_product_first(backend, session):
    for query, expected in [
        ("programmable drip coffee maker", "B0TEST0007"),
        ("lightweight 2 person backpacking tent", "B0TEST0001"),
        ("large orthopedic dog bed", "B0TEST0019"),
        ("noise cancelling headphones", "B0TEST0024"),
        ("yoga mat non slip", "B0TEST0029"),
    ]:
        results = await backend.search_products(session, query)
        assert results and results[0].product_id == expected, query
    shoes = await backend.search_products(session, "womens running shoes black")
    assert "B0TEST0013" in {p.product_id for p in shoes[:2]}
    assert await backend.search_products(session, "zzzqqq") == []
    assert await backend.search_products(session, "the of") == []


async def test_a_misspelt_query_falls_back_to_trigrams(backend, session):
    results = await backend.search_products(session, "headphnes")
    assert "B0TEST0024" in {p.product_id for p in results}
    assert trigram_similarity(["headphnes"], "Aurex Headphones") > 0.6
    # A word that merely shares a few letters with some titles is not a match.
    assert trigram_similarity(["cafetera"], "Paper Coffee Filters") < 0.6
    assert await backend.search_products(session, "cafetera") == []


async def test_search_filters_and_sort(backend, session):
    cheap = await backend.search_products(session, "coffee", SearchFilters(max_price=20), limit=20)
    assert all(p.price <= 20 for p in cheap)
    pets = await backend.search_products(session, "bed", SearchFilters(category="pet"), limit=20)
    assert pets and all(p.category == "pet-supplies" for p in pets)
    by_price = await backend.search_products(
        session, "coffee", SearchFilters(sort="price_asc"), limit=20
    )
    assert [p.price for p in by_price] == sorted(p.price for p in by_price)
    brand = await backend.search_products(
        session, "running shoes", SearchFilters(attributes={"brand": "Stridewell"}), limit=20
    )
    assert brand and all(p.brand == "Stridewell" for p in brand)
    grey = await backend.search_products(
        session, "running shoes", SearchFilters(attributes={"color": "grey"}), limit=20
    )
    assert grey and grey[0].product_id == "B0TEST0014"


async def test_out_of_stock_products_are_searchable_but_not_sellable(backend, session):
    tent = backend.products["B0TEST0001"]
    tent.in_stock = False
    results = await backend.search_products(session, "backpacking tent")
    assert any(p.product_id == "B0TEST0001" and p.in_stock is False for p in results)
    with pytest.raises(Unavailable):
        await backend.add_to_cart(session, "B0TEST0001", 1)
    with pytest.raises(KeyError):
        await backend.add_to_cart(session, "B0NOPE0000", 1)


async def test_cart_flow(backend, session, other_session):
    cart = await backend.add_to_cart(session, "B0TEST0007", 1)
    cart = await backend.add_to_cart(session, "B0TEST0007", 2)
    assert cart.items[0].quantity == 3 and cart.item_count == 3
    cart = await backend.update_cart_item(session, "B0TEST0007", 1)
    assert cart.items[0].quantity == 1
    assert (await backend.get_cart(other_session)).items == []  # carts are per session
    cart = await backend.remove_from_cart(session, "B0TEST0007")
    assert cart.items == []
    backend.reset_session(session.session_id)


async def test_orders_reference_catalog_products(backend, session, other_session):
    orders = await backend.get_orders(session)
    assert len(orders) == 4 and orders[0].placed_at >= orders[-1].placed_at
    for order in orders:
        for item in order.items:
            assert item.product_id in backend.products and item.variant_of is None
    assert (await backend.get_order(session, "es-77410")).status == "delivered"
    assert await backend.get_order(other_session, "ES-77410") is None  # not theirs
    assert len(await backend.get_orders(other_session)) == 1
    assert len(backend.recent_orders(limit=3)) == 3


async def test_policies_and_fulfillment(backend, session):
    returns = await backend.search_policies(session, "how do refunds and returns work")
    assert returns and returns[0].policy_id == "returns"
    guides = [p for p in backend._policies if p.policy_id.startswith("guide-")]
    assert len(guides) == 5 and all(p.category == "guide" for p in guides)
    assert not {"guide-tents", "guide-home-coffee", "guide-desk-ergonomics"} & {
        p.policy_id for p in guides
    }
    racks = await backend.search_policies(session, "how do I choose a bike rack without a hitch")
    assert racks and racks[0].policy_id == "guide-bike-racks"
    pricey = max(backend.products.values(), key=lambda p: p.price)
    options = await backend.get_fulfillment_options(session, [pricey.product_id])
    assert [o.method for o in options][:3] == ["delivery", "delivery", "pickup"]
    assert pricey.price > FREE_SHIPPING_OVER and options[0].fee == 0.0
    cheapest = min(backend.products.values(), key=lambda p: p.price)
    (paid, *_rest) = await backend.get_fulfillment_options(session, [cheapest.product_id])
    assert paid.fee == STANDARD_SHIPPING.fee or cheapest.price > FREE_SHIPPING_OVER


def test_delivery_promises_are_stamped_on_in_stock_products(backend):
    for product in backend.products.values():
        promise = product.attributes.get("delivery")
        assert (promise is not None and promise.startswith("Get it by ")) is product.in_stock


def test_config_recognizes_asins_and_keeps_the_defaults():
    config = build_shopping_config()
    assert config.brand_name == "ACME" and len(config.product_id_patterns) == 3
    assert re.search(ASIN_PATTERN, "is B0TEST0001 in stock?")
    assert re.search(config.product_id_patterns[0], "what about AR-1201?")
    assert not re.search(ASIN_PATTERN, "BREWLINE12 is not an id")


def test_a_missing_catalog_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="esci_retail.pipeline"):
        EsciStorefront(tmp_path)
