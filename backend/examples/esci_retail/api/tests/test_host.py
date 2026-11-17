# SPDX-License-Identifier: Apache-2.0

from demo_common.tests.fixtures import start_shopper


def test_health_names_the_store_and_counts_the_catalog(main, client):
    health = client.get("/api/health").json()
    assert health["store"] == "ACME" and health["products"] == len(main.backend.products)
    assert "search-discovery" in health["skills"]


def test_catalog_routes_read_the_database_backed_listings(client):
    page = client.get("/api/products", params={"category": "pet-supplies", "limit": 10}).json()
    assert page["products"] and all(p["category"] == "pet-supplies" for p in page["products"])
    assert "long_description" not in page["products"][0]
    detail = client.get("/api/products/B0TEST0007").json()
    assert (
        detail["title"].startswith("BrewLine 12-Cup") and detail["specs"]["Capacity"] == "12 cups"
    )
    assert client.get("/api/products/B0NOPE0000").status_code == 404


def test_session_binds_a_profile_and_the_add_button_goes_through_the_gates(client, shopper):
    headers = start_shopper(client, "demo-user")
    assert client.get("/api/cart", headers=headers).json()["items"] == []
    unseen = client.post(
        "/api/cart/add", json={"product_id": "B0TEST0007", "quantity": 1}, headers=headers
    )
    assert unseen.status_code == 400  # not in this session's results yet
    seen = shopper("B0TEST0007")
    added = client.post(
        "/api/cart/add", json={"product_id": "B0TEST0007", "quantity": 2}, headers=seen
    ).json()
    assert added["ok"] and added["cart"]["items"][0]["quantity"] == 2
    orders = client.get("/api/orders", headers=seen).json()["orders"]
    assert len(orders) == 4
