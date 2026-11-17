# SPDX-License-Identifier: Apache-2.0

"""ACME over ESCI, the example API: the catalog backend behind the shared storefront
routes. No merchant side and no web app; the catalog is browsed through the API or by
the agent.

    uvicorn esci_retail.api.main:app --app-dir examples --reload --port 8004

Memory is in-process and re-seeded from ``data/memory-seed.json`` at every boot.
``ESCI_RETAIL_DATA_DIR`` points the app at a catalog built elsewhere.
"""

from __future__ import annotations

from commerce_common.memory import InMemoryMemoryStore
from demo_common import (
    REPO_ROOT,
    CartAddRequest,
    MemorySeeder,
    build_storefront_host,
    load_demo_env,
)
from shopping_agent_runtime import ShoppingAgent

from .agent_config import build_shopping_config
from .esci_backend import EsciStorefront, default_data_dir

DATA_DIR = default_data_dir()
load_demo_env(DATA_DIR.parent)

backend = EsciStorefront(DATA_DIR)
agent = ShoppingAgent(
    backend=backend,
    skills_dir=REPO_ROOT / "shopping-agent" / "skills",
    config=build_shopping_config(),
    memory_store=InMemoryMemoryStore(),
)
host = build_storefront_host(
    title="ACME ESCI demo API",
    example_root=DATA_DIR.parent,
    backend=backend,
    agent=agent,
    memory_seeder=MemorySeeder(DATA_DIR / "memory-seed.json"),
)
app = host.app


@app.post("/api/cart/add")
async def cart_add(request: CartAddRequest, record: host.CurrentSession) -> dict:
    return await host.direct_add(
        record,
        request,
        note="Customer tapped the add-to-cart button on {title} ({product_id}), quantity {quantity}.",
    )
