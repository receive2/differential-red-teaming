# SPDX-License-Identifier: Apache-2.0

"""The ESCI deployment's shopping agent config. The catalog's ids are Amazon ASINs (B0 and eight more characters), which
the default id patterns (``AR-1201`` and the like) do not cover, so the catalog grounding
gate gets a pattern for them; otherwise a message naming an ASIN would not force a read."""

from __future__ import annotations

from shopping_agent import ShoppingAgentConfig

ASIN_PATTERN = r"\bB0[0-9A-Z]{8}\b"


def build_shopping_config() -> ShoppingAgentConfig:
    defaults = ShoppingAgentConfig.model_fields["product_id_patterns"].default
    return ShoppingAgentConfig(
        brand_name="ACME",
        assistant_name="ACME Assistant",
        brand_voice="professional, warm, and brief",
        domain_search_notes=(
            "The catalog is general merchandise across twenty categories; filters.attributes "
            "accepts `brand` and `color`, and filters.category a store category such as "
            "electronics, home-kitchen or pet-supplies."
        ),
        product_id_patterns=(*defaults, ASIN_PATTERN),
    )
