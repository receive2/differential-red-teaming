# SPDX-License-Identifier: Apache-2.0

"""The shopping agent's deployment config for the WANDS example."""

from __future__ import annotations

from shopping_agent import ShoppingAgentConfig

# The catalog's ids have the shape of the ESCI example's: "B0" and eight characters.
PRODUCT_ID_PATTERN = r"\bB0[0-9A-Z]{8}\b"


def build_shopping_config() -> ShoppingAgentConfig:
    defaults = ShoppingAgentConfig.model_fields["product_id_patterns"].default
    return ShoppingAgentConfig(
        brand_name="ACME Home",
        assistant_name="ACME Home Assistant",
        brand_voice="professional, warm, and brief",
        domain_search_notes=(
            "The catalog is home goods (furniture, decor, lighting, rugs, outdoor, kitchen, "
            "bed and bath, storage, home improvement); filters.attributes accepts `color`, "
            "`material` and `style`, and filters.category a store category such as desks, "
            "sectionals or area-rugs."
        ),
        product_id_patterns=(*defaults, PRODUCT_ID_PATTERN),
    )
