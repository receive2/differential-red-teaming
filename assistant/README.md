# ACME shopping assistant

The assistant that talks with ACME's customers inside the store's app and website while
they shop: it searches the catalog, compares options, fills the cart, stages checkout, and
answers order and store-terms questions. Everything it says about a product, a price, an
order or a term comes from a tool result in the conversation; the storefront behind the
tools is an integration (`StorefrontBackend`) the deployment supplies.

## Layout

| Path | Holds |
|---|---|
| `shopping-agent/core/shopping_agent/` | the domain types, `StorefrontBackend`, the deployment config, the system prompt, the tool registry and presentation payloads (`tools/`), the gates, payload enrichment, and the tool executor |
| `shopping-agent/skills/` | the flows, one `SKILL.md` each, listed in the prompt's skill index and loaded with `load_skill` |
| `shopping-agent/runtime-messages-api/` | `ShoppingAgent`, the turn loop on the Messages API |
| `commerce-common/` | the shared library the assistant is built on: fencing, streaming events, the executor frame, skills, memory store, turn helpers. Consumed as is; changes to it are out of scope for this repository |

## Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
ruff format --check . && ruff check .
```

The tests script the model with `commerce_common.testing.FakeClient` and need no credentials.
A live run needs `ANTHROPIC_API_KEY` in the environment (see `.env.example`).

## Rules the code keeps

- Facts come from tool results: the prompt forbids describing availability, prices, terms or
  order details the conversation has not fetched, and the grounding rules force the first read.
- Writes are gated in code: cart writes need session provenance and respect the quantity and
  line caps; `checkout` stages a summary and never places an order.
- Everything a tool returns reaches the model inside the `storefront_data` fence; third-party
  text is information, never instruction.
- Components identify products by `product_id`; the app fills in prices and availability.
