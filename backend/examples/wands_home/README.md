# ACME Home over WANDS (`wands_home`)

A storefront whose catalog is a few thousand real home-goods products from the Wayfair
ANnotation Dataset (WANDS), chosen by real search queries that carry human relevance labels.
It is the second backend of the self-evolving-agent research harness, beside `esci_retail`:
another catalog, another kind of goods, the same services and the same demo customers, so
that a finding on one backend can be checked on the other. It is not one of the upstream
verticals and is not meant as an upstream contribution: the product names carry real brands,
which the repository's "everything is fictional" rule excludes.

## Provenance and license

- Source: [wayfair/WANDS](https://github.com/wayfair/WANDS), MIT License. Three tab-separated
  files of its `dataset/` directory are read: `product.csv` (90.6 MB; 42,994 products),
  `query.csv` (480 queries) and `label.csv` (5.7 MB; 233,448 judgments). The build in place
  was made from commit `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`; the sha256 digests of the
  three files are recorded in `data/raw/selection.json` and `data/manifest.json` by the build.
- WANDS gives a product's name, class, path in the category tree, description, a list of
  features, and its rating and review count, and for each query the products a human judged
  **Exact**, **Partial** or **Irrelevant**. It has no prices, stock or images; price and
  stock are synthesized here (below), deterministically from a seed.
- The data files are not committed. `data/raw/` and `data/catalog.sqlite` are ignored; the
  small derived files (`queries_gold.json`, `orders.json`, `manifest.json`) are kept so the
  gold is reviewable without the dataset.

## Build

```bash
# 1. The three files, into an empty directory outside the repository.
mkdir -p ~/Documents/data/wands && cd ~/Documents/data/wands
B=https://raw.githubusercontent.com/wayfair/WANDS/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset
for f in product.csv query.csv label.csv; do curl -sSL -o "$f" "$B/$f"; done
shasum -a 256 *.csv
#   c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef  label.csv
#   d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6  product.csv
#   63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95  query.csv

# 2. From the repository root. The extract and build stages need the standard library only.
PYTHONPATH=examples python -m wands_home.pipeline all --source-dir ~/Documents/data/wands
```

`all` runs the three stages; each is also a command of its own:

| Stage | Command | Reads | Writes |
|---|---|---|---|
| extract | `python -m wands_home.pipeline extract --source-dir DIR` | the three WANDS files | `data/raw/products.jsonl`, `data/raw/examples.jsonl`, `data/raw/selection.json` |
| build | `python -m wands_home.pipeline build` | `data/raw/` | `data/catalog.sqlite`, `data/queries_gold.json`, `data/orders.json`, `data/manifest.json` |
| eval-search | `python -m wands_home.pipeline eval-search [-k 8]` | the catalog and the gold file | a report on stdout |

Selection parameters (`--n-queries`, `--seed`, the thresholds) are flags of `extract` and
`all`, and `--anchor` (the day the demo orders are dated from) of `build` and `all`; `--help`
lists them. A build is a function of the three files, the parameters, the seed and the anchor
and holds no time of its own: the same inputs reproduce the catalog and the tracked files
byte for byte, and `records_sha256` in the manifest is one digest over every product. The
build in place has `c0719b36a119bfc68bf43a997de501b1bc594e208eb4b114f10df6d7d13d2180`.

## Run

```bash
uvicorn wands_home.api.main:app --app-dir examples --port 8005
curl -s localhost:8005/api/health
curl -s 'localhost:8005/api/products?category=desks&limit=3'
```

There is no web app and no merchant side; the shared storefront routes (`/api/session`,
`/api/chat`, `/api/products`, `/api/cart`, `/api/orders`, `/api/memory`) are all of it. Chat
needs `ANTHROPIC_API_KEY` as for the other examples. `WANDS_HOME_DATA_DIR` points the app
at a catalog built in another directory. Memory is in-process and re-seeded at every boot.

## How the catalog is chosen

The catalog is chosen by query, not by product, as the ESCI example's is: every query in the
catalog comes with every product a human judged for it.

- A query qualifies when it has at least 2 Exact and 1 Irrelevant judgments and between 8
  and 300 judgments in all. WANDS judges far more products for a query than ESCI does (486
  on average), hence the higher ceiling and the smaller number of queries.
- WANDS judges some pairs of a query and a product more than once, a few of them with two
  different labels; the first judgment in the file's order stands.
- Queries and product names are screened with the ESCI example's rules (not English, not
  ASCII, content a general store's assistant should not be evaluated on), applied to the text
  with its accents folded, since WANDS writes "décor" in ordinary English names. The default
  build excludes nothing by these rules.
- 50 queries are drawn with the seed from the 176 that qualify.

The default build: 50 queries, 7,625 products, 8,416 judgments (2,639 Exact, 4,514 Partial,
1,263 Irrelevant), 7,005 products in stock, across 439 categories. `data/manifest.json` has
the counts of the build in place.

## What is taken from WANDS, and what is synthesized

| Field | Source |
|---|---|
| `product_id` | `B0WF` and the WANDS product id in six digits (`B0WF000123`), the ten-character shape of the ESCI example's ids |
| `title` | the product's name. WANDS ships its text lowercased and split into tokens (`it 's`, `32 '' wide`), part of it read in the wrong encoding (`dã©cor`); `pipeline/text.py` puts the spacing, the characters and the case back and drops a statement of a length (`Arlo 47.2" W Standing Desk` becomes `Arlo Standing Desk`; see Width and weight); otherwise it changes no wording |
| `long_description`, `short_description` | the description, cleaned likewise, less the sentences that state a length, and cut at 800 characters; its first sentence; the title for the 1,065 products without a description (1,030 have none in the source; every sentence of 35 more stated a length) |
| `category` | the slug of the product's first class (`desks`, `area-rugs`, `sideboards-buffets`); for the 521 products without a class the last step of its category path; `home-goods` for the 41 with neither |
| `attributes` | `color`, `material` and `style`, from the features of those names, up to three values each |
| `specs` | up to twelve of a fixed list of features (product type, materials, finish, shape, counts of drawers, shelves, doors and lights, capacity, power source, assembly, origin, warranty); a value that states a length is left out |
| `rating`, `review_count` | WANDS's own figures; 1,670 products have none |
| `price` | synthesized: log-normal around a median for the product's class or, for a class without one, its department; rounded to .99 |
| `in_stock` | synthesized: 92% of the products |
| `brand`, `labels`, `image_url`, option families | none; every product is a plain product |

Price and stock are hashed from the seed and the product id, so a product keeps its figures
in any build with that seed, whichever other products were drawn. Prices are plausible for a
class, not real, and a class without a band of its own takes its department's: a small
accessory filed under furniture can carry a furniture-sized price.

## Width and weight

Two features stay out of every record: a product's overall width (inches) and its weight
(pounds). They are columns of the catalog that the backend reads and no service returns:
`WandsStorefront.width_of(product_id)` and `weight_of(product_id)`. 4,764 products have a
width (a product listed in several sizes has none) and 6,943 a weight. The specifications
leave out every value that states a length for the same reason. The research harness asks
the assistant for things this backend does not answer (which of these fit a 40-inch
alcove?), and that question is only a test while no record answers it. In the source, the
name or a description of 2,743 of the 7,625 products states a size (`Neville 48" Console
Table`, `measures 18" h x 40" w x 16" d`), and a search for "benches 40 inches wide" ranks
those first. The pipeline drops the statement from a name and the sentence that makes it
from a description (`text.without_lengths`, `without_length_sentences`), so that a width is
read from the column or not at all. This is the one place the wording changes.

## Search

`search_products` runs an FTS5 query over a porter-stemmed index of title, classes and
category path, attribute and specification values, and description, ranked by BM25 with
column weights 10 / 2 / 2 / 1; terms of four or more characters also match as prefixes. When
no word matches at all, a trigram index of the titles stands in, so a misspelt query still
finds its product; a trigram candidate counts only when some query term shares at least 60%
of its trigrams with the title. The repository's shared ranking rules then apply: price and
rating filters, a relevance cutoff at half the best score, category and attribute filters
that are dropped if they would empty the result, and the requested sort.

- `filters.category` is matched against everything a product is filed under: its category
  slug, all of its classes and every step of its category path. Every word of the wanted
  category must be one of those words, give or take a plural, so `desks`, `desk`,
  `area rugs`, `lighting` and `office furniture` all filter.
- `filters.attributes` takes `color`, `material` and `style`; a value matches as a part of
  the product's own value, and a product without that attribute is matched on its other
  attributes and its title.

`eval-search` scores the search against the WANDS labels (k = 8, default build):

| Measure | Value |
|---|---|
| recall@8 of Exact products | 64.8% |
| recall@8 of Exact or Partial | 88.2% |
| share of Irrelevant products in the top 8 | 5.2% |
| MRR of the first Exact hit | 69.7% |
| queries with no results | 0 |

Recall@8 is the number of a query's Exact products among the first eight results over the
smaller of eight and the number it has; with dozens of Exact products for most queries it
reads as the share of the eight results that are Exact. The weakest queries are the ones
whose words say more than the product names do: "queen wingback chair", "tall buffet
cabinet", "small wardrobe grey", "46 inch closet door".

## Delivery, pickup and the clock

- A product of 50 lb or more ships by freight ($49 per order, 1 to 3 weeks, delivery by
  appointment), and an order with one such product has that option alone: 2,185 products of
  the default build. Everything else ships as a parcel: standard ($5.99, free over $49, 3-5
  business days), express ($9.99, 2 business days) or free pickup at the customer's saved
  showroom, ready two hours after the order within showroom hours (9 AM to 9 PM). A product
  without a weight is a parcel. `data/policies.json` states the same terms, and a test holds
  the two together.
- The backend reads no clock of its own. The time is the session's (`now_of`): what the host
  gives the session, and for a session without a clock 10 AM on the day the orders are dated
  from (`orders_anchored_to` in the manifest; 2026-11-16, a Monday, in the default build).
  The orders are served with the dates they were written with. A run of the harness
  therefore sees the same pickup time and the same order dates on any day it is made.

## What the labels give the harness

The gold file, `data/queries_gold.json`, lists for each query the product ids under each
label, under the letters the ESCI example's gold file uses so that one reader serves both:

- **E** (Exact): what a search for the query should surface.
- **S** (Partial): the alternatives, close to the query without meeting all of it.
- **I** (Irrelevant): what must not be presented as a match for it.
- **C**: always empty; WANDS has no complement label.

## Configuration

`api/agent_config.py` names the deployment (ACME Home), adds the id pattern (`B0` and eight
more characters) to the catalog grounding gate, and tells the model through
`domain_search_notes` what the catalog holds, that `color`, `material` and `style` are
attribute filters, and what a category looks like.

## Layout

| Path | Contents |
|---|---|
| `api/wands_backend.py` | `WandsStorefront`, the `StorefrontBackend` over the database |
| `api/agent_config.py`, `api/main.py` | the deployment's config and the FastAPI app |
| `api/tests/` | tests over a synthetic, WANDS-shaped sample in `tests/fixtures/wands_sample/` (fictional products; no dataset needed): `pytest examples/wands_home/api/tests` |
| `pipeline/select.py` | query eligibility, the screening rules (the ESCI example's), deterministic sampling |
| `pipeline/extract.py` | the three tab-separated files to the raw JSONL files |
| `pipeline/text.py` | WANDS's tokenized, lowercased text back to names and sentences |
| `pipeline/catalog.py` | records, the hidden columns, synthesis, the SQLite and FTS5 writer, the gold and order files |
| `pipeline/search_eval.py` | the search quality report |
| `data/users.json`, `data/memory-seed.json` | the ESCI example's two demo profiles, unchanged |
| `data/policies.json` | the store's terms as this backend serves them (freight, showroom pickup) and five buying guides for kinds this catalog carries (area rugs, desks, sofas and sectionals, outdoor furniture, lighting) |

## For the research harness

This example is the world, not the subject. Mount `data/` read-only into the scoring
service, keep it outside the developer agent's workspace, and treat `catalog.sqlite` as
frozen for a run: the manifest's seed, parameters, source digests and `records_sha256`
identify it. The package imports the ESCI example's screening rules
(`esci_retail.pipeline.select`), so both examples are on the path together.

## Limits

- Prices are synthesized by class or department and are not the products' real prices.
- A product's category is its first class; a product filed under several classes is found
  under the others by the category filter, not by its `category` field.
- The trigram index needs SQLite 3.34 or newer (Python's bundled SQLite on current macOS and
  Linux is).
- The source files are downloaded data: only ever read, by path, never imported or executed.
