# ACME over ESCI (`esci_retail`)

A storefront whose catalog is a few thousand real listings from the Amazon Shopping Queries
Dataset (ESCI), chosen by real search queries that carry human relevance labels. It exists
for the self-evolving-agent research harness: a store large enough that the agent has to
search rather than remember the catalog, with objective gold for what a search should return.
It is not one of the upstream verticals and is not meant as an upstream contribution: the
listings carry real brands, which the repository's "everything is fictional" rule excludes.

## Provenance and license

- Source: [amazon-science/esci-data](https://github.com/amazon-science/esci-data), Apache-2.0.
  Two parquet files are read: `shopping_queries_dataset_products.parquet` (1.1 GB) and
  `shopping_queries_dataset_examples.parquet` (51 MB). Their sha256 digests are recorded in
  `data/raw/selection.json` and `data/manifest.json` by the build.
- ESCI gives a listing's title, description, bullet points, brand and color, and for each
  query the products a human judged **E**xact, **S**ubstitute, **C**omplement or
  **I**rrelevant. It has no prices, categories, ratings, stock or images; those are
  synthesized here (below), deterministically from a seed.
- The data files are not committed. `data/raw/` and `data/catalog.sqlite` are ignored; the
  small derived files (`queries_gold.json`, `orders.json`, `manifest.json`) are kept so the
  gold is reviewable without the dataset.

## Build

```bash
# 1. The two parquet files, into an empty directory outside the repository (git-lfs media URLs).
mkdir -p ~/Documents/data/esci-data && cd ~/Documents/data/esci-data
B=https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset
curl -sSL -o shopping_queries_dataset_examples.parquet "$B/shopping_queries_dataset_examples.parquet"
curl -sSL -o shopping_queries_dataset_products.parquet "$B/shopping_queries_dataset_products.parquet"
shasum -a 256 *.parquet
#   4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a  ...examples.parquet
#   25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265  ...products.parquet

# 2. From the repository root. Only the extract step needs pyarrow.
pip install pyarrow
PYTHONPATH=examples python -m esci_retail.pipeline all --source-dir ~/Documents/data/esci-data
```

`all` runs the three stages; each is also a command of its own:

| Stage | Command | Reads | Writes |
|---|---|---|---|
| extract | `python -m esci_retail.pipeline extract --source-dir DIR` | the parquet files | `data/raw/products.jsonl`, `data/raw/examples.jsonl`, `data/raw/selection.json` |
| build | `python -m esci_retail.pipeline build` | `data/raw/` | `data/catalog.sqlite`, `data/queries_gold.json`, `data/orders.json`, `data/manifest.json` |
| eval-search | `python -m esci_retail.pipeline eval-search [-k 8]` | the catalog and the gold file | a report on stdout |

Selection parameters (`--n-queries`, `--seed`, `--version small|large`, the thresholds) are
flags of `extract` and `all`; `--help` lists them. The same parameters and seed reproduce the
same store byte for byte.

## Run

```bash
uvicorn esci_retail.api.main:app --app-dir examples --port 8004
curl -s localhost:8004/api/health
curl -s 'localhost:8004/api/products?category=electronics&limit=3'
```

There is no web app and no merchant side; the shared storefront routes (`/api/session`,
`/api/chat`, `/api/products`, `/api/cart`, `/api/orders`, `/api/memory`) are all of it. Chat
needs `ANTHROPIC_API_KEY` as for the other examples. `ESCI_RETAIL_DATA_DIR` points the app
at a catalog built in another directory. Memory is in-process and re-seeded at every boot.

## How the store is chosen

The catalog is chosen by query, not by product: every query in the store comes with every
product a human judged for it, so the store has labelled results for each of its queries.

- Locale `us`, the `small` version (the queries ESCI kept after dropping the easy ones).
- A query qualifies when it has at least 2 Exact and 1 Irrelevant judgments and between 8
  and 40 judgments in all, so no single query floods the catalog.
- Queries and listings are left out when they are not English (two or more Spanish function
  words), not ASCII (a few Korean, Chinese, Arabic and Hebrew queries, and a handful of
  English ones written with curly quotes), or name content a general store's assistant
  should not be evaluated on (sexual content, firearms and similar weapons, tobacco, vaping
  and drugs; the word lists are in `pipeline/select.py`). Bare words that also name ordinary
  goods (gun, weed, lube, nude) are deliberately not on the list.
- 300 queries are drawn with the seed from the 14,759 that qualify; a query is dropped again
  when the product filters remove the judgments it needed.

The default build: 296 queries, 6,191 listings, 6,219 judgments (2,783 E, 1,555 S, 169 C,
1,712 I), 5,690 listings in stock, across 21 categories. `data/manifest.json` has the
counts of the build in place.

## What is synthesized, and how

| Field | Source |
|---|---|
| `title`, `brand`, `long_description`, `specs`, `attributes.color` | ESCI, cleaned: HTML stripped, bullets split on newlines, `Key: value` bullets become specs |
| `short_description` | the first bullet, else the first sentence of the description |
| `category` | a keyword taxonomy of 20 categories over the title (weight 3) and bullets (weight 1); a phrase counts its words, ties go to the earlier category; no hit is `general-merchandise` (14% of the default build) |
| `price` | log-normal per category, median and spread in `pipeline/catalog.py`, rounded to .99 |
| `rating`, `review_count`, `in_stock`, `labels` | hashed from the seed and the product id, so a listing keeps its figures in any build with that seed |
| `image_url`, option families | none; every listing is a plain product |

Prices and ratings are plausible, not real. A listing's figures do not depend on which other
listings were drawn, so two builds with different query samples agree on the products they share.

## Search

`search_products` runs an FTS5 query over a porter-stemmed index of title, brand, category,
bullets and description, ranked by BM25 with column weights 10 / 4 / 2 / 2 / 1; terms of four
or more characters also match as prefixes. When no word matches at all, a trigram index of
title and brand stands in, so a misspelt query still finds its product; a trigram candidate
counts only when some query term shares at least 60% of its trigrams with the listing's title
and brand, so a word that merely shares a few letters with the catalog returns nothing rather
than noise. The repository's
shared ranking rules then apply: price and rating filters, a relevance cutoff at half the
best score, category and attribute filters (`brand`, `color`) that are dropped if they would
empty the result, and the requested sort.

`eval-search` scores this against the ESCI labels (k = 8, default build):

| Measure | Value |
|---|---|
| recall@8 of Exact products | 57.0% |
| recall@8 of Exact or Substitute | 61.8% |
| share of Irrelevant products in the top 8 | 14.5% |
| MRR of the first Exact hit | 73.1% |
| queries with no results | 0 |

Recall is a lower bound: a result judged for a different query counts as a miss. The weakest
queries are the ones that lexical search cannot read, such as "without hitch", "not more
than three dollars" or "$1 items for men"; interpreting those is the agent's job. An
AND-first variant (all terms required, OR matches down-weighted) was measured and was worse
on every figure, so the plain OR ranking ships.

## What the labels give the harness

The gold file, `data/queries_gold.json`, lists for each query the product ids under each
label. They map onto agent behaviors without an LLM judge:

- **E**: what a search for the query should surface (search and discovery).
- **I**: what must not be presented as a match for it (hallucinated or careless matches).
- **S**: the alternatives to offer when the exact product is out of stock (purchase research).
- **C**: what goes with the product (planning and bundles).

Real queries carry misspellings and vague phrasing ("chikdrens watch", "lockers stuff"), a
natural source of hard cases.

## Configuration

`api/agent_config.py` adds an ASIN pattern (`B0` and eight more characters) to the catalog
grounding gate, since the default id patterns do not cover these ids, and tells the model
through `domain_search_notes` that `brand` and `color` are attribute filters. Listings whose
ASIN is an ISBN (digits only; 237 of the 6,191 in the default build) do not trigger the gate;
they still resolve by id.

## Layout

| Path | Contents |
|---|---|
| `api/esci_backend.py` | `EsciStorefront`, the `StorefrontBackend` over the database |
| `api/agent_config.py`, `api/main.py` | the deployment's config and the FastAPI app |
| `api/tests/` | tests over a synthetic, ESCI-shaped sample in `tests/fixtures/esci_sample/` (fictional listings; no dataset needed): `pytest examples/esci_retail` |
| `pipeline/select.py` | query eligibility, exclusion rules, deterministic sampling |
| `pipeline/extract.py` | the parquet stage (pyarrow) and the raw JSONL files |
| `pipeline/catalog.py` | cleaning, taxonomy, synthesis, the SQLite and FTS5 writer, the gold and order files |
| `pipeline/search_eval.py` | the search quality report |
| `data/users.json`, `data/policies.json`, `data/memory-seed.json` | static fixtures: the retail profiles and store terms, with five buying guides written for categories this catalog carries (bike racks, portable air conditioners, wireless chargers, patio umbrellas, flashlights) in place of the retail example's guides |

## For the research harness

This example is the world, not the subject. Mount `data/` read-only into the scoring
service, keep it outside the developer agent's workspace, and treat `catalog.sqlite` as
frozen for a run: the manifest's seed, parameters and source digests identify it.

## Limits

- Category is a keyword heuristic; `general-merchandise` catches what it misses.
- The trigram index needs SQLite 3.34 or newer (Python's bundled SQLite on current macOS and
  Linux is).
- The source files are downloaded data: only ever read, by path, never imported or executed.
