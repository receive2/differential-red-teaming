# The backends

The shopping assistant and the services behind its tools come from the open-source
[Claude Commerce Agents](https://github.com/anthropics/commerce-agents) (Apache-2.0), at
commit `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf`. This directory holds the two stores that
this work adds to the `examples/` directory of that project. No other file of the project
is changed.

| Path | Store | Catalog |
|---|---|---|
| `examples/esci_retail/` | ACME over ESCI | 6,191 products of the Amazon Shopping Queries Dataset (ESCI), selected by 296 of its queries |
| `examples/wands_home/` | ACME Home over WANDS | 7,625 products of the Wayfair WANDS dataset, selected by 50 of its queries |

The `README.md` of each store gives its provenance and license, what is taken from the
dataset and what is synthesized, and how the catalog is built. A build is deterministic.
`data/manifest.json` records the seed, the selection parameters, the sha256 digests of the
source files and the counts, so that a rebuilt catalog can be checked against the one that
the experiments used. The datasets and the built catalogs are not in this repository.

A store answers every request that it is given. The backend conditions of the experiments
are variants defined in `harness/worlds.py`, which wraps the services of a store: a service
that ignores a request parameter, a service that has no such parameter, and a service that
does what was asked. The stores themselves know nothing of these variants.

## Assembling the backend

Run the following from the root of this repository. It leaves the backend, with its Python
environment and the two catalogs, in `../commerce-agents`.

```bash
git clone https://github.com/anthropics/commerce-agents ../commerce-agents
git -C ../commerce-agents checkout fd4d59224ab96b43c6dc6888207c67b3bd5a24cf
cp -R backend/examples/esci_retail backend/examples/wands_home ../commerce-agents/examples/

cd ../commerce-agents
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt jinja2 pyarrow

# The two catalogs, from the public datasets (the README of each store has the download lines)
PYTHONPATH=examples .venv/bin/python -m esci_retail.pipeline all --source-dir <directory with the two ESCI parquet files>
PYTHONPATH=examples .venv/bin/python -m wands_home.pipeline all --source-dir <directory with the three WANDS files>
```

`jinja2` is needed by the shell in which the agents work (`harness/mini/`), and `pyarrow` by
the extraction of ESCI. Everything else is pinned in the requirements of the upstream
project. The experiments ran on Python 3.12.
