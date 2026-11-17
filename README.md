# Catching Capability Facades: Differential Red Teaming for Self-Evolving Shopping Assistants

This repository contains the code, the experiment definitions and the results of the paper
*Catching Capability Facades: Differential Red Teaming for Self-Evolving Shopping
Assistants*, which is under review.

The paper studies self-evolving systems in which LLM agents repair a shopping assistant
from failed customer conversations. It identifies the *capability facade*: an update that
makes the assistant claim to satisfy a request parameter that its backend does not support.
It proposes *differential red teaming* (DRT), a release check that detects such updates
before they are released.

## Contents

| Path | Contents |
|---|---|
| [`experiment/`](experiment/) | The repair system, the release checks of the seven architectures (DRT is one of them), and the outcome evaluation |
| [`harness/`](harness/) | The backend conditions, the code that conducts and grades a conversation with the assistant, the shell in which the agents work, and the tests |
| [`assistant/`](assistant/) | The deployed shopping assistant, from which every run starts |
| [`backend/`](backend/) | The two shopping backends, built from Amazon ESCI and Wayfair WANDS |
| [`failures/`](failures/) | The failed customer conversations that the repair system is given |
| [`paper/`](paper/) | The analysis scripts, and in `jobs/` the command line of each of the 2,352 runs |
| [`results/`](results/) | The labelled outcome of every run, and the numbers and tables computed from the outcomes |

The transcripts of the runs take several gigabytes and are not included.
[`experiment/DATA.md`](experiment/DATA.md) describes every file in `results/` and every
file that a run writes.

## From the paper to the code

| In the paper | In this repository |
|---|---|
| The repair system: orchestrator, developer and evaluator; rounds, release, withdrawal, round budget | `experiment/loop.py` (the rounds), `experiment/roles_evolve.py` (the instructions of each agent), `experiment/config.py` (the budgets) |
| Acceptance tests and the fixed safety suite | `experiment/run_eval.py` (running the tests), `experiment/evaluation.py` (`gate`: whether a candidate passes), `experiment/workspace.py` (`SAFETY_CASES`) |
| Differential probing: the tests of the red-team agent, the protected checker, the recording and the replay of backend calls | `experiment/compare.py` |
| Detection criteria C1 and C2, claim verification, the release decision | `experiment/compare.py`: `unused_inputs` (C1), `unreached` (C2), `rubric` and `value_rubric` (the claim rubric), `verdict`; `experiment/loop.py`: `compare_check` |
| The shopping backends and the 17 backend conditions | `backend/examples/esci_retail/` and `backend/examples/wands_home/` (the backends), `harness/worlds.py` (the conditions) |
| The failed customer conversations | `failures/`, `experiment/failures.py` |
| The baselines and the Oracle | `experiment/config.py` (the seven architectures), `experiment/loop.py` (`monitor_check`, `redteam_check`, `sealed_check`). The option `--arrangement` selects the architecture |
| The experimental protocol and the 2,352 runs | `paper/jobs/`: one command line per run (`R-*.jobs` on ESCI, `W-*.jobs` on WANDS), and the condition, architecture, repair model and seed of each run (`R-runs.tsv`, `W-runs.tsv`) |
| Outcome evaluation: the held-out customer questions, the rubric of the judge, the five outcomes | `experiment/facade.py` and `experiment/wands.py` (the questions), `experiment/claims.py` (the rubric and the labels), `paper/endings.py` (the outcome of a run) |
| Table 1 | `paper/endings_table.py`, which writes `results/tables/tab_endings.tex` (ESCI) and `results/tables/tab_endings_wands.tex` (WANDS) |
| The rates, p-values and confidence interval in the abstract and in the results | `paper/against_each.py` and `paper/stats.py`. The output is in `results/both-against-each.txt`, `results/esci-against-each.txt` and `results/wands-against-each.txt` |

## Names used in the code

The code, the command lines and the result files use names of their own for several terms
of the paper. [`paper/GLOSSARY.md`](paper/GLOSSARY.md) has the complete list.

| In the code and the result files | In the paper |
|---|---|
| arrangement (`--arrangement`) | architecture |
| `solo`, `pair`, `isolated`, `monitor`, `redteam` | Solo Agent, Paired, Isolated, Monitor, RedTeam |
| `compare` | DRT |
| `sealed` | Oracle |
| role model (`--role-model`); `lu`, `ds` | repair model; GPT-5.6-Luna, DeepSeek-V4.1-Flash |
| subject (`--subject-model`) | the assistant |
| store; world (`--world`) | backend; backend condition |
| `ign`, `non` | parameter-ignoring backend, parameter-absent backend: no repair exists |
| `cap` | capable backend: a repair exists |
| topic, feature | target capability |
| `F3`, `F8`, `F9`, `F10` | on ESCI: ZIP-specific delivery quotes, best-selling products, products on promotion, store-specific inventory |
| `F21`, `F22`, `F23`, `F24` | on WANDS: products within a width limit, delivery by a specified date, new arrivals, inventory at a specified showroom |
| comparison | differential test |
| first rule, `unused`; second rule, `unreached` | criterion C1 (unused argument); criterion C2 (unreached parameter) |
| held | rejected by the release check |
| `false claim shipped` | a facade was released |
| `truth shipped` | a disclosure was released |
| `real shipped` | the real fix was released |
| `other shipped` | another update was released (Other) |
| `withdrawn`, `cancelled` | no release: the system withdrew, or it exhausted the round budget |
| reading (`results/*-reading.*`) | the outcome of every run on one backend |
| `R-0001` to `R-1088`, `W-0001` to `W-1264` | the runs on ESCI, the runs on WANDS |

## Reproducing the numbers of the paper

The numbers of the paper are computed from the labelled outcomes in `results/labels/`. The
commands below need Python 3.10 or later and no other package.

```bash
# The outcome of every run
python3 paper/reading.py results/labels/esci-[0-9].json --json results/esci-reading.json > results/esci-reading.txt
python3 paper/reading.py results/labels/wands-[0-9].json --json results/wands-reading.json > results/wands-reading.txt

# Table 1: the ESCI rows, then the WANDS rows
python3 paper/endings_table.py --reading results/esci-reading.json --out results/tables/tab_endings.tex --where "on the Amazon ESCI backend"
python3 paper/endings_table.py --reading results/wands-reading.json --out results/tables/tab_endings_wands.tex --label tab:endings-wands --where "on the Wayfair WANDS backend"

# DRT against each other architecture over both backends: the rates, the p-values and the
# confidence interval of the abstract
python3 paper/against_each.py results/esci-reading.json results/wands-reading.json

# The same comparison on each backend
python3 paper/against_each.py results/esci-reading.json
python3 paper/against_each.py results/wands-reading.json
```

The first four commands write files that are already in `results/`, with the same content,
so `git status` shows no change afterwards. The last three print their result, which is
also stored in `results/both-against-each.txt`, `results/esci-against-each.txt` and
`results/wands-against-each.txt`. For the facade rates, the output gives Fisher's exact
test of DRT against each baseline, with Holm's correction over the five baselines. For the
real-fix rates, it gives the difference between DRT and each other architecture, with its
95% Newcombe interval.

## Running the system

A run needs macOS, because the shell commands of the agents run under `sandbox-exec`. It
also needs Python 3.12, and API keys for Anthropic, OpenAI and DeepInfra. Set up from the
root of this repository:

```bash
# 1. Assemble the backend in ../commerce-agents, with its Python environment and the two
#    catalogs, as backend/README.md describes

# 2. Put the starting repository of the assistant beside it
cp -R assistant ../commerce-agents-subject

# 3. Make the two copies of the backend that the command lines of the runs name
mkdir -p runs
rsync -a --exclude .venv --exclude .git ../commerce-agents/ runs/world-esci-e57457b/
rsync -a --exclude .venv --exclude .git ../commerce-agents/ runs/world-wands-f4b4055/

# 4. Write ANTHROPIC_API_KEY, OPENAI_API_KEY and DEEPINFRA_API_KEY into a file named .env
```

Then:

```bash
# The interpreter of the backend, as an absolute path: the sandbox refuses a path with ".."
PY="$(cd ../commerce-agents/.venv/bin && pwd)/python"

# The tests, which call no model
"$PY" -m pytest harness/tests -q

# An offline run with scripted agents, which calls no model and needs no API key
PYTHONPATH=. "$PY" -m experiment.run --name dry --source failures --arrangement pair --dry-run --rounds 2 --features F1 --subject-model dry --min-cases 1 --end-when-resolved --no-adjudication --gold-rounds never

# One run: each line of a file in paper/jobs/ is the command line of one run
PYTHONPATH=. "$PY" -m $(head -1 paper/jobs/R-01.jobs)

# All runs of one file, one after another
experiment/lane.sh R-01 paper/jobs/R-01.jobs

# After the runs on a backend (R is ESCI, W is WANDS): put the held-out questions to the
# final assistants, label the replies, and compute the outcomes. This replaces
# results/esci-reading.* with the outcomes of the new runs.
paper/pipeline.sh R
```

A run writes its transcripts and logs to `runs/<name>/`.

## Documentation

| Document | Contents |
|---|---|
| [`experiment/README.md`](experiment/README.md) | The modules of `experiment/` |
| [`experiment/DATA.md`](experiment/DATA.md) | The files in `results/` with their fields, and the files that a run writes |
| [`paper/GLOSSARY.md`](paper/GLOSSARY.md) | Every name used in the code, with the term of the paper |
| [`backend/README.md`](backend/README.md) | How to assemble the backend and build the two catalogs |

## Third-party material and licenses

The assistant and the services behind its tools come from the open-source
[Claude Commerce Agents](https://github.com/anthropics/commerce-agents) (Apache-2.0), at
commit `fd4d59224ab96b43c6dc6888207c67b3bd5a24cf`.

- `assistant/` is a reduced copy of that project's shopping assistant (`shopping-agent/`,
  with the shared library `commerce-common/`), with its `LICENSE` beside it. Four flows are
  removed: planning, purchase research, shopper memory, and the returns section of customer
  care. In a backend condition whose service has the request parameter, the harness first
  adds that parameter to the backend interface of the assistant, as a repository written
  against such a backend would show it (`harness/worlds.py`, `PATCHES`).
- `backend/` holds the two stores that this work adds to that project: `esci_retail` (6,191
  products of Amazon ESCI) and `wands_home` (7,625 products of Wayfair WANDS). Nothing else
  of the project is changed. The datasets and the built catalogs are not in this repository.
- `harness/mini/` is a vendored core of mini-swe-agent (MIT, `harness/mini/LICENSE`).

The code of this repository is under the Apache License 2.0 (`LICENSE`). The catalogs are
built from the Amazon Shopping Queries Dataset (Apache-2.0) and the Wayfair WANDS dataset
(MIT).
