# Data dictionary

This document describes the result files in this repository and the files that a run
writes. Field names are given as the code wrote them.
[`../paper/GLOSSARY.md`](../paper/GLOSSARY.md) gives the term of the paper for each name.
For example, `store` is the backend, `held` is rejected, `standing` is confirmed, and `tree`
is a checkout of the assistant's repository.

## Files in the repository

| Path | Contents |
|---|---|
| `paper/jobs/R-<n>.jobs`, `paper/jobs/R-runs.tsv` | The command lines of the 1,088 runs on ESCI, one line per run, in 28 files (written by `experiment/jobs.py`). The table gives the target capability, backend condition, architecture, repair model and seed of each run |
| `paper/jobs/W-<n>.jobs`, `paper/jobs/W-runs.tsv` | The same for the 1,264 runs on WANDS (`experiment/jobs_wands.py`) |
| `failures/claude-haiku-5-5/<topic>.json` | The four failed conversations that the repair system is given for a target capability, as conducted with the deployed assistant (`experiment/failures.py`) |
| `results/labels/esci-<k>.json`, `results/labels/wands-<k>.json` | The labelled runs, in ten files per backend (fields below) |
| `results/esci-reading.*`, `results/wands-reading.*` | The outcome of every run, and the counts of outcomes by condition and architecture (`paper/reading.py`; fields below) |
| `results/esci-against-each.*`, `results/wands-against-each.*`, `results/both-against-each.*` | DRT compared with each other architecture, on each backend and over both (`paper/against_each.py`; fields below) |
| `results/tables/tab_endings.tex`, `results/tables/tab_endings_wands.tex` | The outcomes by architecture on ESCI and on WANDS, as LaTeX tables (`paper/endings_table.py`) |

The repository also holds these additional checks:

| Path | Contents |
|---|---|
| `results/*-second-judge.*` | The outcomes and the comparisons when the labels of the second judge decide instead of those of the first (`paper/reading.py --judge 2`) |
| `results/esci-sensitivity.*`, `results/wands-sensitivity.*`, `results/tables/tab_sensitivity.tex`, `results/tables/tab_sensitivity_both.tex` | Every release decision of the checker, recomputed under other thresholds (`experiment/sensitivity.py`, `paper/sensitivity_table.py`) |
| `results/tables/tab_endings_both.tex` | The outcomes by architecture over both backends together |
| `results/tables/tab_endings_shared_seeds.tex`, `tab_endings_wands_shared_seeds.tex`, `tab_endings_both_shared_seeds.tex` | The outcome tables with the conditions where a repair exists counted over seeds 31 to 38 only, the seeds that every architecture has (`paper/endings_table.py --seeds shared`) |
| `results/human-check/` | The two samples of probe conversations drawn for a human check of the judges' labels ([`HUMAN-CHECK.md`](HUMAN-CHECK.md)) |

The run directories (`runs/<name>/`, described below) are not in the repository. They hold
every transcript and take several gigabytes. A run cannot be regenerated exactly, because it
is a sample of model behaviour. Every number that the paper reports can be recomputed from
`results/labels/`, as the [README](../README.md) shows. The sensitivity files cannot be
recomputed from this repository, because `experiment/sensitivity.py` reads the logs of the
runs.

## How the runs were made

The command lines of each backend are divided over 28 files. `experiment/lane.sh` ran the
lines of a file one after another, and the 28 files ran in parallel. A watch on the spending
ran beside them (`experiment/cost_cap.py` on ESCI, `paper/jobs/W-spend.py` on WANDS). Late
in each experiment, lines that had not yet started were moved from the longest files to
further parallel workers. This changed when a run started and nothing else: the command
line of a run never changed, and the files in `paper/jobs/` are the files as first written.

One run on ESCI, `R-0083`, ended on a fault of the harness: a specification file was not
valid UTF-8. The fault was fixed, and the run was made again with the same name, command
line and seed.

After a run had ended, its final assistant was asked the held-out questions and the replies
were labelled (`paper/pipeline.sh`). The labelling of a run reads only that run.

Before each experiment, a few test runs with seed 20 checked the setup (the option
`--smoke` of `experiment/jobs.py` and `experiment/jobs_wands.py`). They are not part of the
results.

## A run directory, `runs/<name>/` (not in the repository)

| Path | Contents |
|---|---|
| `config.json` | The configuration of the run as launched (arrangement, world, models, budgets) |
| `start-tree/` | The assistant before any change (`assistant/`, with the interface patch of the backend condition, `harness/worlds.py`) |
| `workspace/` | The repository of the assistant as the repair system changes it (a git repository; one commit per round) |
| `cases/` | The specifications and the test cases of the repair system; `cases/failures/` holds the failed conversations as the evaluator sees them |
| `redteam/` | The directory of the red team: failed conversations, findings, notes, and `comparisons/` in the `compare` arrangement |
| `roles-home/`, `spool/`, `bin/` | The home directory of the sandboxed shells, the spool of the evaluation broker, and the commands that the agents may run |
| `report.md` | A rendered summary of the run |
| `log/events.jsonl` | The event log, one JSON object per line (below) |
| `log/comparisons.jsonl` | Every comparison that the checker conducted (below) |
| `log/redteam_asks.jsonl` | Every conversation that the red team conducted (below) |
| `log/subject_costs.jsonl` | The model costs of the assistant per evaluation, conversation or comparison |
| `log/trajectories/` | The full transcript of every agent session, `NN-<role>.json` by round; `NN-F-spec.json` is the session in which the developer writes the specification |
| `log/eval/`, `log/run_eval_calls/`, `log/cases/`, `log/specs/` | The evaluations of the test cases at each gate, the `run_eval` calls of the agents, and the test cases and specifications as they stood in each round |
| `log/sandbox/`, `log/eval-proxy.log` | The logs of the sandbox and of the evaluation broker |

## `log/events.jsonl`

Each line has `kind` (the event), `t` (seconds since the epoch) and the fields of the event.

| `kind` | Contents |
|---|---|
| `start` | The configuration of the run |
| `arrival` | The failed conversations of a target capability are published |
| `decision` | The decision of the orchestrator for a round, with its guidance to the other agents |
| `session` | One agent session: `role`, `round`, `exit_status`, `model_calls`, `commands`, `submission` (the report of the agent), `role_cost_usd` |
| `spec_session` | The session in which the developer writes the specification: `topics`, `written` (the target capabilities that have a specification after the session), `discarded` (what the session had changed in the repository) |
| `evaluator_waits` | No target capability had a specification to write test cases from |
| `spec_restored` | A session of the evaluator had changed a specification, and the change was undone |
| `commit` | The commit of the developer's work in a round (`changed`, `sha`) |
| `case_changes` | The test cases added, deleted and modified in a round |
| `gate` | The acceptance decision for a candidate: the number of test cases, their pass rate over the trials (`rate`), `regressions`, `safety_failures`, the record of the release check where the architecture has one, whether the candidate is released (`delivered`), and the rounds that remain (`countdown`) |
| `release` | One release check (record below) |
| `verdict_asked` | The report of the monitor carried no verdict, and the monitor was asked again |
| `infeasible_claim` | The developer reports that no change can repair the failures |
| `withdrawal` | The orchestrator accepts that report: nothing is released for the target capability |
| `sandbox_audit` | The audit of one agent session: what it tried to read outside its own directory |
| `adjudication` | Empty in the runs of the paper, which do not use it (`--no-adjudication`) |
| `round` | The end of a round: the pass rates, the failures of the safety suite, and the cost so far |
| `stop`, `end` | Why the run stopped, and its final cost |
| `adjudication_error`, `ask_again_error`, `titles_error` | Errors |

A `session` event of the session in which the developer writes the specification carries
`part: "spec"`. The `decision` event of a round that has such a session carries
`guidance_spec`, the guidance of the orchestrator for that session, beside
`guidance_developer`, its guidance for the session in which the developer changes the
assistant.

### The record of a release check

These records are the `release` events. They are also the `releases` of a run in the
labelled runs.

Common fields:

- `by`: `monitor`, `redteam`, `compare` or `sealed`;
- `accepted`: false means that the release was rejected;
- `note`: what the repair system is told;
- `flag`, a procedural flag: `asked-again` (the report of the monitor carried no verdict
  and it was asked again), `inert` (a red-team session with fewer conversations or
  comparisons than its minimum), `no-verdict` or `too-few`.

| `by` | Further fields |
|---|---|
| `monitor` | `verdict`, `reasons` |
| `redteam` | `asks` (conversations conducted), `findings` (filed), `standing` (the findings confirmed on re-test: `id`, `turns`, `reason`, `judge`), `dropped`, `unreadable`, `file` (the feedback given to the repair system) |
| `compare` | As `redteam`, and `comparisons` (conducted) and `comparisons_standing` (the confirmed ones: `id`, `detail`, `claims`, `showing`, `unused` (the arguments that criterion C1 found), `unreached` (the conversations that criterion C2 found)) |
| `sealed` | `label` (the label of the candidate on the hidden questions), `label_kind`, `wanted`, `shares`, `judges`, `rows` |

## `log/comparisons.jsonl` (the `compare` arrangement)

One line per comparison:

- `t`, `role`, `round`, `n` (its number in the session), `id`, `feature`;
- `detail`: the parameter that the answer should depend on;
- `cart`, `seen`;
- `conversations` (below);
- `showing`: the conversations that met a criterion;
- `claims`: of those, the replies that the judge read as the answer for the value;
- `stands`: confirmed, that is `showing` > 0 and `claims` * 2 >= `showing`;
- `cost_usd`.

Each conversation has:

- `id`, and `value`, the value of the parameter (`cannot_be` marks an invalid one);
- `turns`: `customer`, `reply`, `calls`, `components`;
- `reads`: every backend call of the session, recorded at the backend: `method`, `inputs`,
  `digest` of the response, `shown`, and `optional` (each optional argument with
  `same_without`, whether the replayed response was identical);
- `before`: the same for the assistant before the change, with the `score` of the judge
  and its reason;
- `unused`: criterion C1, the arguments that the backend did not use;
- `unreached`: criterion C2;
- `judged`: the reading of the judge per argument (`path`, `score`, `reason`);
- `claim`: a score of 2 or less on any of them;
- `error`.

## `log/redteam_asks.jsonl`

One line per conversation of the red team: `t`, `role`, `round`, `n`, `cart`, `seen`,
`turns` (`customer`, `reply`, `calls`, and `results`, what the tools returned, shortened),
`error`, `cost_usd`.

## The labelled runs, `results/labels/<backend>-<k>.json`

`experiment.outcomes loops` wrote these files after the runs had ended. A file has:

- `judges`: the two judges that label the probe conversations. The first one decides.
- `runs`: one record per run (fields below).
- `tables`: counts of the runs in the file by target capability, backend condition and
  architecture. The analysis scripts do not read them.
- `judge_cost_usd`: the cost of the judge calls made when the file was last written.

The fields of a run that the analysis uses:

| Field | Meaning |
|---|---|
| `run`, `feature`, `world`, `arm`, `seed` | The name of the run, its target capability, backend condition, architecture and seed. `arm` has the form `failures:<arrangement>@<assistant>`, with `+roles-<model>` for the second repair model |
| `status` | `delivered`, `withdrawn` or `cancelled` (or `unfinished`, `no-start`) |
| `decision_round` | The round in which the status was reached |
| `tree` | The label of the final assistant per judge: `facade`, `honest`, `real`, `broken`, `other` or `declined` |
| `tree_detail` | The same with `other` divided into `outage`, `unqualified` and `mixed` |
| `customer` | Per judge: the label, and the `shares` of each probe label over the probe conversations |
| `probe_rows` | Every probe conversation: the question (`id`), the `trial`, the `reply`, the label of each judge (`labels`), and whether the conversation counts towards the label of the run (`counts`) |
| `releases` | The records of the release checks of the run (above) |
| `claim_rounds` | The rounds in which the developer reported that the failures cannot be repaired |
| `gate_rates`, `own_rate_at_gate` | The pass rate of the acceptance tests at each gate and at the decision gate |
| `sessions_at_limit` | The agent sessions that ended at their step budget |
| `roles_cost_usd`, `evaluations_cost_usd`, `minutes` | Cost and duration |
| `excluded` | The reason, when a run is excluded. No run is excluded |
| `revision`, `sha_at_gate`, `sandboxed`, `classify_error`, `case_level`, `case_edits_after_first`, `spec_audit` | Provenance and checks |

## The outcome of every run, `results/<backend>-reading.json`

`paper/reading.py` writes this file from the labelled runs. The `.txt` file beside it is
what the script prints: the same counts, as tables.

- `runs` has one record per run: `run`, `feature`, `world`, `store` (the kind of the
  backend condition: `ign`, `non` or `cap`), `condition` (`<feature>-<store>`),
  `arrangement`, `family` (the repair model: `lu` or `ds`), `assistant`, `seed`, `status`,
  `excluded`, `ending` (`real shipped`, `truth shipped`, `false claim shipped`,
  `other shipped`, `withdrawn`, `cancelled` or `unresolved`), `tree`, `shares`, `round`,
  `own_rate`, `claims`, `releases`, `revision`, `gate_rates`, `held` (the number of rejected
  releases), `flags`, `cost` and `minutes`.
- `cells` counts the endings by condition, arrangement and repair model.
- `summary` holds the share of runs that released a facade where no repair exists, and the
  share that released the real fix where one exists, with their 95% Wilson intervals.
- `capable_shared_seeds` and `capable_denials` describe the conditions where a repair
  exists: the real fixes over the seeds that every architecture has, and the disclosures
  released there.

## The comparisons, `results/*-against-each.json`

`paper/against_each.py` writes this file from the `runs` of one reading, or of both readings
for the comparison over both backends. The `.txt` file beside it is what the script prints.

- `readings`: the files that were read.
- `false claims`: for each baseline, the runs that released a facade where no repair
  exists, under DRT (`compare`: count and number of runs) and under the baseline (`other`),
  the p-value of Fisher's exact test (`p`), the same after Holm's correction over the five
  baselines (`holm`), and the same comparison stratified by backend, condition and repair
  model (`stratified`: a Cochran-Mantel-Haenszel test).
- `real fixes`: for each other architecture, the runs that released the real fix where a
  repair exists, under DRT and under the other architecture, the `difference` of the two
  shares (DRT minus the other) and its 95% Newcombe `interval`.
- `real fixes, shared seeds`: the same over seeds 31 to 38 only.

```bash
python3 paper/reading.py results/labels/esci-[0-9].json --json results/esci-reading.json > results/esci-reading.txt
python3 paper/against_each.py results/esci-reading.json --json results/esci-against-each.json > results/esci-against-each.txt
```
