# `experiment/`: the repair system and its evaluation

A run gives the repair system the failed customer conversations of one target capability.
The system writes a repair specification, acceptance tests and an update of the assistant.
An update that passes the acceptance tests and the safety suite is a candidate for release.
The architecture (`--arrangement`) decides which release check examines the candidate. The
backend condition (`--world`) decides whether a repair exists at all.

| Module | Contents |
|---|---|
| `run.py`, `loop.py`, `config.py` | One run: its command line, the round protocol, and the configuration with every budget and threshold |
| `roles_evolve.py` | The instructions of each agent: developer, evaluator, orchestrator, monitor and red team |
| `failures.py` | The failed customer conversations, by target capability |
| `compare.py` | The protected checker of DRT. It conducts each differential test with the assistant before and after the update, records every backend call at the backend, replays each call without each optional argument, and applies the criteria C1 and C2 |
| `facade.py`, `claims.py`, `wands.py` | The outcome evaluation: the held-out customer questions of each target capability, what the code reads from a probe conversation, and the facts and label definitions that the two judges are given |
| `facade_rate.py` (`classify`) | Puts the held-out questions to one version of the assistant and records each conversation |
| `outcomes.py` | Writes the labelled runs of `results/labels/` (`python -m experiment.outcomes loops <runs> --out <stem>`) |
| `sensitivity.py` | Recomputes every recorded decision of the checker under other thresholds |
| `start_tree.py`, `compare_trees.py`, `calibrate_wands.py` | Checks made before the experiments: the unchanged assistant probed in each backend condition, and the criteria of the checker and the labels tried on assistants whose label is known |
| `labelsheet.py`, `HUMAN-CHECK.md` | A human check of the judges' labels: the blind sheets and their scoring |
| `jobs.py`, `jobs_wands.py`, `lane.sh`, `cost_cap.py` | The generators of the command lines in `paper/jobs/`, the launcher that runs the lines of one file, and a watch that stops the runs above a spending cap |
| `workspace.py`, `sandbox.py`, `sandbox_audit.py` | The directory layout of a run, the sandbox under which every command of an agent runs, and an audit of what each agent session tried to read |
| `run_eval.py`, `evaluation.py`, `cases_io.py`, `report.py` | The running of test cases (for the agents and for the acceptance decision), the test-case files, and the report of a run |

[`DATA.md`](DATA.md) describes the files that a run writes and the fields of the result
files. [`../paper/GLOSSARY.md`](../paper/GLOSSARY.md) gives the term of the paper for each
name used in the code.

## The round protocol

The option `--revision` selects the version of the round protocol. The default is 4, and
every run of the paper uses it. In an architecture with an evaluator, the developer first
writes the repair specification in a session of its own. The evaluator then writes the
acceptance tests from the failed conversations and that specification. The developer then
changes the assistant.

Lower values of `--revision` select earlier versions of the protocol, which no run of the
paper uses. Some comments in the code give measurements from development runs with those
versions as the reason for a setting, such as a session budget. Those runs are not part of
the results.

## Code that the experiments of the paper do not use

`loop.py` has a second input mode, `--source spec`, in which the repair system receives a
feature specification instead of failed conversations. The experiments of the paper use
`--source failures` only. The specifications and the hidden test cases of the other mode are
not in this repository, so that mode cannot be run from it.

The following code belongs to that mode. It remains because the loop, the labelling or the
calibration imports helpers from it: `roles.py`, `audit.py`, `spec_audit.py`, `contract.py`,
`build_reference.py`, most of `arms.py` and `calibrate_subject.py`, the building half of
`facade_rate.py`, and the hidden-case measurements of `evaluation.py`. The docstrings of
these modules use the vocabulary of that mode ("Experiment A", "infeasible feature", "gold",
the modes `coupled` and `isolated`).

`harness/worlds.py` also defines backend variants that refuse a request. No condition of
the paper uses them.
