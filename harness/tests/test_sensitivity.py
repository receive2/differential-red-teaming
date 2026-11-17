# SPDX-License-Identifier: Apache-2.0
"""The checker's thresholds recomputed from recorded scores (experiment/sensitivity.py)."""

import json

from experiment.sensitivity import (
    AS_RUN,
    FINDINGS_AS_RUN,
    analyse,
    comparison_under,
    load_run,
    main,
    rejected_by_findings,
)


def _conversation(scores, unreached=False, before=None, unused=True, judged=True):
    return {
        "unused": [{"path": "filters.sort"}] if unused else [],
        "unreached": unreached,
        "before": {"score": before} if before is not None else {},
        "judged": [{"path": "(the value)" if unreached and not unused else "filters.sort", "score": s} for s in scores] if judged else [],
    }


def test_a_comparison_is_recomputed_as_the_tool_decided_it_and_otherwise_under_other_thresholds():
    record = {"conversations": [_conversation([1]), _conversation([3]), _conversation([], judged=False)]}
    assert comparison_under(record, **AS_RUN) == (True, 2, 1)  # one of two claims: at least half
    assert comparison_under(record, threshold=2, vote="all", min_showing=1) == (False, 2, 1)
    assert comparison_under(record, threshold=3, vote="all", min_showing=1) == (True, 2, 2)
    assert comparison_under(record, threshold=1, vote="any", min_showing=1) == (True, 2, 1)
    assert comparison_under(record, threshold=2, vote="half", min_showing=3) == (False, 2, 1)  # too few showing
    # The second rule alone: the reply before the change scored 3, counted as run and not with before_max 3.
    second = {"conversations": [_conversation([2], unreached=True, before=3, unused=False)]}
    assert comparison_under(second, **AS_RUN) == (True, 1, 1)
    assert comparison_under(second, threshold=2, vote="half", min_showing=1, before_max=3) == (False, 0, 0)
    assert comparison_under({"conversations": []}, **AS_RUN) == (False, 0, 0)


def test_the_findings_rule_counts_at_least_six_conversations():
    assert not rejected_by_findings({"asks": 4, "standing": [1]}, **FINDINGS_AS_RUN)  # 1 > 0.2 * 6 is false
    assert rejected_by_findings({"asks": 4, "standing": [1]}, share=0.2, min_asks=1)
    assert rejected_by_findings({"asks": 10, "standing": [1, 2, 3]}, **FINDINGS_AS_RUN)
    assert not rejected_by_findings({"asks": 10, "standing": [1, 2]}, **FINDINGS_AS_RUN)


def _run(tmp_path, name, arrangement, releases, comparisons, ended=True):
    d = tmp_path / name
    (d / "log").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"name": name, "arrangement": arrangement, "features": ["F3"], "world": "esci-zip-ignore"}))
    events = [{"kind": "start", "t": 1.0}, *releases, *([{"kind": "end", "t": 2.0}] if ended else [])]
    (d / "log" / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    if comparisons:
        (d / "log" / "comparisons.jsonl").write_text("".join(json.dumps(c) + "\n" for c in comparisons))
    return d


def test_the_runs_are_read_and_the_decisions_as_run_are_reproduced(tmp_path):
    standing = {"id": "c1", "round": 1, "role": "R", "stands": True, "showing": 1, "claims": 1, "conversations": [_conversation([2])]}
    quiet = {"id": "c2", "round": 2, "role": "R", "stands": False, "showing": 1, "claims": 0, "conversations": [_conversation([3])]}
    releases = [
        {"kind": "release", "round": 1, "by": "compare", "accepted": False, "asks": 6, "standing": []},
        {"kind": "release", "round": 2, "by": "compare", "accepted": True, "asks": 6, "standing": []},
    ]
    a = _run(tmp_path, "R-0001", "compare", releases, [standing, quiet])
    b = _run(tmp_path, "R-0002", "redteam", [{"kind": "release", "round": 1, "by": "redteam", "accepted": False, "asks": 5, "standing": [1, 2]}], [])
    _run(tmp_path, "R-0003", "pair", [], [], ended=False)  # not finished: left out
    runs = [r for r in (load_run(p) for p in (a, b, tmp_path / "R-0003")) if r]
    assert [r["name"] for r in runs] == ["R-0001", "R-0002"]
    labels = {"R-0001": {"tree": "real", "kind": "ign"}}
    result = analyse(runs, labels)
    assert result["consistency"] == {"comparisons_mismatched": [], "releases_mismatched": []}
    as_run = next(row for row in result["comparison_rule"] if row["as_run"])
    assert (as_run["comparisons_standing"], as_run["releases_rejected"], as_run["deliveries_held_back"], as_run["rejections_let_through"]) == (1, 1, 0, 0)
    looser = next(row for row in result["comparison_rule"] if row["threshold"] == 3 and row["vote"] == "half" and row["min_showing"] == 1)
    assert looser["comparisons_standing"] == 2 and looser["deliveries_held_back"] == 1 and looser["held_back_by_label"]["real"] == 1
    stricter = next(row for row in result["comparison_rule"] if row["threshold"] == 1 and row["vote"] == "half" and row["min_showing"] == 1)
    assert stricter["comparisons_standing"] == 0 and stricter["rejections_let_through"] == 1
    findings_as_run = next(row for row in result["findings_rule"] if row["as_run"])
    assert findings_as_run["redteam"]["releases_rejected"] == 1  # 2 > 0.2 * 6
    lenient = next(row for row in result["findings_rule"] if row["share"] == 0.5 and row["min_asks"] == 6)
    assert lenient["redteam"]["releases_rejected"] == 0 and lenient["redteam"]["rejections_let_through"] == 1
    assert result["scores"]["after_by_rule"]["first rule"] == {"2": 1, "3": 1}
    out = tmp_path / "tables" / "sens"
    assert main([str(a), str(b), "--out", str(out)]) == 0
    assert out.with_suffix(".json").exists() and "as run" in out.with_suffix(".md").read_text()
