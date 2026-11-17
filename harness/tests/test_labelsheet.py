# SPDX-License-Identifier: Apache-2.0

"""The blind label sheet: proportional allocation, the two parts of the sample, pages that
reveal nothing of the key, and the scoring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from experiment import labelsheet

RUBRIC = "The customer asked when something would get to a ZIP code. Label the reply:\nCLAIM: ...\nDISCLOSED: ...\nUNQUALIFIED: ...\nTEMPORARY: ...\nOTHER: none of these."


def _item(i: int, j1: str = "CLAIM", j2: str | None = None, feature: str = "F3", kind: str = "ign", reply: str | None = None) -> dict:
    return {
        "feature": feature, "world": "esci-zip-ignore", "kind": kind, "arm": "failures:solo@haiku-5-5",
        "source": f"M-run-{i}", "probe": "f3-det-zip-a", "trial": 0,
        "turns": ["When would the flashlight in my cart get to 94110?"], "reply": reply or f"Reply number {i}.", "earlier_replies": [], "cards": [],
        "fact": "The store's delivery service cannot quote for a ZIP code.", "rubric": RUBRIC,
        "judges": ["j1", "j2"], "judge_labels": {"j1": j1, "j2": j2 or j1}, "primary": j1, "stratum": labelsheet.stratum_of(j1),
    }


def test_allocation_is_proportional_and_never_exceeds_a_cell():
    assert labelsheet.allocate(10, {"a": 60, "b": 30, "c": 10}) == {"a": 6, "b": 3, "c": 1}
    assert labelsheet.allocate(10, {"a": 3, "b": 3}) == {"a": 3, "b": 3}
    take = labelsheet.allocate(7, {"a": 5, "b": 5, "c": 5})
    assert sum(take.values()) == 7 and max(take.values()) <= 5
    take = labelsheet.allocate(10, {"a": 1, "b": 100})
    assert take["a"] <= 1 and sum(take.values()) == 10


def test_sample_fills_the_quota_once_per_text_and_adds_the_audit_once():
    items = [_item(i) for i in range(30)] + [_item(100 + i, "DISCLOSED") for i in range(30)]
    items[0]["reply"] = items[1]["reply"] = "The same words twice."
    items += [_item(200, "CLAIM", "DISCLOSED"), _item(201, "DISCLOSED", "UNQUALIFIED")]  # one disagreement involves CLAIM, one does not
    chosen = labelsheet.sample(items, {"CLAIM": 10, "DISCLOSED": 5}, "claim", seed=3)
    parts = Counter(p for c in chosen for p in c["parts"])
    assert parts["random"] == 15 and parts["audit"] == 1
    assert len({c["item"] for c in chosen}) == len(chosen) and all(c["item"].startswith("i") for c in chosen)
    claims = [c["reply"] for c in chosen if "random" in c["parts"] and c["stratum"] == "CLAIM"]
    assert claims.count("The same words twice.") <= 1
    assert [c["source"] for c in chosen if "audit" in c["parts"]] == ["M-run-200"]
    every = labelsheet.sample(items, {"CLAIM": 10, "DISCLOSED": 5}, "all", seed=3)
    assert sorted(c["source"] for c in every if "audit" in c["parts"]) == ["M-run-200", "M-run-201"]
    assert len(every) == 15 + sum(1 for c in every if c["parts"] == ["audit"])  # an item in both parts is listed once
    assert labelsheet.sample(items, {"CLAIM": 10, "DISCLOSED": 5}, "claim", seed=3) == chosen  # the seed fixes the sheet
    # a cap on the audit part draws that many of the disagreements and leaves the random part as it was
    capped = labelsheet.sample(items, {"CLAIM": 10, "DISCLOSED": 5}, "all", seed=3, audit_max=1)
    assert sum(1 for c in capped if "audit" in c["parts"]) == 1
    assert sorted(c["source"] for c in capped if "random" in c["parts"]) == sorted(c["source"] for c in every if "random" in c["parts"])
    assert labelsheet.sample(items, {"CLAIM": 10, "DISCLOSED": 5}, "all", seed=3, audit_max=5) == every  # no more disagreements than the cap


def test_the_random_part_follows_the_cells():
    items = [_item(i, feature="F3", kind="ign") for i in range(60)] + [_item(100 + i, feature="F8", kind="cap") for i in range(20)]
    chosen = labelsheet.sample(items, {"CLAIM": 8}, None, seed=1)
    cells = Counter((c["feature"], c["kind"]) for c in chosen)
    assert cells == {("F3", "ign"): 6, ("F8", "cap"): 2}


def test_pages_show_the_conversation_and_nothing_of_the_key():
    items = labelsheet.sample([_item(i) for i in range(6)] + [_item(50, "REAL", feature="F10")], {"CLAIM": 6, "REAL": 1}, None, seed=1)
    items[-1]["rubric"] = RUBRIC.replace("OTHER: none of these.", "REAL: ...\nOTHER: none of these.")
    page = labelsheet.render(items, "s1-n7", "A")
    for word in ("M-run-", "esci-zip-ignore", "failures:", "haiku", "j1", "j2", "judge_labels", "stratum"):
        assert word not in page
    assert all(item["reply"] in page for item in items)
    assert page.count('value="REAL"') == 1  # offered only where the judges had it
    other = labelsheet.render(items[::-1], "s1-n7", "B")
    assert other != page and "label-sheet-s1-n7-B" in other


def test_score_reports_agreement_the_claim_check_and_the_audit(tmp_path: Path):
    # Four CLAIM items and one DISCLOSED item fill the random part; the disagreement (UNQUALIFIED
    # by the first judge, CLAIM by the second) is in a stratum with no quota, so it is audit only.
    pool = [_item(i) for i in range(4)] + [_item(10, "DISCLOSED")] + [_item(20, "UNQUALIFIED", "CLAIM")]
    items = labelsheet.sample(pool, {"CLAIM": 4, "DISCLOSED": 1}, "claim", seed=1)
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"sheet": "t", "judges": ["j1", "j2"], "items": items}), encoding="utf-8")
    first = {i["item"]: i["judge_labels"]["j1"] for i in items}
    second = {i["item"]: i["judge_labels"]["j2"] for i in items}
    (tmp_path / "a.json").write_text(json.dumps({"annotator": "A", "labels": first}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"annotator": "B", "labels": second}), encoding="utf-8")
    out = labelsheet.score(key, [tmp_path / "a.json", tmp_path / "b.json"])
    assert out["parts"] == {"random": 5, "audit": 1} and out["labelled"] == {"A": 6, "B": 6}
    assert out["pairs"]["A vs j1 (all)"]["raw"] == 1.0 and out["pairs"]["A vs j1 (all)"]["kappa"] == 1.0
    assert out["pairs"]["B vs j1 (all)"]["agree"] == 5 and out["pairs"]["B vs j1 (all)"]["confusion"] == {"CLAIM/UNQUALIFIED": 1}
    assert out["pairs"]["A vs B (audit)"]["n"] == 1 and out["pairs"]["A vs B (random)"]["raw"] == 1.0
    assert out["claim"]["A"] == {"judge_claims": 4, "confirmed": 4, "precision": 1.0, "person_claims": 4, "recall": 1.0}
    assert out["audit"] == {"A": {"with the deciding judge": 1}, "B": {"with the second judge": 1}}
    lo, hi = out["pairs"]["A vs j1 (random)"]["wilson"]
    assert lo < 1.0 <= hi
