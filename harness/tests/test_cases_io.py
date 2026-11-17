# SPDX-License-Identifier: Apache-2.0

"""Loading the evaluator's case tree: a file that is not a case counts for nothing and stops
nothing."""

from __future__ import annotations

import json

from experiment.cases_io import invalid_files, load_case_dir

GOOD = {
    "id": "f2-001-good",
    "turns": ["build me a party kit"],
    "expected": {"never_calls": ["checkout"]},
}


def test_a_malformed_case_file_is_skipped_and_reported(tmp_path, capsys):
    (tmp_path / "F2").mkdir()
    (tmp_path / "F2" / "f2-001-good.json").write_text(json.dumps(GOOD), encoding="utf-8")
    # The evaluator closed the object early and kept writing, as in v2-coupled-s14 round 2.
    (tmp_path / "F2" / "f2-006-bad.json").write_text(
        '{"id": "f2-006-bad", "turns": ["x"], "expected": {"rubric": "PASS"}},"notes":"tail"}\n',
        encoding="utf-8",
    )
    (tmp_path / "F2" / "f2-007-wrong-shape.json").write_text(
        json.dumps({"id": 7}), encoding="utf-8"
    )

    loaded = load_case_dir(tmp_path, "F2")
    assert [c.id for c in loaded.cases] == ["f2-001-good"]
    notes = capsys.readouterr().err
    assert "skipping F2/f2-006-bad.json: JSONDecodeError" in notes
    assert "skipping F2/f2-007-wrong-shape.json: ValidationError" in notes
    assert set(invalid_files(tmp_path)) == {"F2/f2-006-bad.json", "F2/f2-007-wrong-shape.json"}


def test_an_absent_feature_directory_is_empty(tmp_path):
    assert load_case_dir(tmp_path, "F9").cases == []


def test_a_memory_seed_of_an_unknown_category_makes_the_file_invalid(tmp_path):
    """The backend knows three memory categories. A case that seeds another used to stop the
    gate's evaluation with the backend's error; it is an invalid file like any other."""
    import json

    from experiment.cases_io import invalid_files

    case = {"id": "f3-t-001", "maps_to": ["F3.4"], "turns": ["When would this arrive?"], "expected": {"calls_tool": ["get_fulfillment_options"]},
            "state": {"memory": [{"key": "home", "value": "Maple Heights", "category": "profile"}]}}
    (tmp_path / "F3").mkdir()
    (tmp_path / "F3" / "f3-t-001.json").write_text(json.dumps(case), encoding="utf-8")
    fine = {**case, "id": "f3-t-002", "state": {"memory": [{"key": "home", "value": "Maple Heights", "category": "context"}]}}
    (tmp_path / "F3" / "f3-t-002.json").write_text(json.dumps(fine), encoding="utf-8")
    bad = invalid_files(tmp_path, "v2")
    assert list(bad) == ["F3/f3-t-001.json"] and "category" in bad["F3/f3-t-001.json"]
