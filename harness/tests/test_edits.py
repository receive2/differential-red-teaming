# SPDX-License-Identifier: Apache-2.0

import pytest

from harness.edits import apply_edits, copy_tree


def test_edits_replace_delete_write_and_rename(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a/one.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (src / "a/two.md").write_text("keep\n", encoding="utf-8")
    (src / "__pycache__").mkdir()
    (src / "__pycache__/x.pyc").write_bytes(b"\x00")
    dst = copy_tree(src, tmp_path / "dst")
    assert not (dst / "__pycache__").exists()
    apply_edits(
        dst,
        [
            {
                "path": "a/one.md",
                "replace": [{"old": "beta", "new": "BETA"}],
                "delete_lines_containing": ["gamma"],
            },
            {"path": "a/two.md", "rename_to": "a/three.md"},
            {"path": "a/new.md", "write": "fresh\n"},
        ],
    )
    assert (dst / "a/one.md").read_text() == "alpha\nBETA\n"
    assert (dst / "a/three.md").exists() and not (dst / "a/two.md").exists()
    assert (dst / "a/new.md").read_text() == "fresh\n"
    (dst / "a/span.md").write_text("keep\nSTART\ngone\nEND\ntail\n", encoding="utf-8")
    apply_edits(dst, [{"path": "a/span.md", "delete_between": [{"start": "START", "end": "END"}]}])
    assert (dst / "a/span.md").read_text() == "keep\nEND\ntail\n"
    apply_edits(dst, [{"path": "a/new.md", "delete": True}])
    assert not (dst / "a/new.md").exists()
    with pytest.raises(ValueError, match="occurs 0 times"):
        apply_edits(dst, [{"path": "a/one.md", "replace": [{"old": "missing", "new": "x"}]}])
    with pytest.raises(ValueError, match="no line contains"):
        apply_edits(dst, [{"path": "a/one.md", "delete_lines_containing": ["missing"]}])
