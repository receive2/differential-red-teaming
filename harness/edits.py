# SPDX-License-Identifier: Apache-2.0

"""Edit sets: the data that turns one tree into another. The reference implementation is
the world branch plus the reskin edits; a knockout is the reference minus one clause. Each
edit names a file and either replacements (each ``old`` must occur exactly once), lines to
delete by substring, a whole-file deletion, or a file to write.

    {"edits": [
      {"path": "shopping-agent/skills/x/SKILL.md", "replace": [{"old": "...", "new": "..."}]},
      {"path": "...", "delete_lines_containing": ["..."]},
      {"path": "...", "delete_between": [{"start": "...", "end": "..."}]},   # end kept
      {"path": "...", "delete": true},
      {"path": "...", "write": "<content>"},
      {"path": "...", "rename_to": "..."}
    ]}
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

IGNORED = shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"
)


def copy_tree(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=IGNORED, symlinks=True)
    return dst


def apply_edits(tree: Path, edits: list[dict[str, Any]], dry_run: bool = False) -> None:
    """Apply the edits in place; with ``dry_run`` only check that every anchor is found."""
    for edit in edits:
        path = tree / edit["path"]
        if edit.get("delete"):
            if not path.exists():
                raise ValueError(f"{edit['path']}: nothing to delete")
            if dry_run:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            continue
        if "write" in edit:
            if dry_run:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit["write"], encoding="utf-8")
            continue
        if "rename_to" in edit:
            if not path.exists():
                raise ValueError(f"{edit['path']}: nothing to rename")
            if dry_run:
                continue
            target = tree / edit["rename_to"]
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            continue
        text = path.read_text(encoding="utf-8")
        for pair in edit.get("replace", []):
            old, new = pair["old"], pair["new"]
            count = text.count(old)
            if count != 1:
                raise ValueError(f"{edit['path']}: anchor occurs {count} times: {old[:80]!r}")
            text = text.replace(old, new)
        for span in edit.get("delete_between", []):
            start, end = text.find(span["start"]), text.find(span["end"])
            if start < 0 or end < 0 or end <= start:
                raise ValueError(
                    f"{edit['path']}: span not found: {span['start'][:60]!r} .. {span['end'][:60]!r}"
                )
            text = text[:start] + text[end:]
        for needle in edit.get("delete_lines_containing", []):
            lines = text.splitlines(keepends=True)
            kept = [line for line in lines if needle not in line]
            if len(kept) == len(lines):
                raise ValueError(f"{edit['path']}: no line contains {needle[:80]!r}")
            text = "".join(kept)
        if not dry_run:
            path.write_text(text, encoding="utf-8")


def load_edit_file(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["edits"]


def materialize(base: Path, edit_files: list[Path], dst: Path) -> Path:
    """``base`` copied to ``dst`` with every edit file applied in order."""
    copy_tree(base, dst)
    for edit_file in edit_files:
        apply_edits(dst, load_edit_file(edit_file))
    return dst
