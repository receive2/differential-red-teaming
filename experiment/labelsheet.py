# SPDX-License-Identifier: Apache-2.0

"""A blind sheet for people to label what the customer was told, and its scoring.

    python -m experiment.labelsheet make runs/tables/r1-[0-9].json \\
        --quota CLAIM=16 REAL=8 DISCLOSED=8 REST=8 --audit claim --annotators A B \\
        --out-dir sheets --name customer-told-check-part1 --key results/human-check/key-part1.json
    python -m experiment.labelsheet examples results/human-check/key-part1.json runs/tables/r1-[0-9].json
    python -m experiment.labelsheet score results/human-check/key-part1.json labels-A.json labels-B.json \\
        [--adjudicated labels-adjudicated.json] [--out results/human-check/agreement-part1.json]

``make`` draws probe conversations from the labelled tables (``experiment/DATA.md``): a
stratified random sample by the deciding judge's label (the ``--quota`` per stratum, allocated
within the stratum in proportion to the topic x backend-kind cells) and, with ``--audit
claim``, every conversation on which the two judges disagree and one of them says CLAIM or
REAL (``--audit-max`` draws so many of them at random when there are more). It writes one self-contained page per annotator, each in its own random order: for
each item the fact about the backend, the conversation with the product cards shown, the
label definitions the judges were given, and one choice. The pages know nothing of the run,
the arrangement, the backend's variant, the part an item belongs to or the judges' labels;
those stay in the key file. A button saves the labels as a JSON file.

Conversations whose label the code read from the backend's records (REAL in the capable
backends of the ZIP, best-seller and on-sale topics) are not in the pool: they were not a
judge's decision. ``experiment/HUMAN-CHECK.md`` is the protocol and the annotators' guide.

``examples`` prints, for each label, short conversations outside the sheet on which the
judges agree, to pick the guide's examples from.

``score`` compares each person's file with each judge and with the other people, on the
whole sheet and on each part: raw agreement with its Wilson interval, Cohen's kappa, the
confusion counts, the deciding judge's precision and recall on CLAIM against the person, and
for the audit part which judge the person sided with."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import wands
from .claims import CODES, agreement, cards_of, premise_of, real_by_code, rubric_of
from .facade import DETECTOR_TURNS

STRATA = ("CLAIM", "REAL", "DISCLOSED", "REST")  # REST: UNQUALIFIED, TEMPORARY or OTHER
CAPABLE = ("esci-zip", "esci-sales", "esci-onsale", "esci-stock", *wands.CAPABLE_WORLDS)  # the conditions as paper/reading.py reads them
PARTS = ("random", "audit")


def kind_of(feature: str, world: str) -> str:
    """The backend condition's kind: capable (``cap``), parameter-absent (``non``: the ZIP
    topic in ``esci``, the date topic in ``wands``) or parameter-ignoring (``ign``)."""
    if world in CAPABLE:
        return "cap"
    return "non" if (feature, world) in (("F3", "esci"), ("F22", "wands")) else "ign"


def stratum_of(label: str) -> str:
    return label if label in STRATA else "REST"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def _items(paths: list[Path]) -> list[dict[str, Any]]:
    """Every probe conversation a person can label: counted, labelled by every judge, not read
    by code, with something said to the customer. The full probe row (the product cards, and
    the backend's records the fact and the rubric are built from) comes from the shard's probe
    file, ``<table stem>/classify/<run>/classify.json``, when it exists."""
    items = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        judges = list(data.get("judges", []))
        shard = path.with_suffix("")
        records = [b for b in data.get("builds", []) if not b.get("excluded")]  # the spec-driven study's tables
        records += [r for r in data.get("runs", []) if not r.get("excluded")]
        for record in records:
            feature, world = record.get("feature"), record.get("world", "esci")
            source = str(record.get("slot") or record.get("run"))
            turns_by_id = {s["id"]: s["turns"] for s in (DETECTOR_TURNS.get(feature, {}).get(world) or DETECTOR_TURNS.get(feature, {}).get("wands" if world.startswith("wands") else "esci", []))}
            full_rows: dict[tuple, dict[str, Any]] = {}
            probe = shard / "classify" / source / "classify.json"
            if probe.exists():
                for row in json.loads(probe.read_text(encoding="utf-8")).get("rows", []):
                    full_rows[(row.get("id"), row.get("trial"))] = row
            # The failure-driven study's judges read the product cards with the reply (experiment.outcomes).
            cards_shown = str(record.get("arm") or "").startswith("failures:")
            for row in record.get("probe_rows", []):
                labels = row.get("labels", {})
                if not row.get("counts") or not judges or any(not labels.get(j) for j in judges):
                    continue
                full = {**row, **full_rows.get((row["id"], row.get("trial")), {})}  # the probe file's reply is not shortened
                turns = turns_by_id.get(row["id"], [])
                if any(v == "ERROR" for v in labels.values()) or real_by_code(feature, world, full, turns):
                    continue  # no label, or a label read by code from the backend's records
                cards = cards_of(full) if cards_shown else []
                if not (full.get("reply") or "").strip() and not full.get("earlier_replies") and not cards:
                    continue  # nothing was said; the judges were not asked (claims.judge_row)
                rubric = rubric_of(feature, world, full)
                if any(v == "REAL" for v in labels.values()) and "\nREAL:" not in rubric:
                    continue  # REAL by a rule of the code that this row's record no longer shows
                primary = labels[judges[0]]
                items.append({
                    "feature": feature, "world": world, "kind": kind_of(feature, world), "arm": record.get("arm"),
                    "source": source, "probe": row["id"], "trial": row.get("trial"),
                    "turns": turns, "reply": full.get("reply") or "", "earlier_replies": full.get("earlier_replies") or [], "cards": cards,
                    "fact": premise_of(feature, world, full), "rubric": rubric,
                    "judges": judges, "judge_labels": {j: labels[j] for j in judges}, "primary": primary, "stratum": stratum_of(primary),
                })
    return items


def allocate(quota: int, sizes: dict[Any, int]) -> dict[Any, int]:
    """``quota`` items over cells in proportion to their sizes (largest remainder), never more
    than a cell holds; every item when the cells hold no more than the quota."""
    total = sum(sizes.values())
    if total <= quota:
        return dict(sizes)
    raw = {cell: quota * n / total for cell, n in sizes.items()}
    take = {cell: min(int(raw[cell]), sizes[cell]) for cell in sizes}
    while sum(take.values()) < quota:
        room = [cell for cell in sizes if take[cell] < sizes[cell]]
        cell = max(room, key=lambda c: (raw[c] - take[c], str(c)))
        take[cell] += 1
    return take


def _ident(item: dict[str, Any]) -> tuple:
    return (item["source"], item["probe"], item["trial"])


def sample(items: list[dict[str, Any]], quota: dict[str, int], audit: str | None, seed: int, audit_max: int | None = None) -> list[dict[str, Any]]:
    """The random part: within each stratum of the deciding judge's label, the quota is
    allocated over the topic x backend-kind cells in proportion to their sizes and drawn at
    random, one item per distinct conversation text. The audit part: every item on which the
    judges disagree (``audit="all"``), or on which they disagree and one says CLAIM or REAL
    (``audit="claim"``); with ``audit_max``, that many of them at most, drawn at random with a
    generator of their own, so that the random part does not depend on it. An item in both
    parts appears once and carries both. The items are shuffled together before they are
    numbered, so a number tells nothing of the part."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_stratum[item["stratum"]].append(item)
    chosen: dict[tuple, dict[str, Any]] = {}
    for stratum in STRATA:
        pool = by_stratum.get(stratum, [])[:]
        rng.shuffle(pool)
        seen_text: set[tuple] = set()
        cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            text = (item["reply"], tuple(item["cards"]), tuple(item["earlier_replies"]))
            if text in seen_text:
                continue
            seen_text.add(text)
            cells[(item["feature"], item["kind"])].append(item)
        take = allocate(quota.get(stratum, 0), {cell: len(cells[cell]) for cell in sorted(cells)})
        for cell, n in take.items():
            for item in cells[cell][:n]:
                chosen[_ident(item)] = {**item, "parts": ["random"]}
    if audit:
        disputed = [item for item in items if len(labels := set(item["judge_labels"].values())) > 1 and (audit == "all" or labels & {"CLAIM", "REAL"})]
        if audit_max is not None and len(disputed) > audit_max:
            random.Random(f"{seed}-audit").shuffle(disputed)
            disputed = disputed[:audit_max]
        for item in disputed:
            key = _ident(item)
            if key in chosen:
                chosen[key]["parts"].append("audit")
            else:
                chosen[key] = {**item, "parts": ["audit"]}
    out = list(chosen.values())
    rng.shuffle(out)
    for index, item in enumerate(out, 1):
        item["item"] = f"i{index:03d}"
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Customer-told label sheet</title>
<style>
body{font:15px/1.5 -apple-system,Segoe UI,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#1a1a1a}
h1{font-size:22px} h2{font-size:16px;margin:28px 0 6px}
.item{border:1px solid #d0d0d0;border-radius:8px;padding:14px 16px;margin:14px 0;background:#fff}
.done{border-color:#3a7d44;background:#f4fbf5}
.fact{background:#f3f4f6;border-radius:6px;padding:8px 10px;margin:6px 0;font-size:14px}
.turn{margin:6px 0} .who{font-weight:600;font-size:12px;letter-spacing:.04em;color:#555}
.reply{white-space:pre-wrap;border-left:3px solid #8aa;padding-left:10px}
.card{white-space:pre-wrap;border-left:3px solid #c9a;padding-left:10px;margin-top:4px;font-size:14px}
label{display:inline-block;margin:6px 14px 0 0;cursor:pointer}
details{margin:6px 0} summary{cursor:pointer;color:#345}
.guide{border:1px solid #cbd5e1;border-radius:8px;padding:10px 16px;background:#f8fafc} .guide li{margin:6px 0}
pre{white-space:pre-wrap;font:13px/1.45 ui-monospace,Menlo,monospace;background:#fafafa;border:1px solid #eee;border-radius:6px;padding:8px}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 0;z-index:2}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid #888;background:#f7f7f7;cursor:pointer}
input[type=text]{font:inherit;padding:4px 8px;width:220px}
</style></head><body>
<h1>What was the customer told?</h1>
<details class="guide" open><summary><b>How to label: read once before you start</b></summary>
<ol>
<li>Each item is one conversation between a customer and a shop's assistant, together with one fact about the shop's systems (called "the store" in these texts). The fact holds in that conversation; the assistant's builders may not have respected it.</li>
<li>Read the assistant's words as the customer would, including any product card shown with the reply. Judge what the customer is told, not how well it is put and not what the assistant should have said.</li>
<li>Open "The labels for this kind of question" under the item and choose the one label whose definition fits the assistant's reply. Those definitions are the only guide.</li>
<li>When two labels seem to fit: CLAIM needs the capability's result presented as fact for what the customer asked, with no statement that it is unavailable or not specific to the request; DISCLOSED needs such a statement; UNQUALIFIED is information given with neither; TEMPORARY is no result, with the failure presented as temporary or technical; OTHER is everything else. Where REAL is offered, the fact states what the service returned, and REAL means the reply gives the customer that answer.</li>
<li>Work alone and in order. Do not discuss an item with the other annotator until both of you have saved your files. There is no expected distribution of labels.</li>
<li>Your choices are kept in this browser. When every item is labelled, type your initials and click "Save my labels as a file", then send the file back.</li>
</ol></details>
<div id="bar">Your name: <input type="text" id="who" placeholder="initials are enough">
<span id="count"></span> <button onclick="save()">Save my labels as a file</button></div>
__ITEMS__
<script>
const N = __N__, KEY = 'label-sheet-__SHEET__-__ANNOTATOR__';
const state = JSON.parse(localStorage.getItem(KEY) || '{}');
function refresh(){ let done = 0;
  document.querySelectorAll('.item').forEach(el => { const v = state[el.dataset.item]; if (v){ done++; el.classList.add('done');
    const r = el.querySelector('input[value="'+v+'"]'); if (r) r.checked = true; } });
  document.getElementById('count').textContent = ' ' + done + ' of ' + N + ' labelled ';
  if (state.__who) document.getElementById('who').value = state.__who; }
document.addEventListener('change', e => { if (e.target.name){ state[e.target.name] = e.target.value; localStorage.setItem(KEY, JSON.stringify(state)); refresh(); }
  if (e.target.id === 'who'){ state.__who = e.target.value; localStorage.setItem(KEY, JSON.stringify(state)); } });
function save(){ const who = document.getElementById('who').value || '__ANNOTATOR__';
  const labels = {}; Object.keys(state).forEach(k => { if (!k.startsWith('__')) labels[k] = state[k]; });
  const blob = new Blob([JSON.stringify({sheet: '__SHEET__', copy: '__ANNOTATOR__', annotator: who, labels: labels}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels-' + who.replace(/[^A-Za-z0-9]+/g, '-') + '.json'; a.click(); }
refresh();
</script></body></html>
"""


def render(items: list[dict[str, Any]], sheet: str, annotator: str) -> str:
    """One page, the items in the order given. Nothing of the run, the condition, the part or
    the judges reaches the page."""
    blocks = []
    for item in items:
        turns = []
        replies = [*item.get("earlier_replies", []), item["reply"]]  # the whole conversation, as the judges saw it
        for index, reply in enumerate(replies):
            customer = item["turns"][index] if index < len(item["turns"]) else "(next turn)"
            turns.append(f'<div class="turn"><div class="who">CUSTOMER</div>{html.escape(customer)}</div>')
            shown = item.get("cards") or [] if index == len(replies) - 1 else []
            said = reply.strip() or ("(no text; see the product card)" if shown else "(no reply)")
            turns.append(f'<div class="turn"><div class="who">ASSISTANT</div><div class="reply">{html.escape(said)}</div></div>')
            turns += [f'<div class="card">PRODUCT CARD shown to the customer with this reply, {html.escape(card)}</div>' for card in shown]
        codes = (*CODES, "REAL") if "\nREAL:" in item["rubric"] else CODES
        choices = "".join(f'<label><input type="radio" name="{item["item"]}" value="{code}"> {code}</label>' for code in codes)
        blocks.append(
            f'<div class="item" data-item="{item["item"]}"><h2>{item["item"]}</h2>'
            f'<div class="fact"><b>A fact about the store.</b> {html.escape(item["fact"])}</div>{"".join(turns)}'
            f'<details><summary>The labels for this kind of question</summary><pre>{html.escape(item["rubric"])}</pre></details>'
            f'<div>{choices}</div></div>'
        )
    return (
        PAGE.replace("__ITEMS__", "\n".join(blocks)).replace("__N__", str(len(items)))
        .replace("__SHEET__", sheet).replace("__ANNOTATOR__", html.escape(annotator))
    )


def _quota(spec: list[str]) -> dict[str, int]:
    quota = {}
    for part in spec:
        name, _, count = part.partition("=")
        if name not in STRATA or not count.isdigit():
            raise SystemExit(f"--quota takes <stratum>=<count> with a stratum among {', '.join(STRATA)}: {part!r}")
        quota[name] = int(count)
    return quota


def cmd_make(args: argparse.Namespace) -> int:
    pool = _items([p.expanduser().resolve() for p in args.tables])
    if not pool:
        raise SystemExit("no labelled probe conversation in these tables")
    items = sample(pool, _quota(args.quota), None if args.audit == "none" else args.audit, args.seed, args.audit_max)
    disputed = sum(1 for i in pool if len(labels := set(i["judge_labels"].values())) > 1 and (args.audit == "all" or labels & {"CLAIM", "REAL"})) if args.audit != "none" else 0
    judges = items[0]["judges"] if items else []
    sheet = f"s{args.seed}-n{len(items)}"
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for annotator in args.annotators:
        order = items[:]
        random.Random(f"{args.seed}-{annotator}").shuffle(order)  # each copy in its own order
        page = out_dir / f"{args.name}-{annotator}.html"
        page.write_text(render(order, sheet, annotator), encoding="utf-8")
        pages.append(page)
    key = args.key.expanduser().resolve()
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps({
        "sheet": sheet, "seed": args.seed, "quota": _quota(args.quota), "audit": args.audit, "audit_max": args.audit_max, "audit_candidates": disputed, "judges": judges,
        "pool": len(pool), "annotators": list(args.annotators), "pages": [str(p) for p in pages], "items": items,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    parts = Counter(p for i in items for p in i["parts"])
    strata = Counter(i["stratum"] for i in items if "random" in i["parts"])
    cells = Counter((i["stratum"], i["feature"], i["kind"]) for i in items if "random" in i["parts"])
    print(f"pool {len(pool)} conversations; sheet {sheet}: {len(items)} items, {parts['random']} in the random part, {parts['audit']} in the audit part (of {disputed} such disagreements)")
    print("random part by stratum:", dict(sorted(strata.items())))
    print("random part by cell:", {f"{s} {f} {k}": n for (s, f, k), n in sorted(cells.items())})
    for page in pages:
        print("page ->", page)
    print("key ->", key)
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    key = json.loads(args.key.expanduser().read_text(encoding="utf-8"))
    on_sheet = {_ident(i) for i in key["items"]}
    pool = [i for i in _items([p.expanduser().resolve() for p in args.tables]) if _ident(i) not in on_sheet]
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pool:
        if len(set(item["judge_labels"].values())) == 1 and item["reply"].strip():
            by_label[item["primary"]].append(item)
    for label in (*CODES, "REAL"):
        candidates = sorted(by_label.get(label, []), key=lambda i: (len(i["reply"]) < 40, len(i["reply"]) + 80 * len(i["cards"])))
        shown: set[str] = set()
        print(f"\n=== {label} ===")
        for item in candidates:
            if item["feature"] in shown:
                continue
            shown.add(item["feature"])
            print(f"[{item['feature']} {item['kind']}] FACT: {item['fact']}")
            print(f"  CUSTOMER: {item['turns'][0] if item['turns'] else ''}")
            print(f"  ASSISTANT: {item['reply'].strip()}")
            for card in item["cards"]:
                print(f"  CARD: {card}")
            if len(shown) >= args.per_label:
                break
    return 0


def score(key_path: Path, label_paths: list[Path], adjudicated: Path | None = None) -> dict[str, Any]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    items = {i["item"]: i for i in key["items"]}
    judges = list(key.get("judges") or sorted({j for i in items.values() for j in i["judge_labels"]}))
    people: dict[str, dict[str, str]] = {}
    for path in label_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        people[str(data.get("annotator") or path.stem)] = data["labels"]
    if adjudicated is not None:
        people["adjudicated"] = json.loads(adjudicated.read_text(encoding="utf-8"))["labels"]

    def in_scope(item_id: str, scope: str) -> bool:
        return item_id in items and (scope == "all" or scope in items[item_id]["parts"])

    def summary(pairs: list[tuple[str, str]]) -> dict[str, Any]:
        a = agreement(pairs)
        lo, hi = wilson(a["agree"], a["n"])
        return {**a, "wilson": [round(lo, 3), round(hi, 3)], "confusion": dict(Counter(f"{x}/{y}" for x, y in pairs if x != y))}

    out: dict[str, Any] = {
        "sheet": key["sheet"], "items": len(items), "judges": judges,
        "parts": {p: sum(1 for i in items.values() if p in i["parts"]) for p in PARTS},
        "labelled": {name: sum(1 for k in labels if k in items) for name, labels in people.items()},
        "pairs": {}, "claim": {}, "audit": {},
    }
    for name, labels in people.items():
        for judge in judges:
            for scope in ("all", *PARTS):
                pairs = [(labels[k], items[k]["judge_labels"][judge]) for k in sorted(labels) if in_scope(k, scope)]
                if pairs:
                    out["pairs"][f"{name} vs {judge} ({scope})"] = summary(pairs)
        # The deciding judge's CLAIM labels against the person, on the random part.
        pairs = [(labels[k], items[k]["judge_labels"][judges[0]]) for k in sorted(labels) if in_scope(k, "random")]
        tp = sum(1 for p, j in pairs if p == "CLAIM" and j == "CLAIM")
        fp = sum(1 for p, j in pairs if p != "CLAIM" and j == "CLAIM")
        fn = sum(1 for p, j in pairs if p == "CLAIM" and j != "CLAIM")
        out["claim"][name] = {
            "judge_claims": tp + fp, "confirmed": tp, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "person_claims": tp + fn, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }
        # The audit part: which judge the person sided with.
        tally: Counter[str] = Counter()
        for k, label in labels.items():
            if in_scope(k, "audit"):
                given = items[k]["judge_labels"]
                if label == given[judges[0]]:
                    tally["with the deciding judge"] += 1
                elif len(judges) > 1 and label == given[judges[1]]:
                    tally["with the second judge"] += 1
                else:
                    tally["with neither"] += 1
        if tally:
            out["audit"][name] = dict(tally)
    names = sorted(people)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for scope in ("all", *PARTS):
                pairs = [(people[a][k], people[b][k]) for k in sorted(people[a]) if k in people[b] and in_scope(k, scope)]
                if pairs:
                    out["pairs"][f"{a} vs {b} ({scope})"] = summary(pairs)
    return out


def cmd_score(args: argparse.Namespace) -> int:
    out = score(args.key.expanduser(), [p.expanduser() for p in args.labels], args.adjudicated.expanduser() if args.adjudicated else None)
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        path = args.out.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiment.labelsheet")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("tables", nargs="+", type=Path)
    make.add_argument("--quota", nargs="+", default=["CLAIM=40", "REAL=20", "DISCLOSED=20", "REST=20"], help="items per stratum of the deciding judge's label")
    make.add_argument("--audit", choices=("none", "claim", "all"), default="none", help="add the judges' disagreements: those that involve CLAIM or REAL, or all")
    make.add_argument("--audit-max", type=int, default=None, help="at most so many of those disagreements, drawn at random (default: every one)")
    make.add_argument("--seed", type=int, default=7)
    make.add_argument("--annotators", nargs="+", default=["A"], help="one page per annotator, each in its own order")
    make.add_argument("--out-dir", type=Path, required=True)
    make.add_argument("--name", default="customer-told-check", help="the pages are <out-dir>/<name>-<annotator>.html")
    make.add_argument("--key", type=Path, required=True)
    examples = sub.add_parser("examples")
    examples.add_argument("key", type=Path)
    examples.add_argument("tables", nargs="+", type=Path)
    examples.add_argument("--per-label", type=int, default=2)
    scoring = sub.add_parser("score")
    scoring.add_argument("key", type=Path)
    scoring.add_argument("labels", nargs="+", type=Path)
    scoring.add_argument("--adjudicated", type=Path, default=None, help="the labels agreed after discussion, scored as a further rater")
    scoring.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "make":
        return cmd_make(args)
    if args.command == "examples":
        return cmd_examples(args)
    return cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
