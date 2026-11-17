#!/bin/bash
# The labels and the reading of an experiment: once its lanes have ended, probe and label the
# final assistant builds in ten shards at once, then read the tables (paper/reading.py).
# Started beside the lanes.
#
#   nohup paper/pipeline.sh R >> runs/tables/r1-pipeline.log 2>&1 < /dev/null &    # the first backend (Amazon ESCI)
#   nohup paper/pipeline.sh W >> runs/tables/w1-pipeline.log 2>&1 < /dev/null &    # the second (Wayfair WANDS)
#
# The shards are fixed by the run names of the table of runs (paper/jobs/<series>-runs.tsv),
# not by what is on disk, so that a second start finds a run's probe
# (runs/tables/<tag>-<k>/classify/<run>/) and its judge calls (runs/tables/<tag>-<k>.cache.jsonl)
# where the first left them. The labelling of a run reads only that run, so when it is
# labelled does not change its label.
series="${1:?R or W}"
case "$series" in
  R) tag=r1; name=esci ;;
  W) tag=w1; name=wands ;;
  *) echo "the series is R (the first backend) or W (the second)" >&2; exit 1 ;;
esac
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root" || exit 1
PY="$(cd "$root/../commerce-agents/.venv/bin" && pwd)/python"
[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 1; }
mkdir -p runs/tables results
awk -F'\t' 'NR > 1 {print $1}' "paper/jobs/$series-runs.tsv" | sort -u > "runs/tables/$tag-names.txt"
total="$(wc -l < "runs/tables/$tag-names.txt" | tr -d ' ')"
for k in 0 1 2 3 4 5 6 7 8 9; do
  awk -v k="$k" 'NR % 10 == k' "runs/tables/$tag-names.txt" | sed 's|^|runs/|' > "runs/tables/$tag-shard-$k.txt"
done
ended() { grep -l '"kind": "end"' runs/"$series"-[0-9]*/log/events.jsonl 2>/dev/null | wc -l | tr -d ' '; }
label_pass() {  # each shard's runs that have ended, ten shards at once
  for k in 0 1 2 3 4 5 6 7 8 9; do
    list=""
    while IFS= read -r r; do
      grep -q '"kind": "end"' "$r/log/events.jsonl" 2>/dev/null && list="$list $r"
    done < "runs/tables/$tag-shard-$k.txt"
    [ -n "$list" ] || { echo "pipeline: shard $k has no ended run"; continue; }
    # shellcheck disable=SC2086
    (env -i HOME="$HOME" PATH="/usr/bin:/bin:/opt/homebrew/bin" PYTHONPATH=. HTTP_PROXY="${HTTP_PROXY:-}" HTTPS_PROXY="${HTTPS_PROXY:-}" NO_PROXY="${NO_PROXY:-}" \
      "$PY" -u -m experiment.outcomes loops $list --out "runs/tables/$tag-$k" --cost-cap 15 >> "runs/tables/$tag-$k.log" 2>&1; echo "pipeline: shard $k exit $? $(date '+%H:%M')") &
  done
  wait
}
echo "pipeline: waiting for the lanes; $(ended) of $total runs have ended $(date '+%m-%d %H:%M')"
until ! pgrep -f "lane.sh $series-" > /dev/null; do sleep 120; done
echo "pipeline: the lanes have ended; $(ended) of $total runs have ended $(date '+%m-%d %H:%M')"
label_pass
echo "pipeline: labels done $(date '+%m-%d %H:%M')"
env -i HOME="$HOME" PATH="/usr/bin:/bin:/opt/homebrew/bin" PYTHONPATH=. "$PY" paper/reading.py runs/tables/"$tag"-[0-9].json --json "results/$name-reading.json" > "results/$name-reading.txt" 2>&1
echo "pipeline: reading written, exit $? $(date '+%m-%d %H:%M')"
