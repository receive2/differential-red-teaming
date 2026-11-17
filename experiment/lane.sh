#!/bin/bash
# One lane: run the jobs of a file one after another, from the repository root of this script.
# The interpreter is the one of the backend's checkout beside this repository
# (../commerce-agents/.venv).
#
#   experiment/lane.sh <lane name> <jobs file>
#
# A job is one line: what follows `python -u -m`, for example
#   experiment.run --name R-0509 --arrangement compare --seed 38 --features F3 --arrivals F3=1 --world esci-zip ...
# Blank lines and lines starting with # are skipped. A run whose event log already has its
# end event is skipped (experiment.run would otherwise start it over). A file named STOP in
# runs/lanes ends every lane after its current job. A job's standard input is /dev/null: the
# loop reads this file's lines from its own, and a command of a job that read it would take
# the next lines.
lane="$1"; jobs="$2"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root" || exit 1
PY="$(cd "$root/../commerce-agents/.venv/bin" && pwd)/python"
[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 1; }
mkdir -p runs/lanes
log="runs/lanes/$lane.out"
echo "== lane $lane: $(grep -c -v -e '^#' -e '^$' "$jobs") jobs from $jobs, start $(date '+%m-%d %H:%M')" >> "$log"
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  if [ -e runs/lanes/STOP ]; then echo "== lane $lane: STOP file found, ending $(date +%H:%M)" >> "$log"; break; fi
  name="$(printf '%s\n' "$line" | sed -n 's/.*--name \([^ ]*\).*/\1/p')"
  if [ -n "$name" ] && [ -f "runs/$name/log/events.jsonl" ] && grep -q '"kind": "end"' "runs/$name/log/events.jsonl"; then
    echo "== lane $lane: $name already finished" >> "$log"; continue
  fi
  echo "== lane $lane: $line  [start $(date +%H:%M)]" >> "$log"
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH="/usr/bin:/bin" PYTHONPATH=. HTTP_PROXY="${HTTP_PROXY:-}" HTTPS_PROXY="${HTTPS_PROXY:-}" NO_PROXY="${NO_PROXY:-}" \
    "$PY" -u -m $line >> "$log" 2>&1 < /dev/null
  echo "== lane $lane: exit $? [$(date +%H:%M)]" >> "$log"
done < "$jobs"
echo "== lane $lane finished $(date '+%m-%d %H:%M')" >> "$log"
