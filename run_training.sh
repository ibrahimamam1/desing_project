#!/bin/bash
# Step-by-step training runner.
#
# Trains one variant at a time and stops cleanly between them, so the machine
# can be shut down (thunderstorm, power cut) and the sweep picked up later.
# Re-running this script resumes each variant from its last good checkpoint;
# variants that already finished are skipped.
#
#   ./run_training.sh            # work through every variant in order
#   ./run_training.sh <variant>  # just one
#   ./run_training.sh --status   # how far along everything is

set -u
cd "$(dirname "$0")"

PY=/home/muntasir/miniconda3/envs/flow/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"

VERSIONS=(
  heuristic_continous
  heuristic_discrete
  attention_continous
  attention_discrete
  heuristic_attention_continous
  heuristic_attention_discrete
  shielded_attention_continous
)

if [ "${1:-}" = "--status" ]; then
    exec "$PY" -u src/full_pipeline.py --mode status
fi

if [ $# -ge 1 ]; then
    VERSIONS=("$1")
fi

# Clear out anything left behind by a hard stop, so TraCI ports are free.
pkill -f sumo 2>/dev/null
sleep 2

for v in "${VERSIONS[@]}"; do
    if [ -f "checkpoints/v0_1_full/$v/final_model.zip" ]; then
        echo "== $v already finished, skipping"
        continue
    fi

    echo "== training $v  ($(date '+%Y-%m-%d %H:%M:%S'))"
    PYTHONUNBUFFERED=1 "$PY" -u src/full_pipeline.py --mode train --version "$v" \
        >> "$LOG_DIR/train_${v}.log" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "== $v exited with code $rc — stopping here."
        echo "   Progress is checkpointed; re-run this script to resume."
        exit $rc
    fi

    echo "== $v complete"
    pkill -f sumo 2>/dev/null
    sleep 2
done

echo
echo "All requested variants are done."
"$PY" -u src/full_pipeline.py --mode status
