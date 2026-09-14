#!/bin/bash
# Evaluate in the pre-defence environment.
#
# Waits for the Attention + Continuous model trained with --train-profile
# ibrahima, then evaluates it in the same environment (--eval-profile ibrahima)
# without the shield, with the shield, and with the rear-aware shield. The
# report is regenerated and committed after each evaluation.
#
# Resumable: after a power cut, run it again.
#
#   ./run_complex.sh

set -u
cd "$(dirname "$0")"

PY=/home/muntasir/miniconda3/envs/flow/bin/python
GIT=/home/muntasir/miniconda3/bin/git
export SUMO_HOME=/usr/share/sumo
export PYTHONUNBUFFERED=1
export SUMO_SLEEP=0.2
export EVAL_SLEEP=0.5
LOG=logs/complex.log
CKPT=checkpoints/v0_1_full_ibrahima/attention_continous/final_model.zip
COMMON=(--policy-from attention_continous --policy-file "$CKPT"
        --eval-profile ibrahima --episodes 252 --paired-seeds)
mkdir -p logs

say() { echo "$(date '+%F %H:%M:%S')  $*" | tee -a "$LOG"; }

# Matching on the process name being python keeps this from matching itself.
busy() { ps -eo comm,args | awk '$1 ~ /^python/ && /full_pipeline.py --mode/' | grep -q .; }

publish() {
    say "REPORT $1"
    "$PY" scripts/make_report.py >> "$LOG" 2>&1
    mkdir -p experiments/eval_complex
    cp eval_results/*__complex*_results.csv experiments/eval_complex/ 2>/dev/null
    "$GIT" add docs/ experiments/submission/ experiments/eval_complex/ >> "$LOG" 2>&1
    if "$GIT" commit -q -m "$1" >> "$LOG" 2>&1; then
        timeout 180 "$GIT" push -q origin muntasir_210041265_v2 >> "$LOG" 2>&1 \
            && say "PUSHED $("$GIT" rev-parse --short HEAD)" \
            || say "PUSH FAILED, committed locally"
    fi
}

ev() {
    local tag=$1; shift
    while busy; do sleep 30; done
    say "EVAL   $tag"
    "$PY" -u src/full_pipeline.py --mode eval "$@" "${COMMON[@]}" --tag "$tag" >> "$LOG" 2>&1
    say "EVAL   $tag  rc=$?"
}

say "=== complex queue started ==="

# Training must finish first; resuming it here is harmless if it already has.
while busy; do sleep 60; done
if [ ! -f "$CKPT" ]; then
    say "TRAIN  resuming attention_continous in the pre-defence environment"
    "$PY" -u src/full_pipeline.py --mode train --version attention_continous \
        --train-profile ibrahima --workers 24 --n-steps 341 >> logs/complex_train.log 2>&1
fi
if [ ! -f "$CKPT" ]; then
    say "no trained model at $CKPT, stopping"
    exit 1
fi

# Smoke test the rear-aware shield so a failing configuration cannot
# be recorded as a collision-free run.
say "SMOKE  rear-aware shield"
SHIELD_REAR_AWARE=1 "$PY" -u src/full_pipeline.py --mode eval --version shielded_attention_continous \
    "${COMMON[@]}" --episodes 4 --intentions all_left --tag smoke_rear > logs/smoke_rear.log 2>&1
rm -f eval_results/*smoke_rear*
REAR_OK=1
if grep -q "failed" logs/smoke_rear.log || ! grep -q "col=" logs/smoke_rear.log; then
    say "SMOKE  rear-aware FAILED, see logs/smoke_rear.log; skipping that arm"
    REAR_OK=0
else
    say "SMOKE  ok"
fi

ev complex --version attention_continous
publish "add Attention + Continuous in the pre-defence environment"

ev complex --version shielded_attention_continous
publish "add shield results in the pre-defence environment"

if [ "$REAR_OK" = "1" ]; then
    SHIELD_REAR_AWARE=1 ev complex_rear --version shielded_attention_continous
    publish "add rear-aware shield results in the pre-defence environment"
fi

say "=== complex queue finished ==="
