#!/bin/bash
# Shield-aware training in the pre-defence environment.
#
# Trains Attention + Continuous with the rear-aware + commit-zone shield active,
# using the same training settings as the no-shield model in that environment,
# evaluates it there with the shield on, then regenerates and commits the report.
# Resumable: after a power cut, run it again.
set -u
cd "$(dirname "$0")"
PY=/home/muntasir/miniconda3/envs/flow/bin/python
GIT=/home/muntasir/miniconda3/bin/git
export SUMO_HOME=/usr/share/sumo SUMO_SLEEP=0.2 EVAL_SLEEP=0.5 PYTHONUNBUFFERED=1
export SHIELD_REAR_AWARE=1 SHIELD_COMMIT_ZONE=1
LOG=logs/shieldtrained.log
CKPT=checkpoints/v0_1_full_ibrahima/shielded_attention_continous/final_model.zip
mkdir -p logs
say() { echo "$(date '+%F %H:%M:%S')  $*" | tee -a "$LOG"; }
busy() { ps -eo comm,args | awk '$1 ~ /^python/ && /full_pipeline.py --mode/' | grep -q .; }

while busy; do sleep 30; done
say "TRAIN  shield-aware"
"$PY" -u src/full_pipeline.py --mode train --version shielded_attention_continous \
    --train-profile ibrahima --workers 24 --n-steps 341 >> "$LOG" 2>&1
say "TRAIN  rc=$?"
[ -f "$CKPT" ] || { say "no model at $CKPT"; exit 1; }

# Rule fixed before any result: if training ends after 20:15, evaluate 126
# episodes per intention (504 total) so the report is ready before 22:00.
EPISODES=252
[ "$(date +%H%M)" -gt 2015 ] && EPISODES=126
say "EVAL   shield-aware, $EPISODES episodes per intention"
"$PY" -u src/full_pipeline.py --mode eval --version shielded_attention_continous \
    --policy-from shielded_attention_continous --policy-file "$CKPT" \
    --eval-profile ibrahima --episodes "$EPISODES" --paired-seeds --tag complex_shieldtrained >> "$LOG" 2>&1
say "EVAL   rc=$?"

"$PY" scripts/make_report.py >> "$LOG" 2>&1
mkdir -p experiments/eval_complex
cp eval_results/*__complex*_results.csv experiments/eval_complex/ 2>/dev/null
"$GIT" add docs/ experiments/submission/ experiments/eval_complex/ scripts/make_report.py >> "$LOG" 2>&1
"$GIT" commit -q -m "add shield-aware training results in the pre-defence environment" >> "$LOG" 2>&1
timeout 180 "$GIT" push -q origin muntasir_210041265_v2 >> "$LOG" 2>&1 && say "PUSHED $("$GIT" rev-parse --short HEAD)"
say "DONE"
