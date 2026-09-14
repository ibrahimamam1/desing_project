#!/bin/bash
# Deadline cutoff for shield-aware training.
#
# At 21:10 stops the shield-aware training queue, evaluates the newest
# checkpoint in the pre-defence environment with the refined shield on
# (126 episodes per intention, 504 total), then regenerates and commits the
# report, so results exist before the 22:00 deadline.
set -u
cd "$(dirname "$0")"
PY=/home/muntasir/miniconda3/envs/flow/bin/python
GIT=/home/muntasir/miniconda3/bin/git
export SUMO_HOME=/usr/share/sumo SUMO_SLEEP=0.2 EVAL_SLEEP=0.5 PYTHONUNBUFFERED=1
export SHIELD_REAR_AWARE=1 SHIELD_COMMIT_ZONE=1
LOG=logs/shieldtrained.log
DIR=checkpoints/v0_1_full_ibrahima/shielded_attention_continous
say() { echo "$(date '+%F %H:%M:%S')  $*" | tee -a "$LOG"; }

until [ "$(date +%H%M)" -ge 2110 ]; do sleep 20; done

# If training already produced a final model, the queue evaluates it itself.
if [ -f "$DIR/final_model.zip" ]; then
    say "CUTOFF final model exists; leaving the queue to finish"
    exit 0
fi

say "CUTOFF stopping shield-aware training for the deadline"
for pid in $(ps -eo pid,comm,args | awk '$2 == "bash" && /run_shieldtrained\.sh/ {print $1}'); do
    kill -9 "$pid" 2>/dev/null
done
for pid in $(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /--version shielded_attention_continous/ {print $1}'); do
    kill "$pid" 2>/dev/null
done
while ps -eo comm,args | awk '$1 ~ /^python/ && /full_pipeline.py --mode/' | grep -q .; do sleep 5; done
sleep 5

CKPT=$(ls "$DIR"/rl_model_*_steps.zip 2>/dev/null | sed 's/.*rl_model_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1 | awk '{print $2}')
STEPS=$(basename "$CKPT" | sed 's/rl_model_\([0-9]*\)_steps.zip/\1/')
[ -n "$CKPT" ] || { say "CUTOFF no checkpoint found"; exit 1; }
say "CUTOFF evaluating $CKPT ($STEPS steps), 126 episodes per intention"
echo "$STEPS" > logs/shieldtrained_steps.txt

"$PY" -u src/full_pipeline.py --mode eval --version shielded_attention_continous \
    --policy-from shielded_attention_continous --policy-file "$CKPT" \
    --eval-profile ibrahima --episodes 126 --paired-seeds --tag complex_shieldtrained >> "$LOG" 2>&1
say "EVAL   rc=$?"

"$PY" scripts/make_report.py >> "$LOG" 2>&1
mkdir -p experiments/eval_complex
cp eval_results/*__complex*_results.csv experiments/eval_complex/ 2>/dev/null
"$GIT" add docs/ experiments/submission/ experiments/eval_complex/ >> "$LOG" 2>&1
"$GIT" commit -q -m "add shield-aware training results in the pre-defence environment

Evaluated at the $STEPS-step checkpoint, stopped at the deadline, against
a no-shield model trained for 1.5M steps." >> "$LOG" 2>&1
timeout 180 "$GIT" push -q origin muntasir_210041265_v2 >> "$LOG" 2>&1 && say "PUSHED $("$GIT" rev-parse --short HEAD)"
say "CUTOFF DONE"
