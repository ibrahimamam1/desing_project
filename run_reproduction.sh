#!/bin/bash
# Reproduction under the pre-defence training environment.
#
# Trains and evaluates the report's controllers with --train-profile ibrahima,
# which reproduces the training environment of src/configs/v0_1_single_agent.py,
# then evaluates on the pre-defence benchmark, regenerates the results report
# and commits it. Steps run strictly one at a time.
#
# Resumable: after a power cut just run it again. Finished training is skipped,
# evaluation resumes per configuration, and the report is regenerated.
#
#   ./run_reproduction.sh

set -u
cd "$(dirname "$0")"

PY=/home/muntasir/miniconda3/envs/flow/bin/python
GIT=/home/muntasir/miniconda3/bin/git
export SUMO_HOME=/usr/share/sumo
export PYTHONUNBUFFERED=1
LOG=logs/reproduction.log
REPRO=checkpoints/v0_1_full_ibrahima
mkdir -p logs

say() { echo "$(date '+%F %H:%M:%S')  $*" | tee -a "$LOG"; }

# Wait for any running full_pipeline process. Matching on the process name
# being python keeps this loop from matching its own command line.
wait_idle() {
    while ps -eo comm,args | awk '$1 ~ /^python/ && /full_pipeline.py --mode/' | grep -q .; do
        sleep 60
    done
}

train() {
    local v=$1
    wait_idle
    say "TRAIN  $v"
    "$PY" -u src/full_pipeline.py --mode train --version "$v" --train-profile ibrahima >> "logs/repro_train_${v}.log" 2>&1
    say "TRAIN  $v  rc=$?"
}

ev() {
    local env_version=$1 policy=$2
    local ckpt="$REPRO/$policy/final_model.zip"
    if [ ! -f "$ckpt" ]; then
        say "SKIP   eval $env_version: no checkpoint at $ckpt"
        return 1
    fi
    wait_idle
    say "EVAL   $env_version <- $policy"
    "$PY" -u src/full_pipeline.py --mode eval --version "$env_version" --policy-from "$policy" \
        --policy-file "$ckpt" --tag repro >> "logs/repro_eval.log" 2>&1
    say "EVAL   $env_version <- $policy  rc=$?"
}

publish() {
    local msg=$1
    say "REPORT $msg"
    "$PY" scripts/make_report.py >> "$LOG" 2>&1
    mkdir -p experiments/eval_repro
    cp eval_results/*__repro_results.csv experiments/eval_repro/ 2>/dev/null
    "$GIT" add docs/ experiments/submission/ experiments/eval_repro/ >> "$LOG" 2>&1
    if "$GIT" commit -q -m "$msg" >> "$LOG" 2>&1; then
        timeout 180 "$GIT" push -q origin muntasir_210041265_v2 >> "$LOG" 2>&1 \
            && say "PUSHED $("$GIT" rev-parse --short HEAD)" \
            || say "PUSH FAILED, committed locally"
    else
        say "nothing new to commit"
    fi
}

say "=== reproduction queue started ==="

# 1. Attention + Continuous, with and without the shield
train attention_continous
ev attention_continous          attention_continous
ev shielded_attention_continous attention_continous
publish "add shield results under the pre-defence training environment

Attention + Continuous trained with the pre-defence training environment
and evaluated on the pre-defence benchmark, with and without the shield."

# 2. Attention + Discrete
train attention_discrete
ev attention_discrete attention_discrete
publish "add attention discrete under the pre-defence training environment"

# 3. Heuristic baselines, completing the ordering in one environment
train heuristic_continous
ev heuristic_continous heuristic_continous
train heuristic_discrete
ev heuristic_discrete heuristic_discrete
publish "add heuristic baselines under the pre-defence training environment"

say "=== reproduction queue finished ==="
