#!/bin/bash
# Final evaluation queue for the submission.
#
# Runs every remaining evaluation strictly one after another, in priority
# order. Each evaluation resumes per configuration, so after a power cut just
# run this script again: finished work is skipped, and it carries on from
# the configuration it was on.
#
#   ./run_final.sh
#
# Do not start it while another full_pipeline run is active; two SUMO-based
# runs at once collide on TraCI ports.

set -u
cd "$(dirname "$0")"

PY=/home/muntasir/miniconda3/envs/flow/bin/python
export SUMO_HOME=/usr/share/sumo
export PYTHONUNBUFFERED=1
LOG=logs/final.log
mkdir -p logs

if pgrep -f "src/full_pipeline.py --mode" >/dev/null 2>&1; then
    echo "Another full_pipeline run is active. Wait for it to finish first."
    exit 1
fi

say() { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

ev() {
    say "START  $*"
    "$PY" -u src/full_pipeline.py --mode eval "$@" >> "$LOG" 2>&1
    local rc=$?
    say "END    rc=$rc  $*"
    return $rc
}

# Abort before a long run if a smoke run shows episodes failing. A failing
# episode records empty telemetry, which reads as zero collisions and looks
# like a result instead of an error.
smoke() {
    local tag="smoke_$1"; shift
    say "SMOKE  $tag"
    "$PY" -u src/full_pipeline.py --mode eval "$@" --tag "$tag" --episodes 2 \
        --intentions all_left --scenarios "${SMOKE_SCENARIO}" > "logs/${tag}.log" 2>&1
    rm -f eval_results/*__"${tag}"_*
    if grep -q "failed" "logs/${tag}.log" || ! grep -q "col=" "logs/${tag}.log"; then
        say "SMOKE FAILED ($tag). See logs/${tag}.log. Stopping."
        exit 1
    fi
    say "SMOKE  ok"
}

# ---- 0. 3M comparison (resumes if a power cut interrupted it) ------------
ev --version attention_continous         --policy-from attention_continous --tag 3m
ev --version shielded_attention_continous --policy-from attention_continous --tag 3m

# ---- choose the continuous checkpoint for the remaining runs ------------
# Rule, fixed before any stress results exist: use whichever continuous
# checkpoint had fewer collisions without the shield on the standard 24
# configurations. Ties go to 1.5M, the checkpoint already in the main table.
CHOICE=$("$PY" - <<'PYEOF'
import csv, os
def coll(f):
    # Only a complete 24-configuration result counts; a partial file would
    # compare collisions over different numbers of episodes.
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    return sum(int(r["collisions"]) for r in rows) if len(rows) == 24 else None
c15 = coll("eval_results/attention_continous_results.csv")
c30 = coll("eval_results/attention_continous__policy_attention_continous__3m_results.csv")
if c30 is not None and c15 is not None and c30 < c15:
    print("3m checkpoints/v0_1_full/attention_continous/final_model.zip %s %s" % (c15, c30))
else:
    print("1500k checkpoints/v0_1_full/attention_continous/final_model_1500k.zip %s %s" % (c15, c30))
PYEOF
)
LABEL=$(echo "$CHOICE" | awk '{print $1}')
CKPT=$(echo "$CHOICE" | awk '{print $2}')
say "continuous checkpoint: $LABEL  (collisions 1.5M=$(echo "$CHOICE" | awk '{print $3}')  3M=$(echo "$CHOICE" | awk '{print $4}'))"

CONT=(--policy-from attention_continous --policy-file "$CKPT")
STRESS=(--stress --paired-seeds)

# ---- pre-flight: stress traffic and the commit zone must actually run ----
SMOKE_SCENARIO=St3_All_850
smoke stress        --version attention_continous "${CONT[@]}" "${STRESS[@]}"
SHIELD_COMMIT_ZONE=1 smoke commit --version shielded_attention_continous "${CONT[@]}" "${STRESS[@]}"

# ---- 1-3. stress: continuous without shield, with shield, with commit zone
ev --version attention_continous          "${CONT[@]}" "${STRESS[@]}" --tag "stress_${LABEL}"
ev --version shielded_attention_continous "${CONT[@]}" "${STRESS[@]}" --tag "stress_${LABEL}"
SHIELD_COMMIT_ZONE=1 ev --version shielded_attention_continous "${CONT[@]}" "${STRESS[@]}" --tag "stress_commit_${LABEL}"

# ---- 4. stress: attention discrete, for the full ordering ---------------
ev --version attention_discrete --policy-from attention_discrete "${STRESS[@]}" --tag stress

# ---- 5. standard 24 configurations with the commit zone -----------------
SHIELD_COMMIT_ZONE=1 ev --version shielded_attention_continous "${CONT[@]}" --tag "commit_${LABEL}"

# ---- 6. stress: heuristic baselines, if there is time --------------------
ev --version heuristic_continous --policy-from heuristic_continous "${STRESS[@]}" --tag stress
ev --version heuristic_discrete  --policy-from heuristic_discrete  "${STRESS[@]}" --tag stress

say "ALL DONE"
