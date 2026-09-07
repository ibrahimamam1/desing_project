#!/bin/bash
# Kill any old training processes
pkill -f "full_pipeline" 2>/dev/null
sleep 2
pkill -f sumo 2>/dev/null
sleep 2

cd /home/muntasir/thesis/desing_project-main

# Use PYTHONUNBUFFERED so output appears in real-time in the log file
export PYTHONUNBUFFERED=1

echo "Starting 1.5M training at $(date)..."

# Use conda activate instead of conda run to avoid output buffering
eval "$(conda shell.bash hook)"
conda activate flow

nohup python -u src/full_pipeline.py --mode all >> training_log_1.5m.txt 2>&1 &
echo "PID: $!"
echo ""
echo "Training started! Watch progress with:"
echo "  tail -f /home/muntasir/thesis/desing_project-main/training_log_1.5m.txt"
