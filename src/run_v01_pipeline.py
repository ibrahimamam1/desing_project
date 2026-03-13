#!/usr/bin/env python3
"""
Unified pipeline for training and evaluating all v0.1 variants.

Usage:
    python run_v01_pipeline.py --mode train --versions all
    python run_v01_pipeline.py --mode train --versions heuristic_continous heuristic_discrete
    python run_v01_pipeline.py --mode eval --versions all --checkpoints auto
    python run_v01_pipeline.py --mode eval --versions heuristic_continous --checkpoints /path/to/ckpt
    python run_v01_pipeline.py --mode both --versions all
"""
import argparse
import subprocess
import os
import sys
import glob

parser = argparse.ArgumentParser(description="V0.1 Training & Evaluation Pipeline")
parser.add_argument("--mode", required=True, choices=["train", "eval", "both"],
                    help="Run training, evaluation, or both.")
parser.add_argument("--versions", nargs="+", default=["all"],
                    choices=["all", "heuristic_continous", "heuristic_discrete",
                             "attention_continous", "attention_discrete",
                             "heuristic_attention_continous", "heuristic_attention_discrete"],
                    help="Which versions to process.")
parser.add_argument("--checkpoints", nargs="*", default=None,
                    help="Checkpoint paths for eval. Use 'auto' to find latest best checkpoints.")
parser.add_argument("--n_sims", type=int, default=42,
                    help="Number of eval sims per scenario (passed to v0_1_evaluate.py).")
args = parser.parse_args()

ALL_VERSIONS = ["heuristic_continous", "heuristic_discrete",
                "attention_continous", "attention_discrete",
                "heuristic_attention_continous", "heuristic_attention_discrete"]
versions = ALL_VERSIONS if "all" in args.versions else args.versions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_MAP = {
    "heuristic_continous":  os.path.join(SCRIPT_DIR, "configs", "v0_1_heuristic_continous.py"),
    "heuristic_discrete":   os.path.join(SCRIPT_DIR, "configs", "v0_1_heuristic_discrete.py"),
    "attention_continous":  os.path.join(SCRIPT_DIR, "configs", "v0_1_attention_continous.py"),
    "attention_discrete":   os.path.join(SCRIPT_DIR, "configs", "v0_1_attention_discrete.py"),
    "heuristic_attention_continous": os.path.join(SCRIPT_DIR, "configs", "v0_1_heuristic_attention_continous.py"),
    "heuristic_attention_discrete":  os.path.join(SCRIPT_DIR, "configs", "v0_1_heuristic_attention_discrete.py"),
}

EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "test", "v0_1_evaluate.py")

CHECKPOINT_DIR_MAP = {
    "heuristic_continous":  "checkpoints/v0_1",
    "heuristic_discrete":   "checkpoints/v0_1_discrete",
    "attention_continous":  "checkpoints/v0_1_attention_continous",
    "attention_discrete":   "checkpoints/v0_1_attention_discrete",
    "heuristic_attention_continous":  "checkpoints/v0_1_heuristic_attention_continous",
    "heuristic_attention_discrete":   "checkpoints/v0_1_heuristic_attention_discrete",
}

def find_latest_best_checkpoint(version):
    root = os.path.join(os.getcwd(), CHECKPOINT_DIR_MAP[version])
    if not os.path.exists(root):
        print(f"  [WARN] No checkpoint dir found: {root}")
        return None
    best_dirs = sorted(glob.glob(os.path.join(root, "*", "best")),
                       key=os.path.getmtime, reverse=True)
    if not best_dirs:
        print(f"  [WARN] No 'best' checkpoints found in {root}")
        return None
    return best_dirs[0]

def run_training(version):
    script = CONFIG_MAP[version]
    print(f"\n{'='*70}")
    print(f"  TRAINING: {version}")
    print(f"  Script: {script}")
    print(f"{'='*70}\n")
    cmd = [sys.executable, script, "--train"]
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"  [ERROR] Training failed for {version} (exit code {result.returncode})")
        return False
    print(f"  [OK] Training complete: {version}")
    return True

def run_evaluation(version, checkpoint_path):
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {version}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoint", checkpoint_path,
        "--version", version,
        "--n_sims", str(args.n_sims),
    ]
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"  [ERROR] Evaluation failed for {version}")
        return False
    print(f"  [OK] Evaluation complete: {version}")
    return True

def main():
    print(f"\n{'#'*70}")
    print(f"  V0.1 PIPELINE — Mode: {args.mode} | Versions: {versions}")
    print(f"{'#'*70}\n")

    if args.mode in ("train", "both"):
        for version in versions:
            run_training(version)

    if args.mode in ("eval", "both"):
        for i, version in enumerate(versions):
            if args.checkpoints and args.checkpoints[0] != "auto":
                ckpt = args.checkpoints[i] if i < len(args.checkpoints) else args.checkpoints[-1]
            else:
                ckpt = find_latest_best_checkpoint(version)
            if ckpt is None:
                print(f"  [SKIP] No checkpoint found for {version}")
                continue
            run_evaluation(version, ckpt)

    print(f"\n{'#'*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
