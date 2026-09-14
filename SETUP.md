# Setup

Notes for getting the training and evaluation pipeline running from a clean shell.

## Environment

The project runs in the `flow` conda environment (Python 3.7):

```bash
conda activate flow
```

Key versions: `stable-baselines3` 2.0.0, `gymnasium` 0.28.1, `torch` 1.13.1+cu117.

## Flow

Flow is used from a source checkout at `~/flow` rather than from PyPI, and is
expected to be installed in editable mode:

```bash
pip install -e ~/flow --no-deps
```

If `import flow` fails with `ModuleNotFoundError` even though `~/flow` exists,
the editable link into `site-packages` has been lost. Either re-run the command
above, or drop the path in directly:

```bash
echo "$HOME/flow" > "$(python -c 'import site; print(site.getsitepackages()[0])')/flow.pth"
```

Verify with:

```bash
python -c "import flow; print(flow.__file__)"
```

## SUMO

`SUMO_HOME` must be set (currently `/usr/share/sumo`). TraCI comes from the SUMO
install, not from pip.

## Running

All commands run from the repository root, so that `networks/` resolves:

```bash
python src/full_pipeline.py --mode train    # 1.5M steps per variant
python src/full_pipeline.py --mode eval     # 6 scenarios x 4 intentions
python src/full_pipeline.py --mode plot     # line plots + heatmap
python src/full_pipeline.py --mode all
```

`start_training.sh` wraps the `--mode all` run with process cleanup and logging.

Evaluation opens and closes a SUMO instance per episode. If a run dies partway,
stray processes can hold the TraCI ports:

```bash
pkill -f sumo
```

Only one `full_pipeline.py` run at a time: two concurrent runs collide on TraCI
ports and one dies with a `BrokenPipeError`.

## Resuming after interruption

Training writes a checkpoint every 25k steps and resumes from the newest one that
passes a CRC check. Evaluation records each finished configuration and resumes
from the next. After a power cut, rerun the same command.

```bash
python src/full_pipeline.py --mode status   # training and evaluation progress
```

## Training options

| Option | Effect |
|---|---|
| `--version NAME` | Train one variant |
| `--timesteps N` | Total training steps (default 1.5M) |
| `--workers N` | Parallel environments (default 8) |
| `--extend` | Continue a variant that already finished, up to `--timesteps` |
| `--train-profile ibrahima` | Train in the pre-defence training environment (see below); checkpoints go to `checkpoints/v0_1_full_ibrahima/` |

Training profiles:

| | `pipeline` (default) | `ibrahima` |
|---|---|---|
| Cross traffic | 275 veh/h | 400 veh/h |
| Background traffic in agent's lane | none | 275 veh/h |
| Background vehicle `speed_mode` | 31 | 0 |
| RL spawn probability | 0.8 | 0.3 |
| Environment warmup | 50 | 5 |

`ibrahima` reproduces `src/configs/v0_1_single_agent.py`, which produced the
pre-defence results.

## Evaluation options

| Option | Effect |
|---|---|
| `--episodes N` | Episodes per configuration (default 42) |
| `--policy-from VARIANT` | Evaluate `--version`'s environment with another variant's policy, e.g. the same weights with and without the shield |
| `--policy-file PATH` | Evaluate a specific checkpoint |
| `--tag NAME` | Suffix for output files, so runs do not overwrite each other |
| `--intentions ...` / `--scenarios ...` | Restrict to a subset |
| `--stress` | Use the high-flow scenarios (550 / 700 / 850 veh/h) |
| `--paired-seeds` | Give every variant identical traffic draws |

## Shield configuration

The shield's thresholds are read from environment variables. Defaults reproduce
the reported results.

| Variable | Default | Meaning |
|---|---|---|
| `SHIELD_TTC_THRESHOLD` | 2.0 | Time-to-conflict that activates layer 1 (s) |
| `SHIELD_RSS_MIN_GAP` | 3.0 | RSS minimum gap (m) |
| `SHIELD_RSS_REACTION` | 0.5 | RSS reaction time (s) |
| `SHIELD_EMERGENCY_DECEL` | -4.5 | Maximum braking (m/s²) |
| `SHIELD_TTC_ETA` / `SHIELD_RSS_ETA` / `SHIELD_ROW_ETA` | 0.5 / 0.6 / 0.15 | Arrival-time windows per layer |
| `SHIELD_ROW_DIST` / `SHIELD_ROW_CROSS` | 15.0 / 0.3 | Right-of-way distance and heading thresholds |
| `SHIELD_COMMIT_ZONE` | 0 | Set to 1 to stop braking once the vehicle can no longer stop before the conflict point |

## Automated queues

| Script | Runs |
|---|---|
| `run_training.sh` | Every variant in the default profile, one at a time |
| `run_final.sh` | Controlled shield, high-flow and commit-zone evaluations |
| `run_reproduction.sh` | Train and evaluate in the pre-defence training environment, then regenerate the report and commit |

All three are resumable: rerun after an interruption.

## Report

```bash
python scripts/make_report.py
```

Regenerates `docs/results_tables.md` and the figures in `experiments/submission/`
from `eval_results/`. The written report is `docs/RESULTS.md`.
