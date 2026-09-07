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
