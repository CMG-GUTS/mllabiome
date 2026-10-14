# Federated learning on LAMPP CRC

Run a federated colorectal cancer (CRC) classification experiment using **mllabiome**, the **curated-microbiota** benchmark, and **Flower SecAgg+**.

The example follows leave-one-dataset-out (LODO) evaluation: one study is held out for testing, while the remaining studies act as separate training clients. The prediction model is a regularized SGD logistic classifier. Flower coordinates training, and SecAgg+ securely aggregates model updates.

**Start with the quick demo** to check that the workflow runs in your environment. Use the full example for the complete set of LODO folds.

## Before you start

You need Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and a checkout of mllabiome that includes the `federated-evaluate` stage. Run the following commands from the repository root.

For a new environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

If your environment is already active, you can skip those two commands.

Install the optional federated dependencies, Flower application, and benchmark data package:

```bash
uv pip install -e ".[federated]"
uv pip install --no-deps -e ./examples/federated/flower-secagg
uv pip install "git+https://github.com/CMG-GUTS/curated-microbiota.git@v0.2.0"
```

Flower is optional: a standard mllabiome installation does not require it.

## Quick demo

```bash
uv run --no-sync mllabiome examples/federated/fl__lampp_lodo_crc_quick.py --stage federated-evaluate
```

The quick demo holds out LAMPP study `2` and trains on the other studies using **one local epoch and one SecAgg+ round**. It is intended to check that the end-to-end protocol works, not to produce a publishable benchmark result.

## Full LODO evaluation

```bash
uv run --no-sync mllabiome examples/federated/fl__lampp_lodo_crc.py --stage federated-evaluate
```

By default, the full example evaluates all studies as held-out folds, with ten federated rounds per fold. It will take substantially longer than the quick demo.

## Choose your settings

You can edit the uppercase variables at the top of either example script. There is no separate dataset CSV to locate or download manually: `curated-microbiota` supplies the LAMPP CRC dataset and benchmark splits.

| Setting | What it controls |
| --- | --- |
| `EXPERIMENT_DIR` | Where evaluation results and logs are written |
| `HELD_OUT_STUDY` | A study identifier such as `"2"`; `None` runs all folds |
| `NUM_ROUNDS` | Number of secure aggregation rounds per fold |
| `LOCAL_EPOCHS` | Local training epochs per participating study |
| `LEARNING_RATE` | Learning rate for the logistic model |
| `L2_ALPHA` | Model regularization strength |
| `CLIENT_CPUS` | CPU allocation per simulated client; start with integer `1` |
| `NUM_SHARES`, `RECONSTRUCTION_THRESHOLD` | SecAgg+ sharing and reconstruction settings |

The quick example writes its results separately from the full experiment, so you can test first without replacing your longer run's outputs.

## Follow the run

The terminal reports dataset loading, MPDR preparation, the current held-out study, client staging, and Flower activity. Flower output is also saved while the run is active.

For the quick demo, look in:

```text
runs/fl_lampp_lodo_crc_quick/federated/
```

After a successful evaluation, the results include:

| File | Contents |
| --- | --- |
| `fold_metrics.csv` | Metrics for each completed held-out study |
| `predictions.csv` | Per-sample held-out predictions and probabilities |
| `mpma_configs.csv` | mllabiome configuration identifiers |
| `run.json` | Settings and benchmark provenance |
| `flower-logs/*.log` | Detailed Flower messages for each fold |

The full example uses `runs/fl_lampp_lodo_crc/federated/` by default.

If you see `Flower SecAgg+ running` for a long time, check the log being printed in the terminal or the corresponding file in `flower-logs/`. A running spinner alone does not establish that a secure round has completed. Metrics files are written after the requested folds finish successfully.

## What this demonstration does and does not show

This is a **single-machine federated simulation**. Each training study is represented as a client, and the held-out study never trains the model. SecAgg+ protects individual model updates during aggregation under its protocol assumptions.

The benchmark is still loaded on the machine running the simulation. This is not equivalent to a deployment in which hospitals keep data on independent systems, and SecAgg+ does not provide differential privacy. The quick demo is for checking functionality; use the complete LODO evaluation and appropriate validation for scientific results.
