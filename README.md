<img src="https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/favicon.svg" width="80" height="80" alt="mllabiome icon">

## Example configs

Runnable example configurations are available under `examples/`:

```bash
mllabiome examples/ptsd_configs_sweep.py
mllabiome examples/ptsd_configs_sweep.py --stage evaluate
mllabiome examples/ibs_lodo_configs_sweep.py --stage evaluate
```


<i>Example report sections generated with `mllabiome`.</i>

![XAI report preview](https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/report-preview-xai.png)

![ALE report preview](https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/report-preview-ale.png)

Install is recommended with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install mllabiome
```

For local development from a checked-out source tree:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv run pytest -q
```

```python
from mllabiome import mll
```

## Terms

- **MPDR — Microbiome Profile Data Representation**: one taxonomic resolution plus one count transformation.
- **MPMA — Microbiome Profile Modelling Algorithm**: one MPDR plus one learner.
- **MPMAs Ensemble**: a selected and aggregated pool of qualified MPMAs.


## Package layout

The implementation is split by responsibility:

```text
mllabiome/
  transformations.py   abundance transformations and display abbreviations
  resolutions.py       taxonomic-resolution parsing and MPDR materialisation
  learners.py          learner construction and user-supplied estimator handling
  configs_sweep.py     configured MPMA sweep
  ensemble.py          ensemble-facing public API
  ensemble_sweep.py    MPMAs Ensemble search
  explainability.py    strict multi-method explainability
  data.py              microbiome/metadata loading
  figures.py           sweep and explainability figures
  metrics.py           prediction metrics
  pipeline.py          end-to-end execution
```

## Workflow

A sweep is an ordinary Python config. Running the config executes:

```text
configured MPMA sweep -> ensemble sweep -> out-of-fold explainability -> report
```

Configuration files are intentionally editable: add or remove resolution sets, transformations, and learners by editing the Python lists or commenting/uncommenting entries. Runs are resumable; completed MPMA/split evaluations are kept and newly enabled or missing configurations are evaluated on rerun.

```bash
mllabiome examples/sweep_config.py
```

Stage entry points are available for restart/reuse:

```bash
mllabiome examples/sweep_config.py --stage evaluate
mllabiome examples/sweep_config.py --stage ensemble
mllabiome examples/sweep_config.py --stage explain
```

## Config style

Use plain Python lists and functions: define resolution sets, build count transformations, build models.

```python
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from mllabiome import mll

_RESOLUTION_SETS = [
    # ("genus", ("genus",)),
    ("species", ("species",)),
    # ("genus-species", ("genus", "species")),
    # ("family+species", ("family", "species")),
]


def _build_count_transformations():
    T = mll.Transformation
    return [
        # T("none"),             # RA
        T("arcsin_sqrt"),  # $\arcsin\sqrt{x}$
    ]


def _build_models():
    M = []
    M.append(
        (
            "RF_1000_msl5",
            RandomForestClassifier(
                n_estimators=1000,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=42,
            ),
        )
    )
    # M.append(("RF_500_msl3", RandomForestClassifier(
    #     n_estimators=500, min_samples_leaf=3, n_jobs=1, random_state=42,
    # )))
    return M


TITLE = "example mllabiome sweep"
EXPERIMENT_DIR = Path("examples/runs/EXAMPLE")

DATA = mll.Data(
    abundance_path=Path("data/microbiome.tsv"),
    metadata_path=Path("data/metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sample_id",
    target_col="label",
    label_map={"control": 0, "case": 1},
    class_labels=("control", "case"),
    positive_class=1,
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
)
GATE = mll.QualificationGate(enabled=False, metric="nMCC", threshold=0.51)
ENSEMBLE = mll.Ensemble(sizes=(3,), optimize_metric="nMCC")
EXPLAINABILITY = mll.Explainability(targets="auto", top_k=30)
```

`Transformation(name)` uses a provided abundance transformation. `Transformation(name, fn)` applies a custom `fn(X)` independently to train and test. `Transformation(name, fn, True)` applies `fn(X_train, X_test)` and is intended for fold-fitted transformations where parameters must be estimated from the training fold only.

Provided transformations expose only compact labels for outputs:

- `count_transformation`: implementation key used in configs and result files.
- `transformation_abbreviation`: figure/table label used in sweep summaries.

Transformation abbreviations are written alongside every MPMA row in `configs.tsv` and downstream result tables. No separate transformation-space table is emitted by default.


## Data formats

`Data(format=...)` supports:

- `"metaphlan_tsv"`: a taxonomic profile table with clades/features as rows and samples as columns, plus a separate metadata table. The first column is used as the feature/clade index.
- `"wide_csv"`: one row per sample, with metadata columns plus numeric feature columns in the same CSV.
- `"auto"`: chooses `metaphlan_tsv` when a TSV profile and metadata file are supplied, otherwise `wide_csv`.

`"matrix_tsv"`, `"profile_tsv"`, and `"csv"` are accepted aliases for compatibility, but examples use the clearer names above.

Label encoding should be explicit:

```python
DATA = mll.Data(
    abundance_path=Path("examples/data/PTSD/PTSD_profiles.tsv"),
    metadata_path=Path("examples/data/PTSD/PTSD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col="group",
    label_map={"Placebo": 0, "Active": 1},
    class_labels=("Placebo", "Active"),
    positive_class=1,
)
```

Repeated nested-CV is stratified by the target label by default. To additionally balance a metadata column, set `stratify_col` on `Data`; the splitter then stratifies on the composite target-plus-column label.

```python
DATA = mll.Data(
    abundance_path=Path("examples/data/PTSD/PTSD_profiles.tsv"),
    metadata_path=Path("examples/data/PTSD/PTSD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col="group",
    stratify_col="batch",
    label_map={"Placebo": 0, "Active": 1},
    class_labels=("Placebo", "Active"),
    positive_class=1,
)
```

## Strict default explainability

By default, explainability uses `targets="auto"`, which runs the standard comparison suite: MPMA-E when an ensemble has been selected, MPMA-B, and the strict Baseline RF when that baseline exists. For each target, the configured methods run on outer-test, out-of-fold predictions: every explained row is predicted by a model fitted only on that row's corresponding outer-training fold. The configured methods run SHAP, LIME, one-dimensional ALE, permutation importance, and two-dimensional ALE interactions. The package does not replace one requested method with another. If a required dependency is unavailable, or if a method cannot explain the selected target, the stage raises an error.

```python
EXPLAINABILITY = mll.Explainability(
    targets="auto",
    top_k=30,
    # default methods=("shap", "lime", "ale", "permutation", "interactions")
)
```

For a targeted rerun, users may explicitly reduce `methods`; nothing is substituted automatically.

## Outputs

Each run writes tables and figures under `experiment_dir`:

- `configs.tsv`, `configs.db`
- `tables/mpma_rankings.tsv`
- `tables/qualification_gate.tsv`
- `tables/representation_impact_cells.tsv`
- `figures/mpma_top_metric.png`
- `figures/representation_impact.{png,pdf,svg}`
- `ensembling/final_model_comparison.tsv`
- `ensembling/selected_unit.json`
- `ensembling/ensemble_candidates.png`
- `explainability/<target>/oof_predictions.tsv`
- `explainability/<target>/feature_importance_{shap,lime,ale,permutation}.tsv`
- `explainability/<target>/feature_importance.tsv`
- `explainability/<target>/ale_curves.tsv`
- `explainability/<target>/feature_interactions_current.csv`
- `explainability/<target>/feature_interactions_corrected.csv`
- `explainability/<target>/feature_interactions_fixed_pairs.csv`
- `explainability/<target>/feature_interactions_corrected_fixed.csv`
- `explainability/<target>/feature_interactions_by_target_all_methods.csv`
- `explainability/<target>/figures/feature_importance.png`
- `explainability/<target>/figures/interaction_network_current.png`



## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

