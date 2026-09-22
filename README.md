<img src="https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/favicon.svg" width="80" height="80" alt="mllabiome icon">

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


## Example configs

Runnable example configurations are available under `examples/`:

```bash
mllabiome examples/ptsd_configs_sweep.py
mllabiome examples/ptsd_configs_sweep.py --stage evaluate
mllabiome examples/ptsd_configs_sweep.py --stage ensemble
mllabiome examples/ptsd_configs_sweep.py --stage explain

mllabiome examples/ibs_lodo_configs_sweep.py --stage evaluate
```


<i>Example report sections generated with `mllabiome`.</i>

![XAI report preview](https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/report-preview-xai.png)

![ALE report preview](https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/report-preview-ale.png)


## Features already included

The framework automatically supports:
* Leakage-aware nested model selection
* Fold-isolated preprocessing
* Outer-fold performance evaluation
* Leave-one-dataset-out validation
* Compositional microbiome transformations
* Taxonomic-resolution selection
* Multi-model benchmarking
* Ensemble learning
* Out-of-fold model explainability
* Cross-fold feature stability analysis
* Multimodal late integration
* Reproducible configuration-driven workflows


## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

