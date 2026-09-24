<img src="https://raw.githubusercontent.com/CMG-GUTS/mllabiome/main/assets/favicon.svg" width="80" height="80" alt="mllabiome icon">

<b>E</b>valuate, <b>e</b>nsemble, <b>e</b>xplain machine learning for microbiota data analysis in single-modality and multimodality scenarios.

***

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



## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

