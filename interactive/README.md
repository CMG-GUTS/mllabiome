# Interactive inference

The original `mllabiome-ii` Next.js frontend is retained under `interactive/frontend/`, including its layout, styles, and light/dark themes. It uses fitted models and results directly from the selected experiment run.

```bash
mllabiome examples/base/ncv__lampp_scz_cm_configs_sweep.py --stage interactive-inference
```

Normal `uv sync` does not install Node.js or any frontend dependencies. The first interactive launch requires Node.js 20+ and npm and automatically runs `npm ci` and the Next.js static build. Subsequent launches reuse `interactive/frontend/out/`. Ollama chat is optional and lists only installed local models.

`node_modules/`, `.next/`, and `out/` remain untracked. Only frontend source and its lockfile are committed. The Python wheel does not contain the frontend.
