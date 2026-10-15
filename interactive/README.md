# Interactive inference

```bash
mllabiome examples/base/ncv__lampp_scz_cm_configs_sweep.py --stage interactive-inference
```

Normal `uv sync` does not install Node.js or any frontend dependencies. The first interactive launch requires Node.js 20+ and npm and automatically runs `npm ci` and the Next.js static build. Subsequent launches reuse `interactive/frontend/out/`. Ollama chat is optional and lists only installed local models.

