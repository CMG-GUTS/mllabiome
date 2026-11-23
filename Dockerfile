FROM nvidia/cuda:12.9.2-base-ubuntu22.04 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential pkg-config libgomp1 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --all-groups --no-install-project --no-editable

COPY README.md LICENSE ./
COPY mllabiome/ ./mllabiome/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --all-groups --no-editable

FROM nvidia/cuda:12.9.2-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git libgomp1 libgl1 libglib2.0-0 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv
COPY --from=hf_local / /opt/mllabiome-models/huggingface/hub/
COPY --from=tabpfn_local / /opt/mllabiome-models/tabpfn/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HF_HOME=/opt/mllabiome-models/huggingface \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TABPFN_MODEL_CACHE_DIR=/opt/mllabiome-models/tabpfn \
    TABPFN_NO_BROWSER=1

WORKDIR /workspace

RUN python - <<'PY'
from pathlib import Path

root = Path('/opt/mllabiome-models/huggingface/hub/models--nums-ai--causilo')
required = ('classifier/config.json', 'classifier/model.safetensors')
ref = root / 'refs' / 'main'
snapshots = root / 'snapshots'

if not snapshots.is_dir():
    raise RuntimeError(f'Causilo snapshot directory missing: {snapshots}')

complete = sorted(
    snapshot for snapshot in snapshots.iterdir()
    if snapshot.is_dir() and all(
        (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
        for name in required
    )
)

current = (snapshots / ref.read_text().strip()) if ref.is_file() else None
if current not in complete:
    if len(complete) != 1:
        raise RuntimeError(
            f'Causilo has {len(complete)} complete cached revisions; '
            f'expected one, found {[p.name for p in complete]}. '
            'The Causilo classifier configuration and weights are required.'
        )
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(complete[0].name)
    print('Causilo main reference repaired:', complete[0].name)
else:
    print('Causilo main reference valid:', current.name)
PY

RUN python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from mllabiome.limix2 import LIMIX2_FILENAME, LIMIX2_REPO_ID, LIMIX2_REVISION

for repo_id, revision in (
    ('Neuralk-AI/seldon', None),
    ('Neuralk-AI/seldon-mini', None),
    ('nvidia/Kumo-Tabular', 'v1.0.1'),
    ('nums-ai/causilo', None),
):
    snapshot = Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    if not snapshot.is_dir():
        raise RuntimeError(f'Cache missing for {repo_id}')
    print(f'Offline cache resolved: {repo_id}')

for repo_id in ('Neuralk-AI/seldon', 'Neuralk-AI/seldon-mini'):
    snapshot = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    if not (snapshot / 'config.json').is_file():
        raise RuntimeError(f'Missing config.json for {repo_id}')
    if not any(p.is_file() and p.stat().st_size > 0 for p in snapshot.rglob('*.safetensors')):
        raise RuntimeError(f'Missing Seldon weights for {repo_id}')

kumo = Path(snapshot_download(repo_id='nvidia/Kumo-Tabular', revision='v1.0.1', local_files_only=True))
for size in ('small', 'medium', 'large'):
    for task in ('classifier', 'regressor'):
        path = kumo / size / f'{task}.pt'
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Missing cached Kumo weights: {size}/{task}.pt')

causilo = Path(snapshot_download(repo_id='nums-ai/causilo', local_files_only=True))
for filename in ('classifier/config.json', 'classifier/model.safetensors'):
    path = causilo / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f'Missing cached Causilo weights: {filename}')

limix = Path(hf_hub_download(
    repo_id=LIMIX2_REPO_ID,
    filename=LIMIX2_FILENAME,
    revision=LIMIX2_REVISION,
    local_files_only=True,
))
if not limix.is_file() or limix.stat().st_size == 0:
    raise RuntimeError('LimiX-2-1016 checkpoint is missing')

tabpfn = Path('/opt/mllabiome-models/tabpfn')
if not any(p.is_file() and p.stat().st_size > 0 and p.suffix.lower() in {'.ckpt', '.pt', '.pth', '.safetensors'} for p in tabpfn.rglob('*')):
    raise RuntimeError('No TabPFN model checkpoint in the supplied cache')

print('Required pretrained checkpoints are available offline')
PY

RUN python -c "import mllabiome, pyarrow, skbio, torch, sdm, tabpfn, neuralk_oss, causilo, limix; print('mllabiome backends import OK')" \
    && mllabiome --help >/dev/null

CMD ["mllabiome", "--help"]
