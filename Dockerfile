# syntax=docker/dockerfile:1
# Build from the ROOT of your local mllabiome repository.
# Only pyproject.toml, uv.lock, README.md, LICENSE, and mllabiome/ enter the build.
FROM nvidia/cuda:12.9.2-base-ubuntu22.04 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.12

WORKDIR /app
# Install/cache all locked third-party dependencies, not the local project yet.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --all-groups --no-install-project --no-editable

# Explicit copies: no data/, results/, sandbox/, tests/, docs/, etc.
COPY README.md LICENSE ./
COPY mllabiome/ ./mllabiome/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --all-groups --no-editable

# Runtime contains installed mllabiome + dependencies, not source checkout.
FROM nvidia/cuda:12.9.2-base-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libgl1 libglib2.0-0 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
WORKDIR /workspace
CMD ["mllabiome", "--help"]
