FROM ghcr.io/astral-sh/uv:trixie-slim

ENV UV_PROJECT_ENVIRONMENT="/usr/local"

# Warm the cache so the agent's first `uv run --with requests` is instant
RUN uv run --with requests python -c "import requests"

RUN useradd -ms /bin/bash agent
WORKDIR /home/agent
