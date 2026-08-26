FROM ghcr.io/astral-sh/uv:trixie-slim

ENV UV_PROJECT_ENVIRONMENT="/usr/local"

RUN uv pip install --system requests
RUN useradd -ms /bin/bash agent
USER agent
WORKDIR /home/agent
