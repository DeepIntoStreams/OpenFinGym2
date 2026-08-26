FROM ghcr.io/astral-sh/uv:trixie-slim

ENV UV_PROJECT_ENVIRONMENT="/usr/local"

WORKDIR /broker
COPY episode.json episode.json
COPY realtime/ open_fin_gym/realtime/
ENV PYTHONPATH=/broker
RUN uv run --with fastapi --with uvicorn --with requests python -c "import fastapi, uvicorn, requests"
CMD ["uv", "run", "--with", "fastapi", "--with", "uvicorn", "--with", "requests", \
     "uvicorn", "open_fin_gym.realtime.server:app", "--host", "0.0.0.0", "--port", "8000"]
