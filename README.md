# OpenFinGym

## Running

OpenFinGym uses uv to handle dependencies and run scripts. See
installation instructions
[here](https://docs.astral.sh/uv/getting-started/installation/).

The pipeline can then be run using

```commandline
uv run pipeline
```

If using MlFlow for LLM tracking you should run a client with

```commandline
uv run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///./mlruns.db
```

## Developers

Developer notes can be found [here](.github/docs/developers.md)
