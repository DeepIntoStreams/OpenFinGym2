# OpenFinGym

Pipeline for automatic creation of finance related machine learning tasks for
agent benchmarking and training.

Full documentation can be found [here](https://deepintostreams.github.io/OpenFinGym2/).

## Task Generation Pipeline

The task generation is designed to scrape Arxiv papers (that match a given scope) and extract
machine learning tasks suitable for LLM agent assessment or training. See
[here](https://deepintostreams.github.io/OpenFinGym2/pipeline.html) for more details of the
pipeline.

The generated tasks implement the [Harbor task structure](https://www.harborframework.com/docs/tasks)
with isolated assessment of the agent output, allowing for the use of Harbor task execution tooling
and training integrations.

### Running

OpenFinGym uses uv to handle dependencies and run scripts. See
installation instructions
[here](https://docs.astral.sh/uv/getting-started/installation/).

The pipeline uses [Hydra](https://hydra.cc/docs/intro/) for configuration, with configuration
files located in `./conf` by default.

Any LLM API secrets should be placed in a `.env` file.

See [here](https://deepintostreams.github.io/OpenFinGym2/configuration.html) for details of
configuration pipeline runs.

The pipeline can then be run using

```commandline
uv run task pipeline
```

If using MlFlow for LLM tracking you should run a client with

```commandline
uv run task mlflow
```

## Developers

Contributions are very welcome. Developer notes can be found [here](.github/docs/developers.md).

Please raise any issues or suggested features [here](https://github.com/DeepIntoStreams/OpenFinGym2/issues).
