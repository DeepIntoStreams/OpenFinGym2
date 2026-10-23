# OpenFinGym

Pipeline for automatic creation of finance related machine learning tasks for
agent benchmarking and training.

This is the update project based on the paper
[OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents](https://arxiv.org/pdf/2606.26350).

The original implementation of the pipeline and gym used in the paper
can be found [here](https://github.com/DeepIntoStreams/OpenFinGym).

## Documentation

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

#### Installation

OpenFinGym uses uv to handle dependencies and run scripts. See
installation instructions
[here](https://docs.astral.sh/uv/getting-started/installation/).

#### Configuration

The pipeline uses [Hydra](https://hydra.cc/docs/intro/) for configuration, with configuration
files located in `./conf` by default.

Any API keys should be placed in a `.env` file:

- LLM access token
- Github pat token to allow agent to use the [Github MCP service](https://github.com/github/github-mcp-server)
- Alpaca paper trading account tokens to allow agents to access the [Alpaca MCP service](https://github.com/alpacahq/alpaca-mcp-server)
- Kaggle API token to allow agents access the [Kaggle MCP service](https://www.kaggle.com/blog/kaggles-official-mcp-server)

The MCP related tokens are optional (relevant tools will be skipped if not provided),
but they allow agents to lookup docs, repos, datasets etc.

**Note: Agents are only provided MCP tools that can read data relevant to task generation,
but it still may be sensible to generate pipeline specific tokens with low access rights.**

See [here](https://deepintostreams.github.io/OpenFinGym2/configuration.html) for details of
configuration pipeline runs.

If using MlFlow for LLM tracking you should run a client with

```commandline
uv run task mlflow
```

#### Batch Pipeline

The pipeline can then be run using

```commandline
uv run task pipeline
```

#### Single Paper Pipeline

To run the pipeline and produce a task for a specific paper you should use

```commandline
uv run task paper scopes=<SCOPE-ID> arxiv_id=<PAPER-ARXIV-ID>
```

Where `SCOPE-ID` specifies the scope to use when generated the task,
and `PAPER-ARXIV-ID` is the arxiv assigned id of the paper, e.g. `1706.03762v7`.

## Developers

Contributions are very welcome. Developer notes can be found [here](.github/docs/developers.md).

Please raise any issues or suggested features [here](https://github.com/DeepIntoStreams/OpenFinGym2/issues).
