.. _running:

********************
Running the Pipeline
********************

=====
Setup
=====

OpenFinGym uses uv to handle dependencies and to run scripts. See
installation instructions `here <https://docs.astral.sh/uv/getting-started/installation/>`_.

Any API keys should be placed in a `.env` file in the repo root, including:

- LLM access token
- Github pat token to allow agent to use the `Github MCP service <https://github.com/github/github-mcp-server>`_
- Alpaca paper trading account tokens to allow agents to access the `Alpaca MCP service <https://github.com/alpacahq/alpaca-mcp-server>`_
- Kaggle API token to allow agents access the `Kaggle MCP service <https://www.kaggle.com/blog/kaggles-official-mcp-server>`_

The MCP related tokens are optional (relevant tools will be skipped if not provided),
but if provided they allow agents to lookup docs, repos, datasets etc.

.. caution::

   Agents are only provided MCP tools that can read data relevant to task generation,
   but it still may be sensible to generate pipeline specific tokens with low access rights.

See :ref:`configuration` for further details of configuring the pipeline.

=======
Running
=======

If using MlFlow for LLM tracking you should run a client with

.. code-block::

   uv run task mlflow

To run the pipeline with the current configuration with

.. code-block::

   uv run task pipeline

You can override configuration values from the command line,
for example

.. code-block::

   uv run task pipeline scraping.max_papers_per_scope=20

See :ref:`configuration` or the `Hydra docs <https://hydra.cc/docs/patterns/extending_configs/>`_
for more details.

======
Output
======

Each run of the pipeline will initialise an output folder in ``outputs/`` folder of the repo
root, indexed by run date and time. Debug outputs for each step are written to this folder.

Tasks are written to a ``tasks/`` folder in the repo root. Each folder being a task in the
layout expected by `Harbor <https://docs.harborframework.com/tasks/overview>`_.
