.. _single_paper:

************************
Process a Specific Paper
************************

Rather than running the pipeline across a batch of scraped papers, you can run the
pipeline to generate a task for a specific paper.

The default single paper configuration is specified in ``conf/manual_pipeline_config.yaml``
using the same step configuration files as the main pipeline (see :ref:`configuration` for details).

The pipeline can then be run using:

.. code-block::

   uv run task paper scopes=<SCOPE-ID> arxiv_id=<PAPER-ARXIV-ID>

where ``SCOPE-ID`` specifies the scope to use when generated the task,
and ``PAPER-ARXIV-ID`` is the arxiv assigned id of the paper, e.g. ``1706.03762v7``.

To skip the paper-judgement stage (i.e. to automatically approve the paper for task generation)
set the ``enable_judge`` flag in the configuration file.
