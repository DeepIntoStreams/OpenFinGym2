********
Concepts
********

Scopes
======

A scope is a research area the pipeline harvests papers from. Scopes are defined
in ``conf/pipeline_config.yaml`` and are processed independently at every stage,
so adding a scope does not affect the others.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``id``
     - Identifier used for database rows and run output filenames
   * - ``name``
     - Human-readable name, passed to the judge and task extractor as context
   * - ``task_type``
     - ``forecasting`` or ``generation``, see :ref:`task-types`
   * - ``description``
     - Prose definition of the area, passed to the LLM stages as context
   * - ``enabled``
     - Scopes with ``false`` are skipped by every stage
   * - ``categories``
     - arXiv categories the search is restricted to, e.g. ``q-fin.ST``
   * - ``queries``
     - arXiv search expressions, see :ref:`arxiv-queries`

Papers are keyed by ``(paper_id, scope_id)``, so the same paper can be collected
by several scopes and is judged separately for each. Full text is retrieved and
chunked only once per paper, and shared across the scopes that collected it.

The task type as an invariant
-----------------------------

A scope declares one task type, and every task built from it is of that type.
The invariant is enforced twice:

- **At retrieval**, by excluding the other type from the scope's queries.
- **At judgement**, by rejecting papers whose experiment is of the other type.

The task extractor does not re-derive the type; it takes ``scope.task_type``.
This keeps a single source of truth, at the cost of rejecting papers that are
good but land in the wrong scope.

Both stages judge a paper by *what its experiment does*, not by the method it
uses. A generative model used to forecast is a forecasting paper.

Tasks
=====

A task is a machine learning assessment derived from one experiment in one
paper. The agent under assessment is given a description and the training data,
writes its outputs to a file, and is scored against ground truth it never sees.

Datasets are grouped by role:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Role
     - Given to agent
     - Purpose
   * - ``training_inputs``
     - yes
     - Data the model is fit on
   * - ``training_targets``
     - yes
     - Ground truth paired with the training inputs
   * - ``test_inputs``
     - yes
     - Inputs the agent conditions its output on
   * - ``test_outputs``
     - produced
     - What the agent writes to ``/logs/artifacts/``
   * - ``test_targets``
     - **withheld**
     - Ground truth the outputs are scored against

.. _task-types:

Task types
----------

The two task types differ in whether the agent's output rows correspond to the
ground-truth rows, which decides both the shape of the task and how it can be
scored.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - ``forecasting``
     - ``generation``
   * - Agent produces
     - Predictions from given inputs
     - Samples from a fitted distribution
   * - ``test_inputs``
     - Required
     - Usually empty, sampling is unconditional
   * - ``training_targets``
     - Required
     - Optional, the model is often fit to the inputs alone
   * - Row correspondence
     - Row *i* of the output matches row *i* of the target
     - None, the two sets are compared as distributions
   * - Typical metrics
     - MSE, MAE, directional accuracy
     - Wasserstein distance, ACF error

Because the row correspondence differs, the two types cannot share a scoring
script. The task generator selects the wording for the metric prompt from the
task type, and the task critic checks a different set of required dataset roles
for each.
