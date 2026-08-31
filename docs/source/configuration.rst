.. _configuration:

*************
Configuration
*************

The pipeline is configured with `Hydra <https://hydra.cc/>`_. ``conf/pipeline_config.yaml``
is the root: it defines the scopes and selects one configuration file per stage
from the ``conf/`` subdirectories.

.. code-block:: text

    conf/
        pipeline_config.yaml       scopes, database, stage selection
        scraping/                  paper search
        retrieval/                 full-text download
        judge/                     paper acceptance
        task_extractor/            paper to task specification
        task_critic/               candidate gating
        task_generator/            code generation
        task_exporter/             Harbor output

Any value can be overridden on the command line without editing a file ::

    uv run task pipeline scraping.max_papers_per_scope=20 judge.sift_budget=8

and a whole stage can be swapped for another file in its directory ::

    uv run task pipeline judge=my_judge

Secrets are not held in these files. ``.env`` in the repository root is loaded at
startup, and the LLM clients read their credentials from the environment.

Scopes
======

Scopes are defined under ``scopes:`` in ``conf/pipeline_config.yaml``. See
:doc:`concepts` for what the fields mean. A scope can be disabled with
``enabled: false`` rather than deleted, which keeps its history in the database.

.. _arxiv-queries:

arXiv queries
-------------

Each entry in a scope's ``queries`` is an arXiv search expression, passed to the
API largely as written. The syntax has several traps that fail *silently* by
returning a plausible but wrong set of papers, so queries are worth testing
against the API before relying on them.

**Multi-word terms must be quoted.** An unquoted multi-word term is parsed as a
disjunction of its words, so ``all:financial time series`` matches any paper
containing any one of those three words.

**Only the first clause needs a field prefix.** Later clauses inherit ``all:``.
Use ``ti:`` to restrict a clause to titles.

**Wildcards are unsupported**, and truncating a word to stem it manually makes
the query worse rather than broader. arXiv already applies stemming.

**ANDNOT binds more loosely than AND.** This is the trap most likely to go
unnoticed, because the pipeline appends the category and date filters to every
query. Written unparenthesised ::

    <terms> ANDNOT ti:"forecasting"

the expression the API receives is ::

    <terms> ANDNOT (ti:"forecasting" AND <categories> AND <dates>)

which subtracts almost nothing and, worse, leaves ``<terms>`` unconstrained by
category and date. An exclusion must therefore wrap the whole expression ::

    (<terms> ANDNOT ti:"forecasting")

A useful check is that adding an exclusion should only ever remove papers. If
papers appear that were not in the result set before, the exclusion is being
parsed the wrong way.

Stages
======

Scraping
--------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``since`` / ``until``
     - Submission date window, ``YYYY-MM-DD``
   * - ``max_papers_per_scope``
     - Papers kept per scope, split across its queries
   * - ``arxiv.sort_by``
     - How the API ranks within the window, e.g. ``Relevance``
   * - ``arxiv.request_interval_sec``
     - Delay between API requests
   * - ``semantic_scholar.enabled``
     - Whether to enrich with citation and venue data

A narrow date window starves the more specific scopes, since the budget cannot
be filled. Widening the window is usually better than raising the budget.

Retrieval
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``source_preference``
     - Order sources are tried in, e.g. ``[html, pdf]``
   * - ``arxiv_html_url``
     - Template for the LaTeXML rendering
   * - ``request_interval_sec``
     - Delay between downloads
   * - ``user_agent``
     - Sent with every request; identify the crawler honestly

Source tarballs are not an option, as arXiv's ``robots.txt`` disallows
``/e-print``. ``/html`` and ``/pdf`` are both permitted.

Judge
-----

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``prefilter_enabled``
     - Whether to screen on title and abstract first
   * - ``sift_budget``
     - Papers per scope that reach the full-text pass
   * - ``threshold_default``
     - Minimum score, out of 10, for acceptance
   * - ``ranking_citation_boost``
     - How far citations can amplify a paper's rank
   * - ``ranking_recency_weight``
     - Weight of the recency term
   * - ``ranking_recency_half_life_days``
     - Age at which the recency term halves

``sift_budget`` is the main cost control: it caps the number of full-text LLM
calls per scope regardless of how many papers were scraped.

Task critic
-----------

``threshold_default`` sets the minimum critique score for a candidate to reach
generation. Structural checks run before the LLM and are not configurable, as
they encode requirements the later stages depend on.

Task generator and exporter
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``task_generator.templates_path``
     - Jinja templates for instructions and Dockerfiles
   * - ``task_exporter.export_path``
     - Where task directories are written
   * - ``task_exporter.task_config.org_name``
     - Organisation prefix in the exported task name

Language models
---------------

Every LLM stage takes an ``llm`` block instantiated by Hydra, so the provider can
be changed per stage ::

    llm:
      _target_: langchain_openai.chat_models.ChatOpenAI
      model: gpt-4.1-mini
      temperature: 0

The judge and critic run at ``temperature: 0``, since they are scoring gates and
should not vary between runs. The extractor and generator run warmer.

Database
========

``db_engine`` is a SQLAlchemy URL, by default a local SQLite file. The database
carries state between stages and between runs: papers already scraped are not
re-fetched, text already retrieved is not re-downloaded, and a stage resumes from
whatever the previous one left. Deleting the file starts from scratch.
