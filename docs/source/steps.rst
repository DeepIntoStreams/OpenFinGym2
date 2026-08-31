*****
Steps
*****

The pipeline runs seven stages in order, each over every enabled scope. Stages
communicate through the database rather than in memory, so a stage picks up
whatever the previous run left behind and an interrupted run can be resumed.

Every stage also writes its results as JSON under the Hydra run directory
(``outputs/<date>/<time>/``), which is the fastest way to inspect what an LLM
stage actually decided.

Scraping
========

Searches arXiv for each of the scope's queries, restricted to the scope's
categories and to the configured date window.

Each query is allocated ``max_papers_per_scope`` divided by the number of
queries, rounded up. Results are deduplicated across queries and the budget is
applied to the deduplicated set, so overlapping queries do not waste it.

Records are then enriched with citation counts, venue, and peer-review status
from Semantic Scholar, and inserted into the database. Papers already present
for that scope are skipped, so re-running does not re-fetch them.

Papers enter as ``SCRAPED``.

Retrieval
=========

Downloads the full text of newly scraped papers and splits it into chunks.

Sources are tried in the order given by ``source_preference``. The arXiv LaTeXML
rendering at ``/html/{id}`` is preferred over the PDF: it preserves table
structure and hyperlinks that PDF text extraction loses, which matters because
dataset locations are frequently given as links. The PDF path remains as a
fallback for papers arXiv has not rendered.

The markdown is split on headings, and the chunks stored against the paper. Text
is retrieved once per paper and shared by every scope that collected it.

Papers become ``EXTRACTED``, or ``REJECTED`` if they have no retrievable source.

Judgement
=========

Decides which papers are worth building a task from, in two passes.

**Prefilter.** Every paper is screened on title and abstract alone. This is the
cheap pass, and it rejects papers that are off-topic, are surveys or position
papers, or whose experiment is of the other task type.

**Ranking and cutoff.** Surviving papers are ranked, and only the top
``sift_budget`` per scope go through to the expensive pass. Ranking is driven
primarily by the prefilter relevance score, amplified by a citation and venue
quality term and nudged by recency. The boost is multiplicative, so a
weakly-relevant famous paper cannot overtake a strongly-relevant unknown one.
Papers below the cutoff are rejected as ``JudgeCutoff``.

**Sift.** The remaining papers are judged on their full text. A paper is
accepted only if it shows strong evidence of a relevant experiment, describes
its datasets and metrics in enough detail to reconstruct them, matches the scope
task type, and uses data that is public or reconstructible from public sources.

Data availability is decided independently of relevance and overrides the label:
a paper judged relevant but whose data cannot be obtained is rejected. Access
through a paid vendor does not by itself make data proprietary, as long as the
same series is obtainable from a free, scriptable source.

Accepted papers become ``ACCEPTED``; the rest are ``REJECTED`` with a reason.

Task Extraction
===============

Reads an accepted paper's full text and specifies one assessment from it: the
task description, the datasets in each role, and the metrics.

References, acknowledgements, and appendices are dropped before the text reaches
the LLM. If a paper contains several experiments, one is selected.

The task type is not inferred here; it is taken from the scope. Dataset
descriptions must be detailed enough for a later stage to write a download
script, so direct links found in the paper are preferred.

Produces a task candidate with status ``NEW``.

Task Critic
===========

Gates candidates before the expensive generation stage, in two passes.

**Structural checks**, which use no LLM: that every dataset role required by the
task type is populated, that every dataset has a source or a download link, that
metrics exist and reference only datasets that exist, and that filenames are not
reused across roles. A candidate failing any of these is rejected outright.

**Critique**, which scores the surviving candidates on internal consistency,
completeness, and data availability. Candidates scoring below
``threshold_default`` are rejected.

Candidates become ``APPROVED`` or ``REJECTED``.

Task Generation
===============

Turns an approved candidate into runnable code:

- a training script that downloads the training and test input datasets,
- a testing script that downloads the withheld test targets,
- an assessment script that scores the agent's output and writes the result to
  ``/logs/verifier/reward.json``,
- the agent-facing instructions, a short description, and a difficulty note.

The metric prompt states whether output rows correspond to target rows, which
comes from the task type, so a generation task is scored distributionally rather
than row by row.

Candidates become ``PROCESSED``, or ``FAILED`` if generation errored.

Task Export
===========

Writes each generated task to ``export_path`` as a Harbor task directory:

.. code-block:: text

    <task_name>/
        task.toml          metadata, resource limits, timeouts
        instruction.md     what the agent is asked to do
        environment/       agent image
            Dockerfile
            data.py        downloads training and test input data
        tests/             verifier image
            Dockerfile
            data.py        downloads withheld test targets
            grader.py      scores the output

The agent and verifier are separate images, so the agent cannot reach the test
targets. Task names come from the LLM, so they are reduced to lowercase words
joined by underscores before being used as a directory name.
