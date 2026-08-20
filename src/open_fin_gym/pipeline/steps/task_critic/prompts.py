from pydantic import BaseModel, Field

from open_fin_gym.pipeline.config import Scope, scope_context
from open_fin_gym.pipeline.db.tables import JudgeLabel
from open_fin_gym.pipeline.steps.task_extraction.utils import TaskSpecification


def _join(items) -> str:
    return "\n\t".join(f"- {x.model_dump()}" for x in items) or "\t(none)"


def build_critique_prompt(scope: Scope, spec: TaskSpecification) -> str:
    return f"""
You are reviewing a machine-learning task specification extracted from a research
paper, to check whether it is well-formed enough to generate task code from. Do not
invent details; judge only what is written below.

{scope_context(scope)}

Task name: {spec.task_name}
Task type: {spec.task_type}
Task description: {spec.task_description}

Training input datasets:
{_join(spec.training_inputs)}

Training target datasets:
{_join(spec.training_targets)}

Test input datasets:
{_join(spec.test_inputs)}

Test target datasets (withheld, used for grading):
{_join(spec.test_targets)}

Test output datasets (produced by the agent):
{_join(spec.test_outputs)}

Assessment metrics:
{_join(spec.metrics)}

Checks to make:
1) Consistency: do the datasets, metrics, and description match the declared task
   type? (e.g. a "generation" task samples outputs unconditionally and should not
   have test inputs; a "forecasting" task predicts from test inputs and should.)
2) Completeness: is each dataset specific enough (entities, period, frequency, real
   source/download path) to script against, without guessing?
3) Metric validity: do the metrics make sense given the test outputs/targets, and do
   they reference datasets actually present above?
4) Scope fit: does this task belong to the stated scope?
5) Data Availability: for each dataset, does the stated source/download_link
   plausibly point to something a script could retrieve without authentication
   or a paid subscription? A named vendor (e.g. Bloomberg, Refinitiv, WRDS) is
   not itself disqualifying if the same series is a standard public-market
   quantity (prices, returns, volumes, OHLCV, standard macro indicators) —
   only flag it if the specific instrument, period, or frequency is vague or
   unspecified. Flag genuinely proprietary content (hand-curated labels,
   broker-internal order flow, full-depth order-book data) as an issue.

If evidence is weak, ambiguous, or incomplete, reject by default.

Task:
1) Provide `consistency_assessment`.
2) Provide `completeness_assessment`.
3) Provide 'data_availability_assessment'.
4) Provide `issues`: concrete problems found, empty string if none.
5) Decide `label`: accepted/rejected, consistent with the assessments and issues.
6) Give `score` in `[0, 10]` and `confidence` in `[0.0, 1.0]`.
""".strip()


class TaskCritique(BaseModel):
    consistency_assessment: str
    completeness_assessment: str
    data_availability_assessment: str
    issues: str
    label: JudgeLabel
    score: float = Field(0.0, ge=0.0, le=10.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
