from pydantic import BaseModel, Field, model_validator

from open_fin_gym.pipeline.config import Scope, scope_context
from open_fin_gym.pipeline.db.tables import JudgeLabel, Paper


def build_prefilter_prompt(scope: Scope, paper: Paper) -> str:
    return f"""
You are a strict prefilter for financial-ML paper screening.
Use only title + abstract evidence. Do not use assumptions beyond the given text.

{scope_context(scope)}

Paper title: {paper.title}
Paper abstract: {paper.abstract}
Paper categories: {", ".join(paper.categories) or "(none)"}

Decision policy:
- Accept only if there is **STRONG EVIDENCE** of in-scope ML work.
- Reject obvious surveys, position papers, theory-only papers, or clearly off-scope domains.
- When uncertain or evidence is weak, reject by default.

Task:
1) Write detailed `reasons` grounded in title/abstract words. Be specific about what evidence supports relevance or irrelevance, and how it relates to the scope definition, queries, and categories. Decide whether this paper has **STRONG EVIDENCE** of relevance while writing.
2) Decide `label`: accepted/rejected, consistent with `reasons`.
3) Estimate `relevance_score` in the range `[0, 10]`.
4) Estimate `confidence` in the range `[0.0, 1.0]`.
""".strip()


def build_sift_judge_prompt(
    scope: Scope,
    paper: Paper,
    excerpt: str,
) -> str:
    return f"""
You are judging whether the paper is suitable for research task construction, based on the provided fulltext excerpt and scope context. Do not invent details.

{scope_context(scope)}

Paper title: {paper.title}
Paper fulltext excerpt:
{excerpt}

Hard acceptance rule:
- Accept only if ALL THREE conditions are satisfied:
  (1) the paper has STRONG EVIDENCE of relevance to this scope, especially in terms of experiments, datasets, and evaluation metrics; AND
  (2) there is STRONG EVIDENCE of detailed setups for ALL three: experiments, datasets, and evaluation metrics; AND
  (3) 'data_publicly_available' (decided below) is 'true'.
- Reject papers that are off-topic, surveys, position papers, or theory-only work.
- If evidence is weak, incomplete, or ambiguous, reject by default.

Data availability criterion - decide this INDEPENDENTLY of the overall relevance judgement, using this exact rule:
- If the data used in the paper is proprietary AND there is no public source for it, reject (`data_publicly_available = false`).
- If the data is proprietary BUT it can be recreated/reconstructed from public sources (i.e. a public-data equivalent exists), accept (`data_publicly_available = true`).
- If the data is already public, accept.
- If more than one dataset is used, `data_publicly_available = true` as long as at least one benchmark-relevant dataset satisfies the above.

Treat the following as positive evidence that data is public or publicly reconstructible:
- the dataset is from an open-data body or public exchange API (e.g. FRED, IMF, World Bank, ECB, BIS, exchange public REST endpoints)
- the dataset is a commonly used public benchmark (e.g. ACL18, CIKM18, KDD17, LOBSTER samples)
- the dataset is a standard public-market series — prices, returns, volumes, OHLCV, or standard macro indicators for publicly traded instruments (equities, indices, futures, FX, crypto, rates, commodities). The canonical values of such series are vendor-independent: a paid or terminal vendor named in the paper (e.g. Bloomberg, Refinitiv, WRDS, CRSP, MetaTrader, ricequant, JoinQuant, Pinnacle CLC) is proprietary access but NOT proprietary data — it does not block acceptance as long as the same instrument identifier(s), period, and frequency are obtainable from a free, scriptable, no-interactive-login source. Representative free families include yfinance for global equities/indices/FX/futures front-month, public exchange REST or ccxt for crypto, akshare/tushare/baostock for Chinese markets, FRED/IMF/World Bank for macro, and Frankfurter/exchangerate.host for FX; this list is illustrative, not exhaustive.
- the authors directly open-source the dataset, or the paper contains a repository that lays out procedures to download the data
- Hugging Face, Kaggle, or well-known academic datasets with clear naming and source context, if the excerpt gives enough clues about accessibility
- any other strong evidence that the dataset is reproducibly downloadable without a paywall or interactive-auth barrier

Treat the following as evidence the data is proprietary with no public source (reject):
- the dataset content itself (not merely the access vendor) is paper-specific or otherwise non-reconstructible from public sources: hand-curated labels, proprietary signal/factor libraries, broker-internal order flow, full-depth limit-order-book or tick-by-tick data behind paid feeds, or vendor-licensed historical universes such as historical index constituents that the paper does not include
- the data is described too vaguely to identify the canonical entity — instruments, period, and frequency are all unspecified or only hinted at
- after considering vendor-substitutable public-market series, public reconstructability remains genuinely ambiguous — treat ambiguous as reject, not accept

Evidence quality requirements:
- evidence.experiments must be a detailed, concrete string covering the experimental setup, compared methods or baselines, splits or backtest windows, and evaluation protocol when present.
- evidence.datasets must be a detailed string with specific dataset names and source clues from the excerpt.
- evidence.metrics must be a detailed string with specific metric names and enough detail to tell what is measured and how the metric is used.

Task:
1) Provide detailed `evidence` strings for experiments, datasets, and metrics — these are factual observations from the excerpt.
2) Decide `data_publicly_available` using the data availability criterion above, and provide `data_availability_reasoning`: cite the specific dataset(s), whether they are proprietary or public, and, if proprietary, whether a public reconstruction exists and from where.
3) Provide `reasons` grounded in the evidence above. Be specific about what evidence supports acceptance or rejection, especially for experiments, datasets, and metrics.
4) Decide `label`: accepted/rejected, consistent with the evidence and reasons.
5) Give `score` in the range `[0, 10]` and `confidence` in the range `[0.0, 1.0]`.
""".strip()


class PrefilterDecision(BaseModel):
    reasons: str
    label: JudgeLabel
    relevance_score: float = Field(0.0, ge=0.0, le=10.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class Evidence(BaseModel):
    experiments: str
    datasets: str
    metrics: str


class SiftJudgement(BaseModel):
    evidence: Evidence
    data_publicly_available: bool = Field(
        description=(
            "True if the paper's data is public, or proprietary but "
            "reconstructible from public sources. False if proprietary with "
            "no public source."
        )
    )
    data_availability_reasoning: str = Field(
        description=(
            "Reasoning for `data_publicly_available`, citing the specific "
            "dataset(s), whether they are proprietary or public, and, if "
            "proprietary, whether and from where a public reconstruction exists."
        )
    )
    reasons: str
    label: JudgeLabel
    score: float = Field(0.0, ge=0.0, le=10.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _data_availability_gates_label(self) -> "SiftJudgement":
        if not self.data_publicly_available and self.label == JudgeLabel.ACCEPTED:
            self.label = JudgeLabel.REJECTED
            self.reasons = (
                f"{self.reasons}\n\n"
                "Overridden to be rejected: data_publicly_available is False"
            )
        return self
