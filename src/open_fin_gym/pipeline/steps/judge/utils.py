import math
from datetime import datetime, timezone

from open_fin_gym.pipeline.db.tables import Chunk, Paper

HEADING_FILTERS = {
    "references",
    "acknowledgement",
    "acknowledgements",
    "appendix",
    "related",
    "review",
}


def filter_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Filter chunks not used in judgement, e.g. references or acknowledgements

    Args:
        chunks: List of paper text chunks

    Returns:
        List of filtered chunks
    """
    return [
        x for x in chunks if not HEADING_FILTERS.intersection(set(x.header.split()))
    ]


def rank_papers_for_llm(
    papers: list[Paper],
    *,
    citation_boost: float,
    recency_weight: float,
    recency_half_life_days: int,
) -> list[Paper]:
    """
    Sort papers in acceptance priority

    Args:
        papers: list of papers
        citation_boost: Citation boost factor
        recency_weight: Recency score bias
        recency_half_life_days: Recency score scaling

    Returns:
        Sorted list of papers
    """
    return sorted(
        papers,
        key=lambda p: _hybrid_priority_score(
            p,
            citation_boost=citation_boost,
            recency_weight=recency_weight,
            recency_half_life_days=recency_half_life_days,
        ),
        reverse=True,
    )


def _hybrid_priority_score(
    paper: Paper,
    *,
    citation_boost: float,
    recency_weight: float,
    recency_half_life_days: int,
) -> float:
    """Multiplicative-boost ranking score.

    Prefilter relevance is the primary axis. A maximally-cited paper receives
    a ``(1 + citation_boost)x`` rank amplification but cannot overtake a
    sufficiently more relevant paper, so a weakly-relevant famous paper never
    dominates a strongly-relevant unknown one. Recency is a small additive
    sweetener on a separate axis.
    """
    prefilter = _normalize_prefilter_score(paper.prefilter_score)
    quality = _paper_quality_score(paper)
    recency = _recency_score(paper.published_at, recency_half_life_days)
    return prefilter * (1.0 + citation_boost * quality) + recency_weight * recency


def _normalize_prefilter_score(score_0_10: float | None) -> float:
    if score_0_10 is None:
        return 0.0
    return min(1.0, max(0.0, float(score_0_10) / 10.0))


def _paper_quality_score(paper: Paper) -> float:
    citation = _citation_score(paper.citation_count)
    influential = _citation_score(paper.influential_citation_count, scale=200)
    venue_signal = 1.0 if (paper.journal_name or paper.venue) else 0.0
    peer_review_signal = 1.0 if paper.peer_reviewed else 0.0
    return (
        0.45 * citation
        + 0.25 * influential
        + 0.15 * venue_signal
        + 0.15 * peer_review_signal
    )


def _citation_score(value: int | None, *, scale: int = 100) -> float:
    if not value or value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(scale))


def _recency_score(published_at: datetime | None, half_life_days: int) -> float:
    if not published_at:
        return 0.0

    now = datetime.now(tz=timezone.utc)
    age_days = max(
        0.0, (now - published_at.astimezone(timezone.utc)).total_seconds() / 86400.0
    )

    if half_life_days <= 0:
        return 0.0

    return 0.5 ** (age_days / float(half_life_days))
