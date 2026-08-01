import pytest
from pydantic import ValidationError

from open_fin_gym.pipeline.db.tables import JudgeLabel
from open_fin_gym.pipeline.steps.judge import utils
from open_fin_gym.pipeline.steps.judge.prompts import Evidence, SiftJudgement


def test_section_filtering():

    chunks = [
        utils.Chunk(
            paper_id="foo",
            chunk_index=0,
            header="Review",
            text="some text",
        ),
        utils.Chunk(
            paper_id="foo",
            chunk_index=1,
            header="References",
            text="some text",
        ),
        utils.Chunk(
            paper_id="foo",
            chunk_index=2,
            header="Appendix: ",
            text="some text",
        ),
        utils.Chunk(
            paper_id="foo",
            chunk_index=3,
            header="##Appendix## - ",
            text="some text",
        ),
        utils.Chunk(
            paper_id="foo",
            chunk_index=4,
            header="Introduction",
            text="some text",
        ),
    ]

    filtered = utils.filter_chunks(chunks)

    assert len(filtered) == 1
    assert filtered[0].chunk_index == 4


def test_sift_judgement_requires_data_availability_fields():
    with pytest.raises(ValidationError):
        SiftJudgement(
            evidence=Evidence(experiments="", datasets="", metrics=""),
            reasons="",
            label=JudgeLabel.ACCEPTED,
            score=8.0,
            confidence=0.9,
        )


def test_sift_judgement_accepts_valid_payload():
    judgement = SiftJudgement(
        evidence=Evidence(experiments="e", datasets="d", metrics="m"),
        data_publicly_available=JudgeLabel.ACCEPTED,
        data_availability_reasoning="reconstructible from yfinance",
        reasons="looks good",
        label=JudgeLabel.ACCEPTED,
        score=8.0,
        confidence=0.9,
    )
    assert judgement.data_publicly_available == JudgeLabel.ACCEPTED
