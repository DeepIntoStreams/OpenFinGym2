from open_fin_gym.pipeline.steps.judge import utils


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
