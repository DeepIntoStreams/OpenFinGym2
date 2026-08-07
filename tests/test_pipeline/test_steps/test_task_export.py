import tomllib

import pytest
from jinja2 import Environment, FileSystemLoader

from open_fin_gym.pipeline.steps.task_export.pipeline import slugify


@pytest.fixture
def task_meta_template():
    return Environment(loader=FileSystemLoader("templates")).get_template(
        "task.toml.j2"
    )


@pytest.mark.parametrize(
    "description",
    [
        "Predicts equity returns from a 30-day window.",
        'Predicts the "alpha" signal from order flow.',
        "Predicts returns.\n\nUses an LSTM.",
        r"Uses a \tau-scaled loss.",
    ],
)
def test_task_config_survives_llm_prose(task_meta_template, description: str) -> None:
    # Both description fields are free text written by an LLM
    config = task_meta_template.render(
        org_name="org",
        task_name="task",
        description=description,
        keywords=[],
        difficulty_explanation=description,
    )

    assert tomllib.loads(config)["task"]["description"] == description


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Stock Return Forecasting", "stock_return_forecasting"),
        ("LOB/mid-price prediction", "lob_mid_price_prediction"),
        ("VaR (95%) backtest", "var_95_backtest"),
        ("../escape", "escape"),
        ("!!!", "task"),
    ],
)
def test_slugify_keeps_the_name_inside_the_export_directory(
    name: str, expected: str
) -> None:
    assert slugify(name) == expected
