import pytest

from open_fin_gym.pipeline.config import Scope, TaskType, scope_context
from open_fin_gym.pipeline.task_types import (
    ForecastingParams,
    GenerationParams,
)


def test_scope_init():
    a = Scope(id="a", name="A", task_type="forecasting", description="")

    assert isinstance(a.task_type, TaskType)
    assert a.task_type == TaskType.FORECASTING
    assert a.task_params == ForecastingParams

    b = Scope(id="b", name="B", task_type="generation", description="")

    assert isinstance(b.task_type, TaskType)
    assert b.task_params == GenerationParams

    with pytest.raises(ValueError):
        Scope(id="c", name="C", task_type="foo", description="")


def test_scope_context_carries_task_type():
    # The judge rejects papers that do not match it
    scope = Scope(id="a", name="A", task_type="generation", description="")

    assert "generation" in scope_context(scope)
