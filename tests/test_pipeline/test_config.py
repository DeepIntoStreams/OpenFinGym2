import pytest

from open_fin_gym.pipeline.config import Scope, TaskType


def test_scope_init():
    a = Scope(id="a", name="A", task_type="forecasting", description="")

    assert isinstance(a.task_type, TaskType)
    assert a.task_type == TaskType.FORECASTING

    b = Scope(id="b", name="B", task_type="generation", description="")

    assert isinstance(b.task_type, TaskType)

    with pytest.raises(ValueError):
        Scope(id="c", name="C", task_type="foo", description="")
