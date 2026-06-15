import pytest
from sqlalchemy import Engine, create_engine

from open_fin_gym.pipeline.db.tables import Base


@pytest.fixture(scope="function")
def db() -> Engine:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine
