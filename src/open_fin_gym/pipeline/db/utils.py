from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from .tables import Paper, PaperStatus


def set_paper_status(db: Engine, paper_id: str, status: PaperStatus, **kwargs) -> None:
    """
    Update paper status in the DB

    Args:
        db: SqlAlchemy db engine
        paper_id: Id of the paper
        status: Status value
    """
    with Session(db) as session:
        stmt = (
            update(Paper)
            .values({"status": status, **kwargs})
            .where(Paper.paper_id == paper_id)
        )
        session.execute(stmt)
        session.commit()
