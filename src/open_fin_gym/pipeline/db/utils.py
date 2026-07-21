from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from .tables import Paper, PaperStatus


def set_paper_status(db: Engine, paper: Paper, status: PaperStatus, **kwargs) -> None:
    """
    Update paper status in the DB

    Args:
        db: SqlAlchemy db engine
        paper: Paper to be updated
        status: Status value
    """
    with Session(db) as session:
        stmt = (
            update(Paper)
            .values({"status": status, **kwargs})
            .where(Paper.scope_id == paper.scope_id, Paper.paper_id == paper.paper_id)
        )
        session.execute(stmt)
        session.commit()
