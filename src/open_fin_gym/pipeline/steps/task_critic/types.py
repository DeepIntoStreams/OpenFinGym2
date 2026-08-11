from pydantic import BaseModel


class CritiqueResult(BaseModel):
    task_id: int
    scope_id: str
    approved: bool
    reasoning: str
