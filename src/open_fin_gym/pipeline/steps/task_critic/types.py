from pydantic import BaseModel


class CritiqueResult(BaseModel):
    task_id: int
    scope_id: str
    approved: bool
    score: float | None = None
    confidence: float | None = None
    reasoning: str
