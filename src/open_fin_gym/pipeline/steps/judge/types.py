from typing import Optional

from pydantic import BaseModel

from open_fin_gym.pipeline.db.tables import RejectionReason


class JudgeResult(BaseModel):
    paper_id: str
    scope_id: str
    accepted: bool
    rejection_reason: Optional[RejectionReason]
    reasoning: str
