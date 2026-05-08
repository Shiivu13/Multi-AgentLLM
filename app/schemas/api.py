import uuid
from typing import Any, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: Optional[uuid.UUID] = None

class TraceResponse(BaseModel):
    job_id: uuid.UUID
    traces: list[dict[str, Any]]

class EvalSummaryResponse(BaseModel):
    run_id: uuid.UUID
    overall_score: float
    category_scores: dict[str, float]
    dimension_scores: dict[str, float]

class RewriteApprovalRequest(BaseModel):
    approved: bool

class ReEvalResponse(BaseModel):
    new_run_id: uuid.UUID
