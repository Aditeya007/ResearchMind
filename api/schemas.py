from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    query: str
    query_type: str
    answer: str
    sources: List[str]
    confidence: str


class IngestURLRequest(BaseModel):
    url: str
    session_id: str = "default"


class IngestResponse(BaseModel):
    message: str
    chunks_added: int


class DeleteSourceRequest(BaseModel):
    source: str
    session_id: str = "default"


class FeedbackRequest(BaseModel):
    query_log_id: int
    is_helpful: bool


class FeedbackResponse(BaseModel):
    message: str


class MetricsResponse(BaseModel):
    avg_faithfulness: Optional[float] = None
    avg_answer_relevance: Optional[float] = None
    avg_context_precision: Optional[float] = None
    total_queries: Optional[int] = 0