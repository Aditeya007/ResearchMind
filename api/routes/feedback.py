# api/routes/feedback.py

from fastapi import APIRouter, HTTPException
from evaluation.metrics_logger import log_feedback, get_avg_scores
from api.schemas import FeedbackRequest, FeedbackResponse, MetricsResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    try:
        log_feedback(request.query_log_id, request.is_helpful)
        return FeedbackResponse(message="Feedback recorded. Thank you!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    try:
        scores = get_avg_scores()
        return MetricsResponse(**scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))