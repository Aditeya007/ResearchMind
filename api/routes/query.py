from fastapi import APIRouter, HTTPException
from generation.rag_chain import run
from evaluation.ragas_eval import evaluate_response
from evaluation.metrics_logger import log_query, log_eval
from api.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = run(request.query)

        query_log_id = log_query(
            query=result["query"],
            query_type=result["query_type"],
            answer=result["answer"],
            sources=result["sources"],
        )

        try:
            scores = evaluate_response(result["query"], result["answer"], result["chunks"])
            log_eval(query_log_id, scores)
        except Exception:
            pass

        return QueryResponse(
            query=result["query"],
            query_type=result["query_type"],
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))