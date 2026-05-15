from fastapi import APIRouter, HTTPException, BackgroundTasks
from generation.rag_chain import run
from evaluation.ragas_eval import evaluate_response
from evaluation.metrics_logger import log_query, log_eval
from retrieval.bm25_retriever import has_index
from api.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


def _evaluate_and_log(query_log_id, query, answer, chunks):
    try:
        scores = evaluate_response(query, answer, chunks)
        log_eval(query_log_id, scores)
    except Exception:
        pass


@router.post("/", response_model=QueryResponse)
def query(request: QueryRequest, background_tasks: BackgroundTasks):
    if not has_index(request.session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please ingest a URL or PDF first."
        )
    try:
        result = run(request.query, session_id=request.session_id)

        query_log_id = log_query(
            query=result["query"],
            query_type=result["query_type"],
            answer=result["answer"],
            sources=result["sources"],
        )

        background_tasks.add_task(
            _evaluate_and_log,
            query_log_id,
            result["query"],
            result["answer"],
            result["chunks"],
        )

        return QueryResponse(
            query=result["query"],
            query_type=result["query_type"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))