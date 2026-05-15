from fastapi import APIRouter, HTTPException
from api.schemas import IngestURLRequest, IngestResponse
from ingestion.pipeline import ingest_url as pipeline_ingest_url
from retrieval.vector_store import add_chunks, get_all_chunks
from retrieval.bm25_retriever import build_index

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/url", response_model=IngestResponse)
def ingest_url(request: IngestURLRequest):
    try:
        chunks = pipeline_ingest_url(request.url)
        add_chunks(chunks)
        build_index(get_all_chunks())
        return IngestResponse(message="URL ingested", chunks_added=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))