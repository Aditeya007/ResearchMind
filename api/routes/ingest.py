from fastapi import APIRouter, HTTPException
from api.schemas import IngestURLRequest, IngestResponse
from ingestion.pipeline import ingest_url as pipeline_ingest_url

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/url", response_model=IngestResponse)
def ingest_url(request: IngestURLRequest):
    try:
        chunks = pipeline_ingest_url(request.url)
        return IngestResponse(message="URL ingested", chunks_added=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))