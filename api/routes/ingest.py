from fastapi import APIRouter, UploadFile, File, HTTPException
from ingestion.pipeline import ingest_pdf, ingest_url
from retrieval.vector_store import add_chunks, get_session_sources, delete_source
from retrieval.bm25_retriever import build_index, get_corpus
from api.schemas import IngestURLRequest, IngestResponse, DeleteSourceRequest
import tempfile
import os

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _rebuild_index(new_chunks, session_id):
    existing = get_corpus(session_id)
    build_index(existing + new_chunks, session_id=session_id)


@router.post("/url", response_model=IngestResponse)
def ingest_from_url(request: IngestURLRequest):
    try:
        chunks = ingest_url(request.url)
        add_chunks(chunks, session_id=request.session_id)
        _rebuild_index(chunks, request.session_id)
        return IngestResponse(message="URL ingested successfully.", chunks_added=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pdf", response_model=IngestResponse)
def ingest_from_pdf(file: UploadFile = File(...), session_id: str = "default"):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        chunks = ingest_pdf(tmp_path)
        add_chunks(chunks, session_id=session_id)
        _rebuild_index(chunks, session_id)
        os.remove(tmp_path)
        return IngestResponse(message="PDF ingested successfully.", chunks_added=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
def list_sources(session_id: str = "default"):
    sources = get_session_sources(session_id)
    return {"sources": sources}


@router.delete("/source")
def remove_source(request: DeleteSourceRequest):
    try:
        delete_source(request.source, session_id=request.session_id)
        return {"message": f"Source '{request.source}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))