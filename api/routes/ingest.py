from fastapi import APIRouter, HTTPException, UploadFile, File
from api.schemas import IngestURLRequest, IngestResponse
from ingestion.pipeline import ingest_url as pipeline_ingest_url, ingest_pdf as pipeline_ingest_pdf
from retrieval.vector_store import add_chunks, get_all_chunks
from retrieval.bm25_retriever import build_index
import tempfile
import shutil
from pathlib import Path

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


@router.post("/pdf", response_model=IngestResponse)
def ingest_pdf(file: UploadFile = File(...)):
    temp_file = None
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_file = tmp.name
            # Write uploaded content to temp file
            content = file.file.read()
            tmp.write(content)
        
        # Process the PDF
        chunks = pipeline_ingest_pdf(temp_file)
        add_chunks(chunks)
        build_index(get_all_chunks())
        return IngestResponse(message="PDF ingested", chunks_added=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
            except Exception:
                pass