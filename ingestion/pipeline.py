from typing import List, Dict, Any
from ingestion.pdf_loader import load_pdf
from ingestion.web_loader import load_url, load_urls
from ingestion.chunker import chunk_documents


def ingest_pdf(file_path: str) -> List[Dict[str, Any]]:
    
    docs = load_pdf(file_path)
    chunks = chunk_documents(docs)
    print(f"[pipeline] PDF → {len(chunks)} chunks")
    return chunks


def ingest_url(url: str) -> List[Dict[str, Any]]:
    
    docs = load_url(url)
    chunks = chunk_documents(docs)
    print(f"[pipeline] URL → {len(chunks)} chunks")
    return chunks


def ingest_urls(urls: List[str]) -> List[Dict[str, Any]]:
    
    docs = load_urls(urls)
    chunks = chunk_documents(docs)
    print(f"[pipeline] {len(urls)} URLs → {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Error: Missing arguments.")
        sys.exit(1)

    source_type = sys.argv[1].lower()
    source = sys.argv[2]

    if source_type == "pdf":
        chunks = ingest_pdf(source)
    elif source_type == "url":
        chunks = ingest_url(source)
    else:
        print(f"Error: Unknown type '{source_type}'. Use 'pdf' or 'url'.")
        sys.exit(1)

    print(f"\nTotal chunks : {len(chunks)}")
    print(f"Sample chunk :\n{chunks[0]['text'][:200]}")
    print(f"Metadata     : {chunks[0]['metadata']}")