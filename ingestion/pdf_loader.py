# ingestion/pdf_loader.py

import fitz  
from typing import List, Dict, Any
from pathlib import Path

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    documents = []

    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if not text:          
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": path.name,
                    "source_type": "pdf",
                    "page": page_num,
                    "total_pages": len(pdf),
                    "file_path": str(path),
                },
            })

    print(f"[pdf_loader] Loaded {len(documents)} pages from '{path.name}'")
    return documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    docs = load_pdf(sys.argv[1])
    print(f"Total pages loaded: {len(docs)}")
    print(f"First page preview:\n{docs[0]['text'][:200]}")
    print(f"Metadata: {docs[0]['metadata']}")