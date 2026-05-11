# ingestion/chunker.py

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configs.settings import get_settings
settings = get_settings()
def chunk_text(
    text: str,
    metadata: Dict[str, Any] = {},
) -> List[Dict[str, Any]]:
    """
    Split a plain string into chunks.
    Returns a list of dicts with 'text' and 'metadata' keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(text)

    return [
        {"text": chunk, "metadata": metadata}
        for chunk in chunks
        if chunk.strip()           # drop empty chunks
    ]

def chunk_documents(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Takes a list of {text, metadata} dicts (output of any loader)
    and returns a flat list of chunked {text, metadata} dicts.
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            metadata=doc.get("metadata", {}),
        )
        all_chunks.extend(chunks)

    return all_chunks
if __name__ == "__main__":
    # quick sanity check
    sample = [
        {
            "text": "Retrieval Augmented Generation (RAG) is a technique that combines "
                    "retrieval systems with language models. " * 20,
            "metadata": {"source": "test", "page": 1},
        }
    ]

    chunks = chunk_documents(sample)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk preview: {chunks[0]['text'][:80]}...")
    print(f"Metadata: {chunks[0]['metadata']}")