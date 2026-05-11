from pathlib import Path
import sys
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter


if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from configs.settings import get_settings

settings = get_settings()


def chunk_text(
    text: str,
    metadata: Dict[str, Any] = {},
) -> List[Dict[str, Any]]:
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
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            text=doc["text"],
            metadata=doc.get("metadata", {}),
        )
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
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