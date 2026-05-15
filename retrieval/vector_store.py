

from typing import List, Dict, Any, Optional
import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from retrieval.embedder import embed_texts, embed_query
from configs.settings import get_settings

settings = get_settings()

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunks: List[Dict[str, Any]]) -> None:
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = np.asarray(embed_texts(texts), dtype=np.float32)
    ids = [f"{meta.get('source', 'doc')}_{meta.get('page', meta.get('row', i))}_{i}"
           for i, meta in enumerate(metadatas)]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[vector_store] Added {len(chunks)} chunks to '{settings.COLLECTION_NAME}'")


def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    collection = get_collection()
    k = top_k if top_k is not None else settings.TOP_K_DENSE
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    output = []
    for text, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        output.append({
            "text": text,
            "metadata": meta,
            "score": round(1 - distance, 4),
        })

    return output


def get_collection_count() -> int:
    return get_collection().count()


def get_all_chunks() -> List[Dict[str, Any]]:
    collection = get_collection()
    results = collection.get(include=["documents", "metadatas"])

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    return [
        {"text": text, "metadata": meta}
        for text, meta in zip(documents, metadatas)
        if text is not None and meta is not None
    ]


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url

    print("Ingesting a sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

    add_chunks(chunks)
    print(f"Total vectors in store: {get_collection_count()}")

    print("\nSearching: 'What is RAG?'")
    results = search("What is RAG?", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']}")
        print(f"    Source: {r['metadata'].get('source')}")
        print(f"    Text: {r['text'][:150]}")