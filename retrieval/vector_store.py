from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from retrieval.embedder import embed_texts, embed_query
from configs.settings import get_settings
import uuid

settings = get_settings()

_client = None
_collections = {}


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection(session_id: str = "default"):
    if session_id not in _collections:
        client = get_client()
        collection_name = f"{settings.COLLECTION_NAME}_{session_id}"
        _collections[session_id] = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[session_id]


def add_chunks(chunks: List[Dict[str, Any]], session_id: str = "default") -> None:
    collection = get_collection(session_id)
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"[vector_store] Added {len(chunks)} chunks for session '{session_id}'")


def search(query: str, session_id: str = "default", top_k: int = None) -> List[Dict[str, Any]]:
    collection = get_collection(session_id)
    k = top_k or settings.TOP_K_DENSE
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()) if collection.count() > 0 else 1,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": text, "metadata": meta, "score": round(1 - distance, 4)})
    return output


def get_collection_count(session_id: str = "default") -> int:
    return get_collection(session_id).count()


def get_session_sources(session_id: str = "default") -> List[str]:
    collection = get_collection(session_id)
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    sources = list({m.get("source", "unknown") for m in results["metadatas"]})
    return sources


def delete_source(source: str, session_id: str = "default") -> None:
    collection = get_collection(session_id)
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(results["ids"], results["metadatas"])
        if meta.get("source") == source
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"[vector_store] Deleted {len(ids_to_delete)} chunks for source '{source}'")