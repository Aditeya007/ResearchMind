

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from configs.settings import get_settings

settings = get_settings()

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[reranker] Loading model '{settings.RERANKER_MODEL}'...")
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
        print(f"[reranker] Model loaded.")
    return _reranker


def rerank(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url
    from retrieval.vector_store import add_chunks
    from retrieval.bm25_retriever import build_index
    from retrieval.hybrid_retriever import search as hybrid_search

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    add_chunks(chunks)
    build_index(chunks)

    query = "How does RAG reduce hallucinations?"
    hybrid_results = hybrid_search(query)

    print(f"\nBefore reranking:")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  [{i}] RRF Score: {r['score']} | {r['text'][:80]}")

    reranked = rerank(query, hybrid_results)

    print(f"\nAfter reranking:")
    for i, r in enumerate(reranked, 1):
        print(f"  [{i}] Rerank Score: {r['rerank_score']} | {r['text'][:80]}")