# retrieval/hybrid_retriever.py

from typing import List, Dict, Any, Optional
from retrieval import vector_store
from retrieval import bm25_retriever
from configs.settings import get_settings

settings = get_settings()


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    scores = {}
    all_chunks = {}

    for rank, result in enumerate(dense_results):
        key = result["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        all_chunks[key] = result

    for rank, result in enumerate(sparse_results):
        key = result["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        all_chunks[key] = result

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [
        {**all_chunks[key], "score": round(scores[key], 6)}
        for key in sorted_keys
    ]


def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    k = top_k if top_k is not None else settings.TOP_K_RERANK

    dense_results = vector_store.search(query, top_k=settings.TOP_K_DENSE)
    sparse_results = bm25_retriever.search(query, top_k=settings.TOP_K_SPARSE)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    return fused[:k]


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url
    from retrieval.bm25_retriever import build_index
    from retrieval.vector_store import add_chunks

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

    add_chunks(chunks)
    build_index(chunks)

    query = "How does RAG reduce hallucinations?"
    print(f"\nQuery: {query}")

    results = search(query)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']}")
        print(f"    Source: {r['metadata'].get('source')}")
        print(f"    Text: {r['text'][:150]}")