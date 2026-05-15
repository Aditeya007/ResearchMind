from typing import List, Dict, Any
from retrieval import vector_store
from retrieval import bm25_retriever
from configs.settings import get_settings

settings = get_settings()


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
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
    return [{**all_chunks[key], "score": round(scores[key], 6)} for key in sorted_keys]


def search(query: str, session_id: str = "default", top_k: int = None) -> List[Dict[str, Any]]:
    k = top_k or settings.TOP_K_RERANK
    dense_results = vector_store.search(query, session_id=session_id, top_k=settings.TOP_K_DENSE)
    sparse_results = bm25_retriever.search(query, session_id=session_id, top_k=settings.TOP_K_SPARSE)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:k]