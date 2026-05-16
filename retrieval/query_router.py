from typing import List, Dict, Any
from retrieval.hybrid_retriever import search as hybrid_search
from retrieval.reranker import rerank
from configs.settings import get_settings

settings = get_settings()


def classify_query(query: str) -> str:
    query_lower = query.lower()

    factual_keywords = ["what is", "who is", "when did", "where is", "define", "meaning of"]
    summary_keywords = ["summarize", "summary", "overview", "explain", "describe"]
    comparison_keywords = ["compare", "difference between", "vs", "versus", "better"]

    for kw in factual_keywords:
        if kw in query_lower:
            return "factual"

    for kw in summary_keywords:
        if kw in query_lower:
            return "summary"

    for kw in comparison_keywords:
        if kw in query_lower:
            return "comparison"

    return "general"


def route_and_retrieve(query: str, session_id: str = "default") -> Dict[str, Any]:
    # Keep retrieval scoped to the same session that ingested the documents.
    query_type = classify_query(query)

    if query_type == "factual":
        top_k = 3
    elif query_type == "summary":
        top_k = 6
    elif query_type == "comparison":
        top_k = 6
    else:
        top_k = settings.TOP_K_DENSE

    chunks = hybrid_search(query, session_id=session_id, top_k=top_k)
    reranked = rerank(query, chunks)

    return {
        "query": query,
        "query_type": query_type,
        "chunks": reranked,
    }


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url
    from retrieval.vector_store import add_chunks
    from retrieval.bm25_retriever import build_index

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    add_chunks(chunks)
    build_index(chunks)

    test_queries = [
        "What is retrieval augmented generation?",
        "Summarize how RAG works",
        "Compare RAG vs fine tuning",
        "How does RAG handle hallucinations?",
    ]

    for query in test_queries:
        result = route_and_retrieve(query)
        print(f"\nQuery      : {result['query']}")
        print(f"Type       : {result['query_type']}")
        print(f"Chunks     : {len(result['chunks'])}")
        print(f"Top chunk  : {result['chunks'][0]['text'][:120]}")