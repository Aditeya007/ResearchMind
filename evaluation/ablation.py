from typing import Dict, Any
from retrieval.vector_store import search as dense_search
from retrieval.hybrid_retriever import search as hybrid_search
from retrieval.reranker import rerank
from generation.prompt_builder import build_prompt, SYSTEM_PROMPT
from generation.llm_client import generate
from evaluation.ragas_eval import evaluate_response


def run_pipeline(query: str, retriever: str = "hybrid") -> Dict[str, Any]:
    if retriever == "dense":
        chunks = dense_search(query)
    else:
        chunks = hybrid_search(query)
        chunks = rerank(query, chunks)

    prompt = build_prompt(query, chunks)
    answer = generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    scores = evaluate_response(query, answer, chunks)

    return {
        "retriever": retriever,
        "answer": answer,
        "scores": scores,
        "num_chunks": len(chunks),
    }


def compare(query: str) -> None:
    print(f"Query: {query}\n")
    print("Running dense-only pipeline...")
    dense_result = run_pipeline(query, retriever="dense")

    print("Running hybrid pipeline...")
    hybrid_result = run_pipeline(query, retriever="hybrid")

    print("\n" + "=" * 50)
    print(f"{'Metric':<25} {'Dense':>10} {'Hybrid':>10}")
    print("=" * 50)

    metrics = ["faithfulness", "answer_relevance", "context_precision"]
    for metric in metrics:
        d = dense_result["scores"].get(metric, 0)
        h = hybrid_result["scores"].get(metric, 0)
        winner = "✓" if h >= d else "✗"
        print(f"{metric:<25} {d:>10.4f} {h:>10.4f}  {winner}")

    print("=" * 50)
    print(f"{'Chunks retrieved':<25} {dense_result['num_chunks']:>10} {hybrid_result['num_chunks']:>10}")


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url
    from retrieval.vector_store import add_chunks
    from retrieval.bm25_retriever import build_index

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    add_chunks(chunks)
    build_index(chunks)

    compare("How does RAG reduce hallucinations in language models?")