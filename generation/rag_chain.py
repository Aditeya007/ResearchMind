from typing import Dict, Any
from retrieval.query_router import route_and_retrieve
from generation.prompt_builder import build_prompt, SYSTEM_PROMPT
from generation.llm_client import generate


def run(query: str) -> Dict[str, Any]:
    routed = route_and_retrieve(query)
    chunks = routed["chunks"]

    if not chunks:
        return {
            "query": query,
            "query_type": routed["query_type"],
            "answer": "I could not find any relevant information to answer your question.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    answer = generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    sources = []
    for chunk in chunks:
        source = chunk["metadata"].get("source", "unknown")
        if source not in sources:
            sources.append(source)

    return {
        "query": query,
        "query_type": routed["query_type"],
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url
    from retrieval.vector_store import add_chunks
    from retrieval.bm25_retriever import build_index

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    add_chunks(chunks)
    build_index(chunks)

    query = "How does RAG reduce hallucinations in language models?"
    print(f"\nQuery: {query}\n")

    result = run(query)
    print(f"Query Type : {result['query_type']}")
    print(f"Sources    : {result['sources']}")
    print(f"\nAnswer:\n{result['answer']}")