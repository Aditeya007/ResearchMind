
from typing import List, Dict, Any


SYSTEM_PROMPT = """You are ResearchMind, an intelligent research assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough information to answer this."
Always cite which source your answer comes from."""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(f"[{i}] Source: {source}\n{chunk['text']}")
    return "\n\n".join(context_parts)


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    context = format_context(chunks)
    return f"""Context:
{context}

Question: {query}

Answer:"""


if __name__ == "__main__":
    sample_chunks = [
        {
            "text": "RAG combines a retrieval system with a language model to generate accurate answers.",
            "metadata": {"source": "wikipedia", "page": 1},
        },
        {
            "text": "The retrieval step fetches relevant documents from a knowledge base before generation.",
            "metadata": {"source": "arxiv_paper", "page": 3},
        },
    ]

    prompt = build_prompt("How does RAG work?", sample_chunks)
    print("System Prompt:\n", SYSTEM_PROMPT)
    print("\nUser Prompt:\n", prompt)