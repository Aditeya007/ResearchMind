# generation/prompt_builder.py

from typing import List, Dict, Any


SYSTEM_PROMPT = """You are ResearchMind, an intelligent research assistant.
Your job is to answer questions clearly and concisely in your own words.
Use the provided context to form your answer — do NOT copy paste from it.
If the context doesn't have enough information, say "I don't have enough information to answer this."
Keep answers under 4 sentences unless the question specifically needs more detail."""


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

Answer in your own words, concisely:"""