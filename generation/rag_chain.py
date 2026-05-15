from typing import Dict, Any, List
from retrieval.query_router import route_and_retrieve
from generation.prompt_builder import build_prompt, SYSTEM_PROMPT
from generation.llm_client import generate

_chat_histories = {}


def get_history(session_id: str) -> List[Dict[str, str]]:
    return _chat_histories.get(session_id, [])


def update_history(session_id: str, query: str, answer: str) -> None:
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []
    _chat_histories[session_id].append({"query": query, "answer": answer})
    _chat_histories[session_id] = _chat_histories[session_id][-4:]


def build_history_text(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    lines = []
    for h in history:
        lines.append(f"User: {h['query']}\nAssistant: {h['answer']}")
    return "Previous conversation:\n" + "\n\n".join(lines) + "\n\n"


def run(query: str, session_id: str = "default") -> Dict[str, Any]:
    routed = route_and_retrieve(query, session_id=session_id)
    chunks = routed["chunks"]

    if not chunks:
        return {
            "query": query,
            "query_type": routed["query_type"],
            "answer": "I could not find any relevant information to answer your question.",
            "sources": [],
            "chunks": [],
            "confidence": "low",
        }

    history_text = build_history_text(session_id)
    prompt = history_text + build_prompt(query, chunks)
    answer = generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    update_history(session_id, query, answer)

    sources = list({chunk["metadata"].get("source", "unknown") for chunk in chunks})

    top_score = chunks[0].get("rerank_score", chunks[0].get("score", 0))
    if top_score > 0.7:
        confidence = "high"
    elif top_score > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "query": query,
        "query_type": routed["query_type"],
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "confidence": confidence,
    }