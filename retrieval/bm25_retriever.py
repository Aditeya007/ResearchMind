from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from configs.settings import get_settings

settings = get_settings()

_indexes = {}
_corpora = {}


def build_index(chunks: List[Dict[str, Any]], session_id: str = "default") -> None:
    if not chunks:
        _corpora[session_id] = []
        _indexes.pop(session_id, None)
        print(f"[bm25] Cleared index for session '{session_id}' (no chunks)")
        return

    _corpora[session_id] = chunks
    tokenized = [c["text"].lower().split() for c in chunks]
    _indexes[session_id] = BM25Okapi(tokenized)
    print(f"[bm25] Index built for session '{session_id}' with {len(chunks)} chunks")


def search(query: str, session_id: str = "default", top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    if session_id not in _indexes:
        return []

    k = top_k or settings.TOP_K_SPARSE
    tokenized_query = query.lower().split()
    scores = _indexes[session_id].get_scores(tokenized_query)
    corpus = _corpora[session_id]

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"text": corpus[i]["text"], "metadata": corpus[i]["metadata"], "score": round(float(scores[i]), 4)}
        for i in top_indices if scores[i] > 0
    ]


def get_corpus(session_id: str = "default") -> List[Dict[str, Any]]:
    return _corpora.get(session_id, [])


def has_index(session_id: str = "default") -> bool:
    return session_id in _indexes


def delete_source(source: str, session_id: str = "default") -> int:
    corpus = _corpora.get(session_id, [])
    if not corpus:
        return 0

    filtered = [c for c in corpus if c.get("metadata", {}).get("source") != source]
    deleted = len(corpus) - len(filtered)

    if deleted == 0:
        return 0

    build_index(filtered, session_id=session_id)
    print(f"[bm25] Deleted {deleted} chunks for source '{source}' in session '{session_id}'")
    return deleted