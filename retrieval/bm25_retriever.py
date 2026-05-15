from typing import List, Dict, Any, Optional
import math
from collections import Counter

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
except ImportError:
    class BM25Okapi:
        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
            self.doc_freqs = [Counter(doc) for doc in corpus]
            self.idf = {}

            num_docs = len(corpus)
            term_doc_counts = Counter(term for doc in corpus for term in set(doc))
            for term, count in term_doc_counts.items():
                self.idf[term] = math.log(1 + (num_docs - count + 0.5) / (count + 0.5))

        def get_scores(self, query_tokens: List[str]) -> List[float]:
            scores: List[float] = []

            for doc_index, doc_freq in enumerate(self.doc_freqs):
                score = 0.0
                doc_length = self.doc_len[doc_index] or 1
                norm = self.k1 * (1 - self.b + self.b * doc_length / self.avgdl) if self.avgdl else self.k1

                for term in query_tokens:
                    if term not in doc_freq:
                        continue

                    tf = doc_freq[term]
                    idf = self.idf.get(term, 0.0)
                    score += idf * (tf * (self.k1 + 1)) / (tf + norm)

                scores.append(score)

            return scores
from configs.settings import get_settings

settings = get_settings()

_bm25 = None
_corpus = None


def build_index(chunks: List[Dict[str, Any]]) -> None:
    global _bm25, _corpus
    _corpus = chunks
    tokenized = [c["text"].lower().split() for c in chunks]
    _bm25 = BM25Okapi(tokenized)
    print(f"[bm25] Index built with {len(chunks)} chunks")


def load_index_from_vector_store() -> None:
    from retrieval.vector_store import get_all_chunks

    chunks = get_all_chunks()
    if not chunks:
        return

    build_index(chunks)


def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    if _bm25 is None or _corpus is None:
        load_index_from_vector_store()

    if _bm25 is None or _corpus is None:
        raise RuntimeError("BM25 index not built. Ingest a source first.")

    k = top_k if top_k is not None else settings.TOP_K_SPARSE
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        {
            "text": _corpus[i]["text"],
            "metadata": _corpus[i]["metadata"],
            "score": round(float(scores[i]), 4),
        }
        for i in top_indices
        if scores[i] > 0
    ]


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

    build_index(chunks)

    query = "How does RAG use external knowledge?"
    results = search(query, top_k=3)

    print(f"\nQuery: {query}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']}")
        print(f"    Text: {r['text'][:150]}")