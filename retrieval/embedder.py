
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from configs.settings import get_settings

settings = get_settings()
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model '{settings.EMBEDDING_MODEL}'...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"[embedder] Model loaded.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = np.asarray(model.encode(texts, show_progress_bar=False, convert_to_numpy=True))
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    model = get_model()
    embedding = np.asarray(model.encode(query, show_progress_bar=False, convert_to_numpy=True))
    return embedding.tolist()


if __name__ == "__main__":
    sample_texts = [
        "Retrieval Augmented Generation combines search with LLMs.",
        "ChromaDB is a vector database for storing embeddings.",
        "FastAPI is a modern web framework for building APIs.",
    ]

    print("Embedding sample texts...")
    vectors = embed_texts(sample_texts)

    print(f"\nNumber of embeddings : {len(vectors)}")
    print(f"Embedding dimension  : {len(vectors[0])}")
    print(f"First vector (first 5 values): {vectors[0][:5]}")

    query_vec = embed_query("What is RAG?")
    print(f"\nQuery embedding dimension: {len(query_vec)}")