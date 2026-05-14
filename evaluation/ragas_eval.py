# evaluation/ragas_eval.py

from typing import Dict, Any, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from configs.settings import get_settings

settings = get_settings()


def evaluate_response(
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    contexts = [chunk["text"] for chunk in chunks]

    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [answer],
    }

    dataset = Dataset.from_dict(data)

    llm = LangchainLLMWrapper(ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.LLM_MODEL_NAME,
        temperature=0,
    ))

    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    ))

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevance": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    return scores


if __name__ == "__main__":
    from ingestion.pipeline import ingest_url
    from retrieval.vector_store import add_chunks
    from retrieval.bm25_retriever import build_index
    from generation.rag_chain import run

    print("Ingesting sample URL...")
    chunks = ingest_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    add_chunks(chunks)
    build_index(chunks)

    query = "How does RAG reduce hallucinations?"
    result = run(query)

    print(f"\nEvaluating response...")
    scores = evaluate_response(query, result["answer"], result["chunks"])

    print(f"\nRAGAS Scores:")
    print(f"  Faithfulness      : {scores['faithfulness']}")
    print(f"  Answer Relevance  : {scores['answer_relevance']}")
    print(f"  Context Precision : {scores['context_precision']}")