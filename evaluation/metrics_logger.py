import json
from typing import Dict, Any, List
from db.session import SessionLocal
from db.models import QueryLog, EvalLog, Feedback


def log_query(
    query: str,
    query_type: str,
    answer: str,
    sources: List[str],
) -> int:
    session = SessionLocal()
    try:
        entry = QueryLog(
            query=query,
            query_type=query_type,
            answer=answer,
            sources=json.dumps(sources),
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        print(f"[metrics_logger] Query logged with id={entry.id}")
        return entry.id
    finally:
        session.close()


def log_eval(query_log_id: int, scores: Dict[str, float]) -> None:
    session = SessionLocal()
    try:
        entry = EvalLog(
            query_log_id=query_log_id,
            faithfulness=scores.get("faithfulness"),
            answer_relevance=scores.get("answer_relevance"),
            context_precision=scores.get("context_precision"),
        )
        session.add(entry)
        session.commit()
        print(f"[metrics_logger] Eval scores logged for query_log_id={query_log_id}")
    finally:
        session.close()


def log_feedback(query_log_id: int, is_helpful: bool) -> None:
    session = SessionLocal()
    try:
        entry = Feedback(
            query_log_id=query_log_id,
            is_helpful=is_helpful,
        )
        session.add(entry)
        session.commit()
        print(f"[metrics_logger] Feedback logged for query_log_id={query_log_id}")
    finally:
        session.close()


def get_avg_scores() -> Dict[str, float]:
    session = SessionLocal()
    try:
        logs = session.query(EvalLog).all()
        if not logs:
            return {}

        return {
            "avg_faithfulness": round(sum(l.faithfulness for l in logs if l.faithfulness) / len(logs), 4),
            "avg_answer_relevance": round(sum(l.answer_relevance for l in logs if l.answer_relevance) / len(logs), 4),
            "avg_context_precision": round(sum(l.context_precision for l in logs if l.context_precision) / len(logs), 4),
            "total_queries": len(logs),
        }
    finally:
        session.close()


if __name__ == "__main__":
    from db.session import init_db
    init_db()

    qid = log_query(
        query="What is RAG?",
        query_type="factual",
        answer="RAG is a technique that combines retrieval with generation.",
        sources=["wikipedia"],
    )

    log_eval(qid, {
        "faithfulness": 0.91,
        "answer_relevance": 0.88,
        "context_precision": 0.85,
    })

    log_feedback(qid, is_helpful=True)

    print(f"\nAverage scores: {get_avg_scores()}")