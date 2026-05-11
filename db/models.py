from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean  # type: ignore[import-not-found]
from sqlalchemy.orm import declarative_base  # type: ignore[import-not-found]

Base = declarative_base()


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    query_type = Column(String(50))
    answer = Column(Text)
    sources = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_log_id = Column(Integer, nullable=False)
    is_helpful = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvalLog(Base):
    __tablename__ = "eval_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_log_id = Column(Integer, nullable=False)
    faithfulness = Column(Float)
    answer_relevance = Column(Float)
    context_precision = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)