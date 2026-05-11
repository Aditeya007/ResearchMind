from sqlalchemy import create_engine  # type: ignore[import-not-found]
from sqlalchemy.orm import sessionmaker  # type: ignore[import-not-found]
from db.models import Base
from configs.settings import get_settings

settings = get_settings()

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print("[db] Tables created.")


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()