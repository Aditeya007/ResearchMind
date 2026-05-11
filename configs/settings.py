# configs/settings.py

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    
    LLM_PROVIDER: str = "groq"               
    GROQ_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = "llama3-8b-8192"   
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   

    VECTOR_STORE_TYPE: str = "chroma"            
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "researchmind"

    TOP_K_DENSE: int = 5          
    TOP_K_SPARSE: int = 5         
    TOP_K_RERANK: int = 3         
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    
    DATABASE_URL: str = "sqlite:///./researchmind.db"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()