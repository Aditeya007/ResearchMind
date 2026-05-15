from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.session import init_db
from api.routes import ingest, query, feedback
from configs.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="ResearchMind API",
    description="Multi-source Adaptive RAG with Hallucination Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(feedback.router)


@app.on_event("startup")
def startup():
    init_db()
    print("[api] ResearchMind API is running.")


@app.get("/")
def root():
    return {"message": "ResearchMind API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)