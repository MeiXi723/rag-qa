from fastapi import FastAPI
from app.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHROMA_PERSIST_DIR, SQLITE_PATH
)
from app.services.llm import LLMService
from app.db.chroma import VectorStore
from app.db.sqlite import Database

app = FastAPI(
    title="RAG 知识库问答系统 / RAG Knowledge Base Q&A",
    description="上传文档，智能问答，答案可溯源。Upload documents, ask questions, get answers with source citations.",
    version="1.0.0"
)

_llm_service: LLMService | None = None
_vector_store: VectorStore | None = None
_db: Database | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _llm_service


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(CHROMA_PERSIST_DIR)
    return _vector_store


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database(SQLITE_PATH)
    return _db


from app.api.documents import router as documents_router
from app.api.qa import router as qa_router
from app.api.conversations import router as conversations_router

app.include_router(documents_router)
app.include_router(qa_router)
app.include_router(conversations_router)


@app.get("/")
def root():
    return {
        "message": "RAG 知识库问答系统已启动 / RAG Knowledge Base Q&A is running",
        "docs": "/docs"
    }
