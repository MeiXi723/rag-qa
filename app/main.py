from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.database import init_db

@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RAG 知识库问答系统 / RAG Knowledge Base Q&A",
    description="上传文档，智能问答，答案可溯源。Upload documents, ask questions, get answers with source citations.",
    version="2.0.0",
    lifespan=lifespan,
)


from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.qa import router as qa_router
from app.api.conversations import router as conversations_router

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(qa_router)
app.include_router(conversations_router)


@app.get("/")
def root():
    return {
        "message": "RAG 知识库问答系统已启动 / RAG Knowledge Base Q&A is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """存活探针：无鉴权、零依赖，供 Docker HEALTHCHECK 和负载均衡使用。"""
    return {"status": "ok", "version": app.version}
