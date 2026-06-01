from fastapi import APIRouter
from app.models.schemas import AskRequest, AskResponse
from app.services.rag import RAGService

router = APIRouter(prefix="/api/qa", tags=["智能问答 / Q&A"])


@router.post("/ask", response_model=AskResponse,
             summary="提问 / Ask Question",
             description="基于已上传文档提问，返回带来源引用的答案。Ask questions based on uploaded documents, get answers with source citations.")
async def ask_question(req: AskRequest):
    from app.main import get_llm_service, get_vector_store, get_db

    rag = RAGService(
        llm=get_llm_service(),
        vector_store=get_vector_store(),
        db=get_db()
    )
    result = rag.ask(req.question, req.conversation_id)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=result["conversation_id"],
    )
