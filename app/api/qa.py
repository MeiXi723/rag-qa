from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.db.database import get_db
from app.models.db_models import User
from app.models.schemas import AskRequest, AskResponse
from app.services.rag import RAGService
from app.services import get_llm_service, get_vector_store
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/qa", tags=["智能问答 / Q&A"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    req: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rag = RAGService(
        llm=get_llm_service(),
        vector_store=get_vector_store(),
        db=db,
    )

    result = rag.ask(req.question, req.conversation_id, user_id=current_user.id)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=result["conversation_id"],
    )
