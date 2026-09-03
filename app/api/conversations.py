import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db.database import get_db
from app.models.db_models import Conversation, Message, User
from app.models.schemas import ConversationResponse

router = APIRouter(prefix="/api/conversations", tags=["对话历史 / Conversation History"])


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            message_count=db.query(Message).filter(Message.conversation_id == c.id).count(),
        )
        for c in convs
    ]


@router.get("/{conv_id}")
async def get_conversation(
    conv_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "message_count": len(messages),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources if m.sources else None,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    }


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == current_user.id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    db.delete(conv)
    db.commit()
    return {"detail": "已删除"}
