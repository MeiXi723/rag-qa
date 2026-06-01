import json
from fastapi import APIRouter, HTTPException
from app.models.schemas import ConversationResponse

router = APIRouter(prefix="/api/conversations", tags=["对话历史 / Conversation History"])


@router.get("", response_model=list[ConversationResponse],
            summary="对话列表 / List Conversations",
            description="获取所有对话历史列表。Get all conversation history.")
async def list_conversations():
    from app.main import get_db
    return get_db().list_conversations()


@router.get("/{conv_id}",
            summary="对话详情 / Conversation Detail",
            description="获取指定对话的完整消息记录。Get full message history of a conversation.")
async def get_conversation(conv_id: str):
    from app.main import get_db
    db = get_db()
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在 / Conversation not found")
    messages = db.get_messages_by_conversation(conv_id)
    conv["messages"] = [
        {
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "sources": json.loads(m["sources"]) if m["sources"] else None,
            "created_at": m["created_at"],
        }
        for m in messages
    ]
    return conv


@router.delete("/{conv_id}",
               summary="删除对话 / Delete Conversation",
               description="删除指定对话及其所有消息。Delete a conversation and all its messages.")
async def delete_conversation(conv_id: str):
    from app.main import get_db
    db = get_db()
    if not db.get_conversation(conv_id):
        raise HTTPException(status_code=404, detail="对话不存在 / Conversation not found")
    db.delete_conversation(conv_id)
    return {"detail": "已删除 / Deleted"}
