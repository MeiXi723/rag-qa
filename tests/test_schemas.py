from datetime import datetime
from app.models.schemas import (
    DocumentResponse, SourceInfo, AskRequest, AskResponse,
    ConversationResponse, MessageResponse, ErrorResponse
)


def test_document_response():
    doc = DocumentResponse(
        id="doc-1",
        filename="test.pdf",
        file_type="pdf",
        file_size=1024,
        chunk_count=5,
        created_at=datetime(2026, 6, 1, 12, 0, 0)
    )
    data = doc.model_dump()
    assert data["id"] == "doc-1"
    assert data["filename"] == "test.pdf"
    assert data["chunk_count"] == 5


def test_source_info():
    source = SourceInfo(
        document_name="test.pdf",
        chunk_text="这是一段测试文本。",
        page=3
    )
    assert source.page == 3
    source2 = SourceInfo(document_name="a.txt", chunk_text="hello")
    assert source2.page is None


def test_ask_request():
    req = AskRequest(question="什么是RAG？")
    assert req.conversation_id is None

    req2 = AskRequest(question="继续", conversation_id="conv-1")
    assert req2.conversation_id == "conv-1"


def test_ask_response():
    resp = AskResponse(
        answer="RAG是检索增强生成。",
        sources=[
            SourceInfo(document_name="doc.pdf", chunk_text="RAG是...", page=1)
        ],
        conversation_id="conv-1"
    )
    assert len(resp.sources) == 1
    assert resp.conversation_id == "conv-1"


def test_conversation_response():
    conv = ConversationResponse(
        id="conv-1",
        title="关于RAG的讨论",
        created_at=datetime(2026, 6, 1, 12, 0, 0),
        message_count=3
    )
    assert conv.message_count == 3


def test_message_response():
    msg = MessageResponse(
        id="msg-1",
        role="user",
        content="什么是RAG？",
        sources=None,
        created_at=datetime(2026, 6, 1, 12, 0, 0)
    )
    assert msg.role == "user"
    assert msg.sources is None


def test_error_response():
    err = ErrorResponse(detail="文件不存在")
    assert err.detail == "文件不存在"
