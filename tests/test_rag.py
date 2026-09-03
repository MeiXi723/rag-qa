import pytest
from unittest.mock import MagicMock
from app.services.rag import RAGService


@pytest.fixture
def rag_service():
    llm = MagicMock()
    llm.embed.return_value = [0.1] * 10
    llm.chat.return_value = "根据文档内容，答案是：这是答案。"
    vector_store = MagicMock()
    db = MagicMock()
    return RAGService(llm=llm, vector_store=vector_store, db=db)


def test_retrieve(rag_service):
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["文档内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    chunks = rag_service.retrieve("测试问题", n_results=3)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "文档内容片段"
    assert chunks[0]["document_name"] == "a.pdf"


def test_build_messages_no_history(rag_service):
    chunks = [
        {"text": "内容A", "document_name": "a.pdf", "page": 1},
        {"text": "内容B", "document_name": "a.pdf", "page": 2},
    ]
    messages = rag_service._build_messages("问题？", chunks)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "知识库问答助手" in messages[0]["content"]
    assert "问题？" in messages[1]["content"]
    assert "内容A" in messages[1]["content"]
    assert "[来源1]" in messages[1]["content"]
    assert "[来源2]" in messages[1]["content"]


def test_build_messages_with_history(rag_service):
    chunks = [{"text": "内容A", "document_name": "a.pdf", "page": 1}]
    history = [
        {"role": "user", "content": "什么是 Python？"},
        {"role": "assistant", "content": "Python 是一种编程语言。"},
    ]
    messages = rag_service._build_messages("它的优缺点？", chunks, history)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "什么是 Python？"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert "它的优缺点？" in messages[3]["content"]


def test_ask_without_conversation_id(rag_service):
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.llm.chat.return_value = "这是答案。"
    result = rag_service.ask("问题？")
    assert "answer" in result
    assert result["conversation_id"]
    assert result["sources"][0]["document_name"] == "a.pdf"

    call_args = rag_service.llm.chat.call_args[0][0]
    assert len(call_args) == 2  # system + user, no history


def test_ask_with_history(rag_service):
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.llm.chat.return_value = "这是后续回答。"

    rag_service.db.get_conversation.return_value = {"id": "conv-1", "title": "测试"}
    rag_service.db.get_messages_by_conversation.return_value = [
        {"role": "user", "content": "什么是 RAG？", "sources": None},
        {"role": "assistant", "content": "RAG 是检索增强生成。", "sources": "[]"},
    ]

    result = rag_service.ask("它的优势是什么？", conversation_id="conv-1")
    assert result["conversation_id"] == "conv-1"

    call_args = rag_service.llm.chat.call_args[0][0]
    assert len(call_args) == 4  # system + 2 history + user
    assert call_args[0]["role"] == "system"
    assert call_args[1]["role"] == "user"
    assert call_args[1]["content"] == "什么是 RAG？"
    assert call_args[2]["role"] == "assistant"
    assert call_args[3]["role"] == "user"
    assert "它的优势是什么？" in call_args[3]["content"]
