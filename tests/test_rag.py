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


def test_build_prompt(rag_service):
    chunks = [
        {"text": "内容A", "document_name": "a.pdf", "page": 1},
        {"text": "内容B", "document_name": "a.pdf", "page": 2},
    ]
    prompt = rag_service._build_prompt("问题？", chunks)
    assert "问题？" in prompt
    assert "内容A" in prompt
    assert "内容B" in prompt
    assert "[来源1]" in prompt
    assert "[来源2]" in prompt


def test_ask_returns_answer_with_sources(rag_service):
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.llm.chat.return_value = "这是答案。"
    result = rag_service.ask("问题？")
    assert "answer" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_name"] == "a.pdf"
