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
    assert result["conversation_id"] == ""  # 无 user_id（未登录）：不落库直接返回
    assert result["sources"][0]["document_name"] == "a.pdf"

    call_args = rag_service.llm.chat.call_args[0][0]
    assert len(call_args) == 2  # system + user, no history


def test_ask_creates_conversation_on_first_ask(rag_service):
    # 契约：登录用户首次提问（无 conversation_id）必须建会话并返回 ID
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.llm.chat.return_value = "这是答案。"

    # 无既有会话：第一次 query 返回 None
    conv_query = MagicMock()
    conv_query.filter.return_value.first.return_value = None
    rag_service.db.query.side_effect = [conv_query]

    def fake_refresh(obj):
        obj.id = "new-conv-1"  # 模拟数据库写入后回填主键
    rag_service.db.refresh.side_effect = fake_refresh

    result = rag_service.ask("新问题？", user_id="u-1")
    assert result["conversation_id"] == "new-conv-1"
    rag_service.db.add.assert_called()  # 新会话 + 两条消息落库


def test_ask_with_history(rag_service):
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.llm.chat.return_value = "这是后续回答。"

    # ask() 走 SQLAlchemy query 链：mock Conversation 查找与 Message 历史两次查询
    conv_row = MagicMock(id="conv-1")
    conv_query = MagicMock()
    conv_query.filter.return_value.first.return_value = conv_row
    history_rows = [
        MagicMock(role="user", content="什么是 RAG？"),
        MagicMock(role="assistant", content="RAG 是检索增强生成。"),
    ]
    msg_query = MagicMock()
    msg_query.filter.return_value.order_by.return_value.all.return_value = history_rows
    rag_service.db.query.side_effect = [conv_query, msg_query, conv_query]

    result = rag_service.ask("它的优势是什么？", conversation_id="conv-1", user_id="u-1")
    assert result["conversation_id"] == "conv-1"

    call_args = rag_service.llm.chat.call_args[0][0]
    assert len(call_args) == 4  # system + 2 history + user
    assert call_args[0]["role"] == "system"
    assert call_args[1]["role"] == "user"
    assert call_args[1]["content"] == "什么是 RAG？"
    assert call_args[2]["role"] == "assistant"
    assert call_args[3]["role"] == "user"
    assert "它的优势是什么？" in call_args[3]["content"]


def test_ask_returns_answer_confidence(rag_service):
    # v2.0 新增：ask 结果必须带置信度评估
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["内容片段内容片段内容片段内容片段内容片段内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.llm.chat.return_value = "根据 [来源1]，这是答案。"
    result = rag_service.ask("问题？")
    assert result["answer_confidence"] == {"confidence": "high", "cited_sources": [1]}


def test_retrieve_hybrid_fuses_both_legs(rag_service):
    # bm25_index 注入后 retrieve 走混合路径：双路结果都进候选池
    rag_service.bm25_index = MagicMock()
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-v"]],
        "documents": [["向量召回的文本内容，用于混合检索链路的单元测试验证"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    rag_service.bm25_index.search.return_value = [{
        "id": "chunk-b",
        "text": "关键词召回的文本内容，来自 BM25 稀疏检索这一条腿",
        "document_name": "b.pdf", "chunk_index": 3, "page": None, "score": 2.5,
    }]

    chunks = rag_service.retrieve("测试查询")
    by_id = {c["id"]: c for c in chunks}
    assert set(by_id) == {"chunk-v", "chunk-b"}
    assert by_id["chunk-v"]["source"] == "vector"
    assert by_id["chunk-b"]["source"] == "bm25"


def test_retrieve_without_bm25_falls_back_to_vector(rag_service):
    # 不注入 bm25_index（默认 None）→ 与 v1 行为完全一致
    rag_service.vector_store.search.return_value = {
        "ids": [["chunk-0"]],
        "documents": [["文档内容片段"]],
        "metadatas": [[{"document_name": "a.pdf", "chunk_index": 0, "page": 1}]],
        "distances": [[0.1]]
    }
    chunks = rag_service.retrieve("测试问题", n_results=3)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "文档内容片段"
    assert "source" not in chunks[0]  # v1 路径不带融合字段
