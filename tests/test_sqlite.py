import os
import tempfile
import pytest
from app.db.sqlite import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


def test_init_creates_tables(db):
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {t["name"] for t in tables}
    assert "documents" in table_names
    assert "conversations" in table_names
    assert "messages" in table_names


def test_insert_document(db):
    doc_id = db.insert_document("test.pdf", "pdf", 2048, 4)
    assert doc_id is not None
    doc = db.get_document(doc_id)
    assert doc["filename"] == "test.pdf"
    assert doc["file_type"] == "pdf"
    assert doc["file_size"] == 2048
    assert doc["chunk_count"] == 4


def test_list_documents(db):
    db.insert_document("a.pdf", "pdf", 100, 2)
    db.insert_document("b.docx", "docx", 200, 3)
    docs = db.list_documents()
    assert len(docs) == 2


def test_delete_document(db):
    doc_id = db.insert_document("x.pdf", "pdf", 100, 1)
    db.delete_document(doc_id)
    assert db.get_document(doc_id) is None


def test_create_conversation(db):
    conv_id = db.create_conversation("RAG讨论")
    conv = db.get_conversation(conv_id)
    assert conv["title"] == "RAG讨论"


def test_list_conversations(db):
    db.create_conversation("对话1")
    db.create_conversation("对话2")
    convs = db.list_conversations()
    assert len(convs) == 2
    assert "message_count" in convs[0]


def test_add_message(db):
    conv_id = db.create_conversation("测试")
    msg_id = db.add_message(conv_id, "user", "什么是AI？", None)
    msg = db.get_message(msg_id)
    assert msg["role"] == "user"
    assert msg["content"] == "什么是AI？"
    assert msg["sources"] is None


def test_get_messages_by_conversation(db):
    conv_id = db.create_conversation("测试")
    db.add_message(conv_id, "user", "问题1", None)
    db.add_message(conv_id, "assistant", "回答1", '[{"document_name":"a.pdf","chunk_text":"..."}]')
    msgs = db.get_messages_by_conversation(conv_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_delete_conversation(db):
    conv_id = db.create_conversation("待删除")
    db.add_message(conv_id, "user", "hello", None)
    db.delete_conversation(conv_id)
    assert db.get_conversation(conv_id) is None
    assert len(db.get_messages_by_conversation(conv_id)) == 0
