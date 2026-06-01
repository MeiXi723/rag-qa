import io
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_client():
    with patch("app.config.UPLOAD_DIR", tempfile.mkdtemp()), \
         patch("app.config.SQLITE_PATH", os.path.join(tempfile.mkdtemp(), "test.db")), \
         patch("app.config.CHROMA_PERSIST_DIR", tempfile.mkdtemp()):
        from app.main import app
        mock_llm = MagicMock()
        mock_llm.embed.return_value = [0.1] * 512
        mock_llm.chat.return_value = "Answer: Python is a programming language."
        import app.main as am
        am._llm_service = mock_llm
        client = TestClient(app)
        yield client
        # cleanup
        am._llm_service = None
        am._vector_store = None
        am._db = None


def test_root(mock_client):
    resp = mock_client.get("/")
    assert resp.status_code == 200
    assert "RAG" in resp.json()["message"]


def test_full_upload_ask_flow(mock_client):
    # 1. upload document
    content = b"Python is an interpreted programming language."
    upload_resp = mock_client.post(
        "/api/documents/upload",
        files={"file": ("python_intro.txt", io.BytesIO(content), "text/plain")}
    )
    assert upload_resp.status_code == 200
    doc = upload_resp.json()
    assert doc["filename"] == "python_intro.txt"
    assert doc["chunk_count"] > 0

    # 2. ask question
    ask_resp = mock_client.post("/api/qa/ask", json={"question": "What is Python?"})
    assert ask_resp.status_code == 200
    result = ask_resp.json()
    assert "answer" in result
    conv_id = result["conversation_id"]

    # 3. check conversation
    conv_resp = mock_client.get(f"/api/conversations/{conv_id}")
    assert conv_resp.status_code == 200
    assert len(conv_resp.json()["messages"]) == 2

    # 4. conversation list
    list_resp = mock_client.get("/api/conversations")
    assert len(list_resp.json()) == 1

    # 5. delete document
    del_resp = mock_client.delete(f"/api/documents/{doc['id']}")
    assert del_resp.status_code == 200

    # 6. delete conversation
    conv_del = mock_client.delete(f"/api/conversations/{conv_id}")
    assert conv_del.status_code == 200


def test_upload_invalid_file(mock_client):
    resp = mock_client.post(
        "/api/documents/upload",
        files={"file": ("malware.exe", io.BytesIO(b"bad"), "application/x-msdownload")}
    )
    assert resp.status_code == 400


def test_delete_nonexistent_document(mock_client):
    resp = mock_client.delete("/api/documents/nonexistent")
    assert resp.status_code == 404


def test_get_nonexistent_conversation(mock_client):
    resp = mock_client.get("/api/conversations/nonexistent")
    assert resp.status_code == 404


def test_docs_page(mock_client):
    resp = mock_client.get("/docs")
    assert resp.status_code == 200
