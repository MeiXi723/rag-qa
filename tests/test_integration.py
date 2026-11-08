import io
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def mock_client():
    # engine/SessionLocal 在 database.py 模块导入时就绑定真实路径，
    # 只 patch config 会被更早导入的其他测试文件破坏——直接替换模块全局
    import app.db.database as db_module
    test_engine = create_engine(
        f"sqlite:///{tempfile.mkdtemp()}/test.db",
        connect_args={"check_same_thread": False},
    )
    with patch("app.config.UPLOAD_DIR", tempfile.mkdtemp()), \
         patch("app.config.SQLITE_PATH", os.path.join(tempfile.mkdtemp(), "test.db")), \
         patch("app.config.CHROMA_PERSIST_DIR", tempfile.mkdtemp()), \
         patch.object(db_module, "_engine", test_engine), \
         patch.object(db_module, "_SessionLocal",
                      sessionmaker(autocommit=False, autoflush=False, bind=test_engine)):
        from app.main import app
        mock_llm = MagicMock()
        mock_llm.embed.return_value = [0.1] * 512
        mock_llm.chat.return_value = "Answer: Python is a programming language."
        # 单例住在 app.services 命名空间（不是 app.main）——挂错模块 mock 永不生效
        import app.services as services_module
        services_module._llm_service = mock_llm
        # with 块内 lifespan 才会执行（init_db 建表）；受保护接口需 JWT，
        # 先注册拿 token 再挂到后续请求上（v1 遗留：这些用例曾被 403 拦截）
        with TestClient(app) as client:
            reg = client.post(
                "/api/auth/register",
                json={"email": "integration@test.dev", "password": "secret6"},
            )
            assert reg.status_code == 200, reg.text
            client.headers.update({"Authorization": f"Bearer {reg.json()['access_token']}"})
            yield client
        # cleanup
        services_module._llm_service = None
        services_module._vector_store = None
        services_module._bm25_index = None
        test_engine.dispose()


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
