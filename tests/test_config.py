import os
from unittest.mock import patch


def test_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        import importlib
        import app.config
        importlib.reload(app.config)
        from app.config import (
            DEEPSEEK_BASE_URL, CHROMA_PERSIST_DIR, SQLITE_PATH,
            UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
        )
        assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"
        assert CHROMA_PERSIST_DIR == "./data/chroma"
        assert SQLITE_PATH == "./data/rag.db"
        assert UPLOAD_DIR == "./data/uploads"
        assert CHUNK_SIZE == 500
        assert CHUNK_OVERLAP == 50
        assert TOP_K == 3


def test_config_from_env():
    env_vars = {
        "DEEPSEEK_API_KEY": "sk-test-123",
        "DEEPSEEK_BASE_URL": "https://custom.api.com",
        "CHROMA_PERSIST_DIR": "/tmp/chroma",
        "SQLITE_PATH": "/tmp/test.db",
        "UPLOAD_DIR": "/tmp/uploads",
        "CHUNK_SIZE": "300",
        "CHUNK_OVERLAP": "30",
        "TOP_K": "5",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        import importlib
        import app.config
        importlib.reload(app.config)
        from app.config import (
            DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHROMA_PERSIST_DIR,
            SQLITE_PATH, UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
        )
        assert DEEPSEEK_API_KEY == "sk-test-123"
        assert DEEPSEEK_BASE_URL == "https://custom.api.com"
        assert CHROMA_PERSIST_DIR == "/tmp/chroma"
        assert SQLITE_PATH == "/tmp/test.db"
        assert UPLOAD_DIR == "/tmp/uploads"
        assert CHUNK_SIZE == 300
        assert CHUNK_OVERLAP == 30
        assert TOP_K == 5
