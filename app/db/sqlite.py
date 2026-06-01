import sqlite3
import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _uid():
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)

    def execute(self, sql: str, params=None):
        cur = self.conn.execute(sql, params or [])
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()

    # ── documents ──

    def insert_document(self, filename: str, file_type: str, file_size: int, chunk_count: int) -> str:
        doc_id = _uid()
        self.conn.execute(
            "INSERT INTO documents (id, filename, file_type, file_size, chunk_count, created_at) VALUES (?,?,?,?,?,?)",
            (doc_id, filename, file_type, file_size, chunk_count, _now())
        )
        self.conn.commit()
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        rows = self.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
        return rows[0] if rows else None

    def list_documents(self) -> list[dict]:
        return self.execute("SELECT * FROM documents ORDER BY created_at DESC")

    def delete_document(self, doc_id: str):
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.commit()

    # ── conversations ──

    def create_conversation(self, title: str = "") -> str:
        conv_id = _uid()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)",
            (conv_id, title, _now())
        )
        self.conn.commit()
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        rows = self.execute("SELECT * FROM conversations WHERE id=?", (conv_id,))
        if not rows:
            return None
        conv = rows[0]
        count = self.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id=?", (conv_id,)
        )[0]["cnt"]
        conv["message_count"] = count
        return conv

    def list_conversations(self) -> list[dict]:
        convs = self.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        for c in convs:
            c["message_count"] = self.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id=?", (c["id"],)
            )[0]["cnt"]
        return convs

    def delete_conversation(self, conv_id: str):
        self.conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        self.conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self.conn.commit()

    # ── messages ──

    def add_message(self, conv_id: str, role: str, content: str, sources: str | None) -> str:
        msg_id = _uid()
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) VALUES (?,?,?,?,?,?)",
            (msg_id, conv_id, role, content, sources, _now())
        )
        self.conn.commit()
        return msg_id

    def get_message(self, msg_id: str) -> dict | None:
        rows = self.execute("SELECT * FROM messages WHERE id=?", (msg_id,))
        return rows[0] if rows else None

    def get_messages_by_conversation(self, conv_id: str) -> list[dict]:
        return self.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
            (conv_id,)
        )
