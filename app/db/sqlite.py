# =============================================================================
# sqlite.py — 关系型数据存储（文档元数据、对话历史）
# =============================================================================
#
# 【这个文件在干什么】
# 用 SQLite 管理三类数据：
# 1. documents：上传的文档信息（文件名、类型、大小、分块数）
# 2. conversations：对话会话（标题、创建时间）
# 3. messages：对话中的每条消息（用户提问、AI 回答及其引用来源）
#
# 【为什么用 SQLite 而不是 PostgreSQL？】
# SQLite 是一个嵌入式数据库——不需要单独安装数据库服务。
# 整个数据库就是一个文件（./data/rag.db），方便备份和迁移。
# 对于单机原型项目，SQLite 足够用了。
# 如果要支持多人同时写入（比如 SaaS 产品），就需要换 PostgreSQL。
#
# 【三张表的关系】
# documents ← 独立存在，不依赖其他表
# conversations ← 独立存在，不依赖其他表
# messages.conversation_id → conversations.id（外键关联）
#
# 注意：documents 和 conversations/messages 之间没有直接的数据库关联。
# 它们的关联是在 rag.py 的 ask() 方法里，通过向量检索来实现的——
# messages.sources 字段存的是引用的文档信息（JSON 格式），不是外键。
# =============================================================================

import sqlite3
import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _now():
    """
    返回当前的 UTC 时间，ISO 8601 格式的字符串。

    例如："2026-06-06T14:32:15.123456+00:00"

    【为什么用 UTC 而不是本地时间？】
    - 服务器可能在不同时区运行
    - UTC 是全球唯一的时间参考，不存在夏令时等问题
    - 前端拿到 UTC 时间后可以根据用户所在的时区转换显示
    """
    return datetime.now(timezone.utc).isoformat()


def _uid():
    """
    生成一个短唯一 ID。

    uuid.uuid4()       → UUID('a3f2b1c4-5678-9abc-def0-123456789abc')
    .hex               → 'a3f2b1c456789abcdef0123456789abc'
    [:12]              → 'a3f2b1c45678'（前 12 个十六进制字符）

    【为什么不用数据库自增 ID？】
    1. 自增 ID 可预测——别人能猜到你的文档 ID（比如 /api/documents/42）
       → 安全性风险（虽然这个项目没有权限系统，但这是一个好习惯）
    2. UUID 在分布式系统中可以多机同时生成而不会冲突
       → 自增 ID 需要依赖单点数据库

    【12 位十六进制够用吗？】
    16^12 = 约 2.8 × 10^14 种组合。
    在这个量级下，碰撞概率极低。
    如果数据量很大（亿级），建议用完整 32 位的 UUID。
    """
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Database 类
# ---------------------------------------------------------------------------

class Database:
    """
    SQLite 数据库的封装层。

    【设计特点】
    - 直接操作 SQL 字符串，没有用 ORM（SQLAlchemy）
      → 为什么？因为这个项目只有 3 张表，ORM 的"自动建表/迁移/关系管理"优势不明显
        直接用 SQL 更简单、性能更好、学习成本更低
    - 每个方法对应一个具体的业务操作（如 insert_document）
      → 没有通用的 CRUD 接口，因为不同表的操作语义不同

    【WAL 模式】
    PRAGMA journal_mode=WAL 是一个关键设置。
    默认的 SQLite 用 rollback journal：读写互斥——有人在写时其他人不能读。
    WAL (Write-Ahead Log) 允许读写并发——读和写可以同时进行。
    这对 Web 应用来说是基本要求，否则接口一多就会"database is locked"。

    【check_same_thread=False】
    FastAPI 可能在多个线程中调用数据库操作。
    默认情况下 SQLite 禁止跨线程使用同一个连接。
    设置这个参数后允许多线程访问——但要自己保证线程安全（SQLite 的 WAL 模式可以处理）。
    """

    def __init__(self, path: str):
        # 连接 SQLite 文件。如果文件不存在，SQLite 会自动创建。
        self.conn = sqlite3.connect(path, check_same_thread=False)

        # row_factory = sqlite3.Row 让查询结果可以用列名访问
        # 例如 row["id"] 而不是 row[0]（数字下标）
        # 这让代码更可读——你不数数据库有哪些列
        self.conn.row_factory = sqlite3.Row

        # 启用 WAL 模式（允许读写并发）
        self.conn.execute("PRAGMA journal_mode=WAL")
        # 启用外键约束（messages.conversation_id 的 ON DELETE CASCADE 才会生效）
        self.conn.execute("PRAGMA foreign_keys=ON")

        # 建表（如果表不存在的话）
        self._init_tables()

    def _init_tables(self):
        """
        创建数据库表结构。

        execscript() 可以一次执行多条 SQL 语句。
        IF NOT EXISTS 保证重复调用不会报错。

        【三张表的设计意图】
        - documents: 记录文档的元信息（就像一个图书馆的图书目录）
        - conversations: 记录对话会话（就像一个聊天记录的文件夹）
        - messages: 记录每条消息（就像文件夹里的每一条聊天记录）
        """
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
        # ON DELETE CASCADE: 删除 conversations 时自动删除它的所有 messages
        # 这样 delete_conversation 方法就只需要删 conversations 这一条 SQL

    def execute(self, sql: str, params=None):
        """
        执行查询并返回 dict 列表。

        sqlite3.Row 对象 → dict 对象的转换。

        【为什么自己包一层而不是直接用 conn.execute？】
        因为 sqlite3.Row 对象的行为和 dict 有细微差别（比如不能直接 .get()）。
        转成 dict 后业务代码处理起来更顺手。
        """
        cur = self.conn.execute(sql, params or [])
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close(self):
        """关闭数据库连接（应用退出时调用）"""
        self.conn.close()

    # =========================================================================
    # 文档相关操作
    # =========================================================================

    def insert_document(self, filename: str, file_type: str, file_size: int, chunk_count: int) -> str:
        """
        插入一条文档记录，返回文档 ID。

        注意：这个方法只记录元信息。实际文件存在 UPLOAD_DIR 目录里。
        向量存在 FAISS 索引里。数据库只管文档的"身份证信息"。
        """
        doc_id = _uid()
        self.conn.execute(
            "INSERT INTO documents (id, filename, file_type, file_size, chunk_count, created_at) VALUES (?,?,?,?,?,?)",
            (doc_id, filename, file_type, file_size, chunk_count, _now())
        )
        self.conn.commit()  # 每次写操作后立即提交
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        """查单个文档。不存在则返回 None。"""
        rows = self.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
        return rows[0] if rows else None

    def list_documents(self) -> list[dict]:
        """列出所有文档，按创建时间倒序（最新的在前）。"""
        return self.execute("SELECT * FROM documents ORDER BY created_at DESC")

    def delete_document(self, doc_id: str):
        """
        删除文档记录（仅数据库层面）。

        真正的删除需要同时做三件事：
        1. 删除 FAISS 索引中的向量（在 api/documents.py 的 delete_document 路由里调用 vs.delete_by_document_id）
        2. 删除原始文件（当前代码没有做这一步！文件留在 UPLOAD_DIR 里）
        3. 删除数据库记录（这个方法）

        【当前项目的 bug】
        删除文档时没有清理 UPLOAD_DIR 里的原始文件。
        磁盘会慢慢被废弃文件占满。
        """
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.commit()

    # =========================================================================
    # 对话相关操作
    # =========================================================================

    def create_conversation(self, title: str = "") -> str:
        """
        创建新对话。

        title 通常用用户第一条消息的前几个字（见 rag.py 的 ask 方法）。
        创建时没有消息——消息由 add_message 单独添加。
        """
        conv_id = _uid()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)",
            (conv_id, title, _now())
        )
        self.conn.commit()
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        """
        获取对话详情，包含消息数量。

        这里手动计算 message_count（COUNT 查询），
        而不是存一个字段——因为消息数量是动态变化的，存字段需要每次增删消息时更新。
        """
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
        """
        列出所有对话，每条带消息数量。

        【性能注意】
        对每条对话都执行一次 COUNT 查询。如果有 1000 条对话，就是 1000 次查询。
        对于原型阶段没问题，但生产环境应该用 JOIN + GROUP BY 一次查询完成。
        """
        convs = self.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        for c in convs:
            c["message_count"] = self.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id=?", (c["id"],)
            )[0]["cnt"]
        return convs

    def delete_conversation(self, conv_id: str):
        """
        删除对话及其所有消息。

        因为建表时设置了 ON DELETE CASCADE，
        删除 conversations 记录会自动级联删除对应的 messages 记录。
        所以这里只需要删 conversations。
        """
        self.conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
        self.conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self.conn.commit()

    # =========================================================================
    # 消息相关操作
    # =========================================================================

    def add_message(self, conv_id: str, role: str, content: str, sources: str | None) -> str:
        """
        往对话中添加一条消息。

        role 只能是 "user" 或 "assistant"（由建表时的 CHECK 约束保证）。
        sources 是 JSON 字符串（user 消息时为 None）。
        """
        msg_id = _uid()
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) VALUES (?,?,?,?,?,?)",
            (msg_id, conv_id, role, content, sources, _now())
        )
        self.conn.commit()
        return msg_id

    def get_message(self, msg_id: str) -> dict | None:
        """查单条消息"""
        rows = self.execute("SELECT * FROM messages WHERE id=?", (msg_id,))
        return rows[0] if rows else None

    def get_messages_by_conversation(self, conv_id: str) -> list[dict]:
        """
        获取一个对话的所有消息，按发送时间正序。

        为什么是 ASC（正序）？
        - 前端展示需要从旧到新的顺序（就像聊天记录）
        - list_conversations 用 DESC（倒序）是因为列表页需要看最新
        """
        return self.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
            (conv_id,)
        )
