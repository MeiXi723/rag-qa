# =============================================================================
# config.py — 项目配置中心
# =============================================================================
# 
# 【这个文件在干什么】
# 把散落在各处的配置（API密钥、文件路径、分块参数）集中到一个地方。
# 这样做的好处：改配置只需要改这一个文件（或 .env），不用去业务代码里翻。
#
# 【你会反复遇到的模式：环境变量】
# 代码里永远不写死 API key 和密码，而是从环境变量读。
# 开发时：变量写在项目根目录的 .env 文件里（git不会提交它）
# 生产时：变量由 Docker / K8s 注入到容器环境中
# 原则：敏感信息不进代码仓库。
# =============================================================================

import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------
# 
# os.getenv("KEY", "默认值") — 第二个参数是 fallback：
#   如果环境变量里有 KEY，就用它；没有就用默认值。
#   这样就算 .env 文件丢了，程序也不会直接炸，会用默认值兜底。
#
# DEEPSEEK_API_KEY: DeepSeek 的鉴权 token。
#   注意是 "Bearer Token" 模式 — 每个请求的 HTTP Header 里带这个 key。
#   空的默认值意味着：如果没配 .env，LLM 调用会直接报 401。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# DEEPSEEK_BASE_URL: API 的地址。
#   为什么叫 "base_url" 而不是直接写完整 URL？
#   因为 OpenAI SDK 的约定是 base_url + 固定路径（如 /v1/chat/completions）。
#   换一个兼容 OpenAI 的供应商（如硅基流动）只需改这一个值。
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ---------------------------------------------------------------------------
# 存储路径
# ---------------------------------------------------------------------------
#
# CHROMA_PERSIST_DIR: FAISS 索引文件存放目录。
#   注意这个变量名叫 CHROMA 但实际存的是 FAISS 文件 — 历史遗留命名。
#   里面会有两个文件：documents.index（FAISS 二进制索引）和 documents.data.json（元数据）
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

# SQLITE_PATH: SQLite 数据库文件路径。
#   SQLite 是嵌入式数据库 — 整个数据库就是一个文件。
#   不需要单独安装数据库服务，非常适合单机原型项目。
#   生产环境如果要多人并发写，会换成 PostgreSQL。
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/rag.db")

# UPLOAD_DIR: 用户上传的原始文件存放目录。
#   注意：文件存在本地磁盘，不是存在数据库里。
#   数据库只存文件的元信息（文件名、大小、分块数），实际文件在磁盘。
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")

# ---------------------------------------------------------------------------
# 分块参数 — 控制文档怎么被切成小块
# ---------------------------------------------------------------------------
#
# 这三个参数直接影响 RAG 的检索质量。改之前要理解它们的含义：
#
# CHUNK_SIZE: 每个文本块的最大字符数。
#   500 是一个保守值（约 300-400 个中文字）。
#   设太大 → 一个 block 包含多个不相关话题 → 检索不精准。
#   设太小 → 一个完整句子被截断 → LLM 看到碎片。
#   关键约束：embedding 模型的输入上限（bge-small 是 512 tokens）。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

# CHUNK_OVERLAP: 相邻两块之间的重叠字符数。
#   为什么需要 overlap？因为固定长度切割会在句子中间截断。
#   上一块的末尾和下一块的开头重复 50 个字符，
#   保证被截断的信息在至少一个块中是完整的。
#   但当前项目用的是简单字符切割，overlap 解决不了语义截断问题 —
#   这就是为什么第二阶段要改成递归分割。
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# TOP_K: 检索时返回多少个最相关的文档块。
#   这些块会被拼成 prompt 的背景知识喂给 LLM。
#   TOP_K=3 意味着 LLM 最多看到 3 个文档片段。
#   更多片段 = 更全的信息，但也 = 更多无关噪声 + 更多 token 消耗。
TOP_K = int(os.getenv("TOP_K", "3"))

# MAX_HISTORY_MESSAGES: 多轮对话中注入 prompt 的历史消息数量上限。
#   10 条 = 5 轮完整 Q&A。设太大 → token 消耗大 + 可能超出模型上下文窗口。
#   设太小 → LLM 丢失前面的对话内容，回答不连贯。
#   设为 0 可以禁用多轮对话记忆（退化为每次独立提问）。
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/rag.db")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))