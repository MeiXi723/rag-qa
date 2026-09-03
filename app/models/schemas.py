# =============================================================================
# schemas.py — 数据模型定义（Pydantic Models）
# =============================================================================
#
# 【这个文件在干什么】
# 定义整个 API 的"输入输出合同"——请求长什么样、响应长什么样。
# 这些模型不是数据库表，而是 HTTP 传输的数据形状。
#
# 【Pydantic 是什么】
# Pydantic 是一个数据校验库，核心能力：
# 1. 声明式定义：用 Python type hints 描述数据结构
# 2. 自动校验：传入的数据不符合定义就报错（不给业务代码丢脏数据）
# 3. 自动序列化：Pydantic 对象可以 .model_dump() 转成 dict/JSON
# 4. FastAPI 深度集成：用在路由参数上，FastAPI 自动做校验 + 文档生成
#    ^ 这个是最关键的——你在路由里写 response_model=AskResponse，
#      FastAPI 就会自动把返回值转成符合 AskResponse 格式的 JSON。
#      API 文档（/docs）也会自动根据这些模型生成请求/响应的示例。
# =============================================================================

from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# FastAPI 直接用的 request 和 response：
# 这些 model 直接用于：
#   POST /api/qa/ask 的请求体 (@router.post("/ask", response_model=AskResponse))
#   POST /api/qa/ask 的响应体 (返回 AskResponse)
# ---------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """
    一个引用来源的信息。
    
    这是 RAG 系统区别于普通聊天机器人的关键数据结构——
    每个答案都可以追溯到具体的文档和段落。
    
    【Field() 的作用】
    Field(description="...") 不会影响运行时行为（不会真的校验 description 对不对），
    它的作用是生成 API 文档——Swagger 页面会展示这些说明文字。
    但 Field(default=None) 会影响行为：这个字段可以不传，默认是 None。
    """
    document_name: str = Field(description="来源文档名 / Source document name")
    chunk_text: str = Field(description="引用片段内容 / Cited text content")
    page: int | None = Field(default=None, description="页码（PDF）/ Page number (PDF)")


class AskRequest(BaseModel):
    """
    用户提问的请求格式。
    
    【conversation_id 为什么是可选的？】
    因为两种使用场景：
    1. 新对话：用户不传 conversation_id → 系统自动创建新对话
    2. 续写对话：用户传了 conversation_id → 系统往已有对话里追加消息
    
    【注意】当前项目虽然存了 conversation_id，
    但并没有真的把历史对话注入 LLM prompt——多轮对话是"假"的。
    这是第二阶段要修的问题。
    """
    question: str = Field(description="用户问题 / User question")
    conversation_id: str | None = Field(
        default=None,
        description="对话ID（可选，用于续写历史对话）/ Conversation ID (optional)"
    )


class AskResponse(BaseModel):
    """
    问答的响应格式。
    
    三个字段分别承载：
    - answer: LLM 生成的回答正文
    - sources: 支撑这个回答的文档片段（可溯源）
    - conversation_id: 本次对话的 ID（前端可以用它来续写对话）
    """
    answer: str = Field(description="AI 生成的回答 / AI-generated answer")
    sources: list[SourceInfo] = Field(description="引用来源列表 / Source citations")
    conversation_id: str = Field(description="对话ID / Conversation ID")


class DocumentResponse(BaseModel):
    """
    文档上传后的响应格式。
    
    【id 是什么】
    uuid 的前 12 位十六进制字符（看 sqlite.py 的 _uid()）。
    不是数据库自增 ID——因为自增 ID 不安全（别人可以猜 ID），
    uuid 是随机的，无法猜测。这是一个安全意识。
    """
    id: str = Field(description="文档唯一ID / Unique document ID")
    filename: str = Field(description="文件名 / File name")
    file_type: str = Field(description="文件类型 / File type (pdf/docx/txt)")
    file_size: int = Field(description="文件大小（字节）/ File size (bytes)")
    chunk_count: int = Field(description="文本分块数量 / Number of text chunks")
    created_at: datetime = Field(description="上传时间 / Upload time")


class ConversationResponse(BaseModel):
    """对话列表项的格式。只展示摘要信息，详细信息要单独请求。"""
    id: str = Field(description="对话ID / Conversation ID")
    title: str = Field(description="对话标题 / Conversation title")
    created_at: datetime = Field(description="创建时间 / Creation time")
    message_count: int = Field(description="消息数量 / Message count")


class MessageResponse(BaseModel):
    """
    单条消息的格式。
    
    【sources 为什么是可选的？】
    只有 assistant 角色的消息才有 sources（AI 回答的引用）。
    user 角色的消息没有 sources。
    """
    id: str = Field(description="消息ID / Message ID")
    role: str = Field(description="角色 / Role (user/assistant)")
    content: str = Field(description="消息内容 / Message content")
    sources: list[SourceInfo] | None = Field(
        default=None,
        description="引用来源 / Source citations (仅 assistant)"
    )
    created_at: datetime = Field(description="发送时间 / Send time")


class ErrorResponse(BaseModel):
    """
    统一的错误响应格式。
    
    不管什么错误（文件格式不对、文档不存在、服务器内部故障），
    都返回同一个结构，前端只需要处理一个错误格式。
    """
    detail: str = Field(description="错误详情 / Error detail")


class RegisterRequest(BaseModel):
    email: str = Field(description="邮箱")
    password: str = Field(min_length=6, description="密码，至少6位")


class LoginRequest(BaseModel):
    email: str = Field(description="邮箱")
    password: str = Field(description="密码")


class AuthResponse(BaseModel):
    access_token: str = Field(description="JWT 访问令牌")
    user_id: str = Field(description="用户ID")
    email: str = Field(description="用户邮箱")