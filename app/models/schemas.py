from datetime import datetime
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    document_name: str = Field(description="来源文档名 / Source document name")
    chunk_text: str = Field(description="引用片段内容 / Cited text content")
    page: int | None = Field(default=None, description="页码（PDF）/ Page number (PDF)")


class AskRequest(BaseModel):
    question: str = Field(description="用户问题 / User question")
    conversation_id: str | None = Field(default=None, description="对话ID（可选，延续历史对话）/ Conversation ID (optional)")


class AskResponse(BaseModel):
    answer: str = Field(description="AI 生成的回答 / AI-generated answer")
    sources: list[SourceInfo] = Field(description="引用来源列表 / Source citations")
    conversation_id: str = Field(description="对话ID / Conversation ID")


class DocumentResponse(BaseModel):
    id: str = Field(description="文档唯一ID / Unique document ID")
    filename: str = Field(description="文件名 / File name")
    file_type: str = Field(description="文件类型 / File type (pdf/docx/txt)")
    file_size: int = Field(description="文件大小（字节）/ File size (bytes)")
    chunk_count: int = Field(description="文本分块数量 / Number of text chunks")
    created_at: datetime = Field(description="上传时间 / Upload time")


class ConversationResponse(BaseModel):
    id: str = Field(description="对话ID / Conversation ID")
    title: str = Field(description="对话标题 / Conversation title")
    created_at: datetime = Field(description="创建时间 / Creation time")
    message_count: int = Field(description="消息数量 / Message count")


class MessageResponse(BaseModel):
    id: str = Field(description="消息ID / Message ID")
    role: str = Field(description="角色 / Role (user/assistant)")
    content: str = Field(description="消息内容 / Message content")
    sources: list[SourceInfo] | None = Field(default=None, description="引用来源 / Source citations (仅 assistant)")
    created_at: datetime = Field(description="发送时间 / Send time")


class ErrorResponse(BaseModel):
    detail: str = Field(description="错误详情 / Error detail")
