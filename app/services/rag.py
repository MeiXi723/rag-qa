import hashlib
import json
import logging

from app.config import TOP_K, MAX_HISTORY_MESSAGES
from app.models.db_models import Conversation, Message

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, llm, vector_store, db):
        self.llm = llm
        self.vector_store = vector_store
        self.db = db

    def retrieve(self, query: str, n_results: int = TOP_K) -> list[dict]:
        embedding = self.llm.embed(query)
        results = self.vector_store.search(embedding, n_results=n_results)

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunks.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "document_name": results["metadatas"][0][i].get("document_name", ""),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    "page": results["metadatas"][0][i].get("page"),
                })
        return chunks

    def _build_messages(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        system_prompt = (
            "你是一个严谨的知识库问答助手。你的回答必须基于参考资料，不得编造信息。\n"
            "\n"
            "规则：\n"
            "1. 仔细阅读参考资料，从中找到与用户问题最相关的内容\n"
            "2. 回答时引用来源编号，例如「根据 [来源1]……」\n"
            "3. 如果参考资料不足以回答问题，请明确说「参考资料中未找到相关信息」，"
            "不要猜测或编造答案\n"
            "4. 如果参考资料涉及多个来源，请综合多个来源的信息作答\n"
            "5. 回答结构清晰，使用适当的换行和分段"
        )

        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[来源{i+1}] 文档《{chunk['document_name']}》:\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        user_prompt = (
            f"参考资料：\n"
            f"{context}\n"
            f"\n"
            f"用户问题：{question}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def ask(self, question: str, conversation_id: str | None = None, user_id: str | None = None) -> dict:
        from app.cache import cache_get_llm_response, cache_set_llm_response

        chunks = self.retrieve(question)
        context_hash = hashlib.md5(
            "".join(c["text"][:200] for c in chunks).encode()
        ).hexdigest()

        cached_answer = cache_get_llm_response(question, context_hash)
        if cached_answer is not None:
            logger.info("LLM cache hit for question: %s...", question[:50])

        history = []
        if conversation_id and user_id:
            conv = self.db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            ).first()
            if conv:
                messages = (
                    self.db.query(Message)
                    .filter(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at.asc())
                    .all()
                )
                if MAX_HISTORY_MESSAGES > 0:
                    messages = messages[-MAX_HISTORY_MESSAGES:]
                history = [{"role": m.role, "content": m.content} for m in messages]

        if cached_answer is not None:
            answer = cached_answer
        else:
            messages = self._build_messages(question, chunks, history)
            answer = self.llm.chat(messages)
            cache_set_llm_response(question, context_hash, answer)

        sources = [
            {
                "document_name": c["document_name"],
                "chunk_text": c["text"][:200],
                "page": c["page"],
            }
            for c in chunks
        ]

        if not conversation_id or not user_id:
            return {"answer": answer, "sources": sources, "conversation_id": ""}

        conv = self.db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        ).first()
        if not conv:
            conv = Conversation(title=question[:50], user_id=user_id)
            self.db.add(conv)
            self.db.commit()
            self.db.refresh(conv)
            conversation_id = conv.id

        user_msg = Message(
            conversation_id=conversation_id, role="user", content=question
        )
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        self.db.add_all([user_msg, assistant_msg])
        self.db.commit()

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
        }
