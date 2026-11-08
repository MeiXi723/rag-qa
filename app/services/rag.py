import hashlib
import logging

from app.config import (
    TOP_K, MAX_HISTORY_MESSAGES,
    ENABLE_HYBRID, RETRIEVAL_CANDIDATES, RRF_K,
    MIN_CHUNK_LENGTH, ENABLE_RERANK, RERANKER_MODEL, TOP_N_RERANK,
)
from app.models.db_models import Conversation, Message
from app.services.retrieval.fusion import rrf_fuse
from app.services.retrieval.quality import quality_filter
from app.services.retrieval.rerank import cross_encoder_rerank
from app.services.retrieval.validate import assess_answer

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, llm, vector_store, db, bm25_index=None):
        self.llm = llm
        self.vector_store = vector_store
        self.db = db
        # bm25_index 为 None 时走 v1 纯向量路径——排障与回退兜底
        self.bm25_index = bm25_index

    def retrieve(self, query: str, n_results: int = TOP_K) -> list[dict]:
        if ENABLE_HYBRID and self.bm25_index is not None:
            return self._retrieve_hybrid(query, n_results)
        return self._retrieve_vector(query, n_results)

    def _retrieve_vector(self, query: str, n_results: int) -> list[dict]:
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

    def _retrieve_hybrid(self, query: str, n_results: int) -> list[dict]:
        """BM25+向量双路召回 → RRF 融合 → 质量过滤 → CrossEncoder 精排。"""
        embedding = self.llm.embed(query)
        vec_results = self.vector_store.search(
            embedding, n_results=max(RETRIEVAL_CANDIDATES, n_results)
        )
        vector_chunks = []
        if vec_results["ids"] and vec_results["ids"][0]:
            for i in range(len(vec_results["ids"][0])):
                vector_chunks.append({
                    "id": vec_results["ids"][0][i],
                    "text": vec_results["documents"][0][i] if vec_results["documents"] else "",
                    "document_name": vec_results["metadatas"][0][i].get("document_name", ""),
                    "chunk_index": vec_results["metadatas"][0][i].get("chunk_index", 0),
                    "page": vec_results["metadatas"][0][i].get("page"),
                })

        bm25_chunks = self.bm25_index.search(query, top_n=RETRIEVAL_CANDIDATES)

        fused = rrf_fuse(vector_chunks, bm25_chunks, k=RRF_K)
        fused = quality_filter(fused, min_length=MIN_CHUNK_LENGTH)
        if ENABLE_RERANK:
            fused = cross_encoder_rerank(query, fused, top_n=TOP_N_RERANK, model_name=RERANKER_MODEL)
        return fused[:n_results]

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

        # 无 user_id（未登录）：不落库，无法溯源到用户
        if not user_id:
            return {
                "answer": answer,
                "sources": sources,
                "conversation_id": "",
                "answer_confidence": assess_answer(answer, chunks),
            }

        # 首次提问（无 conversation_id）也要建会话——schema 契约：返回会话 ID 供前端续写
        conv = None
        if conversation_id:
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
            "answer_confidence": assess_answer(answer, chunks),
        }
