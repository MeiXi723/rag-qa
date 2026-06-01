import json
from app.config import TOP_K


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

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[来源{i+1}] 来自文档《{chunk['document_name']}》:\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        return f"""你是一个知识库问答助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实说"参考资料中未找到相关信息"。

参考资料：
{context}

用户问题：{question}

请给出清晰的回答，并在回答中引用来源编号。"""

    def ask(self, question: str, conversation_id: str | None = None) -> dict:
        chunks = self.retrieve(question)
        prompt = self._build_prompt(question, chunks)

        messages = [{"role": "user", "content": prompt}]
        answer = self.llm.chat(messages)

        sources = [
            {
                "document_name": c["document_name"],
                "chunk_text": c["text"][:200],
                "page": c["page"],
            }
            for c in chunks
        ]

        if not conversation_id or not self.db.get_conversation(conversation_id):
            conversation_id = self.db.create_conversation(question[:50])
        self.db.add_message(conversation_id, "user", question, None)
        self.db.add_message(conversation_id, "assistant", answer, json.dumps(sources, ensure_ascii=False))

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
        }
