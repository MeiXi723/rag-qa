# =============================================================================
# bm25.py — BM25 稀疏检索腿（关键词精确命中）
# =============================================================================
# 向量检索擅长语义相似，但对专有名词、编号、错误码这类「必须字面匹配」的
# 查询经常排不对——BM25 补的就是这条腿。
#
# 为什么不落盘：afac 是离线管道（索引建一次用一次），本项目是在线服务，
# 文档随时增删。语料直接读 VectorStore.records（它已持久化在 JSON 里），
# 增删文档后 mark_dirty，下次检索前惰性重建。中小知识库重建是毫秒级。
# =============================================================================

import threading

import jieba
from rank_bm25 import BM25Okapi

from app.config import BM25_SCORE_THRESHOLD, RETRIEVAL_CANDIDATES


class BM25Index:
    """内存态 BM25 索引，语料与 VectorStore.records 保持同步。"""

    def __init__(self, vector_store):
        self._vs = vector_store
        self._bm25: BM25Okapi | None = None
        self._records: list[dict] = []
        self._dirty = True
        # 惰性重建用锁保护：FastAPI 多线程并发检索时避免重复重建
        self._lock = threading.Lock()

    def mark_dirty(self):
        """文档增删后调用；下次 search 前重建。"""
        self._dirty = True

    def _ensure_fresh(self):
        with self._lock:
            if not self._dirty:
                return
            # 语料快照：upload/delete 线程可能在检索中途重绑 VectorStore.records
            # （chroma.py delete 是整体重绑），索引与下标必须来自同一份快照，
            # 否则并发下错位/越界
            self._records = list(self._vs.records)
            texts = [r.get("text", "") for r in self._records]
            # 分词器必须索引与查询一致，否则 BM25 分数没有可比性
            tokenized = [jieba.lcut(t) for t in texts]
            # 全空语料（records 有条目但文本全空）会让 BM25Okapi 除零，必须挡住
            self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None
            self._dirty = False

    def search(self, query: str, top_n: int = RETRIEVAL_CANDIDATES) -> list[dict]:
        self._ensure_fresh()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(jieba.lcut(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict] = []
        for i in order:
            # 分数降序排列，低于阈值后全是更低分，直接停
            if scores[i] <= BM25_SCORE_THRESHOLD or len(results) >= top_n:
                break
            record = self._records[i]
            results.append({
                "id": record["id"],
                "text": record.get("text", ""),
                "document_name": record.get("document_name", ""),
                "chunk_index": record.get("chunk_index", 0),
                "page": record.get("page"),
                "score": float(scores[i]),
            })
        return results
