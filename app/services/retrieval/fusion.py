# =============================================================================
# fusion.py — 双路检索结果 RRF 融合
# =============================================================================
# 移植自 afac2026-rag 的 _rrf_fusion。
# 核心教训（afac 实测）：BM25 原始分 [0,~30] 和余弦相似度 [0,1] 量纲完全不同，
# 直接加权求和会被 BM25 劫持。RRF 只看排名不看分值：
#   score = 1 / (k + rank + 1)，双路都命中的块分数累加天然置顶。
# =============================================================================

from app.config import RRF_K


def rrf_fuse(vector_chunks: list[dict], bm25_chunks: list[dict], k: int = RRF_K) -> list[dict]:
    """
    对两路检索结果做 Reciprocal Rank Fusion。

    两路结果都必须是「相关性从高到低」排列的块列表（shape 一致：
    id/text/document_name/chunk_index/page），输出追加 score 和 source 字段。
    """
    combined: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_chunks):
        cid = chunk["id"]
        score = 1.0 / (k + rank + 1)
        if cid in combined:
            combined[cid]["score"] += score
            combined[cid]["sources"].add("vector")
        else:
            combined[cid] = {**chunk, "score": score, "sources": {"vector"}}

    for rank, chunk in enumerate(bm25_chunks):
        cid = chunk["id"]
        score = 1.0 / (k + rank + 1)
        if cid in combined:
            combined[cid]["score"] += score
            combined[cid]["sources"].add("bm25")
        else:
            combined[cid] = {**chunk, "score": score, "sources": {"bm25"}}

    fused = sorted(combined.values(), key=lambda c: c["score"], reverse=True)
    for chunk in fused:
        chunk["source"] = "+".join(sorted(chunk.pop("sources")))
    return fused
