# =============================================================================
# rerank.py — CrossEncoder 精排
# =============================================================================
# 移植自 afac2026-rag 的 _cross_encoder_rerank，降级策略一致：
# 1. sentence-transformers 未安装 / 模型加载失败 → 静默回退 RRF 排序
#    （afac 实测：bge-reranker-large 1.3GB 在 CPU 不可行，用 base 280MB）
# 2. 只精排融合后前 top_n 名——CPU 上全量打分不可承受
# 3. 精排分必须写回 score 字段，否则下游按 score 排序会把精排顺序完全抵消
#    （afac 外审 P1#2 的教训）
# =============================================================================

import logging
import threading

from app.config import ENABLE_RERANK, RERANKER_MODEL, TOP_N_RERANK

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_failed = False
_init_lock = threading.Lock()


def _get_cross_encoder(model_name: str = RERANKER_MODEL):
    """惰性加载 CrossEncoder；失败置 failed 标志后不再重试（每请求都试会拖垮响应）。"""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder_failed:
        return None
    if _cross_encoder is None:
        with _init_lock:
            if _cross_encoder is None and not _cross_encoder_failed:
                try:
                    from sentence_transformers import CrossEncoder
                    # max_length=384：query+passage 场景截断损失极小，CPU 提速 ~25%
                    _cross_encoder = CrossEncoder(model_name, max_length=384)
                except Exception as e:
                    # 常见原因：未装 sentence-transformers、模型下载失败、网络受限
                    logger.warning("CrossEncoder 加载失败，精排降级为 RRF 排序: %s", e)
                    _cross_encoder_failed = True
                    return None
    return _cross_encoder


def reset_reranker():
    """测试用：清空模型单例和失败标志。"""
    global _cross_encoder, _cross_encoder_failed
    with _init_lock:
        _cross_encoder = None
        _cross_encoder_failed = False


def cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    top_n: int = TOP_N_RERANK,
    model_name: str = RERANKER_MODEL,
) -> list[dict]:
    if not candidates or not ENABLE_RERANK:
        return candidates

    model = _get_cross_encoder(model_name)
    if model is None:
        return candidates

    subset = candidates[:top_n]
    pairs = [(query, c.get("text", "")) for c in subset]
    scores = model.predict(pairs)
    for chunk, score in zip(subset, scores):
        chunk["rerank_score"] = float(score)
        chunk["score"] = float(score)

    reranked = sorted(subset, key=lambda c: c["score"], reverse=True)
    # 精排范围外的候选垫底，保证不丢块
    return reranked + candidates[top_n:]
