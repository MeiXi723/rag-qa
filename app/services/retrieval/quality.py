# =============================================================================
# quality.py — 检索结果质量过滤
# =============================================================================
# 移植自 afac2026-rag 的 eval_norm.norm + retriever._quality_filter。
# norm_text 是全链路共用的文本归一化口径：去重、比较都必须用它，
# 否则全角/半角、中文标点的差异会让相同文本被当成不同块。
# =============================================================================

import unicodedata
import re

from app.config import MIN_CHUNK_LENGTH

# 中文标点 → 英文标点映射（NFKC 不覆盖这些）
_PUNCT_MAP = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "（": "(", "）": ")", "「": "\"", "」": "\"", "《": "<", "》": ">",
    "、": ",", "“": "\"", "”": "\"", "‘": "'", "’": "'",
})


def norm_text(text: str) -> str:
    """文本归一化：NFKC 全角→半角 + 中文标点映射 + 去全部空白。幂等。"""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP)
    return re.sub(r"\s+", "", normalized)


def quality_filter(chunks: list[dict], min_length: int = MIN_CHUNK_LENGTH) -> list[dict]:
    """
    对融合后的候选块做去重和碎块过滤。

    - 归一化哈希去重：双路检索常召回同一文本，归一化能抓住全角/半角变体
    - min_length：丢弃过短的碎片块（这类块 embedding 不可靠，喂给 LLM 只是噪声）
    """
    seen: set[int] = set()
    kept: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if text and len(text) < min_length:
            continue
        key = hash(norm_text(text))
        if key in seen:
            continue
        seen.add(key)
        kept.append(chunk)
    return kept
