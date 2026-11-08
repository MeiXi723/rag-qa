# =============================================================================
# validate.py — 生成答案的置信度评估
# =============================================================================
# afac2026-rag 的 Validator 是「双模型交叉验证」（deepseek vs qwen 各答一次比对），
# 本项目只有单一 LLM 服务，直接搬会引入第二个供应商依赖。
# 这里改造成单模型自检：从答案文本本身可验证的三个信号评估置信度——
# 引用编号有效性、证据不足话术、引用覆盖度。
# =============================================================================

import re

_INSUFFICIENT_PATTERNS = ("未找到相关信息", "未找到相关内容", "参考资料中未", "无法回答")

# 匹配 [来源1] / [来源1][来源3] / 「根据 [来源2]」等引用
_CITATION_RE = re.compile(r"\[来源(\d+)\]")


def assess_answer(answer: str, chunks: list[dict]) -> dict:
    """
    评估 LLM 答案的置信度。返回 {"confidence": high|medium|low, "cited_sources": [...]}。

    判定顺序（从严到宽）：
    - 答案含「证据不足」类话术 → low
    - 引用了不存在的来源编号 → low（模型幻觉引用）
    - 至少引用一个有效来源 → high
    - 有答案但零引用 → medium（可能正确但不可溯源）
    """
    if not answer:
        return {"confidence": "low", "cited_sources": []}

    if any(pattern in answer for pattern in _INSUFFICIENT_PATTERNS):
        return {"confidence": "low", "cited_sources": []}

    cited = [int(n) for n in _CITATION_RE.findall(answer)]
    cited = sorted(set(cited))
    if cited and max(cited) > len(chunks):
        return {"confidence": "low", "cited_sources": cited}

    if cited:
        return {"confidence": "high", "cited_sources": cited}

    return {"confidence": "medium", "cited_sources": []}
