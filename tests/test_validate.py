from app.services.retrieval.validate import assess_answer

# 3 个可用来源块
CHUNKS = [{"id": f"c{i}", "text": "内容", "document_name": "a.pdf"} for i in range(3)]


def test_valid_citation_is_high():
    result = assess_answer("根据 [来源1] 和 [来源3] 可知……", CHUNKS)
    assert result["confidence"] == "high"
    assert result["cited_sources"] == [1, 3]


def test_insufficient_evidence_is_low():
    result = assess_answer("参考资料中未找到相关信息。", CHUNKS)
    assert result["confidence"] == "low"
    assert result["cited_sources"] == []


def test_hallucinated_citation_is_low():
    # 引用了不存在的来源编号 → 幻觉引用
    assert assess_answer("根据 [来源9] 的记载……", CHUNKS)["confidence"] == "low"


def test_no_citation_is_medium():
    assert assess_answer("答案是 42。", CHUNKS)["confidence"] == "medium"


def test_empty_answer_is_low():
    assert assess_answer("", CHUNKS)["confidence"] == "low"


def test_duplicate_citations_deduped():
    result = assess_answer("[来源2] 和 [来源2] 都提到……", CHUNKS)
    assert result["cited_sources"] == [2]
