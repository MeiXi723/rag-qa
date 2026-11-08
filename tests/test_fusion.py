from app.services.retrieval.fusion import rrf_fuse


def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "text": text, "document_name": "a.pdf", "chunk_index": 0, "page": None}


def test_vector_only_keeps_order():
    fused = rrf_fuse([_chunk("a", "A"), _chunk("b", "B")], [])
    assert [c["id"] for c in fused] == ["a", "b"]
    assert fused[0]["score"] > fused[1]["score"]
    assert fused[0]["source"] == "vector"


def test_bm25_only_keeps_order():
    fused = rrf_fuse([], [_chunk("x", "X"), _chunk("y", "Y")])
    assert [c["id"] for c in fused] == ["x", "y"]
    assert fused[0]["source"] == "bm25"


def test_dual_hit_aggregates_to_top():
    # b 在两路都命中：分数累加后必须排第一（RRF 的核心行为）
    fused = rrf_fuse(
        [_chunk("a", "A"), _chunk("b", "B")],
        [_chunk("b", "B"), _chunk("c", "C")],
    )
    assert fused[0]["id"] == "b"
    assert fused[0]["source"] == "bm25+vector"
    # b: 向量路 rank1 → 1/(k+2)，bm25 路 rank0 → 1/(k+1)
    assert fused[0]["score"] == 1 / (10 + 2) + 1 / (10 + 1)


def test_rank_position_determines_score():
    # rank 从 0 起：顶分 1/(k+1)
    fused = rrf_fuse([_chunk("a", "A")], [], k=10)
    assert fused[0]["score"] == 1 / 11
    fused_k60 = rrf_fuse([_chunk("a", "A")], [], k=60)
    assert fused_k60[0]["score"] == 1 / 61


def test_fused_chunks_keep_original_shape():
    # 融合输出必须保留 id/text/document_name 等字段——下游 ask()/sources 依赖它们
    fused = rrf_fuse([_chunk("a", "正文内容")], [])
    assert fused[0]["text"] == "正文内容"
    assert fused[0]["document_name"] == "a.pdf"
