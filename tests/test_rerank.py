from app.services.retrieval import rerank


class FakeCrossEncoder:
    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "text": text, "document_name": "a.pdf", "chunk_index": 0, "page": None}


def test_rerank_reorders_by_model_score(monkeypatch):
    chunks = [_chunk("a", "低相关文本"), _chunk("b", "高相关文本内容")]
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda name="": FakeCrossEncoder([-1.0, 5.0]))

    out = rerank.cross_encoder_rerank("查询", chunks)
    assert out[0]["id"] == "b"
    # 精排分必须写回 score（afac 外审 P1#2 教训），否则下游排序会抵消精排
    assert out[0]["rerank_score"] == 5.0
    assert out[0]["score"] == 5.0


def test_rerank_graceful_degradation(monkeypatch):
    # 模型加载失败 → 原序保留，链路不断
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda name="": None)
    chunks = [_chunk("a", "甲"), _chunk("b", "乙")]
    out = rerank.cross_encoder_rerank("查询", chunks)
    assert [c["id"] for c in out] == ["a", "b"]


def test_rerank_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(rerank, "ENABLE_RERANK", False)
    chunks = [_chunk("a", "甲")]
    assert rerank.cross_encoder_rerank("查询", chunks) == chunks


def test_out_of_scope_candidates_are_kept(monkeypatch):
    # 只精排前 top_n 名，范围外的候选垫底不丢块
    chunks = [_chunk(str(i), f"候选文本内容编号{i}") for i in range(30)]
    scores = [float(-i) for i in range(30)]
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda name="": FakeCrossEncoder(scores))

    out = rerank.cross_encoder_rerank("查询", chunks, top_n=25)
    assert len(out) == 30
    assert out[0]["id"] == "0"
    assert out[-1]["id"] == "29"


def test_empty_candidates_passthrough():
    assert rerank.cross_encoder_rerank("查询", []) == []
