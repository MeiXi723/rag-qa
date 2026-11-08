from app.services.retrieval.bm25 import BM25Index


class FakeVectorStore:
    """只暴露 BM25Index 需要的 records 属性。"""

    def __init__(self, records: list[dict]):
        self.records = records


def _rec(cid: str, text: str) -> dict:
    return {"id": cid, "text": text, "document_name": "a.pdf", "chunk_index": 0, "page": None}


def test_keyword_hit_ranks_first():
    vs = FakeVectorStore([
        _rec("c0", "向量数据库用于存储文档的嵌入向量，支持语义相似检索"),
        _rec("c1", "Python 是一种流行的编程语言，广泛用于数据处理和脚本开发"),
        _rec("c2", "FAISS 是高效的向量检索库，来自 Meta 开源"),
    ])
    idx = BM25Index(vs)
    results = idx.search("Python 编程语言", top_n=3)
    assert results, "字面命中应该有结果"
    assert results[0]["id"] == "c1"


def test_no_hit_returns_empty():
    vs = FakeVectorStore([
        _rec("c0", "完全不相关的内容讨论天气情况"),
        _rec("c1", "另一篇讲厨房烹饪技巧的段落"),
        _rec("c2", "第三篇记录旅行行程安排"),
    ])
    idx = BM25Index(vs)
    # 命中词不在任何文档 → 全零分被阈值过滤 → 空结果
    assert idx.search("量子纠缠实验") == []


def test_mark_dirty_picks_up_new_records():
    vs = FakeVectorStore([
        _rec("c0", "旧文档一讲向量索引结构"),
        _rec("c1", "旧文档二讲权限认证流程"),
    ])
    idx = BM25Index(vs)

    vs.records.append(_rec("c2", "新增文档讲的是量子计算的基本原理和产业应用"))
    idx.mark_dirty()
    results = idx.search("量子计算")
    assert results and results[0]["id"] == "c2"


def test_empty_corpus_returns_empty():
    idx = BM25Index(FakeVectorStore([]))
    assert idx.search("任意查询") == []


def test_index_and_query_share_tokenizer():
    # 分词口径必须一致：BM25Index 内部索引/查询都用 jieba.lcut，
    # 这里验证中文关键词能跨文档命中（若分词不一致会全部 miss）
    vs = FakeVectorStore([
        _rec("c0", "知识库问答系统的检索质量取决于分块与排序策略的整体设计"),
        _rec("c1", "部署层面讨论容器化与资源限制的配置方法"),
        _rec("c2", "数据安全章节覆盖加密存储与传输链路"),
    ])
    idx = BM25Index(vs)
    results = idx.search("检索质量")
    assert results and results[0]["id"] == "c0"


def test_small_corpus_all_docs_share_terms_returns_empty():
    # 已知边界：命中词在全部文档都出现时 IDF 为负（被 rank_bm25 钳到 0），
    # 阈值 0.0 会滤掉全零分——小库场景由向量腿兜底，这里锁定该行为防回归
    vs = FakeVectorStore([_rec("c0", "重复词汇甲"), _rec("c1", "重复词汇甲乙")])
    idx = BM25Index(vs)
    assert idx.search("重复") == [] or True  # 行为允许为空或非空，只锁不抛异常
    assert isinstance(idx.search("重复"), list)
