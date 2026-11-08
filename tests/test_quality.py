from app.services.retrieval.quality import norm_text, quality_filter


def test_norm_is_idempotent():
    s = "ＡＢＣ，测试。"
    assert norm_text(norm_text(s)) == norm_text(s)


def test_norm_fullwidth_and_punctuation():
    # NFKC 只做全角→半角，不改变大小写（afac 口径保持一致）
    assert norm_text("ＡＢＣ，。！") == "ABC,.!"


def test_norm_strips_all_whitespace():
    assert norm_text("a b\tc\n") == "abc"


def test_quality_dedup_catches_fullwidth_variant():
    chunks = [
        {"id": "a", "text": "Python 编程。"},
        {"id": "b", "text": "Ｐｙｔｈｏｎ 编程。"},
    ]
    assert [c["id"] for c in quality_filter(chunks, min_length=5)] == ["a"]


def test_quality_min_length_filter():
    chunks = [
        {"id": "a", "text": "短"},
        {"id": "b", "text": "这是足够长的文本内容，不会被过滤"},
    ]
    assert [c["id"] for c in quality_filter(chunks, min_length=5)] == ["b"]


def test_quality_preserves_order():
    chunks = [{"id": str(i), "text": f"完全不重复的文本内容{i}"} for i in range(5)]
    assert [c["id"] for c in quality_filter(chunks, min_length=5)] == ["0", "1", "2", "3", "4"]
