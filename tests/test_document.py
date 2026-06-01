import os
from app.services.document import parse_file, chunk_text


def test_parse_txt():
    text = parse_file("tests/fixtures.txt")
    assert "测试文本文件" in text


def test_chunk_text_basic():
    text = "第一段测试文本。\n\n第二段测试文本。\n\n第三段测试文本。"
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 10


def test_chunk_text_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    assert chunks[0][-3:] in chunks[1]


def test_chunk_text_single_chunk():
    text = "短文本。"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "短文本。"


def test_parse_file_not_found():
    try:
        parse_file("nonexistent.txt")
        assert False, "应该抛出异常"
    except FileNotFoundError:
        pass
