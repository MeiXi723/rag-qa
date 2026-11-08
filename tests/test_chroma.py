import tempfile
import pytest
from app.db.chroma import VectorStore


@pytest.fixture
def store():
    path = tempfile.mkdtemp()
    vs = VectorStore(path, "test_collection")
    yield vs
    vs.close()


def test_add_and_search(store):
    store.add(
        ids=["doc1-chunk0", "doc1-chunk1"],
        texts=["Python是一种编程语言。", "ChromaDB用于语义搜索。"],
        embeddings=[[0.1]*10, [0.9]*10],
        metadatas=[
            {"document_id": "doc1", "document_name": "test.txt", "chunk_index": 0, "page": 1},
            {"document_id": "doc1", "document_name": "test.txt", "chunk_index": 1, "page": 1},
        ]
    )
    results = store.search(query_embedding=[0.1]*10, n_results=1)
    assert len(results["ids"][0]) == 1
    # 向量相近，应匹配到 chunk0
    assert "doc1-chunk0" in results["ids"][0]


def test_search_returns_metadata(store):
    store.add(
        ids=["chunk-0"],
        texts=["内容文本。"],
        embeddings=[[0.5]*10],
        metadatas=[{"document_id": "d1", "document_name": "a.pdf", "chunk_index": 0, "page": 5}]
    )
    results = store.search(query_embedding=[0.5]*10, n_results=1)
    meta = results["metadatas"][0][0]
    assert meta["document_name"] == "a.pdf"
    assert meta["page"] == 5


def test_delete_by_document_id(store):
    store.add(
        ids=["c1", "c2"],
        texts=["文本A", "文本B"],
        embeddings=[[0.1]*10, [0.2]*10],
        metadatas=[
            {"document_id": "d1", "document_name": "a", "chunk_index": 0, "page": 1},
            {"document_id": "d1", "document_name": "a", "chunk_index": 1, "page": 1},
        ]
    )
    store.delete_by_document_id("d1")
    results = store.search(query_embedding=[0.1]*10, n_results=5)
    assert len(results["ids"][0]) == 0
