from app.config import CHROMA_PERSIST_DIR, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from app.db.chroma import VectorStore
from app.services.llm import LLMService

_llm_service: LLMService | None = None
_vector_store: VectorStore | None = None
_bm25_index = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _llm_service


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(CHROMA_PERSIST_DIR)
    return _vector_store


def get_bm25_index():
    global _bm25_index
    if _bm25_index is None:
        from app.services.retrieval.bm25 import BM25Index
        _bm25_index = BM25Index(get_vector_store())
    return _bm25_index
