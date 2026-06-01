import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import DocumentResponse
from app.services.document import parse_file, chunk_text, ALLOWED_EXTENSIONS
from app.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP

router = APIRouter(prefix="/api/documents", tags=["文档管理 / Document Management"])


@router.post("/upload", response_model=DocumentResponse,
              summary="上传文档 / Upload Document",
              description="上传 PDF/Word/TXT 文件，自动解析内容并建立索引。Upload a document, parse its content and build search index.")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext} / Unsupported file format. Supported: {ALLOWED_EXTENSIONS}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = parse_file(file_path)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    from app.main import get_llm_service, get_vector_store, get_db
    llm = get_llm_service()
    vs = get_vector_store()
    db = get_db()

    doc_id = db.insert_document(file.filename, ext.lstrip("."), os.path.getsize(file_path), len(chunks))

    for i, chunk in enumerate(chunks):
        embedding = llm.embed(chunk)
        vs.add(
            ids=[f"{doc_id}-chunk{i}"],
            texts=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "document_id": doc_id,
                "document_name": file.filename,
                "chunk_index": i,
                "page": None
            }]
        )

    return DocumentResponse(
        id=doc_id,
        filename=file.filename,
        file_type=ext.lstrip("."),
        file_size=os.path.getsize(file_path),
        chunk_count=len(chunks),
        created_at=db.get_document(doc_id)["created_at"]
    )


@router.get("", response_model=list[DocumentResponse],
             summary="文档列表 / List Documents",
             description="获取所有已上传的文档列表。Get all uploaded documents.")
async def list_documents():
    from app.main import get_db
    return get_db().list_documents()


@router.delete("/{doc_id}",
                summary="删除文档 / Delete Document",
                description="删除指定文档及其向量索引。Delete a document and its vector index.")
async def delete_document(doc_id: str):
    from app.main import get_db, get_vector_store
    db = get_db()
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在 / Document not found")
    vs = get_vector_store()
    vs.delete_by_document_id(doc_id)
    db.delete_document(doc_id)
    return {"detail": "已删除"}
