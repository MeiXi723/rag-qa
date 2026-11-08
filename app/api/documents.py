import os
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from app.db.database import get_db
from app.services import get_llm_service, get_vector_store, get_bm25_index
from app.models.db_models import Document, User
from app.models.schemas import DocumentResponse
from app.services.document import ALLOWED_EXTENSIONS, chunk_text, parse_file

router = APIRouter(prefix="/api/documents", tags=["文档管理 / Document Management"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}. Supported: {ALLOWED_EXTENSIONS}"
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = parse_file(file_path)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    llm = get_llm_service()
    vs = get_vector_store()

    doc = Document(
        filename=file.filename,
        file_type=ext.lstrip("."),
        file_size=os.path.getsize(file_path),
        chunk_count=len(chunks),
        user_id=current_user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    for i, chunk in enumerate(chunks):
        embedding = llm.embed(chunk)
        vs.add(
            ids=[f"{doc.id}-chunk{i}"],
            texts=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "document_id": doc.id,
                "document_name": file.filename,
                "chunk_index": i,
                "page": None
            }]
        )

    # BM25 语料来自 VectorStore.records，增删文档后必须标脏
    get_bm25_index().mark_dirty()

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    docs = db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.created_at.desc()).all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            file_type=d.file_type,
            file_size=d.file_size,
            chunk_count=d.chunk_count,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(
        Document.id == doc_id, Document.user_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    vs = get_vector_store()
    vs.delete_by_document_id(doc_id)
    db.delete(doc)
    db.commit()

    # 同步标记 BM25 语料失效
    get_bm25_index().mark_dirty()

    return {"detail": "已删除"}
