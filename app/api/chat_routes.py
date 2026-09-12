from fastapi import APIRouter, Depends, Request, status, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import List, Optional
import datetime

from app.db.database import get_db
from app.db.models import User, Message, Conversation, Job
from app.db.repository import SQLJobRepository
from app.api.auth import get_current_user

try:
    import fitz
except ImportError:
    fitz = None

router = APIRouter()

def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}}
    )

class ChatMetadataResponse(BaseModel):
    chat_id: str
    title: Optional[str]
    created_at: str
    updated_at: str

class MessageMetadataResponse(BaseModel):
    message_id: str
    chat_id: str
    role: str
    content: str
    job_id: Optional[str]
    created_at: str

class ChatDetailResponse(ChatMetadataResponse):
    messages: List[MessageMetadataResponse]

class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    doc_id: str
    chunks: int
    page_count: int
    message: str

from app.constants import MAX_QUERY_LENGTH
from app.config import settings

class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    attached_documents: Optional[List[str]] = Field(default_factory=list, description="Optional list of attached document filenames")

class SendMessageResponse(BaseModel):
    message_id: str
    job_id: str
    chat_id: str
    status: str

@router.get("/api/v1/chats", response_model=List[ChatMetadataResponse], tags=["Chats"])
def list_chats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    chats = repo.list_conversations(user_id=user.user_id)
    return [
        ChatMetadataResponse(
            chat_id=chat.chat_id,
            title=chat.title,
            created_at=chat.created_at.isoformat(),
            updated_at=chat.updated_at.isoformat()
        )
        for chat in chats
    ]

@router.post("/api/v1/chats", response_model=ChatMetadataResponse, tags=["Chats"])
def create_chat(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    chat = repo.create_conversation(user_id=user.user_id)
    return ChatMetadataResponse(
        chat_id=chat.chat_id,
        title=chat.title,
        created_at=chat.created_at.isoformat(),
        updated_at=chat.updated_at.isoformat()
    )

@router.get("/api/v1/chats/{chat_id}", response_model=ChatDetailResponse, tags=["Chats"])
def get_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    chat = repo.get_conversation(chat_id=chat_id, user_id=user.user_id)
    if not chat:
        return _error_response("CHAT_NOT_FOUND", "Chat not found.", status.HTTP_404_NOT_FOUND)
    
    msgs = [
        MessageMetadataResponse(
            message_id=m.message_id,
            chat_id=m.chat_id,
            role=m.role,
            content=m.content,
            job_id=m.job_id,
            created_at=m.created_at.isoformat()
        ) for m in chat.messages
    ]
    return ChatDetailResponse(
        chat_id=chat.chat_id,
        title=chat.title,
        created_at=chat.created_at.isoformat(),
        updated_at=chat.updated_at.isoformat(),
        messages=msgs
    )

@router.delete("/api/v1/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Chats"])
def delete_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    deleted = repo.delete_conversation(chat_id=chat_id, user_id=user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return None

@router.post("/api/v1/chats/{chat_id}/messages", response_model=SendMessageResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Chats"])
def send_message(
    chat_id: str,
    body: SendMessageRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message content cannot be empty or whitespace.")

    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter:
        allowed, retry_after = rate_limiter.is_allowed(f"user_chat:{user.user_id}")
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )

    repo = SQLJobRepository(db)
    active_jobs = repo.count_active_jobs(user_id=user.user_id)
    if active_jobs >= settings.max_concurrent_jobs_per_user:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Concurrent job limit exceeded ({settings.max_concurrent_jobs_per_user}). Please wait for active jobs to finish."
        )

    chat = repo.get_conversation(chat_id=chat_id, user_id=user.user_id)
    if not chat:
        return _error_response("CHAT_NOT_FOUND", "Chat not found.", status.HTTP_404_NOT_FOUND)
        
    effective_content = content
    if body.attached_documents:
        docs_str = ", ".join(body.attached_documents)
        if not effective_content.startswith("[Attached Document:"):
            effective_content = f"[Attached Document: {docs_str}]\n\n{content}"

    if not chat.title:
        chat.title = content[:40]
        db.add(chat)
        
    import uuid
    job_id = uuid.uuid4().hex
    job = repo.create_job(job_id=job_id, user_id=user.user_id, goal=effective_content)
    
    msg = Message(
        chat_id=chat.chat_id,
        role="user",
        content=effective_content,
        job_id=job.job_id
    )
    db.add(msg)
    
    chat.updated_at = datetime.datetime.now(datetime.timezone.utc)
    db.commit()
    
    job_manager = getattr(request.app.state, "job_manager", None)
    if job_manager:
        job_manager.submit_job(job_id=job.job_id, user_id=user.user_id, goal=effective_content)
        
    return SendMessageResponse(
        message_id=msg.message_id,
        job_id=job.job_id,
        chat_id=chat.chat_id,
        status="queued"
    )

@router.post(
    "/api/v1/chats/{chat_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Chats", "RAG"]
)
async def upload_chat_document(
    chat_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    chat = repo.get_conversation(chat_id=chat_id, user_id=user.user_id)
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")

    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported for RAG upload."
        )

    # Read and validate file size (20 MB limit)
    MAX_FILE_SIZE = 20 * 1024 * 1024
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds 20MB limit.")

    # Extract text using PyMuPDF
    if fitz is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyMuPDF is not installed on the server."
        )

    try:
        pdf_doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(pdf_doc)
        pages_text = []
        for p in pdf_doc:
            text = p.get_text()
            if text:
                pages_text.append(text)
        extracted_text = "\n\n".join(pages_text).strip()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse PDF document: {exc}")

    if not extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from the PDF. It may be scanned images or password protected."
        )

    # Index into RAG Retriever
    import uuid
    from app.retriever import Retriever
    from rag.documents import Document

    doc_id = f"{chat_id}_{uuid.uuid4().hex[:8]}"
    rag_doc = Document(
        id=doc_id,
        content=extracted_text,
        source=filename,
        metadata={
            "filename": filename,
            "chat_id": chat_id,
            "user_id": user.user_id,
            "page_count": str(page_count),
        }
    )

    retriever = Retriever()
    retriever.index_documents([rag_doc], user_id=user.user_id)
    chunks_indexed = len(retriever.chunk_index) if hasattr(retriever, "chunk_index") else 1

    return DocumentUploadResponse(
        status="indexed",
        filename=filename,
        doc_id=doc_id,
        chunks=chunks_indexed,
        page_count=page_count,
        message=f"Successfully indexed '{filename}' ({page_count} pages) for RAG."
    )

@router.get("/api/v1/messages/{message_id}", response_model=MessageMetadataResponse, tags=["Chats"])
def get_message(
    message_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    msg = repo.get_message(message_id=message_id, user_id=user.user_id)
    if not msg:
        return _error_response("MESSAGE_NOT_FOUND", "Message not found.", status.HTTP_404_NOT_FOUND)
    
    return MessageMetadataResponse(
        message_id=msg.message_id,
        chat_id=msg.chat_id,
        role=msg.role,
        content=msg.content,
        job_id=msg.job_id,
        created_at=msg.created_at.isoformat()
    )
