from fastapi import APIRouter, Depends, Request, status, HTTPException
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

from app.constants import MAX_QUERY_LENGTH
from app.config import settings

class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)

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
        
    if not chat.title:
        chat.title = content[:40]
        db.add(chat)
        
    import uuid
    job_id = uuid.uuid4().hex
    job = repo.create_job(job_id=job_id, user_id=user.user_id, goal=content)
    
    msg = Message(
        chat_id=chat.chat_id,
        role="user",
        content=content,
        job_id=job.job_id
    )
    db.add(msg)
    
    chat.updated_at = datetime.datetime.now(datetime.timezone.utc)
    db.commit()
    
    job_manager = getattr(request.app.state, "job_manager", None)
    if job_manager:
        job_manager.submit_job(job_id=job.job_id, user_id=user.user_id, goal=content)
        
    return SendMessageResponse(
        message_id=msg.message_id,
        job_id=job.job_id,
        chat_id=chat.chat_id,
        status="queued"
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
