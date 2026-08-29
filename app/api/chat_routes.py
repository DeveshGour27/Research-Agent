"""Chat routes for Phase 5 Authorization Testing."""
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import User
from app.db.repository import SQLJobRepository
from app.api.auth import get_current_user

router = APIRouter()

def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}}
    )

@router.get("/api/v1/chats/{chat_id}", tags=["Chats"])
def get_chat(
    chat_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    chat = repo.get_conversation(chat_id=chat_id, user_id=user.user_id)
    if not chat:
        return _error_response("CHAT_NOT_FOUND", "Chat not found.", status.HTTP_404_NOT_FOUND)
    
    return {
        "chat_id": chat.chat_id,
        "title": chat.title,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat()
    }

@router.get("/api/v1/messages/{message_id}", tags=["Chats"])
def get_message(
    message_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    repo = SQLJobRepository(db)
    msg = repo.get_message(message_id=message_id, user_id=user.user_id)
    if not msg:
        return _error_response("MESSAGE_NOT_FOUND", "Message not found.", status.HTTP_404_NOT_FOUND)
    
    return {
        "message_id": msg.message_id,
        "chat_id": msg.chat_id,
        "role": msg.role,
        "content": msg.content,
        "job_id": msg.job_id,
        "created_at": msg.created_at.isoformat()
    }
