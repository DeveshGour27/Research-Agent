from datetime import timedelta
import secrets
from fastapi import APIRouter, Depends, HTTPException, Response, status, Cookie
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import User, UserSession, _utc_now
from app.api.auth_utils import get_password_hash, verify_password, generate_verification_token, hash_verification_token
from app.services.email_service import get_email_service
from app.api.auth import get_current_web_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class VerifyRequest(BaseModel):
    token: str
    email: EmailStr

@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
    existing_user = db.query(User).filter(User.email == req.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    raw_token, hashed_token = generate_verification_token()
    
    user = User(
        email=req.email,
        password_hash=get_password_hash(req.password),
        email_verified=False,
        verification_token_hash=hashed_token,
        verification_token_expires_at=_utc_now() + timedelta(hours=24)
    )
    db.add(user)
    db.commit()
    
    email_service = get_email_service()
    email_service.send_verification_email(user.email, raw_token)
    
    return {"message": "Signup successful. Please check your email to verify."}


@router.post("/verify")
def verify_email(req: VerifyRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
        
    if user.email_verified:
        return {"message": "Email already verified"}
        
    if not user.verification_token_hash or not user.verification_token_expires_at:
        raise HTTPException(status_code=400, detail="Invalid verification token")
        
    import datetime
    exp = user.verification_token_expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=datetime.timezone.utc)
        
    if exp < _utc_now():
        raise HTTPException(status_code=400, detail="Verification token expired")
        
    hashed_provided_token = hash_verification_token(req.token)
    if not secrets.compare_digest(user.verification_token_hash, hashed_provided_token):
        raise HTTPException(status_code=400, detail="Invalid verification token")
        
    user.email_verified = True
    user.verification_token_hash = None
    user.verification_token_expires_at = None
    db.commit()
    
    return {"message": "Email verified successfully"}


@router.post("/login")
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
        
    session_id = secrets.token_urlsafe(64)
    user_session = UserSession(
        session_id=session_id,
        user_id=user.user_id,
        expires_at=_utc_now() + timedelta(days=7)
    )
    db.add(user_session)
    db.commit()
    
    from app.config import settings, Environment
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=settings.environment == Environment.PRODUCTION,
        samesite="lax",
        max_age=7 * 24 * 60 * 60
    )
    
    return {"message": "Login successful"}


@router.post("/logout")
def logout(
    response: Response,
    user: User = Depends(get_current_web_user),
    session_id: str | None = Cookie(None),
    db: Session = Depends(get_db)
):
    if session_id:
        user_session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        if user_session:
            user_session.revoked_at = _utc_now()
            db.commit()
            
    response.delete_cookie("session_id")
    return {"message": "Logged out successfully"}

