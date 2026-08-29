from datetime import timedelta
import secrets
import httpx
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, HTTPException, Response, status, Cookie, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import select, or_

from app.db.database import get_db
from app.db.models import User, UserSession, OAuthIdentity, _utc_now
from app.api.auth_utils import get_password_hash, verify_password, generate_verification_token, hash_verification_token
from app.services.email_service import get_email_service
from app.api.auth import get_current_web_user
from app.config import settings, Environment

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str

class LoginRequest(BaseModel):
    identifier: str
    password: str

class VerifyRequest(BaseModel):
    token: str
    email: EmailStr

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    email: EmailStr
    password: str
    
class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
    existing_user = db.query(User).filter(or_(User.email == req.email, User.username == req.username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
        
    raw_token, hashed_token = generate_verification_token()
    
    user = User(
        username=req.username,
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


def _create_session(user_id: str, db: Session, response: Response):
    session_id = secrets.token_urlsafe(64)
    user_session = UserSession(
        session_id=session_id,
        user_id=user_id,
        expires_at=_utc_now() + timedelta(days=7)
    )
    db.add(user_session)
    db.commit()
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=settings.environment == Environment.PRODUCTION,
        samesite="lax",
        max_age=7 * 24 * 60 * 60,
        path="/"
    )

@router.post("/login")
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(or_(User.email == req.identifier, User.username == req.identifier)).first()
    
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
        
    _create_session(user.user_id, db, response)
    
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
            
    response.delete_cookie("session_id", path="/")
    return {"message": "Logged out successfully"}

@router.get("/me")
def get_me(user: User = Depends(get_current_web_user)):
    return {"email": user.email, "username": user.username, "user_id": user.user_id}

@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if user:
        raw_token, hashed_token = generate_verification_token()
        user.reset_token_hash = hashed_token
        user.reset_token_expires_at = _utc_now() + timedelta(hours=1)
        db.commit()
        
        email_service = get_email_service()
        email_service.send_password_reset_email(user.email, raw_token)
        
    return {"message": "If that email exists, a password reset link has been sent."}

@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.reset_token_hash or not user.reset_token_expires_at:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
    import datetime
    exp = user.reset_token_expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=datetime.timezone.utc)
        
    if exp < _utc_now():
        raise HTTPException(status_code=400, detail="Reset token expired")
        
    hashed_provided_token = hash_verification_token(req.token)
    if not secrets.compare_digest(user.reset_token_hash, hashed_provided_token):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
    user.password_hash = get_password_hash(req.password)
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    db.commit()
    return {"message": "Password reset successfully"}

@router.post("/change-password")
def change_password(req: ChangePasswordRequest, user: User = Depends(get_current_web_user), db: Session = Depends(get_db)):
    if not user.password_hash:
        raise HTTPException(status_code=400, detail="You do not have a password set")
        
    if not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect old password")
        
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
    user.password_hash = get_password_hash(req.new_password)
    db.commit()
    return {"message": "Password changed successfully"}

@router.post("/delete-account")
def delete_account(user: User = Depends(get_current_web_user), db: Session = Depends(get_db)):
    db.delete(user)
    db.commit()
    return {"message": "Account deleted successfully"}

# ------------------------------------------------------------------ #
# Google OAuth
# ------------------------------------------------------------------ #
@router.get("/google/login")
def google_login():
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
        
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "consent"
    })
    return {"url": url}

@router.get("/google/callback")
async def google_callback(code: str, request: Request, db: Session = Depends(get_db)):
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
        
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://oauth2.googleapis.com/token", data={
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.google_redirect_uri,
        })
        if not resp.is_success:
            raise HTTPException(status_code=400, detail="Failed to exchange Google code")
            
        data = resp.json()
        access_token = data.get("access_token")
        
        user_resp = await client.get("https://www.googleapis.com/oauth2/v2/userinfo", headers={"Authorization": f"Bearer {access_token}"})
        if not user_resp.is_success:
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")
            
        userinfo = user_resp.json()
        google_sub = userinfo.get("id")
        email = userinfo.get("email")
        
        if not google_sub or not email:
            raise HTTPException(status_code=400, detail="Invalid Google profile data")
            
    # Check if OAuth identity exists
    oauth_id = db.query(OAuthIdentity).filter(
        OAuthIdentity.provider == "google",
        OAuthIdentity.provider_subject == google_sub
    ).first()
    
    if oauth_id:
        user = oauth_id.user
    else:
        user = db.query(User).filter(User.email == email).first()
        if user:
            return RedirectResponse(url=f"{settings.app_base_url}/login?error=email_exists")
            
        base_username = email.split("@")[0]
        username = base_username
        suffix = 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base_username}{suffix}"
            suffix += 1
            
        user = User(
            username=username,
            email=email,
            password_hash=None,
            email_verified=True
        )
        db.add(user)
        db.flush()
        
        new_oauth = OAuthIdentity(
            user_id=user.user_id,
            provider="google",
            provider_subject=google_sub
        )
        db.add(new_oauth)
        
    db.commit()
    
    response = RedirectResponse(url=f"{settings.app_base_url}/")
    _create_session(user.user_id, db, response)
    
    return response
