import asyncio
import secrets
from app.db.database import SessionLocal
from app.db.models import User, ApiKey
from app.api.auth import get_api_key_hash

async def create_dev_user():
    db = SessionLocal()
    try:
        # Check if dev user exists
        dev_user = db.query(User).filter(User.email == "dev@example.com").first()
        if not dev_user:
            dev_user = User(
                email="dev@example.com"
            )
            db.add(dev_user)
            db.commit()
            db.refresh(dev_user)
            
        # Generate new key
        raw_key = f"ak_dev_{secrets.token_urlsafe(16)}"
        key_hash = get_api_key_hash(raw_key)
        
        api_key = ApiKey(
            user_id=dev_user.user_id,
            key_hash=key_hash
        )
        db.add(api_key)
        db.commit()
        
        print(f"Created dev API Key: {raw_key}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(create_dev_user())