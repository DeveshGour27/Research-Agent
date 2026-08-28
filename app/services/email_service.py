import logging
from typing import Protocol
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class EmailService(Protocol):
    def send_verification_email(self, email: str, token: str) -> None:
        ...

class DevEmailService(EmailService):
    """Development email service that logs the verification link."""
    def send_verification_email(self, email: str, token: str) -> None:
        query = urlencode({"token": token, "email": email})
        verification_link = f"http://localhost:3000/verify-email?{query}"
        
        logger.info("--- DEVELOPMENT EMAIL SENT ---")
        logger.info(f"To: {email}")
        logger.info("Subject: Verify your account")
        logger.info(f"Verification Link: {verification_link}")
        logger.info("------------------------------")

def get_email_service() -> EmailService:
    return DevEmailService()

