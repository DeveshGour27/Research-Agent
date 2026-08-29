import logging
from typing import Protocol
from urllib.parse import urlencode
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import settings, Environment

logger = logging.getLogger(__name__)

class EmailService(Protocol):
    def send_verification_email(self, email: str, token: str) -> None:
        ...
        
    def send_password_reset_email(self, email: str, token: str) -> None:
        ...

class SMTPEmailService(EmailService):
    """Production email service using SMTP."""
    def _send_email(self, to_email: str, subject: str, body: str) -> None:
        msg = MIMEMultipart()
        msg['From'] = settings.email_from
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(msg)
            logger.info(f"Email sent successfully to {to_email}")
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")

    def send_verification_email(self, email: str, token: str) -> None:
        query = urlencode({"token": token, "email": email})
        verification_link = f"{settings.app_base_url}/verify-email?{query}"
        
        html_body = f"""
        <html>
            <body>
                <h2>Verify your email</h2>
                <p>Please click the link below to verify your email address:</p>
                <p><a href="{verification_link}">{verification_link}</a></p>
            </body>
        </html>
        """
        self._send_email(email, "Verify your account", html_body)
        
    def send_password_reset_email(self, email: str, token: str) -> None:
        query = urlencode({"token": token, "email": email})
        reset_link = f"{settings.app_base_url}/reset-password?{query}"
        
        html_body = f"""
        <html>
            <body>
                <h2>Reset your password</h2>
                <p>Please click the link below to reset your password:</p>
                <p><a href="{reset_link}">{reset_link}</a></p>
            </body>
        </html>
        """
        self._send_email(email, "Reset your password", html_body)

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
        
    def send_password_reset_email(self, email: str, token: str) -> None:
        query = urlencode({"token": token, "email": email})
        reset_link = f"http://localhost:3000/reset-password?{query}"
        
        logger.info("--- DEVELOPMENT EMAIL SENT ---")
        logger.info(f"To: {email}")
        logger.info("Subject: Reset your password")
        logger.info(f"Reset Link: {reset_link}")
        logger.info("------------------------------")

def get_email_service() -> EmailService:
    if settings.environment == Environment.PRODUCTION:
        return SMTPEmailService()
    return DevEmailService()

