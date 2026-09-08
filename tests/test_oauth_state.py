import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main_api import app
from app.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def configure_oauth(monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    monkeypatch.setattr(settings, "google_client_secret", "test-client-secret")
    monkeypatch.setattr(settings, "google_redirect_uri", "http://testserver/api/v1/auth/google/callback")

def test_google_login_generates_state_and_cookie():
    res = client.get("/api/v1/auth/google/login")
    assert res.status_code == 200
    data = res.json()
    assert "url" in data
    assert "state=" in data["url"]
    assert "oauth_state" in res.cookies
    # Verify state in url matches the cookie value
    cookie_state = res.cookies["oauth_state"]
    assert f"state={cookie_state}" in data["url"]

def test_google_callback_rejects_missing_state():
    res = client.get("/api/v1/auth/google/callback?code=mock-code")
    assert res.status_code == 400
    assert "Invalid or missing OAuth state" in res.json()["detail"]

def test_google_callback_rejects_mismatched_state():
    client.cookies.set("oauth_state", "expected-secret-state")
    res = client.get("/api/v1/auth/google/callback?code=mock-code&state=wrong-state")
    assert res.status_code == 400
    assert "Invalid or missing OAuth state" in res.json()["detail"]

def test_google_callback_rejects_unverified_email():
    client.cookies.set("oauth_state", "valid-state")
    with patch("httpx.AsyncClient.post") as mock_post, patch("httpx.AsyncClient.get") as mock_get:
        mock_post.return_value = MagicMock(is_success=True, json=lambda: {"access_token": "fake-token"})
        mock_get.return_value = MagicMock(is_success=True, json=lambda: {
            "id": "12345",
            "email": "user@example.com",
            "email_verified": False
        })
        res = client.get("/api/v1/auth/google/callback?code=valid-code&state=valid-state", follow_redirects=False)
        assert res.status_code == 400
        assert "Google account email is not verified" in res.json()["detail"]

def test_google_callback_success_with_valid_state():
    client.cookies.set("oauth_state", "valid-state-123")
    with patch("httpx.AsyncClient.post") as mock_post, patch("httpx.AsyncClient.get") as mock_get:
        mock_post.return_value = MagicMock(is_success=True, json=lambda: {"access_token": "fake-token"})
        mock_get.return_value = MagicMock(is_success=True, json=lambda: {
            "id": "google-user-999",
            "email": "testoauth@example.com",
            "email_verified": True
        })
        res = client.get("/api/v1/auth/google/callback?code=valid-code&state=valid-state-123", follow_redirects=False)
        assert res.status_code == 307
        assert "session_id" in res.cookies
