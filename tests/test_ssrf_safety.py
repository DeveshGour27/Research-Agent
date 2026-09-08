import pytest
from unittest.mock import patch, MagicMock
from app.exceptions import ToolExecutionError
from app.tools.ssrf import is_safe_ip, validate_safe_url
from app.tools.web_fetch import WebFetchTool


@pytest.mark.parametrize("ip", [
    "127.0.0.1",
    "127.0.1.1",
    "::1",
    "10.0.0.1",
    "10.254.0.1",
    "172.16.0.1",
    "172.31.255.255",
    "192.168.0.1",
    "192.168.1.100",
    "169.254.169.254",
    "fe80::1",
    "0.0.0.0",
    "::ffff:127.0.0.1",
    "100.64.0.1",
])
def test_is_safe_ip_rejects_restricted_ips(ip: str):
    assert is_safe_ip(ip) is False


@pytest.mark.parametrize("ip", [
    "8.8.8.8",
    "1.1.1.1",
    "93.184.216.34",
    "2606:4700:4700::1111",
])
def test_is_safe_ip_accepts_public_ips(ip: str):
    assert is_safe_ip(ip) is True


@pytest.mark.parametrize("bad_url", [
    "http://127.0.0.1/admin",
    "http://localhost:8000/metrics",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5:5432/",
    "http://192.168.1.1/",
    "http://[::1]/secret",
    "http://user:password@example.com/",
    "file:///etc/passwd",
    "gopher://127.0.0.1:70/",
])
def test_validate_safe_url_rejects_ssrf(bad_url: str):
    with pytest.raises(ToolExecutionError):
        validate_safe_url(bad_url)


def test_web_fetch_tool_blocks_ssrf_attempt():
    tool = WebFetchTool()
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.execute(url="http://169.254.169.254/latest/meta-data")
    assert "denied" in str(exc_info.value).lower() or "restricted" in str(exc_info.value).lower()
