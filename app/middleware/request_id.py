"""Request-ID propagation middleware for Phase 7.5.

Behaviour:
    1. If the incoming request contains an ``X-Request-ID`` header, use it.
    2. Otherwise, generate a new UUID4 string.
    3. Store the ID in ``request.state.request_id``.
    4. Echo the ID back in the response ``X-Request-ID`` header.

This middleware does NOT:
    - Modify authentication behaviour.
    - Alter Phase 1–6 execution contracts.
    - Introduce any new tracing framework.
    - Include secrets or sensitive data in the generated ID.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach and propagate ``X-Request-ID`` on every request/response cycle."""

    _HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Extract or generate
        request_id = request.headers.get(self._HEADER) or str(uuid.uuid4())

        # Store on request state for downstream access
        request.state.request_id = request_id

        response: Response = await call_next(request)

        # Always echo back
        response.headers[self._HEADER] = request_id
        return response
