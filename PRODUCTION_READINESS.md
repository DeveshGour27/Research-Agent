# Production Readiness Assessment

Audit date: 2026-09-09. Scores reflect the current code path and deployment artifacts, not intended future architecture.

Scale: 0 absent · 1 prototype · 2 functional · 3 production-oriented · 4 strong · 5 excellent.

| Category | Score | Assessment |
|---|---:|---|
| Architecture | 2 | Clear layers and a real job/supervisor path exist, but active and legacy implementations diverge; documented RAG/API/frontend architecture is not fully wired. |
| Authentication | 2 | Argon2, expiry, logout revocation, generic login errors, and email verification are good. OAuth lacks state, auth abuse throttling is missing, and reset does not revoke sessions. |
| Authorization | 4 | Jobs, chats, messages, HITL, events, and SSE are consistently ownership-scoped in reviewed routes and have meaningful tests. |
| Agent safety | 2 | Planner task/capability allowlists and bounded plans help. The runtime remains highly dependent on model output and has no general policy boundary for all side effects. |
| Tool safety | 1 | Search has basic timeouts; fetch lacks SSRF policy and MCP defaults to allow-all. |
| Prompt injection resistance | 2 | Planner and ReAct prompts contain defensive wording and live web retrieval is mostly deterministic, but raw RAG text reaches a synthesizer and evidence claims are not structurally enforced. |
| RAG security | 3 | User metadata filtering exists for dense, BM25, and persisted retrieval. Ingestion/deletion/lifecycle is not exposed/wired in the active API, and hostile-document controls are absent. |
| Memory security | 1 | Per-user JSON partitions exist in legacy/CLI code, but no locking, retention, secret filtering, or production-grade storage exists. It is not part of the primary web path. |
| Data isolation | 4 | Database and vector queries are owner-filtered; cross-user route/SSE/RAG tests exist. Continue testing every future resource endpoint. |
| Reliability | 2 | Atomic claims, heartbeats, stale recovery, state transitions, retry bounds, and failure propagation are solid. Thread timeouts do not cancel underlying work; API/job and frontend integration remain incomplete in places. |
| Observability | 3 | Structured events, redaction, readiness, and metrics exist. The metric endpoint is public and database-polled SSE lacks operational controls/limits. |
| Evaluation | 2 | Replay/golden/evidence tests exist, but the evidence framework is narrowly domain-specific and does not prove factual support for arbitrary answers. |
| Testing | 2 | 573 tests collect and several security/state tests are meaningful. The full run was not completed during this audit; the first executed failure is a stale chat-service expectation. Key SSRF/OAuth/quota/MCP tests are absent. |
| Frontend security | 3 | React Markdown is used without raw HTML and session cookies are HttpOnly. No browser token storage was found. No completed lint/build run; no CSP/security-header policy or frontend container deployment is present. |
| Deployment | 1 | Dockerfile drops root privileges, but compose exposes internal services with predictable defaults, uses a floating SearXNG tag, and lacks a frontend deployment. |
| Documentation | 2 | Architecture/deep-dive material is extensive but materially stale: test counts, routes, and claims do not all match active code. |

## Overall

**2/5 — functional, production-oriented prototype; not production-ready.**

The strongest foundation is owner-scoped persistence and job-state handling. The largest blockers are external-boundary safety (OAuth/SSRF/MCP), resource controls, deployment defaults, generic evidence integrity, and demonstrating a clean full test/build run.
