# Remediation Plan

This is an implementation plan only. No code was changed as part of the audit.

## P0 — Must fix before calling this production-ready

### P0-1: Bind Google OAuth callbacks to the initiating browser

- **Problem:** OAuth callback accepts a code without a one-time `state` value.
- **Why it matters:** Prevents login CSRF/account confusion and data disclosure into an attacker account.
- **Files:** `app/api/auth_routes.py`, DB migration/new transient state store, frontend login integration.
- **Implementation direction:** Create a cryptographic state/nonce, store hash+expiry bound to secure browser session, send state to Google, consume/validate it before token exchange, then validate OIDC claims including `email_verified`.
- **Tests required:** Wrong/missing/expired/replayed/cross-browser state rejection; valid flow; cookie attributes.
- **Dependencies:** OAuth provider configuration and a small state persistence mechanism.
- **Risk of change:** Medium; coordinate callback URL and user-facing error behavior.

### P0-2: Make fetch SSRF-safe and enforce egress boundaries

- **Problem:** URL scheme validation is the only fetch control.
- **Why it matters:** Page fetches can reach private/metadata/local network services.
- **Files:** `app/tools/web_fetch.py`, `app/agent/specialized/web_agent.py`, deployment network policy, tests.
- **Implementation direction:** Canonicalize URLs; resolve/check all IPs; block private/reserved ranges; connect with DNS-rebinding protection; manually validate limited redirects; impose byte/type/decompression limits; run fetches with outbound network restrictions.
- **Tests required:** IPv4/IPv6/private/encoded host/redirect/rebinding fixtures and assertion that blocked endpoints receive no request.
- **Dependencies:** Network policy/container configuration.
- **Risk of change:** High; legitimate scholarly sites/CDNs/redirects need measured allow/deny behavior.

### P0-3: Put hard resource boundaries around authenticated work

- **Problem:** Chat/auth/SSE paths are unbounded or not rate-limited.
- **Why it matters:** Prevents cost, worker, DB, and connection exhaustion.
- **Files:** `app/api/chat_routes.py`, `app/api/auth_routes.py`, `app/api/routes.py`, `app/services/rate_limiter.py`, `app/services/job_manager.py`, frontend error handling.
- **Implementation direction:** Add request/max-size validation; atomic DB job quota per user; IP+account rate limits for unauthenticated auth routes; per-user SSE connection/lifetime limits; shared/distributed limiter before multi-instance deployment.
- **Tests required:** Quota races, oversized chat rejection, 429/retry behavior, connection cleanup, multi-worker behavior.
- **Dependencies:** Database migration/indexes; production rate-limit store if horizontally scaled.
- **Risk of change:** Medium; tune limits from observed workload.

### P0-4: Make production compose safe by default

- **Problem:** Default credentials and published internal services.
- **Why it matters:** Prevents direct database compromise in an operator-default deployment.
- **Files:** `docker-compose.yml`, `searxng/settings.yml`, `.env.example`, deployment documentation.
- **Implementation direction:** Require secrets with startup failure, remove public DB/SearXNG ports, use internal network, pin images, separate local developer override, externalize SearXNG secret.
- **Tests required:** Compose/config policy assertions and a smoke deployment with no public internal ports.
- **Dependencies:** Secret-delivery process.
- **Risk of change:** Low-to-medium; local developer workflow needs explicit replacement compose profile.

### P0-5: Establish a release gate that actually completes

- **Problem:** The 573-test suite was collected but not fully completed during audit; its first executed failure is stale.
- **Why it matters:** Security claims cannot rest on collection counts.
- **Files:** `tests/test_chat_service.py`, CI workflow (currently not identified), dependency tooling.
- **Implementation direction:** Fix the test expectation or make the system-prompt contract explicit; run isolated test groups, then full suite in CI with JUnit output, timeout diagnostics, lint/build, dependency audit, and coverage artifacts.
- **Tests required:** The corrected chat service assertion plus deterministic full-suite CI.
- **Dependencies:** Repair local npm installation and standardize Python environment.
- **Risk of change:** Low for the first failure; medium for test isolation work.

## P1 — Important engineering fixes

### P1-1: Complete account-session security

- **Problem:** Password reset/change leaves prior sessions active; signup/login/reset lack abuse controls; OAuth does not enforce verified email claim.
- **Why it matters:** Account takeover recovery and abuse resistance.
- **Files:** `app/api/auth_routes.py`, `app/db/models.py`, `app/db/repository.py`.
- **Implementation direction:** Revoke sessions transactionally on reset/change, add session version/revocation index, rate-limit by IP/account, normalize generic responses, validate OAuth claims.
- **Tests required:** Multi-session reset, auth throttle, response privacy, OAuth email verification.
- **Dependencies:** P0-1.
- **Risk of change:** Medium.

### P1-2: Default-deny MCP and isolate extensions

- **Problem:** Configured MCP commands may launch automatically with an allow-all policy.
- **Why it matters:** MCP is equivalent to granting a local extension code execution and data access.
- **Files:** `app/mcp/policy.py`, `app/mcp/registry.py`, `app/mcp/transport.py`, `app/tools/registry.py`, `app/agent/specialized/web_agent.py`.
- **Implementation direction:** Require explicit allowlist, avoid MCP initialization unless an agent requests it, use minimal env/absolute command validation/process sandbox, classify tool side effects and require HITL.
- **Tests required:** Default no-spawn, denied tool/server, no environment inheritance, approval behavior.
- **Dependencies:** Deployment policy.
- **Risk of change:** High for existing MCP users; release behind an explicit migration flag.

### P1-3: Enforce generic evidence/citation contracts

- **Problem:** Evidence verifier is quantum-discovery specific and raw RAG/web prose can still become unsupported synthesis.
- **Why it matters:** It is the product’s central reliability promise.
- **Files:** `app/agent/evidence.py`, `app/agent/specialized/web_agent.py`, `app/agent/specialized/rag_agent.py`, `app/agent/specialized/reasoning_agent.py`.
- **Implementation direction:** Pass typed evidence records with source URL/document ID/offset/hash; make output claims map to evidence IDs; render citations deterministically; reject model-generated sources/claims without support. Preserve prompt injection labels as defense-in-depth, not the primary control.
- **Tests required:** Hostile document, fabricated URL, unsupported claim, duplicate source, publication-vs-article date, and cross-domain research cases.
- **Dependencies:** P0-2 for safe source retrieval.
- **Risk of change:** High; answer format and quality evaluation will change.

### P1-4: Make timeout/cancellation real

- **Problem:** `to_thread` work survives job timeout.
- **Why it matters:** Status correctness alone does not bound external cost/resource use.
- **Files:** `app/services/job_manager.py`, LLM providers, tools.
- **Implementation direction:** Deadline budget in execution context; cooperative cancellation at provider/tool boundaries; isolate non-cooperative work in a killable worker process before scale-out.
- **Tests required:** Blocked provider/tool and orphan accounting/cancellation tests.
- **Dependencies:** Provider/client support and possibly worker isolation.
- **Risk of change:** High.

### P1-5: Finish active product wiring

- **Problem:** RAG ingestion/API key lifecycle/frontend deployment are absent or split; `/research.chat_id` is inert.
- **Why it matters:** Prevents documents and architecture claims from drifting away from deployed behavior.
- **Files:** `app/api/routes.py`, `app/api/chat_routes.py`, `app/main_api.py`, `frontend/next.config.ts`, compose/Docker artifacts.
- **Implementation direction:** Choose one supported conversation/job API flow; remove inert fields/routes or implement ownership-checked behavior; add owner-scoped document lifecycle before exposing ingestion; build/deploy frontend with service DNS rather than loopback.
- **Tests required:** End-to-end browser/API/SSE success/failure/HITL and document lifecycle isolation.
- **Dependencies:** P0 resource controls and P1 evidence model.
- **Risk of change:** Medium.

## P2 — Quality improvements

### P2-1: Consolidate duplicate/dead layers

- **Problem:** Root tools/stubs, old app modules, `agent_core`, and duplicate retrieval paths obscure active behavior.
- **Why it matters:** Reduces security review surface and prevents accidental imports.
- **Files:** `agent_core/`, root `tools/`, `app/retriever.py`, stubs under `app/`, prompts, dependency files.
- **Implementation direction:** Build an import/reference map, deprecate with tests, delete one coherent slice at a time, and align documentation.
- **Tests required:** Import smoke tests and active-path integration tests after each deletion.
- **Dependencies:** P1-5 decision on canonical RAG/API path.
- **Risk of change:** Medium-to-high.

### P2-2: Replace JSON long-term memory with a governed store

- **Problem:** No locking, retention policy, secret filtering, or web-path integration.
- **Why it matters:** Durable memory is privacy-sensitive state.
- **Files:** `app/memory.py`, DB migrations, privacy documentation.
- **Implementation direction:** Owner-scoped transactional storage, retention/deletion, sensitivity filter, explicit consent, audit trail.
- **Tests required:** Concurrent writes, delete/export, tenant isolation, poisoning/secret rejection.
- **Dependencies:** Data policy decision.
- **Risk of change:** High for migration.

### P2-3: Pin and audit dependencies

- **Problem:** Multiple manifests and broad/unpinned dependencies.
- **Why it matters:** Repeatability and supply-chain risk.
- **Files:** `pyproject.toml`, `requirements.txt`, `frontend/package.json`, lockfiles, CI.
- **Implementation direction:** Select one Python dependency authority/lock strategy, align manifests, pin base/images, run vulnerability/license checks in CI, upgrade deliberately with compatibility tests.
- **Tests required:** Clean environment install and build matrix.
- **Dependencies:** Functioning npm tooling.
- **Risk of change:** Medium.

## P3 — Future enhancements

### P3-1: Durable event delivery and distributed worker model

- **Problem:** Database-polling SSE and in-process execution have limited scale/cancellation semantics.
- **Why it matters:** Operational resilience at multi-instance scale.
- **Files:** Job/event service, API events, deployment.
- **Implementation direction:** Only after P0/P1, introduce a minimal durable queue/event transport that preserves job ownership/idempotency and supports replay.
- **Tests required:** Multi-worker replay/order/failover/load tests.
- **Dependencies:** Defined scale/SLO requirement.
- **Risk of change:** High.

### P3-2: Broaden research evaluation beyond the quantum-specific suite

- **Problem:** Current evidence heuristics/evaluations do not represent general research quality.
- **Why it matters:** Avoids measuring success only on tailored cases.
- **Files:** Datasets, evaluation engine, evidence contracts.
- **Implementation direction:** Add diverse domains, adversarial sources, citation verification, calibration, and human-reviewed benchmark cases.
- **Tests required:** Deterministic replay plus periodic controlled live evaluation.
- **Dependencies:** P1-3 typed evidence format.
- **Risk of change:** Low code risk, high evaluation-design effort.

## What Not To Change Yet

Do not introduce Kubernetes, microservices, Celery/Redis/Kafka, a new vector database, or another agent framework as a substitute for the P0/P1 controls. The existing FastAPI/SQLAlchemy/supervisor foundation can become correct and demonstrable before scale architecture is justified.
