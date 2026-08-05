# Rules.md
# Production AI Research Agent
Version: 1.0
Status: Active
Last Updated: August 4, 2026

---

# 1. Purpose

This document defines the engineering standards, coding conventions, architectural constraints, and AI-assisted development rules for the project.

All code written for this project must follow these rules unless an explicit architectural decision supersedes them.

---

# 2. Core Engineering Principles

Always prioritize:

1. Readability over cleverness.
2. Simplicity over unnecessary abstraction.
3. Modularity over monolithic code.
4. Explicit behavior over hidden magic.
5. Correctness over optimization.
6. Maintainability over speed of development.

---

# 3. Python Standards

Required Version

Python 3.12+

Follow:

- PEP 8
- Type hints on all public functions
- Descriptive variable names
- Small, focused functions
- Docstrings for public modules/classes/functions

Avoid:

- Global mutable state
- Circular imports
- Wildcard imports (`from x import *`)
- Deeply nested logic (>3 levels when avoidable)

---

# 4. Project Structure Rules

Every module must have a single responsibility.

Examples:

✓ planner.py → planning only

✓ retriever.py → retrieval only

✓ memory.py → memory only

Never mix unrelated responsibilities in one file.

---

# 5. Dependency Rules

Prefer standard library whenever practical.

Approved libraries:

- OpenAI Agents SDK
- openai
- pydantic
- pydantic-settings
- chromadb
- rank-bm25
- numpy
- tiktoken
- PyMuPDF
- python-dotenv
- pytest
- uv

Avoid introducing new dependencies unless they provide significant value.

Every new dependency must have a clear justification.

---

# 6. Configuration Rules

Never hardcode:

- API keys
- File paths
- Model names
- Secrets
- Tokens

Use:

.env

config.py

Environment variables

---

# 7. Logging Rules

Every important action must be logged.

Log:

- Tool execution
- Retrieval
- Planning
- Reflection
- Errors
- Retry attempts

Never log:

- API keys
- Secrets
- User credentials

---

# 8. Error Handling

Never silently ignore exceptions.

Always:

- Catch expected errors.
- Log meaningful information.
- Return helpful messages.
- Fail gracefully.

Do not expose internal stack traces to end users.

---

# 9. Prompt Engineering Rules

System prompts must be stored separately.

Never hardcode prompts inside Python files.

Prompts belong inside:

prompts/

Each prompt should have:

- Purpose
- Version
- Last Updated

---

# 10. Tool Development Rules

Each tool must:

- Perform one task only.
- Validate inputs.
- Validate outputs.
- Handle failures.
- Return structured responses.

Never let tools modify unrelated project state.

---

# 11. Memory Rules

Memory must be separated into:

- Conversation memory
- Task memory
- User preferences

Memory updates should occur only after successful execution unless a failure must also be recorded.

---

# 12. Retrieval Rules

Always prefer:

Query

↓

Query Rewriting

↓

Hybrid Search

↓

Reranking

↓

Reflection

↓

Answer

Avoid generating answers directly from retrieved chunks without synthesis.

---

# 13. Reflection Rules

Before returning the final response, verify:

- Are sources sufficient?
- Is evidence conflicting?
- Are citations present?
- Is another retrieval needed?

If confidence is low, the agent should acknowledge uncertainty rather than invent information.

---

# 14. Testing Rules

Every new feature should include appropriate tests.

Test categories:

- Unit tests
- Integration tests
- End-to-end tests (when applicable)

Bug fixes should include a regression test whenever practical.

---

# 15. Security Rules

Never:

- Execute arbitrary code from user input.
- Expose secrets.
- Trust external input.
- Assume file paths are safe.

Always validate external inputs before use.

---

# 16. Performance Rules

Avoid:

- Duplicate LLM calls
- Duplicate retrieval
- Duplicate embeddings

Cache results where appropriate.

Prefer batching operations when supported.

---

# 17. Git Rules

Small, focused commits.

Recommended commit format:

feat:

fix:

refactor:

docs:

test:

chore:

Examples:

feat: add web search tool

fix: handle empty retrieval results

docs: update architecture

---

# 18. Documentation Rules

Every major module must include:

- Purpose
- Inputs
- Outputs
- Responsibilities

Public APIs should include examples where helpful.

Architecture changes must update:

Architecture.md

Decision changes must update:

Decisions.md

---

# 19. AI Assistant Rules

The AI assistant must:

✓ Follow Architecture.md

✓ Follow PRD.md

✓ Follow these Rules

✓ Prefer modifying existing modules over creating unnecessary new files.

✓ Explain significant architectural changes before implementing them.

✓ Keep code modular and readable.

The AI assistant must NOT:

✗ Rewrite unrelated code.

✗ Introduce unnecessary libraries.

✗ Break existing interfaces without justification.

✗ Duplicate logic.

✗ Add placeholder implementations presented as complete.

---

# 20. Definition of Done

A feature is complete only if:

✓ Code is implemented.

✓ Tests pass.

✓ Documentation is updated.

✓ Logging is added.

✓ Error handling exists.

✓ Type hints are present.

✓ Linting passes.

✓ Feature matches the PRD.

Only then can the feature be considered finished.