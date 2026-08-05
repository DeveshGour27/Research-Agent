"""
Application-wide constants for the Production AI Research Agent.

Responsibilities:
    - Eliminate magic numbers and hardcoded strings from business code
    - Provide a single place to update system-wide defaults
    - Group related constants into namespaced classes to aid discoverability

Design Decision:
    Constants that are *tuneable per deployment* live in `app.config`
    (loaded from environment variables). Constants defined here are
    *structural* — they reflect design decisions that should not change
    without a code review (e.g. the name of a memory sub-directory, the
    supported file extensions for document ingestion).

    Plain module-level variables are used instead of an Enum so that the
    values remain bare strings/ints — no `.value` unwrapping needed at
    call sites.
"""

from __future__ import annotations

from pathlib import Path


# ------------------------------------------------------------------ #
# Directory layout
# ------------------------------------------------------------------ #

# Absolute path to the project root (one level above this file's package).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
MEMORY_DIR: Path = PROJECT_ROOT / "memory"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"

# Sub-directories under memory/
MEMORY_CONVERSATIONS_DIR: Path = MEMORY_DIR / "conversations"
MEMORY_PREFERENCES_DIR: Path = MEMORY_DIR / "preferences"
MEMORY_TASKS_DIR: Path = MEMORY_DIR / "tasks"


# ------------------------------------------------------------------ #
# Prompt file names
# (relative to PROMPTS_DIR — keeps references consistent)
# ------------------------------------------------------------------ #

PROMPT_SYSTEM: str = "system.md"
PROMPT_PLANNER: str = "planner.md"
PROMPT_REFLECTION: str = "reflection.md"
PROMPT_RETRIEVAL: str = "retrieval.md"


# ------------------------------------------------------------------ #
# Supported document types
# ------------------------------------------------------------------ #

SUPPORTED_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".md", ".txt", ".docx"}
)


# ------------------------------------------------------------------ #
# RAG pipeline
# ------------------------------------------------------------------ #

# Separator string used between chunks when reassembling context for the LLM
CHUNK_SEPARATOR: str = "\n\n---\n\n"

# Minimum meaningful chunk length (characters).
# Chunks shorter than this are merged with the next one.
MIN_CHUNK_CHARACTERS: int = 100


# ------------------------------------------------------------------ #
# Tool execution
# ------------------------------------------------------------------ #

# Seconds before a single tool call is considered timed out
TOOL_TIMEOUT_SECONDS: int = 30

# Maximum characters accepted in a single user query
MAX_QUERY_LENGTH: int = 4096


# ------------------------------------------------------------------ #
# Logging / output
# ------------------------------------------------------------------ #

# Application-wide logger name (all child loggers are prefixed with this)
APP_LOGGER_NAME: str = "agent"

# Date-time format used in plain-text log output (not JSON)
LOG_DATETIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"


# ------------------------------------------------------------------ #
# Report generation
# ------------------------------------------------------------------ #

# Markdown heading used for the citations / references section
CITATIONS_HEADING: str = "## References"

# Default report file extension
REPORT_FILE_EXTENSION: str = ".md"
