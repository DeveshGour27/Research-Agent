# Architecture.md
# Production AI Research Agent
Version: 1.0
Status: Draft
Last Updated: August 4, 2026

---

# 1. Purpose

This document describes the technical architecture of the Production AI Research Agent.

It defines:

- Overall system architecture
- Application flow
- Component responsibilities
- File and folder structure
- Technology stack
- Communication between modules
- Data flow

This document acts as the technical blueprint for the project.

---

# 2. High-Level Architecture

                    +----------------------+
                    |        User          |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    Main Application  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |      AI Agent        |
                    +----------+-----------+
                               |
        -------------------------------------------------
        |          |           |           |             |
        v          v           v           v             v
    Planner     Memory      Retriever     Tools     Reflection
        |                      |
        |                      |
        |               Vector Database
        |
        v
     Final Response

---

# 3. Agent Execution Flow

User Request

↓

Input Validation

↓

Load Memory

↓

Planner creates execution plan

↓

Retriever gathers relevant information

↓

Tool Selection

↓

Tool Execution

↓

Reflection

↓

Generate Final Answer

↓

Store Memory

↓

Return Response

---

# 4. Core Components

## Main Application

Responsibilities

- Starts the application
- Loads configuration
- Initializes dependencies
- Creates the agent
- Handles user interaction

---

## Agent

Responsibilities

- Main orchestration loop
- Controls execution
- Calls planner
- Calls tools
- Updates memory
- Generates final response

---

## Planner

Responsibilities

- Understand goal
- Break tasks into steps
- Decide execution order
- Re-plan if needed

---

## Memory

Responsibilities

- Conversation memory
- Session memory
- User preferences
- Future long-term memory

---

## Retriever

Responsibilities

- Query rewriting
- Hybrid search
- Vector retrieval
- Metadata filtering
- Reranking

---

## Reflection

Responsibilities

- Verify retrieved information
- Detect missing evidence
- Retry retrieval if needed
- Improve answer quality

---

## Tool Manager

Responsibilities

- Register tools
- Validate tool arguments
- Execute tools
- Handle failures
- Return standardized outputs

---

## Output Generator

Responsibilities

- Markdown formatting
- Citation formatting
- Tables
- Report generation

---

# 5. Folder Structure

project/

├── app/
│   ├── agent.py
│   ├── planner.py
│   ├── memory.py
│   ├── reflection.py
│   ├── retriever.py
│   ├── tool_manager.py
│   └── output.py
│
├── tools/
│   ├── web_search.py
│   ├── calculator.py
│   ├── file_reader.py
│   ├── file_writer.py
│   ├── pdf_reader.py
│   └── markdown_export.py
│
├── rag/
│   ├── embeddings.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── reranker.py
│   └── hybrid_search.py
│
├── prompts/
│   ├── planner.md
│   ├── reflection.md
│   ├── system.md
│   └── retrieval.md
│
├── memory/
│   ├── conversations/
│   ├── preferences/
│   └── tasks/
│
├── reports/
│
├── tests/
│
├── config.py
├── main.py
└── requirements.txt

---

# 6. Technology Stack

Programming Language

- Python 3.12+

LLM Framework

- OpenAI Agents SDK

LLM Provider

- Groq chat completions through a provider-neutral LLM interface
- Primary and fallback model names are configured centrally
- A fallback request is issued only after transient availability failures

Embeddings

- OpenAI Embeddings (replaceable)

Vector Database

- ChromaDB (local)

Document Parsing

- PyMuPDF
- Markdown
- python-docx (future)

Search

- Hybrid Search
- BM25
- Vector Search

Logging

- Python logging

Configuration

- python-dotenv
- Pydantic Settings

Testing

- pytest

Package Manager

- uv

---

# 7. Data Flow

User Query

↓

Planner

↓

Retriever

↓

Tool Manager

↓

LLM

↓

Reflection

↓

Formatter

↓

User

---

# 8. Design Principles

- Modular architecture
- Single responsibility per module
- Loose coupling
- High cohesion
- Configurable components
- Replaceable model providers
- Replaceable vector databases
- Tool abstraction
- Fail gracefully
- Observable execution

---

# 9. Future Expansion

The architecture is designed so future additions require minimal changes.

Planned additions include:

- Multi-Agent System
- MCP Tool Support
- A2A Communication
- Human Approval Workflow
- Evaluation Framework
- Observability Dashboard
- Cloud Deployment
- Authentication
- Web Interface

---

# 10. Architecture Decisions

Decision | Reason
-------- | ------
Modular structure | Easier maintenance
Planner separated from Agent | Clear responsibilities
Dedicated Tool Manager | Easy tool expansion
Separate Retriever | Independent RAG improvements
Reflection module | Higher answer quality
Prompt directory | Easier prompt management
Config file | Centralized configuration
Groq model fallback | Preserve chat availability for rate limits, timeouts, and server failures without coupling business services to Groq

---

# 11. Out of Scope

The following are intentionally excluded from Version 1:

- Distributed execution
- GPU optimization
- Multi-user authentication
- Cloud deployment
- Mobile application
- Fine-tuning
- Voice interface

These will be considered in future versions.
