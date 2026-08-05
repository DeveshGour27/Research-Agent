import os

directories = [
    "app",
    "tools",
    "rag",
    "prompts",
    "memory/conversations",
    "memory/preferences",
    "memory/tasks",
    "reports",
    "tests"
]

files = {
    "app/__init__.py": "",
    "app/agent.py": "\"\"\"\nAgent module.\nResponsibilities:\n- Main orchestration loop\n- Controls execution\n- Calls planner\n- Calls tools\n- Updates memory\n- Generates final response\n\"\"\"\n\n# TODO: Implement agent logic\n",
    "app/planner.py": "\"\"\"\nPlanner module.\nResponsibilities:\n- Understand goal\n- Break tasks into steps\n- Decide execution order\n- Re-plan if needed\n\"\"\"\n\n# TODO: Implement planner logic\n",
    "app/memory.py": "\"\"\"\nMemory module.\nResponsibilities:\n- Conversation memory\n- Session memory\n- User preferences\n- Future long-term memory\n\"\"\"\n\n# TODO: Implement memory logic\n",
    "app/reflection.py": "\"\"\"\nReflection module.\nResponsibilities:\n- Verify retrieved information\n- Detect missing evidence\n- Retry retrieval if needed\n- Improve answer quality\n\"\"\"\n\n# TODO: Implement reflection logic\n",
    "app/retriever.py": "\"\"\"\nRetriever module.\nResponsibilities:\n- Query rewriting\n- Hybrid search\n- Vector retrieval\n- Metadata filtering\n- Reranking\n\"\"\"\n\n# TODO: Implement retriever logic\n",
    "app/tool_manager.py": "\"\"\"\nTool Manager module.\nResponsibilities:\n- Register tools\n- Validate tool arguments\n- Execute tools\n- Handle failures\n- Return standardized outputs\n\"\"\"\n\n# TODO: Implement tool manager logic\n",
    "app/output.py": "\"\"\"\nOutput Generator module.\nResponsibilities:\n- Markdown formatting\n- Citation formatting\n- Tables\n- Report generation\n\"\"\"\n\n# TODO: Implement output generator logic\n",

    "tools/__init__.py": "",
    "tools/web_search.py": "\"\"\"\nWeb Search tool.\n\"\"\"\n\n# TODO: Implement web search tool\n",
    "tools/calculator.py": "\"\"\"\nCalculator tool.\n\"\"\"\n\n# TODO: Implement calculator tool\n",
    "tools/file_reader.py": "\"\"\"\nFile Reader tool.\n\"\"\"\n\n# TODO: Implement file reader tool\n",
    "tools/file_writer.py": "\"\"\"\nFile Writer tool.\n\"\"\"\n\n# TODO: Implement file writer tool\n",
    "tools/pdf_reader.py": "\"\"\"\nPDF Reader tool.\n\"\"\"\n\n# TODO: Implement pdf reader tool\n",
    "tools/markdown_export.py": "\"\"\"\nMarkdown Export tool.\n\"\"\"\n\n# TODO: Implement markdown export tool\n",

    "rag/__init__.py": "",
    "rag/embeddings.py": "\"\"\"\nEmbeddings module.\nResponsibilities:\n- Generate embeddings\n\"\"\"\n\n# TODO: Implement embeddings logic\n",
    "rag/chunking.py": "\"\"\"\nChunking module.\nResponsibilities:\n- Chunk documents\n\"\"\"\n\n# TODO: Implement chunking logic\n",
    "rag/vector_store.py": "\"\"\"\nVector Store module.\nResponsibilities:\n- Store vectors\n- Retrieve relevant chunks\n\"\"\"\n\n# TODO: Implement vector store logic\n",
    "rag/reranker.py": "\"\"\"\nReranker module.\nResponsibilities:\n- Re-rank retrieved results\n\"\"\"\n\n# TODO: Implement reranking logic\n",
    "rag/hybrid_search.py": "\"\"\"\nHybrid Search module.\nResponsibilities:\n- Perform hybrid retrieval\n\"\"\"\n\n# TODO: Implement hybrid search logic\n",

    "prompts/planner.md": "<!-- Planner Prompt -->\n\n# TODO: Write planner prompt\n",
    "prompts/reflection.md": "<!-- Reflection Prompt -->\n\n# TODO: Write reflection prompt\n",
    "prompts/system.md": "<!-- System Prompt -->\n\n# TODO: Write system prompt\n",
    "prompts/retrieval.md": "<!-- Retrieval Prompt -->\n\n# TODO: Write retrieval prompt\n",

    "tests/__init__.py": "",

    "config.py": "\"\"\"\nConfiguration module.\nLoads configuration via python-dotenv and Pydantic Settings.\n\"\"\"\n\n# TODO: Implement configuration logic\n",
    "main.py": "\"\"\"\nMain Application module.\nResponsibilities:\n- Starts the application\n- Loads configuration\n- Initializes dependencies\n- Creates the agent\n- Handles user interaction\n\"\"\"\n\n# TODO: Implement main application logic\n",
    "requirements.txt": "openai\npydantic\npydantic-settings\nchromadb\nrank-bm25\nnumpy\ntiktoken\nPyMuPDF\npython-dotenv\npytest\n",
    
    ".env.example": "OPENAI_API_KEY=your_openai_api_key_here\n",
    ".gitignore": ".venv/\nvenv/\n__pycache__/\n*.pyc\n.env\nmemory/conversations/*\n!memory/conversations/.gitkeep\nmemory/preferences/*\n!memory/preferences/.gitkeep\nmemory/tasks/*\n!memory/tasks/.gitkeep\nreports/*\n!reports/.gitkeep\n.pytest_cache/\nchroma_db/\n",
    
    "memory/conversations/.gitkeep": "",
    "memory/preferences/.gitkeep": "",
    "memory/tasks/.gitkeep": "",
    "reports/.gitkeep": "",
    
    "README.md": "# Production AI Research Agent\n\nVersion: 1.0\nStatus: Draft\n\n## Overview\nThe Production AI Research Agent is an autonomous AI system capable of planning, reasoning, retrieving knowledge, using external tools, and producing high-quality research reports with verifiable citations.\n",
    
    "pyproject.toml": "[project]\nname = \"production-ai-research-agent\"\nversion = \"1.0.0\"\ndescription = \"Production AI Research Agent\"\nreadme = \"README.md\"\nrequires-python = \">=3.12\"\ndependencies = [\n    \"openai\",\n    \"pydantic\",\n    \"pydantic-settings\",\n    \"chromadb\",\n    \"rank-bm25\",\n    \"numpy\",\n    \"tiktoken\",\n    \"PyMuPDF\",\n    \"python-dotenv\"\n]\n\n[project.optional-dependencies]\ndev = [\n    \"pytest\"\n]\n"
}

for d in directories:
    os.makedirs(d, exist_ok=True)

for filepath, content in files.items():
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

print("Project structure created successfully.")
