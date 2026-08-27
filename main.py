"""Interactive command-line chat application.

This Phase 1 entry point deliberately owns only process lifetime and
in-memory conversation state. Provider-specific API calls live in ``app.llm``.
"""

from __future__ import annotations
import uuid

from app.agent import Agent
from app.config import settings
from app.constants import CLI_EXIT_COMMANDS
from app.exceptions import AgentError, ConfigurationError, InputValidationError, AgentHITLPauseException
from app.llm import create_model_gateway
from app.llm.service import ChatService
from app.logger import get_logger
from app.memory import JsonFileMemoryStore
from app.tools import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.web_search import WebSearchTool
from app.agent.registry import AgentRegistry
from app.agent.routing import CapabilityRouter
from app.agent.communicator import InProcessCommunicator
from app.agent.llm_planner import LLMPlanner
from app.agent.executor import PlanExecutor
from app.agent.specialized.web_agent import WebResearchAgent
from app.agent.specialized.rag_agent import RAGAgent
from app.agent.supervisor import Supervisor
from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext
from app.hitl.service import HITLService
from app.hitl.policy import HITLPolicy
from app.db.database import Base, engine, SessionLocal


logger = get_logger(__name__)


def run_chat() -> None:
    """Run the interactive chat loop until the user exits or input closes."""
    provider = create_model_gateway(settings)
    chat_service = ChatService(provider)
    history: list[dict] = []

    logger.info(
        "CLI started",
        extra={"provider": settings.llm_provider, "model": settings.llm_model},
    )
    print("Research Agent chat. Type 'exit' or 'quit' to end.")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.casefold() in CLI_EXIT_COMMANDS:
                break

            try:
                response = chat_service.chat(history, user_input)
            except AgentError as error:
                logger.exception(
                    "Chat request failed",
                    extra={"error_type": type(error).__name__},
                )
                print(f"Unable to generate a response: {error}")
                continue

            history.extend(
                (
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response.content},
                )
            )
            print(f"Agent: {response.content}")
    finally:
        logger.info("CLI stopped", extra={"conversation_messages": len(history)})


def run_agent() -> None:
    """Run the interactive tool-using agent loop until the user exits."""
    provider = create_model_gateway(settings)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    
    hitl_policy = HITLPolicy()
    hitl_service = HITLService(session_factory=SessionLocal, policy=hitl_policy)
    
    memory_store = JsonFileMemoryStore()
    boundary_email = input("User email (optional, for persistent identity): ").strip()
    if boundary_email:
        user = memory_store.get_or_create_user_by_email(boundary_email)
    else:
        user = memory_store.create_user()
    chat_id = input("Chat ID (default: default): ").strip() or "default"
    agent = Agent(
        provider,
        registry,
        user_id=user.user_id,
        chat_id=chat_id,
        memory_store=memory_store,
        hitl_service=hitl_service,
    )

    logger.info(
        "Agent CLI started",
        extra={
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "user_id": user.user_id,
            "chat_id": chat_id,
        },
    )
    print(
        "Research Agent (tool mode). Commands: "
        "'exit'/'quit', '/reset', '/remember <text>', '/memories'."
    )

    turns = 0
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.casefold() in CLI_EXIT_COMMANDS:
                break
            if user_input == "/reset":
                agent.reset()
                print("Agent: Conversation reset.")
                continue
            if user_input.startswith("/remember"):
                memory_text = user_input.removeprefix("/remember").strip()
                if not memory_text:
                    print("Agent: Usage: /remember <text>")
                    continue
                memory = agent.save_user_memory(memory_text)
                print(f"Agent: Saved memory {memory.memory_id}.")
                continue
            if user_input == "/memories":
                memories = agent.get_user_memories()
                if not memories:
                    print("Agent: No saved memories.")
                    continue
                print("Agent memories:")
                for index, memory in enumerate(memories, start=1):
                    print(f"{index}. {memory.content}")
                continue

            context = AgentExecutionContext(task=user_input, user_id=user.user_id, chat_id=chat_id)
            context.set_metadata("job_id", str(uuid.uuid4()))

            try:
                state = agent.run(user_input, context=context)
            except AgentHITLPauseException as e:
                print(f"Agent paused for human approval: {str(e)}")
                continue
            except AgentError as error:
                logger.exception(
                    "Agent request failed",
                    extra={"error_type": type(error).__name__},
                )
                print(f"Unable to generate a response: {error}")
                continue

            turns += 1
            if state.finished and state.final_answer:
                print(f"Agent: {state.final_answer}")
            else:
                print("Agent: I could not finish within the iteration limit.")

            if state.tool_calls:
                tool_summary = ", ".join(call.name for call in state.tool_calls)
                print(f"Tools used: {tool_summary}")
    finally:
        logger.info("Agent CLI stopped", extra={"turns": turns})


def run_supervisor() -> None:
    """Run the interactive Supervisor loop with Phase 6 planning architecture."""
    provider = create_model_gateway(settings)
    
    hitl_policy = HITLPolicy()
    hitl_service = HITLService(session_factory=SessionLocal, policy=hitl_policy)

    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(WebSearchTool())
    general_agent = Agent(provider=provider, registry=tool_registry, hitl_service=hitl_service)
    
    registry = AgentRegistry()
    registry.register(general_agent)
    registry.register(WebResearchAgent())
    registry.register(RAGAgent())

    communicator = InProcessCommunicator(registry)
    plan_executor = PlanExecutor(
        router=CapabilityRouter(),
        registry=registry,
        communicator=communicator,
    )
    planner = LLMPlanner(provider=provider)

    supervisor = Supervisor(
        registry=registry,
        planner=planner,
        plan_executor=plan_executor,
        hitl_service=hitl_service,
    )

    logger.info(
        "Supervisor CLI started",
        extra={"provider": settings.llm_provider, "model": settings.llm_model},
    )
    print("Research Agent (Supervisor mode). Type 'exit' or 'quit' to end.")

    turns = 0
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.casefold() in CLI_EXIT_COMMANDS:
                break

            context = AgentExecutionContext(task=user_input)
            context.set_metadata("job_id", str(uuid.uuid4()))
            request = AgentRequest(input_text=user_input, context=context)

            try:
                result = supervisor.execute(request)
            except AgentHITLPauseException as e:
                print(f"Agent paused for human approval: {str(e)}")
                continue
            except AgentError as error:
                logger.exception(
                    "Supervisor request failed",
                    extra={"error_type": type(error).__name__},
                )
                print(f"Unable to generate a response: {error}")
                continue

            turns += 1
            if result.success:
                print(f"Agent: {result.output}")
            else:
                print(f"Agent failed: {result.output or 'Unknown error'}")

    finally:
        logger.info("Supervisor CLI stopped", extra={"turns": turns})


def run_api() -> None:
    """Run the FastAPI HTTP server using uvicorn."""
    import uvicorn
    logger.info("API server starting", extra={"host": "127.0.0.1", "port": 8000})
    print("Starting FastAPI service on http://127.0.0.1:8000 (OpenAPI docs: http://127.0.0.1:8000/docs)")
    uvicorn.run("app.main_api:app", host="127.0.0.1", port=8000, reload=False)


def main() -> int:
    """Start the CLI and return a process exit code."""
    Base.metadata.create_all(bind=engine)
    try:
        mode = input("Mode [agent/chat/supervisor/api] (default: agent): ").strip().casefold()
        if mode in {"", "agent", "a"}:
            run_agent()
        elif mode in {"chat", "c"}:
            run_chat()
        elif mode in {"supervisor", "s"}:
            run_supervisor()
        elif mode in {"api", "web", "http"}:
            run_api()
        else:
            print("Unknown mode. Starting agent mode.")
            run_agent()
    except (ConfigurationError, InputValidationError) as error:
        logger.error("Unable to start CLI", extra={"error_type": type(error).__name__})
        print(f"Startup failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
