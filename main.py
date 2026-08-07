"""Interactive command-line chat application.

This Phase 1 entry point deliberately owns only process lifetime and
in-memory conversation state. Provider-specific API calls live in ``app.llm``.
"""

from __future__ import annotations

from app.agent import Agent
from app.config import settings
from app.constants import CLI_EXIT_COMMANDS
from app.exceptions import AgentError, ConfigurationError, InputValidationError
from app.llm import ChatMessage, ChatService, create_chat_provider
from app.logger import get_logger
from app.memory import JsonFileMemoryStore
from app.tools import ToolRegistry
from app.tools.calculator import CalculatorTool


logger = get_logger(__name__)


def run_chat() -> None:
    """Run the interactive chat loop until the user exits or input closes."""
    provider = create_chat_provider(settings)
    chat_service = ChatService(provider)
    history: list[ChatMessage] = []

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
                    ChatMessage(role="user", content=user_input),
                    ChatMessage(role="assistant", content=response.content),
                )
            )
            print(f"Agent: {response.content}")
    finally:
        logger.info("CLI stopped", extra={"conversation_messages": len(history)})


def run_agent() -> None:
    """Run the interactive tool-using agent loop until the user exits."""
    provider = create_chat_provider(settings)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
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

            try:
                state = agent.run(user_input)
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


def main() -> int:
    """Start the CLI and return a process exit code."""
    try:
        mode = input("Mode [agent/chat] (default: agent): ").strip().casefold()
        if mode in {"", "agent", "a"}:
            run_agent()
        elif mode in {"chat", "c"}:
            run_chat()
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
