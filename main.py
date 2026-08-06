"""Interactive command-line chat application.

This Phase 1 entry point deliberately owns only process lifetime and
in-memory conversation state. Provider-specific API calls live in ``app.llm``.
"""

from __future__ import annotations

from app.config import settings
from app.constants import CLI_EXIT_COMMANDS
from app.exceptions import AgentError, ConfigurationError, InputValidationError
from app.llm import ChatMessage, ChatService, create_chat_provider
from app.logger import get_logger


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


def main() -> int:
    """Start the CLI and return a process exit code."""
    try:
        run_chat()
    except (ConfigurationError, InputValidationError) as error:
        logger.error("Unable to start CLI", extra={"error_type": type(error).__name__})
        print(f"Startup failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
