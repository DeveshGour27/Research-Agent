"""Public façade for the tool-using agent engine."""

from __future__ import annotations

from uuid import uuid4

from app.agent.contracts import (
    AgentCapabilities,
    AgentExecutionError,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.loop import AgentLoop
from app.agent.state import AgentState
from app.config import settings
from app.exceptions import AgentError, MemoryReadError, MemoryWriteError
from app.llm.gateway import ModelGateway
from app.logger import get_logger
from app.hitl.service import HITLService
from app.memory import (
    JsonFileMemoryStore,
    Memory,
    MemoryExtractor,
    MemoryStore,
)
from app.tools.registry import ToolRegistry
from app.agent.execution_context import AgentExecutionContext

logger = get_logger(__name__)

_DEFAULT_MAX_ITERATIONS: int = settings.max_retries * 3
_DEFAULT_MEMORY_TOP_K: int = settings.top_k_retrieval
_MEMORY_CONTEXT_PREFIX: str = "Relevant user memories:"


class Agent(BaseAgent):
    """Create and run an agent that answers questions using tools.

    Args:
        provider: Any :class:`~app.llm.base.LLMProvider` instance.
        registry: :class:`~app.tools.registry.ToolRegistry` pre-loaded
            with the tools the agent may call.
        max_iterations: Upper bound on LLM round-trips per :meth:`run` call.
            Defaults to ``settings.max_retries * 3``.

    Example::

        from app.llm import create_chat_provider
        from app.tools.registry import ToolRegistry
        from app.tools.calculator import CalculatorTool
        from app.agent import Agent
        from app import settings

        provider = create_chat_provider(settings)
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        agent = Agent(provider, registry)
        state = agent.run("What is 123 * 456?")
        print(state.final_answer)
    """

    def __init__(
        self,
        provider: ModelGateway,
        registry: ToolRegistry,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        user_id: str | None = None,
        chat_id: str | None = None,
        memory_store: MemoryStore | None = None,
        memory_extractor: MemoryExtractor | None = None,
        memory_top_k: int = _DEFAULT_MEMORY_TOP_K,
        hitl_service: HITLService | None = None,
    ) -> None:
        self._router = provider
        self._registry = registry
        self._max_iterations = max_iterations
        self._hitl_service = hitl_service
        self._loop = AgentLoop(
            self._router,
            self._registry,
            self._max_iterations,
            hitl_service=hitl_service,
        )

        self._memory_store = memory_store or JsonFileMemoryStore()
        self._memory_extractor = memory_extractor or MemoryExtractor(provider)
        self._memory_top_k = memory_top_k

        self.user_id = (user_id or str(uuid4())).strip()
        self.chat_id = (chat_id or "default").strip() or "default"

        self._identity = AgentIdentity(
            name="production-research-agent",
            version="1.0.0",
            description="Production AI Research Agent",
        )
        self._capabilities = AgentCapabilities(
            tool_use=True,
            memory=True,
            multi_turn=True,
            retrieval=True,
        )

        self._memory_store.create_user(user_id=self.user_id)
        self._memory_store.create_chat(
            self.user_id,
            self.chat_id,
            initial_messages=AgentLoop.initial_messages(),
        )

        existing_messages = self._memory_store.get_chat_messages(
            self.user_id,
            self.chat_id,
        )

        self._messages: list[dict[str, object]] = (
            existing_messages
            if existing_messages
            else AgentLoop.initial_messages()
        )
        self._context_agent_ids: dict[int, str] = {}

    @property
    def identity(self) -> AgentIdentity:
        """Return the stable identity for this agent instance."""
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        """Return the capabilities exposed by this agent implementation."""
        return self._capabilities

    def _resolve_context_agent_name(self, context: AgentExecutionContext) -> str:
        """Return a context-safe agent name for shared execution context."""
        if context is None:
            return self.identity.name

        cached_name = self._context_agent_ids.get(id(context))
        if cached_name is not None:
            return cached_name

        base_name = self.identity.name
        existing_output = context.get_agent_output(base_name)

        if existing_output is None:
            self._context_agent_ids[id(context)] = base_name
            return base_name

        suffix = 2
        while True:
            candidate = f"{base_name}-{suffix}"
            if context.get_agent_output(candidate) is None:
                self._context_agent_ids[id(context)] = candidate
                return candidate
            suffix += 1

    def execute(self, request: AgentRequest) -> AgentResult:
        """Execute a structured agent request and return a structured result."""
        normalized_input = request.input_text.strip()

        if not normalized_input:
            raise AgentExecutionError(
                "Agent request input_text must not be empty.",
                request=request,
                details={"request_id": request.request_id},
            )

        context = request.context

        if context is None:
            context = AgentExecutionContext(
                task=normalized_input,
                user_id=self.user_id,
                chat_id=self.chat_id,
            )

        agent_name = self._resolve_context_agent_name(context)
        if agent_name != self.identity.name:
            self._identity = AgentIdentity(
                name=agent_name,
                version=self.identity.version,
                description=self.identity.description,
            )

        context.mark_running()

        from app.exceptions import AgentHITLPauseException
        try:
            state = self.run(normalized_input, context=context)

        except AgentExecutionError:
            context.mark_failed()
            raise
            
        except AgentHITLPauseException:
            # We do NOT mark it as failed here. 
            # The exception bubbles up to AsyncJobManager to pause the job.
            raise

        except AgentError as error:
            context.mark_failed()

            raise AgentExecutionError(
                "Agent execution failed.",
                request=request,
                details={
                    "request_id": request.request_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            ) from error

        except Exception as error:  # pragma: no cover - defensive guard
            context.mark_failed()

            raise AgentExecutionError(
                "Unexpected agent execution failure.",
                request=request,
                details={
                    "request_id": request.request_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            ) from error

        success = bool(
            state.finished and state.final_answer is not None
        )

        result_metadata = {
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "iterations": state.iteration,
            "tool_calls": len(state.tool_calls),
        }

        context.publish_agent_output(
            agent_id=self.identity.name,
            output=state.final_answer,
            success=success,
            metadata=result_metadata,
        )

        if success:
            context.mark_completed()
        else:
            context.mark_failed()

        return AgentResult(
            request=request,
            state=state,
            output=state.final_answer,
            success=success,
            context=context,
            metadata=result_metadata,
        )

    def run(self, user_input: str, context: AgentExecutionContext | None = None) -> AgentState:
        """Execute the agent loop for *user_input* and return final state.

        Args:
            user_input: The user's question or instruction.
            context: Optional execution context for HITL operations.

        Returns:
            :class:`~app.agent.state.AgentState` containing the final answer,
            all tool calls made, their observations, and cumulative token usage.
        """
        logger.info(
            "Agent run started",
            extra={"input_chars": len(user_input)},
        )

        memory_context_message = self._build_memory_context_message(
            user_input
        )

        run_messages = [
            *self._messages,
            *(
                [memory_context_message]
                if memory_context_message is not None
                else []
            ),
            {
                "role": "user",
                "content": user_input,
            },
        ]
        
        from app.exceptions import AgentHITLPauseException

        try:
            state = self._loop.run(run_messages, context=context)
        except AgentHITLPauseException as e:
            # Save the message history up to the tool call before pausing
            if hasattr(e, "partial_state") and e.partial_state:
                self._messages = self._strip_memory_context_messages(e.partial_state.messages)
                self._memory_store.save_chat_messages(self.user_id, self.chat_id, self._messages)
            raise

        self._messages = self._strip_memory_context_messages(
            state.messages
        )

        self._memory_store.save_chat_messages(
            self.user_id,
            self.chat_id,
            self._messages,
        )

        if state.finished and state.final_answer:
            self._maybe_extract_memory(
                user_input=user_input,
                assistant_response=state.final_answer,
            )

        logger.info(
            "Agent run completed",
            extra={
                "finished": state.finished,
                "iterations": state.iteration,
                "tool_calls": len(state.tool_calls),
            },
        )

        return state

    def reset(self) -> None:
        """Clear current chat history while preserving long-term memories."""
        self._messages = AgentLoop.initial_messages()

        self._memory_store.reset_chat_history(
            self.user_id,
            self.chat_id,
            self._messages,
        )

        logger.info("Agent conversation reset")

    def save_user_memory(self, content: str) -> Memory:
        """Persist one explicit long-term memory for the current user."""
        return self._memory_store.save_memory(
            self.user_id,
            content,
        )

    def get_user_memories(self) -> list[Memory]:
        """Return long-term memories for the current user."""
        return self._memory_store.get_memories(self.user_id)

    def _maybe_extract_memory(
        self,
        *,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Extract and safely apply an automatic memory operation.

        Memory extraction is deliberately non-fatal. A failure in automatic
        memory processing must never cause an otherwise successful agent
        request to fail.
        """
        try:
            existing_memories = self._memory_store.get_memories(
                self.user_id
            )

            extraction = self._memory_extractor.extract(
                user_input=user_input,
                assistant_response=assistant_response,
                existing_memories=existing_memories,
            )

            if not extraction.should_store:
                return

            if extraction.operation == "create":
                self._memory_store.upsert_memory(
                    self.user_id,
                    extraction.memory,
                )
                return

            if extraction.operation == "update":
                if not extraction.memory_id:
                    logger.warning(
                        "Skipping memory update without memory_id",
                    )
                    return

                self._memory_store.update_memory(
                    self.user_id,
                    extraction.memory_id,
                    extraction.memory,
                )
                return

            logger.warning(
                "Skipping unsupported memory operation",
                extra={"operation": extraction.operation},
            )

        except (MemoryReadError, MemoryWriteError) as error:
            logger.warning(
                "Automatic memory persistence failed",
                extra={
                    "error_type": type(error).__name__,
                },
            )
        except Exception as error:
            logger.exception(
                "Unexpected automatic memory processing failure",
                extra={
                    "error_type": type(error).__name__,
                },
            )

    def _build_memory_context_message(
        self,
        user_input: str,
    ) -> dict[str, str] | None:
        relevant_memories = self._memory_store.retrieve_relevant_memories(
            self.user_id,
            user_input,
            self._memory_top_k,
        )

        if not relevant_memories:
            return None

        deduped_contents: list[str] = []
        seen: set[str] = set()

        for memory in relevant_memories:
            normalized = memory.content.strip()
            key = normalized.casefold()

            if not normalized or key in seen:
                continue

            seen.add(key)
            deduped_contents.append(normalized)

        if not deduped_contents:
            return None

        lines = "\n".join(
            f"- {content}"
            for content in deduped_contents
        )

        return {
            "role": "system",
            "content": f"{_MEMORY_CONTEXT_PREFIX}\n{lines}",
        }

    @staticmethod
    def _strip_memory_context_messages(
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        filtered: list[dict[str, object]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if (
                role == "system"
                and isinstance(content, str)
                and content.startswith(_MEMORY_CONTEXT_PREFIX)
            ):
                continue

            filtered.append(message)

        return filtered