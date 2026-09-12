"""ReasoningAgent — handles 'reasoning' and 'calculation' task types via LLM and safe evaluation."""

from __future__ import annotations

import re

from app.agent.contracts import (
    AgentCapabilities,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.state import AgentState
from app.exceptions import AgentExecutionError
from app.llm.models import ModelRequest, TaskType
from app.logger import get_logger
from app.tools.calculator import CalculatorTool

logger = get_logger(__name__)


class ReasoningAgent(BaseAgent):
    """General-purpose agent that handles reasoning and calculation steps.

    Equipped with tool_use and memory capabilities so it is eligible for any
    calculation or reasoning plan step.
    """

    def __init__(self, gateway=None) -> None:
        self._identity = AgentIdentity(
            name="reasoning_agent",
            version="1.0.0",
            description=(
                "General-purpose reasoning agent for logic, analysis, calculation, "
                "and conversational tasks."
            ),
        )
        self._capabilities = AgentCapabilities(
            tool_use=True,
            memory=True,
            multi_turn=True,
            retrieval=True,
            task_types=frozenset({"reasoning", "calculation"}),
        )
        self._gateway = gateway
        self._calculator = CalculatorTool()

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    def execute(self, request: AgentRequest) -> AgentResult:
        normalized_input = request.input_text.strip()
        if not normalized_input:
            raise AgentExecutionError(
                "Agent request input_text must not be empty.",
                request=request,
            )

        task_type = request.metadata.get("required_task_type", "reasoning")

        # Fast path for calculation if simple expression
        if task_type == "calculation":
            try:
                calc_result = self._calculator.execute(expression=normalized_input)
                state = AgentState(finished=True, final_answer=calc_result)
                return AgentResult(
                    request=request,
                    state=state,
                    output=calc_result,
                    success=True,
                    context=request.context,
                    error=None,
                )
            except Exception:
                # Fall back to LLM for complex/worded math problems
                pass

        # WebResearchAgent already performs source retrieval, candidate
        # verification, bounded replanning, and evidence-grounded formatting.
        # Preserve that verified artifact instead of asking a second LLM to
        # regenerate facts and potentially introduce unsupported claims.
        verified_artifact = self._extract_verified_research_artifact(normalized_input)
        if verified_artifact:
            state = AgentState(finished=True, final_answer=verified_artifact)
            return AgentResult(
                request=request,
                state=state,
                output=verified_artifact,
                success=True,
                context=request.context,
                error=None,
            )

        if self._gateway is None:
            raise AgentExecutionError(
                "ReasoningAgent has no LLM gateway configured.",
                request=request,
            )

        try:
            prompt_instruction = (
                "You are the AI Research Assistant, an intelligent agent built for research, reasoning, and analysis. "
                "Provide a clear, accurate, and direct answer. Always format your responses using clean, standard Markdown. "
                "When compiling research discoveries, preserve the discovery title, institution name, and source links [Source](url) "
                "from the provided research evidence, and ensure any requested sentence constraints (e.g., exactly two sentences for importance) "
                "are strictly preserved. For mathematical equations, use $$ for block math and $ for inline math. "
                "Avoid returning raw JSON."
            )
            if task_type == "calculation":
                prompt_instruction += " For mathematical questions, compute the exact result."

            llm_request = ModelRequest(
                messages=[
                    {"role": "system", "content": prompt_instruction},
                    {"role": "user", "content": normalized_input},
                ],
                task_type=TaskType.GENERAL,
            )
            response = self._gateway.generate(llm_request)
            output = response.content or ""
            success = bool(output.strip())
            error = None
        except Exception as e:
            logger.warning(
                "ReasoningAgent LLM call failed",
                extra={"error_type": type(e).__name__, "error": str(e)},
            )
            output = None
            success = False
            error = AgentExecutionError(
                "ReasoningAgent execution failed.",
                request=request,
                details={"error_type": type(e).__name__, "message": str(e)},
            )

        state = AgentState(finished=success, final_answer=output)

        return AgentResult(
            request=request,
            state=state,
            output=output,
            success=success,
            context=request.context,
            error=error,
        )

    @staticmethod
    def _extract_verified_research_artifact(text: str) -> str | None:
        """Return the strongest verified web-research artifact, if present."""
        matches = re.findall(
            r"(###\s+Major Scientific Discoveries in [^\n]+\s*\([^\n]+\)[\s\S]*?)(?=\n\[Research from step|\Z)",
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            no_result = re.search(
                r"No verified scientific discoveries meeting all criteria[^\n]*",
                text,
                flags=re.IGNORECASE,
            )
            return no_result.group(0).strip() if no_result else None
        return max(matches, key=lambda value: value.count("#### ")).strip()
