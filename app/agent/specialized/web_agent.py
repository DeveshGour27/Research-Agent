"""WebResearchAgent implementation with evidence verification and constraint enforcement."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.agent.contracts import (
    AgentCapabilities,
    AgentIdentity,
    AgentRequest,
    AgentResult,
    BaseAgent,
)
from app.agent.evidence import CandidateEvidence, EvidenceExtractor, EvidenceVerifier
from app.agent.state import AgentState
from app.exceptions import AgentExecutionError, ToolExecutionError
from app.llm.models import ModelRequest, TaskType
from app.logger import get_logger
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool
from app.tools.web_fetch import WebFetchTool

logger = get_logger(__name__)


class WebResearchAgent(BaseAgent):
    """Specialized agent for empirical web research with candidate verification."""

    def __init__(self, gateway=None) -> None:
        self._identity = AgentIdentity(
            name="web_research_agent",
            version="1.0.0",
            description="Specialized agent for performing external web research with claim-level evidence verification.",
        )
        self._capabilities = AgentCapabilities(
            tool_use=True,
            memory=False,
            multi_turn=False,
            retrieval=False,
            task_types=frozenset({"web_search"}),
        )
        self._tool_registry = ToolRegistry(include_mcp=False)
        self._tool_registry.register(WebSearchTool())
        self._fetch_tool = WebFetchTool()
        self._gateway = gateway
        self._query_cache: dict[str, list[dict[str, Any]]] = {}

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

        tool = self._tool_registry.get("web_search")

        is_discovery_task = bool(
            re.search(r"\b(?:discoveries|discovery|breakthroughs?)\b", normalized_input, re.I)
        )

        if not is_discovery_task:
            try:
                raw_output = tool.execute(query=normalized_input)
                if self._gateway and raw_output and raw_output != "No results found.":
                    prompt_instruction = (
                        "You are the AI Research Assistant. Given the user's query and the following raw web search results, "
                        "synthesize a natural, conversational, and informative answer. Cite your sources where appropriate. "
                        "Always format your responses using clean, standard Markdown. Avoid returning raw JSON."
                    )
                    llm_request = ModelRequest(
                        messages=[
                            {"role": "system", "content": prompt_instruction},
                            {"role": "user", "content": f"Query: {normalized_input}\n\nSearch Results: {raw_output}"},
                        ],
                        task_type=TaskType.GENERAL,
                    )
                    response = self._gateway.generate(llm_request)
                    output = response.content or raw_output
                else:
                    output = raw_output

                state = AgentState(finished=True, final_answer=output)
                return AgentResult(
                    request=request,
                    state=state,
                    output=output,
                    success=True,
                    context=request.context,
                )
            except ToolExecutionError as e:
                state = AgentState(finished=False, final_answer=None)
                return AgentResult(
                    request=request,
                    state=state,
                    output=None,
                    success=False,
                    context=request.context,
                    error=AgentExecutionError(
                        "Web search tool execution failed.",
                        request=request,
                        details={"error_message": str(e)},
                    ),
                )
            except Exception as e:
                state = AgentState(finished=False, final_answer=None)
                return AgentResult(
                    request=request,
                    state=state,
                    output=None,
                    success=False,
                    context=request.context,
                    error=AgentExecutionError(
                        "Unexpected error during web research.",
                        request=request,
                        details={"error_type": type(e).__name__, "message": str(e)},
                    ),
                )

        try:
            # ── 1. Parse Query Constraints ──────────────────────────────────
            year_constraint = self._extract_year(normalized_input) or "2026"
            count_constraint = self._extract_count(normalized_input) or 3
            domain_constraint = self._extract_domain(normalized_input)

            verifier = EvidenceVerifier(target_year=year_constraint, target_domain=domain_constraint)

            # ── 2. Determine Multi-Strategy Search Queries ───────────────────
            search_queries = self._generate_search_strategies(normalized_input, year_constraint, domain_constraint)
            logger.info(
                "WebResearchAgent executing multi-strategy search",
                extra={"query_count": len(search_queries), "strategies": search_queries},
            )

            # ── 3. Collect Raw Search Results with Pacing & Caching ──────────
            raw_items: list[dict[str, Any]] = []
            seen_urls: set[str] = set()

            for q in search_queries:
                cached = self._query_cache.get(q)
                if cached is not None:
                    items = cached
                else:
                    try:
                        time.sleep(0.05)  # Gentle pacing
                        raw_str = tool.execute(query=q)
                        if raw_str and raw_str != "No results found.":
                            items = json.loads(raw_str)
                        else:
                            items = []
                        self._query_cache[q] = items
                    except ToolExecutionError:
                        raise
                    except Exception as err:
                        logger.warning("Search query execution failed", extra={"query": q, "error": str(err)})
                        items = []

                for item in items:
                    url = item.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        raw_items.append(item)

            # ── 4. Candidate Extraction and Verification ─────────────────────
            verified_candidates: list[CandidateEvidence] = []
            seen_candidates: set[str] = set()

            for item in raw_items:
                candidate = EvidenceExtractor.extract_candidate(item)
                if verifier.verify_candidate(candidate):
                    cand_key = self._candidate_key(candidate)
                    if cand_key not in seen_candidates:
                        seen_candidates.add(cand_key)
                        verified_candidates.append(candidate)

            # Sort by evidence quality (primary sources and verified details first)
            verified_candidates.sort(key=lambda c: c.evidence_quality, reverse=True)

            # ── 5. Adaptive Research Loop (if count < requested) ─────────────
            if len(verified_candidates) < count_constraint:
                logger.info(
                    "Verified candidates below target count, executing targeted follow-up research",
                    extra={"verified": len(verified_candidates), "target": count_constraint},
                )
                follow_up_queries = self._generate_follow_up_strategies(
                    normalized_input, year_constraint, domain_constraint
                )
                for fq in follow_up_queries:
                    if len(verified_candidates) >= count_constraint:
                        break
                    if fq in self._query_cache:
                        continue
                    try:
                        time.sleep(0.3)
                        raw_str = tool.execute(query=fq)
                        if raw_str and raw_str != "No results found.":
                            items = json.loads(raw_str)
                            self._query_cache[fq] = items
                            for item in items:
                                url = item.get("url", "")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    cand = EvidenceExtractor.extract_candidate(item)
                                    if verifier.verify_candidate(cand):
                                        cand_key = self._candidate_key(cand)
                                        if cand_key not in seen_candidates:
                                            seen_candidates.add(cand_key)
                                            verified_candidates.append(cand)
                    except Exception as err:
                        logger.warning("Follow-up search failed", extra={"query": fq, "error": str(err)})

            # ── 5b. Deep Web Fetch for Primary Source Verification ───────────
            fetch_tool = self._tool_registry._tools.get("web_fetch") or self._fetch_tool
            for cand in verified_candidates[:count_constraint]:
                if cand.primary_source and cand.source_url.startswith("http"):
                    try:
                        fetched_text = fetch_tool.execute(url=cand.source_url)
                        if fetched_text and not fetched_text.startswith("Error") and not fetched_text.startswith("HTTP error") and not fetched_text.startswith("Connection error"):
                            lines = [line.strip() for line in fetched_text.split("\n\n") if line.strip()]
                            for line in lines:
                                if line.startswith("Title:") and len(line) > 15:
                                    clean_title = line.replace("Title:", "").strip()
                                    clean_title = re.sub(r"\s*[-|–]\s*(?:Newsroom|Press Release|Nature|Science).*$", "", clean_title, flags=re.I).strip()
                                    if clean_title:
                                        cand.title = clean_title
                                    break
                    except Exception as err:
                        logger.debug("Primary source deep fetch skipped", extra={"url": cand.source_url, "error": str(err)})

            # ── 6. Synthesis: Format Output Grounded Strictly in Evidence ─────
            output = self._synthesize_response(
                candidates=verified_candidates[:count_constraint],
                requested_count=count_constraint,
                target_year=year_constraint,
                verifier=verifier,
            )

            success = True
            error = None

        except ToolExecutionError as e:
            output = None
            success = False
            error = AgentExecutionError(
                "Web search tool execution failed.",
                request=request,
                details={"error_message": str(e)},
            )
        except Exception as e:
            output = None
            success = False
            error = AgentExecutionError(
                "Unexpected error during web research.",
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

    def _generate_search_strategies(self, query: str, year: str, domain: str = "quantum computing") -> list[str]:
        """Generate focused, high-precision search query angles."""
        if domain == "quantum computing":
            return [
                f"{year} quantum computing breakthrough press release",
                f"{year} quantum error correction discovery research",
                f"{year} quantum advantage demonstration university",
                f"{year} quantum processor hardware scaling announcement",
            ]
        return [
            f"{year} {domain} breakthrough press release",
            f"{year} {domain} discovery research university",
            f"{year} {domain} experimental demonstration",
            f"{year} {domain} major scientific breakthrough",
        ]

    def _generate_follow_up_strategies(self, query: str, year: str, domain: str = "quantum computing") -> list[str]:
        """Generate targeted follow-up queries when candidate count is below target."""
        if domain == "quantum computing":
            return [
                f"{year} neutral atom quantum processor result",
                f"{year} superconducting qubit logical qubit paper",
                f"{year} trapped ion quantum computer demonstration",
                f"{year} quantum error correction Nature Science paper",
                f"{year} quantum processor hardware scaling announcement",
            ]
        return [
            f"{year} {domain} research demonstration university",
            f"{year} {domain} Nature Science publication breakthrough",
            f"{year} {domain} lab experimental results paper",
            f"{year} {domain} breakthrough announcement",
        ]

    def _candidate_key(self, candidate: CandidateEvidence) -> str:
        """Unique key for deduplicating candidates while preserving distinct discoveries from same institution."""
        inst = (candidate.institution or "").strip().lower()
        title = (candidate.title or "").strip().lower()
        return f"{inst}:{title}"

    def _summary_supported(self, summary: str, candidate: CandidateEvidence) -> bool:
        """Check whether a generated significance summary is grounded in the candidate evidence."""
        if not summary or not summary.strip():
            return False

        summary_lower = summary.lower()
        unsupported_markers = [
            "dramatically transforms",
            "universal practical quantum advantage",
            "transforms all future",
            "proves universal",
            "revolutionizes all",
            "cures all",
            "infinite scalability",
            "solves all problems",
        ]
        for marker in unsupported_markers:
            if marker in summary_lower:
                return False

        corpus = " ".join([
            candidate.title or "",
            candidate.discovery_claim or "",
            candidate.importance_claim or "",
            " ".join(candidate.evidence) if candidate.evidence else "",
            " ".join(candidate.significance_evidence) if candidate.significance_evidence else "",
        ]).lower()

        stop_words = {
            "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "and", "or", "but", "if", "because",
            "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
            "it", "its", "the", "a", "an", "reports", "source", "result", "demonstration"
        }
        words = [w for w in re.findall(r"\b[a-z]{4,}\b", summary_lower) if w not in stop_words]
        if not words:
            return True

        overlap = sum(1 for w in words if w in corpus)
        return (overlap / len(words)) >= 0.4

    def _synthesize_response(
        self,
        candidates: list[CandidateEvidence],
        requested_count: int,
        target_year: str,
        verifier: EvidenceVerifier,
    ) -> str:
        """Synthesize verified discoveries into clean Markdown without hallucinating."""
        domain_title = verifier.target_domain.title()
        if not candidates:
            return (
                f"No verified scientific discoveries meeting all criteria ({verifier.target_domain} domain, "
                f"{target_year} publication, attributed institution, and empirical demonstration) could be "
                f"confirmed from primary sources."
            )

        lines: list[str] = [
            f"### Major Scientific Discoveries in {domain_title} ({target_year})\n"
        ]

        for i, c in enumerate(candidates, 1):
            # Generate 2-sentence significance summary grounded strictly in snippet
            two_sentences = self._generate_two_sentence_summary(c, verifier)

            lines.append(f"#### {i}. {c.title}")
            lines.append(f"- **Institution:** {c.institution}")
            if c.publication_date:
                lines.append(f"- **Date:** {c.publication_date}")
            lines.append(f"- **Why It Matters:** {two_sentences}")
            lines.append(f"- **Source:** [{c.institution} Announcement]({c.source_url})\n")

        if len(candidates) < requested_count:
            lines.append(
                f"> **Research Completeness Notice:** Exactly {len(candidates)} discoveries could be fully "
                f"verified against primary sources and peer-reviewed press releases for {target_year}. "
                f"Additional candidates from general commentary or unverified reports were excluded to maintain scientific accuracy."
            )

        return "\n".join(lines).strip()

    def _generate_two_sentence_summary(self, candidate: CandidateEvidence, verifier: EvidenceVerifier) -> str:
        """Generate or format a concise significance summary constrained to exactly two sentences."""
        if self._gateway:
            try:
                prompt = (
                    "Write a concise summary explaining why this scientific discovery matters.\n\n"
                    "STRICT REQUIREMENTS:\n"
                    "1. Your summary MUST consist of EXACTLY TWO SENTENCES. No more, no less.\n"
                    "2. Base your explanation strictly on the facts in the snippet. Do NOT invent details.\n"
                    "3. Return ONLY the two sentences as plain text.\n\n"
                    f"Discovery: {candidate.title}\n"
                    f"Institution: {candidate.institution}\n"
                    f"Evidence: {candidate.discovery_claim}"
                )
                req = ModelRequest(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    task_type=TaskType.GENERAL,
                )
                resp = self._gateway.generate(req)
                content = (resp.content or "").strip()
                if content and self._summary_supported(content, candidate):
                    return verifier.enforce_two_sentences(content)
            except Exception as e:
                logger.warning("LLM 2-sentence generation failed, using rule-based", extra={"error": str(e)})

        # Fallback to rule-based two-sentence enforcement
        base_claim = candidate.importance_claim or candidate.discovery_claim
        base_claim = base_claim.replace("...", "").strip()
        return verifier.enforce_two_sentences(base_claim)

    @staticmethod
    def _extract_year(text: str) -> str | None:
        match = re.search(r"\b(20\d{2})\b", text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_count(text: str) -> int | None:
        match = re.search(r"\b(\d+)\s+(?:major|scientific|breakthrough|discoveries|items)", text, re.I)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        # Check for worded numbers
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        for word, val in word_map.items():
            if re.search(rf"\b{word}\s+(?:major|scientific|breakthrough|discoveries)", text, re.I):
                return val
        return None

    @staticmethod
    def _extract_domain(text: str) -> str:
        """Extract the research domain from the input text, defaulting to quantum computing if detected."""
        text_lower = text.lower()
        if "quantum" in text_lower or "qubit" in text_lower:
            return "quantum computing"

        domains = [
            "artificial intelligence", "machine learning", "robotics",
            "crispr", "gene editing", "genetics", "biotechnology",
            "materials science", "superconductivity", "nanotechnology",
            "astrophysics", "astronomy", "cosmology", "fusion energy",
            "renewable energy", "climate science", "neuroscience",
        ]
        for d in domains:
            if d in text_lower:
                return d

        cleaned = re.sub(
            r"\b(find|what\s+are|the|major|scientific|discoveries|discovery|breakthroughs|breakthrough|in|of|for|recent|latest|\d{4})\b",
            " ",
            text,
            flags=re.I,
        ).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned or "scientific research"
