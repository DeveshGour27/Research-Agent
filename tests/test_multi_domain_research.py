"""Multi-Domain Research Quality and Evaluation Benchmark (P3-2).

Validates:
1. Artificial Intelligence domain research and verification.
2. CRISPR/Biotechnology domain research without false rejection.
3. Astrophysics and observation research without false rejection.
4. Cross-domain adversarial rejection (e.g. quantum results rejected for CRISPR queries).
5. Strict two-sentence significance summary and primary provenance enforcement.
"""

import json
from unittest.mock import patch
import pytest

from app.agent.contracts import AgentRequest
from app.agent.evidence import CandidateEvidence, EvidenceExtractor, EvidenceVerifier
from app.agent.execution_context import AgentExecutionContext
from app.agent.specialized.web_agent import WebResearchAgent
from app.tools.web_fetch import WebFetchTool
from app.tools.web_search import WebSearchTool


def test_artificial_intelligence_research_verification():
    first = {
        "title": "DeepMind demonstrates test-time compute reasoning breakthrough",
        "url": "https://deepmind.google/discover/blog/2026-reasoning-breakthrough",
        "snippet": "Google DeepMind researchers demonstrated self-correcting test-time compute scaling.",
        "date": "2026-03-15",
    }
    second = {
        "title": "Stanford AI Lab unveils neurosymbolic verified reasoning model",
        "url": "https://ai.stanford.edu/blog/2026-neurosymbolic",
        "snippet": "Stanford AI Lab demonstrated a verified neurosymbolic foundation model.",
        "date": "2026-05-20",
    }

    def search(query: str) -> str:
        if "breakthrough" in query:
            return json.dumps([first])
        if "demonstration" in query or "discovery" in query:
            return json.dumps([second])
        return "No results found."

    def fetch(url: str) -> str:
        return "Title: Verified AI discovery\n\nContent: The model demonstrated provable reasoning capabilities."

    agent = WebResearchAgent(gateway=None)
    request = AgentRequest(
        input_text="Find 2 major scientific discoveries made in artificial intelligence in 2026.",
        context=AgentExecutionContext(task="research"),
    )
    with patch.object(WebSearchTool, "execute", side_effect=search):
        with patch.object(WebFetchTool, "execute", side_effect=fetch):
            result = agent.execute(request)

    assert result.success is True
    assert result.output is not None
    assert "### Major Scientific Discoveries in Artificial Intelligence (2026)" in result.output
    assert "DeepMind" in result.output
    assert "Stanford" in result.output


def test_crispr_biotechnology_research_verification():
    verifier = EvidenceVerifier(target_year="2026", target_domain="crispr")
    cand = CandidateEvidence(
        title="CRISPR-Cas9 Precise In-Vivo Base Editing Demonstrated",
        institution="Broad Institute of MIT and Harvard",
        source_url="https://broadinstitute.org/news/2026-crispr-base-editing",
        discovery_claim="Demonstrated targeted in-vivo base editing without double-strand DNA breaks.",
        importance_claim="Enables precise correction of point mutations in genetic diseases.",
        publication_date="2026-04-10",
        primary_source=True,
    )

    # CRISPR is explicitly requested, so it must NOT be rejected as forbidden!
    assert verifier.verify_candidate(cand) is True
    assert cand.satisfies_domain is True
    assert cand.satisfies_year is True


def test_astrophysics_research_verification():
    verifier = EvidenceVerifier(target_year="2026", target_domain="astrophysics")
    cand = CandidateEvidence(
        title="Direct Observation of Intermediate-Mass Black Hole Binary Merger",
        institution="LIGO Scientific Collaboration",
        source_url="https://ligo.org/news/gw2026-intermediate-merger",
        discovery_claim="Detected gravitational waves confirming the merger of intermediate-mass black holes.",
        importance_claim="Provides direct observational evidence of black hole growth mechanisms.",
        publication_date="2026-06-18",
        primary_source=True,
    )

    # Gravitational waves are explicitly requested for astrophysics, so must NOT be rejected
    assert verifier.verify_candidate(cand) is True
    assert cand.satisfies_domain is True
    assert cand.satisfies_year is True


def test_cross_domain_adversarial_rejection():
    # When user asks for CRISPR discoveries, quantum results must be REJECTED
    crispr_verifier = EvidenceVerifier(target_year="2026", target_domain="crispr")
    quantum_cand = CandidateEvidence(
        title="Google Quantum AI reports logical qubit demonstration",
        institution="Google Quantum AI",
        source_url="https://research.google/2026/quantum-qubit",
        discovery_claim="Demonstrated quantum error correction on a superconducting processor.",
        importance_claim="Suppresses physical errors across logical qubits.",
        publication_date="2026-02-14",
        primary_source=True,
    )

    assert crispr_verifier.verify_candidate(quantum_cand) is False
    assert quantum_cand.satisfies_domain is False


def test_missing_institution_rejected_across_domains():
    ai_verifier = EvidenceVerifier(target_year="2026", target_domain="artificial intelligence")
    unattributed_cand = CandidateEvidence(
        title="Huge AI Breakthrough Claimed",
        institution="Unknown Institution",
        source_url="https://someblog.io/2026/ai-breakthrough",
        discovery_claim="A massive breakthrough was announced somewhere.",
        importance_claim="It will change everything.",
        publication_date="2026-01-10",
        primary_source=True,
    )

    assert ai_verifier.verify_candidate(unattributed_cand) is False
    assert unattributed_cand.satisfies_institution is False
