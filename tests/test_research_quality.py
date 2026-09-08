"""Regression test suite for research quality audit: Tests A through J.

Verifies:
- Test A: CRISPR domain rejection
- Test B: Gravitational waves domain rejection
- Test C: Higgs boson domain rejection
- Test D: 2025 discovery rejection (year constraint)
- Test E: Missing source verification rejection
- Test F: Quantity handling (does not invent missing candidates)
- Test G: Multi-strategy / replanning when candidates < requested
- Test H: Synthesis provenance (claims strictly derived from evidence)
- Test I: Exactly two sentences for significance summary
- Test J: Institution verification and attribution
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from app.agent.evidence import CandidateEvidence, EvidenceExtractor, EvidenceVerifier
from app.agent.specialized.web_agent import WebResearchAgent
from app.agent.specialized.reasoning_agent import ReasoningAgent
from app.tools.web_fetch import WebFetchTool
from app.tools.web_search import WebSearchTool
from app.agent.contracts import AgentRequest
from app.agent.execution_context import AgentExecutionContext


@pytest.fixture
def verifier() -> EvidenceVerifier:
    return EvidenceVerifier(target_year="2026", target_domain="quantum computing")


# ── Test A: CRISPR Domain Rejection ───────────────────────────────────────────
def test_a_crispr_domain_rejection(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="CRISPR-Cas9 Base Editing Breakthrough",
        institution="Broad Institute of MIT and Harvard",
        source_url="https://nature.com/articles/crispr2026",
        discovery_claim="CRISPR-Cas9 provides a versatile method for editing DNA in living cells.",
        importance_claim="Accelerates research into human therapeutics.",
        publication_date="2026-03-01",
    )
    assert not verifier.verify_candidate(candidate)
    assert candidate.satisfies_domain is False
    assert any("crispr" in r.lower() or "forbidden" in r.lower() for r in candidate.rejection_reasons)


# ── Test B: Gravitational Waves Domain Rejection ──────────────────────────────
def test_b_gravitational_waves_domain_rejection(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="Direct Detection of Gravitational Waves from Binary Merger",
        institution="LIGO Scientific Collaboration",
        source_url="https://ligo.org/news/gw2026",
        discovery_claim="First direct observation of gravitational waves from colliding neutron stars.",
        importance_claim="Validates predictions of Einstein's general relativity.",
        publication_date="2026-05-12",
    )
    assert not verifier.verify_candidate(candidate)
    assert candidate.satisfies_domain is False
    assert any("gravitational wave" in r.lower() or "forbidden" in r.lower() for r in candidate.rejection_reasons)


# ── Test C: Higgs Boson Domain Rejection ───────────────────────────────────────
def test_c_higgs_boson_domain_rejection(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="Discovery of the Higgs Boson Particle",
        institution="CERN (ATLAS and CMS experiments)",
        source_url="https://cern.ch/press/higgs",
        discovery_claim="Observed resonance confirming the existence of the Higgs scalar boson.",
        importance_claim="Completes the Standard Model of particle physics.",
        publication_date="2026-07-04",
    )
    assert not verifier.verify_candidate(candidate)
    assert candidate.satisfies_domain is False
    assert any("higgs" in r.lower() or "forbidden" in r.lower() for r in candidate.rejection_reasons)


# ── Test D: Year Filtering (2025 Rejected for 2026 Query) ──────────────────────
def test_d_year_filtering_rejects_past_years(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="Quantum error correction below surface code threshold",
        institution="Google Quantum AI",
        source_url="https://nature.com/articles/s41586-024-08449-y",
        discovery_claim="Demonstrated quantum error correction below surface code threshold on Willow processor.",
        importance_claim="Demonstrates that physical error rates are suppressed exponentially.",
        publication_date="2025-06-10",
    )
    assert not verifier.verify_candidate(candidate)
    assert candidate.satisfies_year is False
    assert any("target year" in r.lower() for r in candidate.rejection_reasons)


# ── Test E: Missing Source URL Rejected ───────────────────────────────────────
def test_e_missing_source_url_rejected(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="Topological Qubit Majorana Zero Modes Demonstrated",
        institution="Microsoft Quantum",
        source_url="",  # Missing URL
        discovery_claim="Created hardware-protected topological qubits in semiconductor nanowires.",
        importance_claim="Enables intrinsic error protection.",
        publication_date="2026-08-20",
    )
    assert not verifier.verify_candidate(candidate)
    assert any("source url" in r.lower() for r in candidate.rejection_reasons)


# ── Test F: Quantity Handling (Refuses to Invent Missing Candidates) ───────────
def test_f_quantity_handling_no_fabrication(verifier: EvidenceVerifier) -> None:
    agent = WebResearchAgent(gateway=None)
    # Only 1 candidate verified
    c1 = CandidateEvidence(
        title="Quantum Advantage on Logical Circuits",
        institution="University of Chicago (in partnership with IBM)",
        source_url="https://newsroom.ibm.com/2026-07-30-ibm-chicago",
        discovery_claim="Demonstrated quantum advantage encoding 70 logical qubits.",
        importance_claim="Solves classically intractable problem.",
        publication_date="30 Jul 2026",
        primary_source=True,
    )
    assert verifier.verify_candidate(c1)

    # Synthesize with requested count = 3 but only 1 verified
    output = agent._synthesize_response(
        candidates=[c1],
        requested_count=3,
        target_year="2026",
        verifier=verifier,
    )

    # Must include the 1 verified discovery
    assert "University of Chicago" in output
    assert "70 logical qubits" in output or "Quantum Advantage" in output
    # Must NOT invent second or third discovery headings
    assert "#### 2." not in output
    assert "#### 3." not in output
    # Must explicitly state that only 1 was verified
    assert "1 discoveries could be fully verified" in output or "Research Completeness Notice" in output


# ── Test G: Replanning Triggered When Candidates Insufficient ──────────────────
def test_g_replanning_triggers_on_insufficient_candidates() -> None:
    agent = WebResearchAgent(gateway=None)
    strategies = agent._generate_search_strategies("Find 3 major quantum computing discoveries in 2026", "2026")
    # Must generate multiple distinct search query angles
    assert len(strategies) >= 3
    assert any("press release" in s for s in strategies)
    assert any("error correction" in s for s in strategies)
    assert any("demonstration" in s for s in strategies)


# ── Test H: Synthesis Provenance (Grounding in Verified Evidence) ──────────────
def test_h_synthesis_provenance(verifier: EvidenceVerifier) -> None:
    agent = WebResearchAgent(gateway=None)
    c1 = CandidateEvidence(
        title="D-Wave Demonstrates Major Hardware Breakthrough for QEC",
        institution="D-Wave Quantum Inc.",
        source_url="https://dwavequantum.com/press/2026-qec",
        discovery_claim="Demonstrated a major hardware breakthrough for flux-qubit quantum error correction.",
        importance_claim="Improves coherence across physical qubits.",
        publication_date="5 Aug 2026",
        primary_source=True,
    )
    assert verifier.verify_candidate(c1)

    output = agent._synthesize_response(
        candidates=[c1],
        requested_count=1,
        target_year="2026",
        verifier=verifier,
    )
    # Output must link to the exact provenance URL
    assert "https://dwavequantum.com/press/2026-qec" in output
    assert "D-Wave Quantum Inc." in output


# ── Test I: Exactly Two Sentences Requirement ─────────────────────────────────
def test_i_exactly_two_sentences_enforced(verifier: EvidenceVerifier) -> None:
    # 1-sentence text should be expanded to exactly 2
    one_sentence = "Demonstrated logical qubit operation below threshold."
    res1 = verifier.enforce_two_sentences(one_sentence)
    assert verifier.count_sentences(res1) == 2

    # 3-sentence text should be trimmed to exactly 2
    three_sentences = "First sentence is here. Second sentence explains why it matters. Third sentence is superfluous."
    res2 = verifier.enforce_two_sentences(three_sentences)
    assert verifier.count_sentences(res2) == 2
    assert "Third sentence" not in res2


# ── Test J: Institution Verification and Attribution ──────────────────────────
def test_j_institution_verification(verifier: EvidenceVerifier) -> None:
    item_with_institution = {
        "title": "IBM and The University of Chicago Demonstrate Quantum Advantage",
        "url": "https://newsroom.ibm.com/2026-07-30-ibm-and-the-university-of-chicago",
        "snippet": "Scientists used a new error correction method to encode 70 logical qubits.",
        "date": "30 Jul 2026",
    }
    cand = EvidenceExtractor.extract_candidate(item_with_institution)
    assert "University of Chicago" in cand.institution
    assert verifier.verify_candidate(cand)

    item_unknown_institution = {
        "title": "Quantum Computing Breakthrough Claimed",
        "url": "https://randomblog.com/post123",
        "snippet": "Someone claimed a quantum computing breakthrough somewhere.",
        "date": "2026-01-01",
    }
    cand_unknown = CandidateEvidence(
        title="Quantum Computing Breakthrough Claimed",
        institution="Unknown Institution",
        source_url="https://randomblog.com/post123",
        discovery_claim="Someone claimed a quantum computing breakthrough.",
        importance_claim="",
        publication_date="2026-01-01",
    )
    assert not verifier.verify_candidate(cand_unknown)
    assert cand_unknown.satisfies_institution is False


def test_secondary_publisher_cannot_be_institution_or_primary_source(verifier: EvidenceVerifier) -> None:
    item = {
        "title": "What is the biggest quantum computing breakthrough in 2026?",
        "url": "https://bqpsim.com/blog/quantum-breakthroughs",
        "snippet": "A blog roundup discusses several quantum computing developments.",
        "date": "2026-04-01",
    }
    candidate = EvidenceExtractor.extract_candidate(item)
    assert candidate.institution == "Unknown Institution"
    assert candidate.article_date == "2026-04-01"
    assert not verifier.verify_candidate(candidate)


def test_secondary_2026_article_about_2025_work_is_rejected(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="2026 report on a quantum computing result",
        institution="University of Example",
        source_url="https://sciencedaily.com/releases/2026/01/report.html",
        discovery_claim="The article discusses a quantum processor paper published in 2025.",
        importance_claim="The paper reports an experimental result.",
        article_date="2026-01-02",
        work_date="2025-11-15",
        evidence=["The underlying paper date is 2025-11-15."],
        significance_evidence=["The paper reports an experimental result."],
        primary_source_url="https://doi.org/10/example",
    )
    assert not verifier.verify_candidate(candidate)
    assert candidate.satisfies_year is False


def test_candidate_without_supporting_evidence_is_rejected(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="Quantum processor breakthrough",
        institution="University of Example",
        source_url="https://example.edu/news/2026-breakthrough",
        discovery_claim="A quantum processor was improved.",
        importance_claim="It matters.",
        work_date="2026-02-01",
        primary_source=True,
    )
    # A primary candidate with an extracted claim is acceptable evidence; an
    # explicit empty claim is not silently filled by the verifier.
    candidate.discovery_claim = ""
    candidate.evidence = []
    assert not verifier.verify_candidate(candidate)


def test_primary_source_content_can_overrule_secondary_article_date(verifier: EvidenceVerifier) -> None:
    candidate = CandidateEvidence(
        title="Quantum error-correction result",
        institution="University of Example",
        source_url="https://example.edu/news/2026-announcement",
        discovery_claim="Quantum computing researchers report a result.",
        importance_claim="The result improves error correction.",
        work_date="2026-06-01",
        article_date="2026-06-02",
        primary_source=True,
        evidence=["Quantum computing researchers report a result."],
        significance_evidence=["The result improves error correction."],
    )
    enriched = EvidenceExtractor.enrich_from_source(
        candidate,
        "Title: Quantum error-correction result\n\nContent: The paper was published in 2025. Quantum computing researchers report a result.",
    )
    assert enriched.work_date == "2025"
    assert not verifier.verify_candidate(enriched)


def test_same_institution_different_discoveries_are_not_collapsed() -> None:
    agent = WebResearchAgent(gateway=None)
    first = CandidateEvidence(
        title="Logical qubit experiment A",
        institution="University of Example",
        source_url="https://example.edu/a",
        discovery_claim="Quantum computing qubit experiment A.",
        importance_claim="Improves logical qubit control.",
        work_date="2026",
        primary_source=True,
    )
    second = CandidateEvidence(
        title="Logical qubit experiment B",
        institution="University of Example",
        source_url="https://example.edu/b",
        discovery_claim="Quantum computing qubit experiment B.",
        importance_claim="Improves logical qubit readout.",
        work_date="2026",
        primary_source=True,
    )
    verifier = EvidenceVerifier()
    assert verifier.verify_candidate(first)
    assert verifier.verify_candidate(second)
    assert agent._candidate_key(first) != agent._candidate_key(second)


def test_unsupported_significance_summary_is_rejected() -> None:
    agent = WebResearchAgent(gateway=None)
    candidate = CandidateEvidence(
        title="Logical qubit experiment",
        institution="University of Example",
        source_url="https://example.edu/a",
        discovery_claim="Quantum computing researchers demonstrated a logical qubit.",
        importance_claim="The source reports a logical-qubit demonstration.",
        work_date="2026",
        primary_source=True,
    )
    assert not agent._summary_supported(
        "This dramatically transforms all future computing. It proves universal practical quantum advantage.",
        candidate,
    )


def test_reasoning_agent_preserves_verified_web_artifact() -> None:
    artifact = (
        "### Major Scientific Discoveries in Quantum Computing (2026)\n\n"
        "#### 1. Verified result\n- **Institution:** University of Example\n"
        "- **Why It Matters:** A reported result. The source supports this result."
    )
    text = "Use ONLY the following verified research evidence.\n[Research from step 1]:\n" + artifact
    assert ReasoningAgent._extract_verified_research_artifact(text) == artifact
    assert ReasoningAgent._extract_verified_research_artifact(
        "[Research from step 1]: No verified scientific discoveries meeting all criteria could be confirmed from primary sources."
    ).startswith("No verified scientific discoveries")


def test_end_to_end_research_replans_and_returns_only_verified_candidates() -> None:
    first = {
        "title": "IBM demonstrates a quantum computing logical-qubit result",
        "url": "https://newsroom.ibm.com/2026/logical-qubit",
        "snippet": "IBM Quantum demonstrated a new quantum computing logical qubit result.",
        "date": "2026-06-01",
    }
    second = {
        "title": "Google Quantum AI reports a quantum processor result",
        "url": "https://research.google/2026/processor",
        "snippet": "Google Quantum AI demonstrated a quantum processor experiment.",
        "date": "2026-07-01",
    }
    third = {
        "title": "University of Sydney reports quantum error correction research",
        "url": "https://research.sydney.edu.au/2026/error-correction",
        "snippet": "University of Sydney researchers demonstrated quantum error correction.",
        "date": "2026-08-01",
    }

    def search(query: str) -> str:
        if "breakthrough press release" in query:
            return json.dumps([first])
        if "neutral atom" in query:
            return json.dumps([second])
        if "superconducting" in query:
            return json.dumps([third])
        return "No results found."

    def fetch(url: str) -> str:
        return (
            "Title: Verified research result\n\nContent: Researchers demonstrated a new quantum computing result. "
            "The experiment improves quantum error correction and supports scalable research."
        )

    agent = WebResearchAgent(gateway=None)
    request = AgentRequest(
        input_text="Find 3 major scientific discoveries made in quantum computing in 2026.",
        context=AgentExecutionContext(task="research"),
    )
    with patch.object(WebSearchTool, "execute", side_effect=search) as search_mock:
        with patch.object(WebFetchTool, "execute", side_effect=fetch):
            result = agent.execute(request)

    assert result.success is True
    assert result.output is not None
    assert "#### 1." in result.output and "#### 2." in result.output and "#### 3." in result.output
    assert "University of Sydney" in result.output
    queried = [call.kwargs["query"] for call in search_mock.call_args_list]
    assert any("neutral atom" in query for query in queried)
    assert any("superconducting" in query for query in queried)


# ── Test K: WebFetchTool Primary-Source Deep Content Extraction ───────────────
def test_k_web_fetch_tool_content_extraction() -> None:
    from app.tools.web_fetch import WebFetchTool
    tool = WebFetchTool()
    
    mock_html = (
        "<html><head><title>Quantum Breakthrough - University Lab</title></head>"
        "<body><nav>Menu</nav><p>Researchers at the university laboratory have achieved fault-tolerant logical qubit operations.</p>"
        "<p>The empirical demonstration shows an exponential decline in error rates as code distance increases.</p>"
        "<footer>Copyright 2026</footer></body></html>"
    ).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_html
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        output = tool.execute(url="https://lab.university.edu/2026-breakthrough")

    assert "Title: Quantum Breakthrough - University Lab" in output
    assert "fault-tolerant logical qubit operations" in output
    assert "Menu" not in output
    assert "Copyright 2026" not in output

