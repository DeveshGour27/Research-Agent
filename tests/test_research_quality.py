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
from unittest.mock import MagicMock, patch

from app.agent.evidence import CandidateEvidence, EvidenceExtractor, EvidenceVerifier
from app.agent.specialized.web_agent import WebResearchAgent
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

