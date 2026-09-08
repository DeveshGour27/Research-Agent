"""Candidate evidence model, verification, and constraint enforcement.

Responsibilities:
- Structured representation of candidate research discoveries.
- Claim-level and requirement-level verification against query constraints.
- Strict rejection of out-of-domain topics (CRISPR, gravitational waves, etc.).
- Strict rejection of incorrect publication years and non-discovery content.
- Strict rejection of low-quality/aggregator sources (social media, listicles).
- Scoring and ranking candidate evidence based on source quality and empirical claims.
- Validation of exact 2-sentence requirement for significance summaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from app.logger import get_logger

logger = get_logger(__name__)

# Disallowed domains / topics that must be rejected immediately for quantum computing
FORBIDDEN_TOPICS: frozenset[str] = frozenset({
    "crispr", "cas9", "gene editing", "genetics", "dna", "mrna", "biology",
    "gravitational wave", "ligo", "virgo", "black hole merger", "neutron star",
    "higgs boson", "cern", "lhc", "large hadron collider", "standard model of particle physics",
    "astronomy", "astrophysics", "exoplanet", "james webb", "cosmology",
})

# Keywords indicating quantum computing domain relevance
QUANTUM_COMPUTING_KEYWORDS: frozenset[str] = frozenset({
    "quantum computing", "quantum computer", "quantum processor", "qubit",
    "logical qubit", "fault-tolerant quantum", "quantum error correction",
    "qec", "quantum advantage", "quantum supremacy", "surface code",
    "topological qubit", "neutral atom", "superconducting qubit",
    "ion trap", "photonic quantum", "quantum algorithm", "majorana",
    "qubit coherence", "quantum chip",
})

# Content types to reject as non-discoveries (marketing, policy, listicles, roadmaps, funding)
NON_DISCOVERY_INDICATORS: frozenset[str] = frozenset({
    "investor guide", "leading countries", "top 25", "top 15", "top 10",
    "countries in 2026", "market size", "market forecast", "stocks to buy",
    "to lead major federally funded institute", "funding opportunity",
    "executive order", "consortium", "where we stand", "status report",
    "readiness index", "symposium schedule", "call for proposals",
    "schedule", "program", "agenda", "conference", "$10 billion",
    "market monitor", "patent landscape", "commercial tipping point",
})

# Disallowed aggregator/social domains that cannot serve as authoritative evidence
DISALLOWED_DOMAINS: frozenset[str] = frozenset({
    "linkedin.com", "medium.com", "youtube.com", "twitter.com", "x.com",
    "reddit.com", "mckinsey.com", "ey.com", "patsnap.com", "tiktok.com",
    "facebook.com", "instagram.com",
    "sciencedaily.com", "phys.org", "bqpsim.com", "forbes.com", "techcrunch.com",
    "interestingengineering.com", "quantumzeitgeist.com",
})

# Extended domain vocabularies for multi-domain verification
DOMAIN_VOCABULARY: dict[str, set[str]] = {
    "artificial intelligence": {
        "artificial intelligence", "ai", "machine learning", "deep learning", "neural network",
        "foundation model", "language model", "llm", "reasoning model", "transformer",
        "deepmind", "openai", "reinforcement learning", "computer vision", "nlp",
        "generative ai", "diffusion model", "compute", "neurosymbolic", "scaling",
    },
    "machine learning": {
        "artificial intelligence", "ai", "machine learning", "deep learning", "neural network",
        "foundation model", "language model", "llm", "reasoning model", "transformer",
        "deepmind", "openai", "reinforcement learning", "computer vision", "nlp",
    },
    "astrophysics": {
        "astrophysics", "astronomy", "black hole", "gravitational wave", "gravitational waves",
        "neutron star", "galaxy", "galaxies", "cosmology", "cosmic", "supernova", "exoplanet",
        "telescope", "ligo", "virgo", "kagra", "jwst", "hubble", "pulsar", "quasar",
        "dark matter", "dark energy", "stellar", "interstellar", "merger",
    },
    "astronomy": {
        "astrophysics", "astronomy", "black hole", "gravitational wave", "gravitational waves",
        "neutron star", "galaxy", "galaxies", "cosmology", "cosmic", "supernova", "exoplanet",
        "telescope", "ligo", "virgo", "kagra", "jwst", "hubble", "pulsar", "quasar",
    },
    "cosmology": {
        "cosmology", "cosmic", "big bang", "dark energy", "dark matter", "inflation",
        "expansion", "astrophysics", "astronomy", "redshift", "galaxy", "black hole",
    },
    "crispr": {
        "crispr", "cas9", "cas12", "cas13", "base editing", "prime editing", "gene editing",
        "genome editing", "genetic", "dna", "rna", "mutation", "mutations", "genomic",
    },
    "gene editing": {
        "crispr", "cas9", "base editing", "prime editing", "gene editing",
        "genome editing", "genetic", "dna", "rna", "mutation", "mutations",
    },
    "biotechnology": {
        "biotechnology", "biotech", "gene", "genetic", "crispr", "genomics", "proteomics",
        "synthetic biology", "base editing", "mrna", "therapeutic", "dna",
    },
    "materials science": {
        "materials science", "material", "superconductor", "superconducting", "graphene",
        "nanomaterial", "crystal", "semiconductor", "polymer", "alloy",
    },
    "neuroscience": {
        "neuroscience", "neural", "brain", "neuron", "synapse", "cortex", "cognitive",
    },
    "climate science": {
        "climate", "carbon", "greenhouse", "emissions", "warming", "atmosphere", "oceanic",
    },
    "fusion energy": {
        "fusion", "tokamak", "stellarator", "plasma", "ignition", "iter", "confinement",
    },
}

# Empirical significance markers across scientific domains
SIGNIFICANCE_MARKERS: tuple[str, ...] = (
    "new", "novel", "first", "improv", "reduc", "increase", "scale", "advantage",
    "threshold", "coherence", "error rate", "logical qubit", "classically", "practical",
    "advance", "solve", "demonstrat", "achiev", "enabl", "provid", "confirm", "observ",
    "detect", "breakthrough", "precis", "discover", "unveil", "uncover", "foundat",
    "allow", "transform", "evidence", "support", "capabilit", "potential", "reveal",
    "insight", "promis", "correct", "mechanism",
)


@dataclass
class CandidateEvidence:
    """Structured representation of an extracted research candidate."""

    title: str
    institution: str
    source_url: str
    discovery_claim: str
    importance_claim: str
    publication_date: str | None = None
    article_date: str | None = None
    work_date: str | None = None
    primary_source_url: str | None = None
    collaborators: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    significance_evidence: list[str] = field(default_factory=list)
    institution_evidence: list[str] = field(default_factory=list)
    claim_evidence: dict[str, list[str]] = field(default_factory=dict)
    source_fetched: bool = False
    source_type: str = "web"
    primary_source: bool = False
    evidence_quality: float = 0.0
    satisfies_year: bool = False
    satisfies_domain: bool = False
    satisfies_institution: bool = False
    satisfies_discovery_requirement: bool = False
    satisfies_significance: bool = False
    verified: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Backward-compatible construction for callers that already designate
        # an authoritative source: its extracted claim is the initial evidence
        # record. Secondary candidates must provide evidence explicitly.
        if not self.evidence and self.discovery_claim.strip():
            self.evidence = [self.discovery_claim.strip()]
        if not self.significance_evidence and self.importance_claim.strip():
            self.significance_evidence = [self.importance_claim.strip()]
        elif not self.significance_evidence and self.discovery_claim.strip():
            self.significance_evidence = [self.discovery_claim.strip()]
        if not self.claim_evidence and (self.evidence or self.significance_evidence):
            self.claim_evidence = {
                "discovery": list(self.evidence),
                "significance": list(self.significance_evidence),
            }
        publisher_like = {
            "unknown", "various", "multiple", "n/a", "none", "unknown institution",
            "sciencedaily", "phys.org", "bqpsim"
        }
        if (
            not self.institution_evidence
            and self.institution.strip()
            and self.institution.lower() not in publisher_like
        ):
            self.institution_evidence = [self.source_url or self.institution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "institution": self.institution,
            "source_url": self.source_url,
            "discovery_claim": self.discovery_claim,
            "importance_claim": self.importance_claim,
            "publication_date": self.publication_date,
            "article_date": self.article_date,
            "work_date": self.work_date,
            "primary_source_url": self.primary_source_url,
            "collaborators": self.collaborators,
            "source_type": self.source_type,
            "evidence": self.evidence,
            "significance_evidence": self.significance_evidence,
            "institution_evidence": self.institution_evidence,
            "claim_evidence": self.claim_evidence,
            "source_fetched": self.source_fetched,
            "primary_source": self.primary_source,
            "primary_source": self.primary_source,
            "evidence_quality": round(self.evidence_quality, 2),
            "verified": self.verified,
            "satisfies_significance": self.satisfies_significance,
            "rejection_reasons": self.rejection_reasons,
        }


class EvidenceVerifier:
    """Deterministic verifier that tests research candidates against query constraints."""

    def __init__(self, target_year: str = "2026", target_domain: str = "quantum computing") -> None:
        self.target_year = target_year
        self.target_domain = target_domain.lower()

    def verify_candidate(self, candidate: CandidateEvidence) -> bool:
        """Run all verification gates on a candidate. Updates candidate fields in-place."""
        reasons: list[str] = []
        combined_text = f"{candidate.title} {candidate.discovery_claim} {candidate.source_url}".lower()

        # ── Gate 1: Source Domain Quality Check ───────────────────────────────
        try:
            domain = urlparse(candidate.source_url).netloc.replace("www.", "").lower()
            if any(d in domain for d in DISALLOWED_DOMAINS) and not candidate.primary_source_url:
                reasons.append(f"Disallowed aggregator/social domain: {domain}")
        except Exception:
            pass

        # ── Gate 2: Forbidden Topic Check (for quantum computing domain) ────
        if self.target_domain == "quantum computing":
            for forbidden in FORBIDDEN_TOPICS:
                if forbidden in combined_text:
                    reasons.append(f"Contains forbidden topic '{forbidden}' outside domain")
                    candidate.satisfies_domain = False
                    break
            else:
                # ── Gate 3: Domain Relevance Check (Quantum) ─────────────────────
                has_quantum_term = "quantum" in combined_text
                computing_terms = (
                    "computing", "computer", "processor", "qubit", "logical", "fault-tolerant",
                    "error correction", "surface code", "qec", "algorithm", "quantum advantage",
                    "quantum chip", "superconducting", "neutral atom", "ion trap", "photonic",
                )
                has_domain_kw = has_quantum_term and any(term in combined_text for term in computing_terms)
                candidate.satisfies_domain = has_domain_kw
                if not has_domain_kw:
                    reasons.append("Lacks specific quantum computing terminology")
        else:
            # General Domain Relevance Check: match against domain vocabulary or target_domain keywords
            vocab = DOMAIN_VOCABULARY.get(self.target_domain)
            if vocab:
                has_domain_kw = any(term in combined_text for term in vocab)
            else:
                domain_keywords = set(re.findall(r"\b[a-z0-9-]{3,}\b", self.target_domain))
                has_domain_kw = any(kw in combined_text for kw in domain_keywords) if domain_keywords else True
            candidate.satisfies_domain = has_domain_kw
            if not has_domain_kw:
                reasons.append(f"Lacks specific {self.target_domain} terminology")

        # ── Gate 4: Year Verification ────────────────────────────────────────
        # Article dates and years embedded in URLs do not establish when the
        # scientific work occurred. Only explicit work/publication dates count.
        year_found = any(
            value and re.search(rf"\b{re.escape(self.target_year)}\b", str(value))
            for value in (candidate.work_date, candidate.publication_date)
        )

        candidate.satisfies_year = year_found
        if not year_found:
            reasons.append(f"No verifiable association with target year {self.target_year}")

        # ── Gate 5: Discovery vs Non-Discovery Check ─────────────────────────
        is_non_discovery = any(indicator in combined_text for indicator in NON_DISCOVERY_INDICATORS)
        if is_non_discovery:
            candidate.satisfies_discovery_requirement = False
            reasons.append("Identified as policy/funding/market report/listicle rather than a scientific discovery")
        else:
            candidate.satisfies_discovery_requirement = True

        # ── Gate 6: Institution Attribution Check ────────────────────────────
        inst = candidate.institution.strip()
        publisher_like = {"unknown", "various", "multiple", "n/a", "none", "unknown institution", "sciencedaily", "phys.org", "bqpsim", "nature publishing group / academic collaboration", "department of energy", "doe", "national quantum research centers"}
        has_institution = bool(inst and inst.lower() not in publisher_like and candidate.evidence and candidate.institution_evidence)
        candidate.satisfies_institution = has_institution
        if not has_institution:
            reasons.append("Missing verified institutional attribution evidence")

        # ── Gate 7: Source Provenance Check ──────────────────────────────────
        has_source = bool(candidate.source_url and candidate.source_url.startswith(("http://", "https://")))
        if not has_source:
            reasons.append("Missing verified source URL")
        has_primary_evidence = candidate.primary_source or bool(
            candidate.primary_source_url and candidate.primary_source_url.startswith(("http://", "https://"))
        )
        if not has_primary_evidence:
            reasons.append("Secondary discovery source is not sufficient for verification")
        if not candidate.evidence:
            reasons.append("No supporting evidence for the discovery claim")
        if not candidate.significance_evidence:
            reasons.append("No evidence supporting significance claims")
        sig_candidates = list(candidate.significance_evidence)
        if candidate.importance_claim:
            sig_candidates.append(candidate.importance_claim)
        significance_text = " ".join(sig_candidates).lower()
        candidate.satisfies_significance = bool(
            candidate.significance_evidence and any(marker in significance_text for marker in SIGNIFICANCE_MARKERS)
        )
        if not candidate.satisfies_significance:
            reasons.append("Evidence does not establish why the result is significant")
        score = 0.0
        if candidate.satisfies_domain:
            score += 0.3
        if candidate.satisfies_year:
            score += 0.3
        if candidate.satisfies_institution:
            score += 0.2
        if candidate.satisfies_discovery_requirement:
            score += 0.2
        if has_primary_evidence:
            score += 0.1
        if candidate.evidence:
            score += 0.1
        if candidate.significance_evidence:
            score += 0.1
        if candidate.satisfies_significance:
            score += 0.1

        candidate.evidence_quality = min(score, 1.0)
        candidate.rejection_reasons = reasons
        candidate.verified = (
            candidate.satisfies_domain
            and candidate.satisfies_year
            and candidate.satisfies_institution
            and candidate.satisfies_discovery_requirement
            and candidate.satisfies_significance
            and has_source
            and len(reasons) == 0
        )
        return candidate.verified

    @staticmethod
    def count_sentences(text: str) -> int:
        """Count the number of sentences in a string using punctuation boundaries."""
        cleaned = text.strip()
        if not cleaned:
            return 0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        return len(sentences)

    @staticmethod
    def enforce_two_sentences(text: str) -> str:
        """Ensure the summary contains exactly two sentences."""
        cleaned = text.strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if len(sentences) >= 2:
            return f"{sentences[0]} {sentences[1]}"
        elif len(sentences) == 1:
            # Complete the format without inventing a scientific consequence.
            first = sentences[0]
            if not first.endswith((".", "!", "?")):
                first += "."
            return f"{first} The available source evidence does not support a more specific significance claim beyond the reported result."
        return "The available source does not provide a supported discovery statement. No broader significance claim can be made from the available evidence."


class EvidenceExtractor:
    """Extracts candidate discoveries and attributes from search result items."""

    DATE_PATTERNS = [
        re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})\b", re.I),
        re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2})\b", re.I),
        re.compile(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b"),
        re.compile(r"\b(20\d{2})\b"),
    ]

    INSTITUTION_MARKERS = [
        ("deepmind", "Google DeepMind"),
        ("chicago", "University of Chicago (in partnership with IBM)"),
        ("ibm", "IBM Quantum / IBM Research"),
        ("dwave", "D-Wave Quantum Inc."),
        ("d-wave", "D-Wave Quantum Inc."),
        ("google", "Google Quantum AI"),
        ("microsoft", "Microsoft Quantum"),
        ("stanford", "Stanford University"),
        ("broad institute", "Broad Institute of MIT and Harvard"),
        ("broadinstitute", "Broad Institute of MIT and Harvard"),
        ("ligo", "LIGO Scientific Collaboration"),
        ("cern", "CERN"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("harvard", "Harvard University"),
        ("mit", "Massachusetts Institute of Technology (MIT)"),
        ("caltech", "California Institute of Technology (Caltech)"),
        ("princeton", "Princeton University"),
        ("oxford", "University of Oxford"),
        ("cambridge", "University of Cambridge"),
        ("waterloo", "Institute for Quantum Computing, University of Waterloo"),
        ("sydney", "University of Sydney"),
        ("quera", "QuEra Computing"),
        ("quantinuum", "Quantinuum"),
        ("xanadu", "Xanadu Quantum Technologies"),
        ("fnal", "Fermilab"),
        ("fermilab", "Fermilab"),
        ("nature", "Nature Publishing Group / Academic Collaboration"),
        ("sciencedaily", "University Research Collaboration"),
        ("phys.org", "Research Lab Collaboration"),
        ("aix global", "AIX Global Innovations"),
        ("national institute of standards and technology", "National Institute of Standards and Technology (NIST)"),
        ("sandia national laboratories", "Sandia National Laboratories"),
        ("los alamos national laboratory", "Los Alamos National Laboratory"),
        ("lawrence berkeley national laboratory", "Lawrence Berkeley National Laboratory"),
        ("university of maryland", "University of Maryland"),
        ("university of bristol", "University of Bristol"),
        ("university of new south wales", "University of New South Wales"),
        ("university of sussex", "University of Sussex"),
    ]

    @classmethod
    def extract_date(cls, text: str, url: str) -> str | None:
        for pat in cls.DATE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).strip()
        url_match = re.search(r"/(20\d{2}[-/]\d{2}[-/]\d{2})/", url) or re.search(r"/(20\d{2})/", url)
        if url_match:
            return url_match.group(1).strip()
        return None

    @classmethod
    def extract_institution(cls, title: str, snippet: str, url: str) -> str:
        combined = f"{title} {snippet}".lower()
        if "ibm" in combined and "chicago" in combined:
            return "University of Chicago (in partnership with IBM)"
        for marker, display_name in cls.INSTITUTION_MARKERS:
            if marker in {"sciencedaily", "phys.org", "nature"}:
                continue
            if marker in combined:
                return display_name
        original = f"{title} {snippet}"
        institution_patterns = (
            r"\b(University of [A-Z][A-Za-z&.' -]{2,80})",
            r"\b([A-Z][A-Za-z&.' -]{2,80}(?:University|Institute|Laborator(?:y|ies)|Lab|Labs|College|Research Center|Research Centre))\b",
        )
        for pattern in institution_patterns:
            match = re.search(pattern, original)
            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")
                if value and not any(bad in value.casefold() for bad in ("sciencedaily", "phys.org", "blog", "publisher")):
                    return value
        host = urlparse(url.lower()).netloc.removeprefix("www.")
        official_hosts = {
            "newsroom.ibm.com": "IBM Quantum / IBM Research",
            "research.google": "Google Quantum AI",
            "deepmind.google": "Google DeepMind",
            "fnal.gov": "Fermilab",
            "dwavequantum.com": "D-Wave Quantum Inc.",
            "xanadu.ai": "Xanadu Quantum Technologies",
            "broadinstitute.org": "Broad Institute of MIT and Harvard",
            "ligo.org": "LIGO Scientific Collaboration",
            "cern.ch": "CERN",
            "openai.com": "OpenAI",
            "anthropic.com": "Anthropic",
        }
        if host in official_hosts:
            return official_hosts[host]
        # Never turn a hostname into a research institution. The institution
        # must be named in the title/snippet/source text or remain unknown.
        return "Unknown Institution"

    @classmethod
    def is_primary_source(cls, url: str) -> bool:
        url_lower = url.lower()
        parsed = urlparse(url_lower)
        host = parsed.netloc.removeprefix("www.")
        if any(host == domain or host.endswith("." + domain) for domain in DISALLOWED_DOMAINS):
            return False
        primary_domains = {
            "newsroom.ibm.com", "ibm.com", "research.google", "deepmind.google", "dwavequantum.com",
            "microsoft.com", "nature.com", "science.org", "arxiv.org", "aps.org", "doi.org",
            "xanadu.ai", "fnal.gov", "broadinstitute.org", "ligo.org", "cern.ch",
            "openai.com", "anthropic.com", "cell.com", "pnas.org", "biorxiv.org", "medrxiv.org",
        }
        if any(
            host == pd or host.endswith("." + pd)
            for pd in primary_domains
        ):
            return True
        if host.endswith(".google"):
            return True
        # University and government research pages can be primary, while a
        # generic publisher/blog domain is not promoted by URL heuristics.
        return (
            host.endswith(".edu")
            or ".edu." in host
            or host.endswith(".ac.uk")
            or ".ac." in host
            or host.endswith(".gov")
            or ".gov." in host
        )

    @classmethod
    def source_type(cls, url: str, primary: bool) -> str:
        host = urlparse(url.lower()).netloc.removeprefix("www.")
        if host == "arxiv.org" or host.endswith(".arxiv.org"):
            return "preprint"
        if host in {"nature.com", "science.org", "aps.org", "doi.org"}:
            return "journal_or_paper"
        if (
            host.endswith(".edu") or ".edu." in host or host.endswith(".ac.uk")
            or ".ac." in host or host.endswith(".gov") or ".gov." in host
        ):
            return "institutional_primary" if primary else "institutional_secondary"
        if primary:
            return "institutional_primary"
        return "secondary"

    @classmethod
    def extract_candidate(cls, item: dict[str, Any]) -> CandidateEvidence:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        snippet = item.get("snippet", "") or item.get("content", "")
        
        extracted_date = item.get("date") or cls.extract_date(f"{title} {snippet}", url)
        institution = item.get("institution") or cls.extract_institution(title, snippet, url)
        primary = bool(item.get("primary_source")) or cls.is_primary_source(url)

        # A date supplied by an authoritative primary result is usable as the
        # work date; a secondary result's date remains article_date only.
        work_date = item.get("work_date") or item.get("publication_date")
        if not work_date and primary:
            work_date = item.get("date") or extracted_date

        pub_date = item.get("publication_date") or item.get("work_date") or (item.get("date") if primary else None)

        evidence_list = [snippet[:500].strip()] if snippet.strip() else []
        sig_evidence_list = [snippet[:500].strip()] if snippet.strip() else []
        inst_evidence_list = [snippet[:500].strip()] if institution != "Unknown Institution" and snippet.strip() else ([url] if institution != "Unknown Institution" and url else [])

        return CandidateEvidence(
            title=title,
            institution=institution,
            source_url=url,
            discovery_claim=snippet[:300].strip(),
            importance_claim=snippet[300:600].strip() if len(snippet) > 300 else snippet[:300].strip(),
            publication_date=pub_date,
            article_date=item.get("article_date") or item.get("date") or extracted_date,
            work_date=work_date,
            primary_source_url=item.get("primary_source_url"),
            primary_source=primary,
            source_type=item.get("source_type") or cls.source_type(url, primary),
            evidence=evidence_list,
            significance_evidence=sig_evidence_list,
            claim_evidence={
                "discovery": list(evidence_list),
                "significance": list(sig_evidence_list),
            },
            institution_evidence=inst_evidence_list,
        )

    @classmethod
    def enrich_from_source(cls, candidate: CandidateEvidence, fetched_text: str) -> CandidateEvidence:
        """Attach evidence extracted from the fetched source page.

        Search snippets are discovery material. This method promotes only
        sentences actually present in the fetched source into claim evidence.
        """
        if not fetched_text or fetched_text.startswith(("Error", "HTTP error", "Connection error", "Failed to", "Non-HTML")):
            return candidate
        candidate.source_fetched = True
        content = re.sub(r"[ \t]+", " ", fetched_text)
        content = re.sub(r"\n{2,}", "\n", content).strip()
        metadata_date_match = re.search(r"Metadata date:\s*([^\n]+)", content, re.I)
        title_match = re.search(r"Title:\s*(.*?)(?:\s+Metadata date:|\s+Content:|$)", content, re.I)
        page_title = title_match.group(1).strip() if title_match else ""
        body_match = re.search(r"Content:\s*(.*)$", content, re.I)
        body = body_match.group(1).strip() if body_match else content
        if page_title and (not candidate.title or candidate.title.lower() in {"untitled", "unknown"}):
            candidate.title = page_title
        if candidate.primary_source:
            explicit_work_date = None
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                if re.search(r"\b(?:paper|publication|published|results?|experiment|demonstrat|research|work)\b", sentence, re.I):
                    explicit_work_date = cls.extract_date(sentence, "")
                    if explicit_work_date:
                        break
            candidate.work_date = explicit_work_date or candidate.work_date or (
                metadata_date_match.group(1).strip() if metadata_date_match else None
            )
            candidate.publication_date = candidate.publication_date or candidate.work_date
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) >= 35]
        domain_terms = tuple(QUANTUM_COMPUTING_KEYWORDS)
        evidence = [s for s in sentences if any(term in s.lower() for term in domain_terms)]
        if not evidence and sentences:
            evidence = sentences[:8]
        if evidence:
            candidate.evidence = evidence[:8]
            if not any(term in candidate.discovery_claim.lower() for term in ("quantum computing", "quantum computer", "qubit", "quantum processor", "quantum error correction")):
                candidate.discovery_claim = evidence[0]
        significance_terms = (
            "demonstrat", "achiev", "improv", "reduc", "increase", "scale", "novel", "first",
            "threshold", "advantage", "error rate", "coherence", "enabl", "provid", "confirm",
            "observ", "detect", "breakthrough", "precis", "discover", "unveil", "uncover",
        )
        significance = [s for s in sentences if any(term in s.lower() for term in significance_terms)]
        if not significance and sentences:
            significance = sentences[:4]
        if significance:
            candidate.significance_evidence = significance[:8]
        candidate.claim_evidence = {
            "discovery": list(candidate.evidence),
            "significance": list(candidate.significance_evidence),
        }
        extracted_institution = cls.extract_institution(page_title, body, candidate.source_url)
        if extracted_institution != "Unknown Institution":
            candidate.institution = extracted_institution
            candidate.institution_evidence = [
                sentence for sentence in sentences
                if extracted_institution.casefold() in sentence.casefold()
            ][:4] or candidate.institution_evidence
        return candidate
