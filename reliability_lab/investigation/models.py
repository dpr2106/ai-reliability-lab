"""Domain entities and value objects for Cross-Document Evidence Investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

EvidenceRelationship = Literal[
    "supports",
    "contradicts",
    "qualifies",
    "not_addressed",
    "insufficient_evidence",
]

VALID_RELATIONSHIPS: set[str] = {
    "supports",
    "contradicts",
    "qualifies",
    "not_addressed",
    "insufficient_evidence",
}


@dataclass(frozen=True)
class EvidenceFinding:
    """A claim or empirical finding extracted from a specific document chunk."""

    finding_id: str
    topic_id: str
    doc_id: str
    doc_title: str
    chunk_id: str
    page_number: int | None
    claim: str
    supporting_passage: str
    relationship: EvidenceRelationship
    confidence: float
    methodology_notes: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert finding to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceFinding:
        """Construct finding from a dictionary with validation."""
        rel = data.get("relationship", "insufficient_evidence")
        if rel not in VALID_RELATIONSHIPS:
            rel = "insufficient_evidence"
        return cls(
            finding_id=str(data.get("finding_id") or f"fnd_{uuid4().hex[:12]}"),
            topic_id=str(data.get("topic_id", "")),
            doc_id=str(data["doc_id"]),
            doc_title=str(data.get("doc_title") or data["doc_id"]),
            chunk_id=str(data.get("chunk_id", "")),
            page_number=int(data["page_number"]) if data.get("page_number") is not None else None,
            claim=str(data.get("claim", "")),
            supporting_passage=str(data.get("supporting_passage", "")),
            relationship=rel,
            confidence=float(data.get("confidence", 0.0)),
            methodology_notes=data.get("methodology_notes"),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ComparisonTopic:
    """A thematic aspect or research dimension compared across documents."""

    topic_id: str
    topic_name: str
    description: str
    findings: list[EvidenceFinding] = field(default_factory=list)
    agreements_summary: str = ""
    disagreements_summary: str = ""
    evidence_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert topic to dictionary."""
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "description": self.description,
            "findings": [f.to_dict() for f in self.findings],
            "agreements_summary": self.agreements_summary,
            "disagreements_summary": self.disagreements_summary,
            "evidence_gaps": list(self.evidence_gaps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonTopic:
        """Construct topic from dictionary."""
        return cls(
            topic_id=str(data.get("topic_id") or f"top_{uuid4().hex[:8]}"),
            topic_name=str(data.get("topic_name", "")),
            description=str(data.get("description", "")),
            findings=[EvidenceFinding.from_dict(f) for f in data.get("findings", [])],
            agreements_summary=str(data.get("agreements_summary", "")),
            disagreements_summary=str(data.get("disagreements_summary", "")),
            evidence_gaps=[str(g) for g in data.get("evidence_gaps", [])],
        )


@dataclass
class InvestigationReport:
    """Structured report of cross-document investigation findings, matrix, and synthesis."""

    investigation_id: str
    title: str
    query: str
    comparison_aspects: list[str]
    document_ids: list[str]
    document_summaries: list[dict[str, Any]]
    topics: list[ComparisonTopic]
    total_findings: int
    relationship_counts: dict[str, int]
    executive_summary: str
    methodology_synthesis: str
    created_at: str
    completed_at: str | None = None
    status: str = "completed"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "investigation_id": self.investigation_id,
            "title": self.title,
            "query": self.query,
            "comparison_aspects": list(self.comparison_aspects),
            "document_ids": list(self.document_ids),
            "document_summaries": [dict(s) for s in self.document_summaries],
            "topics": [t.to_dict() for t in self.topics],
            "total_findings": self.total_findings,
            "relationship_counts": dict(self.relationship_counts),
            "executive_summary": self.executive_summary,
            "methodology_synthesis": self.methodology_synthesis,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvestigationReport:
        """Construct report from dictionary."""
        return cls(
            investigation_id=str(data["investigation_id"]),
            title=str(data.get("title", "")),
            query=str(data.get("query", "")),
            comparison_aspects=[str(a) for a in data.get("comparison_aspects", [])],
            document_ids=[str(d) for d in data.get("document_ids", [])],
            document_summaries=list(data.get("document_summaries", [])),
            topics=[ComparisonTopic.from_dict(t) for t in data.get("topics", [])],
            total_findings=int(data.get("total_findings", 0)),
            relationship_counts=dict(data.get("relationship_counts", {})),
            executive_summary=str(data.get("executive_summary", "")),
            methodology_synthesis=str(data.get("methodology_synthesis", "")),
            created_at=str(data.get("created_at", "")),
            completed_at=data.get("completed_at"),
            status=str(data.get("status", "completed")),
            error_message=data.get("error_message"),
        )

    def to_markdown(self) -> str:
        """Generate a structured, publication-grade Markdown comparison report."""
        lines: list[str] = [
            f"# Cross-Document Evidence Investigation: {self.title or self.query}",
            "",
            f"**Investigation ID:** `{self.investigation_id}`  ",
            f"**Created:** {self.created_at}  ",
            f"**Completed:** {self.completed_at or 'In Progress'}  ",
            f"**Query:** *{self.query}*  ",
            "",
            "## 1. Executive Summary",
            "",
            self.executive_summary or "No executive summary available.",
            "",
            "## 2. Documents Examined",
            "",
            "| Document ID | Filename | Chunks | Findings Count |",
            "|---|---|---|---|",
        ]

        for doc in self.document_summaries:
            did = doc.get("doc_id", "unknown")
            fname = doc.get("filename", did)
            chunks = doc.get("chunk_count", 0)
            findings = doc.get("findings_count", 0)
            lines.append(f"| `{did}` | {fname} | {chunks} | {findings} |")

        lines.extend(
            [
                "",
                "## 3. Evidence Breakdown",
                "",
                f"- **Total Extracted Findings:** {self.total_findings}",
                f"- **Supporting Evidence:** {self.relationship_counts.get('supports', 0)}",
                f"- **Contradicting / Inconsistent Evidence:** {self.relationship_counts.get('contradicts', 0)}",
                f"- **Qualified / Conditioned Findings:** {self.relationship_counts.get('qualifies', 0)}",
                f"- **Topics Not Addressed:** {self.relationship_counts.get('not_addressed', 0)}",
                f"- **Insufficient Evidence:** {self.relationship_counts.get('insufficient_evidence', 0)}",
                "",
                "## 4. Cross-Document Thematic Analysis",
                "",
            ]
        )

        for idx, topic in enumerate(self.topics, start=1):
            lines.extend(
                [
                    f"### 4.{idx} {topic.topic_name}",
                    "",
                    topic.description or "No description provided.",
                    "",
                ]
            )

            if topic.agreements_summary:
                lines.extend(
                    [
                        "**Agreements & Cross-Source Consensus:**",
                        f"> {topic.agreements_summary}",
                        "",
                    ]
                )

            if topic.disagreements_summary:
                lines.extend(
                    [
                        "**Disagreements & Contradictions:**",
                        f"> {topic.disagreements_summary}",
                        "",
                    ]
                )

            if topic.evidence_gaps:
                lines.append("**Evidence Gaps & Missing Coverage:**")
                for gap in topic.evidence_gaps:
                    lines.append(f"- {gap}")
                lines.append("")

            lines.extend(
                [
                    "#### Extracted Evidence Findings",
                    "",
                    "| Status | Source Document | Page | Chunk ID | Claim & Supporting Passage |",
                    "|---|---|---|---|---|",
                ]
            )

            if not topic.findings:
                lines.append("| *None* | - | - | - | *No findings extracted for this aspect.* |")
            else:
                for f in topic.findings:
                    page_str = str(f.page_number) if f.page_number is not None else "N/A"
                    rel_upper = f.relationship.upper().replace("_", " ")
                    passage_preview = (
                        f"**Claim:** {f.claim}<br/>"
                        f"**Passage:** *\"{f.supporting_passage}\"*"
                    )
                    if f.methodology_notes:
                        passage_preview += f"<br/>**Methodology:** {f.methodology_notes}"
                    lines.append(
                        f"| `{rel_upper}` | {f.doc_title} | {page_str} | `{f.chunk_id}` | {passage_preview} |"
                    )

            lines.append("")

        if self.methodology_synthesis:
            lines.extend(
                [
                    "## 5. Methodology & Context Synthesis",
                    "",
                    self.methodology_synthesis,
                    "",
                ]
            )

        lines.extend(
            [
                "---",
                "*Report generated by STING Evidence Intelligence Workbench.*  ",
                "*Grounded in deterministic passage retrieval. Verification status indicates claim-passage alignment, not unverified truth.*",
            ]
        )

        return "\n".join(lines)


@dataclass
class InvestigationRecord:
    """Persistent database record for an investigation entity."""

    investigation_id: str
    title: str
    query: str
    comparison_aspects: list[str]
    document_ids: list[str]
    status: str
    created_at: str
    completed_at: str | None = None
    report: InvestigationReport | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "investigation_id": self.investigation_id,
            "title": self.title,
            "query": self.query,
            "comparison_aspects": list(self.comparison_aspects),
            "document_ids": list(self.document_ids),
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "report": self.report.to_dict() if self.report else None,
            "error_message": self.error_message,
        }
