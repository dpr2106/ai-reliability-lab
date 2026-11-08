"""Domain entities and value objects for Cross-Document Evidence Investigation."""

from __future__ import annotations

import json
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
    doc_id: str
    doc_title: str
    chunk_id: str
    page_number: int | None
    claim: str
    supporting_passage: str
    relationship: EvidenceRelationship
    confidence: float
    topic_id: str = ""
    topic_name: str | None = None
    investigation_id: str | None = None
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
            topic_name=data.get("topic_name"),
            investigation_id=data.get("investigation_id"),
            methodology_notes=data.get("methodology_notes"),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ComparisonTopic:
    """A thematic aspect or research dimension compared across documents."""

    topic_name: str
    description: str
    topic_id: str = field(default_factory=lambda: f"top_{uuid4().hex[:8]}")
    findings: list[EvidenceFinding] = field(default_factory=list)
    agreements_summary: str = ""
    disagreements_summary: str = ""
    evidence_gaps: list[str] = field(default_factory=list)

    def __init__(
        self,
        topic_name: str = "",
        description: str = "",
        topic_id: str | None = None,
        findings: list[EvidenceFinding] | None = None,
        agreements_summary: str = "",
        disagreements_summary: str = "",
        evidence_gaps: list[str] | None = None,
    ) -> None:
        self.topic_name = topic_name
        self.description = description
        self.topic_id = topic_id or f"top_{uuid4().hex[:8]}"
        self.findings = list(findings) if findings else []
        self.agreements_summary = agreements_summary
        self.disagreements_summary = disagreements_summary
        self.evidence_gaps = list(evidence_gaps) if evidence_gaps else []

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
    topics: list[ComparisonTopic]
    total_findings: int
    relationship_counts: dict[str, int]
    executive_summary: str
    comparison_aspects: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    document_summaries: list[dict[str, Any]] = field(default_factory=list)
    methodology_synthesis: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    status: str = "completed"
    error_message: str | None = None

    def __init__(
        self,
        investigation_id: str,
        title: str,
        query: str,
        topics: list[ComparisonTopic],
        total_findings: int,
        relationship_counts: dict[str, int],
        executive_summary: str,
        comparison_aspects: list[str] | None = None,
        document_ids: list[str] | None = None,
        document_summaries: list[dict[str, Any]] | None = None,
        methodology_synthesis: str | None = None,
        created_at: str | None = None,
        completed_at: str | None = None,
        status: str = "completed",
        error_message: str | None = None,
        *,
        documents: list[dict[str, Any]] | None = None,
        generated_at: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.investigation_id = investigation_id
        self.title = title
        self.query = query
        self.topics = list(topics)
        self.total_findings = total_findings
        self.relationship_counts = dict(relationship_counts)
        self.executive_summary = executive_summary
        self.document_summaries = document_summaries or documents or []
        self.document_ids = (
            document_ids
            if document_ids is not None
            else [d.get("doc_id", "") for d in self.document_summaries if "doc_id" in d]
        )
        self.comparison_aspects = (
            comparison_aspects
            if comparison_aspects is not None
            else [t.topic_name for t in self.topics]
        )
        self.methodology_synthesis = methodology_synthesis or ""
        self.created_at = created_at or generated_at or datetime.now(timezone.utc).isoformat()
        self.completed_at = completed_at or self.created_at
        self.status = status
        self.error_message = error_message

    @property
    def documents(self) -> list[dict[str, Any]]:
        """Convenience alias for document_summaries."""
        return self.document_summaries

    @property
    def generated_at(self) -> str:
        """Convenience alias for created_at."""
        return self.created_at

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
            f"# STING Investigation Report: {self.title or self.query}",
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
            "## 2. Investigated Document Collection",
            "",
            "| Document ID | Document Title | Chunks | Findings Count |",
            "|---|---|---|---|",
        ]

        for doc in self.document_summaries:
            did = doc.get("doc_id", "unknown")
            fname = doc.get("title") or doc.get("filename") or did
            chunks = doc.get("chunk_count", 0)
            total = doc.get("total_chunks", chunks)
            mode = doc.get("coverage_mode", "exhaustive")
            findings = doc.get("findings_count", 0)
            chunk_str = f"{chunks}/{total} ({mode})" if total != chunks else str(chunks)
            lines.append(f"| `{did}` | {fname} | {chunk_str} | {findings} |")

        contra_count = self.relationship_counts.get("contradicts", 0)
        qual_count = self.relationship_counts.get("qualifies", 0)
        not_addr_count = self.relationship_counts.get("not_addressed", 0)
        insuff_count = self.relationship_counts.get("insufficient_evidence", 0)
        supports_count = self.relationship_counts.get("supports", 0)

        lines.extend(
            [
                "",
                "## 3. Evidence Breakdown",
                "",
                f"- **Total Extracted Findings:** {self.total_findings}",
                f"- **Supporting Evidence:** {supports_count}",
                f"- **Contradicting / Inconsistent Evidence:** {contra_count}",
                f"- **Qualified / Conditioned Findings:** {qual_count}",
                f"- **Topics Not Addressed:** {not_addr_count}",
                f"- **Insufficient Evidence:** {insuff_count}",
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
                        f'**Claim:** {f.claim}<br/>**Passage:** *"{f.supporting_passage}"*'
                    )
                    if f.methodology_notes:
                        passage_preview += f"<br/>**Methodology:** {f.methodology_notes}"
                    row = (
                        f"| `{rel_upper}` | {f.doc_title} | {page_str} | "
                        f"`{f.chunk_id}` | {passage_preview} |"
                    )
                    lines.append(row)

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
                (
                    "*Grounded in deterministic passage retrieval. "
                    "Verification status indicates claim-passage alignment, "
                    "not unverified truth.*"
                ),
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

    @property
    def aspects(self) -> list[str]:
        """Convenience alias for comparison_aspects."""
        return self.comparison_aspects

    @property
    def report_json(self) -> str | None:
        """Serialized report JSON string if report is present."""
        return json.dumps(self.report.to_dict()) if self.report else None

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
