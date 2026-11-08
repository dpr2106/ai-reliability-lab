"""Core application service orchestrating cross-document evidence investigations."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from reliability_lab.core.models import Chunk
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.investigation.models import (
    VALID_RELATIONSHIPS,
    ComparisonTopic,
    EvidenceFinding,
    EvidenceRelationship,
    InvestigationRecord,
    InvestigationReport,
)
from reliability_lab.investigation.repository import (
    InvestigationNotFoundError,
    InvestigationRepository,
)
from reliability_lab.llm.client import BaseLLMClient, OllamaLLMClient
from reliability_lab.llm.models import LLMGenerationConfig, LLMMessage
from reliability_lab.query.orchestrator import DocumentQueryService

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_ASPECTS: list[str] = [
    "Core Claims & Architecture",
    "Performance Metrics & Results",
    "Methodology & Experimental Setup",
    "Agreements & Common Findings",
    "Contradictions & Opposing Results",
    "Limitations & Open Questions",
]


def _normalize_passage(text: str) -> str:
    """Normalize whitespace and quotes for robust grounded substring comparison."""
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return (
        norm.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


class InvestigationValidationError(ValueError):
    """Raised when investigation parameters or document states fail validation."""


class InvestigationService:
    """Orchestrates multi-document retrieval, evidence extraction, and cross-source synthesis."""

    def __init__(
        self,
        investigation_repository: InvestigationRepository,
        document_repository: DocumentRepository,
        query_service: DocumentQueryService | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.investigation_repository = investigation_repository
        self.document_repository = document_repository
        self.query_service = query_service
        self._llm_client = llm_client

    @property
    def llm_client(self) -> BaseLLMClient:
        """Return active LLM client adapter, defaulting to query service client or Ollama."""
        if self._llm_client is not None:
            return self._llm_client
        if self.query_service and hasattr(self.query_service, "llm_client"):
            return self.query_service.llm_client
        return OllamaLLMClient()

    def create_investigation(
        self,
        query: str,
        document_ids: Sequence[str],
        title: str | None = None,
        comparison_aspects: Sequence[str] | None = None,
        *,
        aspects: Sequence[str] | None = None,
    ) -> InvestigationRecord:
        """Validate documents and create a persistent investigation record.

        Raises:
            InvestigationValidationError: If query or document_ids is invalid.
        """
        effective_aspects = comparison_aspects if comparison_aspects is not None else aspects
        if not isinstance(query, str) or not query.strip():
            raise InvestigationValidationError("Investigation query must be a non-empty string.")

        clean_doc_ids = [d.strip() for d in document_ids if isinstance(d, str) and d.strip()]
        if not clean_doc_ids:
            raise InvestigationValidationError(
                "At least one valid document ID must be selected for investigation."
            )

        # Validate document existence and readiness
        for doc_id in clean_doc_ids:
            doc = self.document_repository.get_document(doc_id)
            if doc is None:
                raise InvestigationValidationError(f"Document '{doc_id}' not found in repository.")
            if doc.status == DocumentStatus.FAILED:
                raise InvestigationValidationError(
                    f"Document '{doc_id}' is in FAILED state and cannot be investigated."
                )
            if doc.chunk_count == 0:
                raise InvestigationValidationError(
                    f"Document '{doc_id}' has 0 chunks and cannot be investigated."
                )

        aspects = (
            [a.strip() for a in effective_aspects if isinstance(a, str) and a.strip()]
            if effective_aspects
            else list(DEFAULT_COMPARISON_ASPECTS)
        )
        if not aspects:
            aspects = list(DEFAULT_COMPARISON_ASPECTS)

        effective_title = (
            title.strip() if title and title.strip() else f"Investigation: {query.strip()[:65]}"
        )

        return self.investigation_repository.create_investigation(
            query=query.strip(),
            document_ids=clean_doc_ids,
            title=effective_title,
            comparison_aspects=aspects,
        )

    def execute_investigation(self, investigation_id: str) -> InvestigationReport:
        """Execute a cross-document evidence investigation to completion."""
        inv = self.investigation_repository.get_investigation(investigation_id)
        if inv is None:
            raise InvestigationNotFoundError(f"Investigation '{investigation_id}' not found.")

        self.investigation_repository.update_status(investigation_id, "running")

        try:
            report = self._run_investigation_pipeline(inv)
            self.investigation_repository.save_report(investigation_id, report)
            return report
        except Exception as exc:
            logger.error(
                "Investigation '%s' failed during pipeline execution: %s",
                investigation_id,
                exc,
                exc_info=True,
            )
            self.investigation_repository.update_status(
                investigation_id, "failed", error_message=str(exc)
            )
            raise

    run_investigation = execute_investigation

    def export_investigation_markdown(self, investigation_id: str) -> str:
        """Export completed investigation report as Markdown."""
        inv = self.investigation_repository.get_investigation(investigation_id)
        if inv is None:
            raise InvestigationNotFoundError(f"Investigation '{investigation_id}' not found.")
        if inv.report is None:
            raise InvestigationValidationError(
                f"Investigation '{investigation_id}' has not produced a report."
            )
        return inv.report.to_markdown()

    def _run_investigation_pipeline(self, inv: InvestigationRecord) -> InvestigationReport:
        """Run retrieval, structured extraction, comparison, and report compilation."""
        doc_summaries: list[dict[str, Any]] = []
        doc_chunks_map: dict[str, list[Chunk]] = {}
        doc_records: dict[str, DocumentRecord] = {}
        doc_coverage: dict[str, dict[str, Any]] = {}

        # 1. Gather document records and evidence chunks per document
        for doc_id in inv.document_ids:
            doc = self.document_repository.get_document(doc_id)
            if doc is None:
                continue
            doc_records[doc_id] = doc

            # Retrieve top relevant chunks for this document
            chunks, retrieval_error, is_exhaustive = self._retrieve_document_evidence(
                doc_id=doc_id,
                query=inv.query,
                aspects=inv.comparison_aspects,
            )
            doc_chunks_map[doc_id] = chunks
            all_chunks = self.document_repository.get_chunks(doc_id)
            total_count = len(all_chunks)
            retrieved_count = len(chunks)

            if retrieval_error is not None:
                cov_mode = "retrieval_error"
            elif is_exhaustive or retrieved_count >= total_count:
                cov_mode = "exhaustive"
            else:
                cov_mode = "sampled_retrieval"

            doc_coverage[doc_id] = {
                "total_chunks": total_count,
                "retrieved_chunks": retrieved_count,
                "coverage_mode": cov_mode,
                "is_exhaustive": cov_mode == "exhaustive",
                "retrieval_error": retrieval_error,
                "coverage_ratio": round(retrieved_count / max(1, total_count), 3),
            }

        # 2. Extract structured findings from each document
        extracted_findings_by_topic: dict[str, list[EvidenceFinding]] = {
            aspect: [] for aspect in inv.comparison_aspects
        }

        doc_findings_count: dict[str, int] = dict.fromkeys(inv.document_ids, 0)

        for doc_id in inv.document_ids:
            chunks = doc_chunks_map.get(doc_id, [])
            doc_record = doc_records.get(doc_id)
            doc_title = doc_record.filename if doc_record else doc_id
            cov = doc_coverage.get(doc_id, {"is_exhaustive": True, "total_chunks": len(chunks)})

            findings = self._extract_findings_for_document(
                doc_id=doc_id,
                doc_title=doc_title,
                query=inv.query,
                aspects=inv.comparison_aspects,
                chunks=chunks,
                is_exhaustive=cov["is_exhaustive"],
            )

            for f in findings:
                # Find matching topic bucket (case-insensitive substring match or fallback)
                matched_topic = self._match_topic_name(f.topic_id, inv.comparison_aspects)
                extracted_findings_by_topic[matched_topic].append(f)
                doc_findings_count[doc_id] += 1

            doc_summaries.append(
                {
                    "doc_id": doc_id,
                    "filename": doc_title,
                    "title": doc_title,
                    "chunk_count": len(chunks),
                    "total_chunks": cov.get("total_chunks", len(chunks)),
                    "retrieved_chunks": len(chunks),
                    "coverage_mode": cov.get("coverage_mode", "exhaustive"),
                    "coverage_ratio": cov.get("coverage_ratio", 1.0),
                    "findings_count": doc_findings_count[doc_id],
                }
            )

        # 3. Construct comparison topics and synthesize cross-document relationships
        topics: list[ComparisonTopic] = []
        rel_counts: dict[str, int] = {
            "supports": 0,
            "contradicts": 0,
            "qualifies": 0,
            "not_addressed": 0,
            "insufficient_evidence": 0,
        }

        total_findings = 0

        for aspect_name in inv.comparison_aspects:
            findings_for_aspect = extracted_findings_by_topic.get(aspect_name, [])

            # Ensure all documents are represented,
            # distinguishing verified absence from bounded retrieval
            covered_doc_ids = {f.doc_id for f in findings_for_aspect}
            for doc_id in inv.document_ids:
                if doc_id not in covered_doc_ids:
                    doc_rec = doc_records.get(doc_id)
                    d_title = doc_rec.filename if doc_rec else doc_id
                    cov = doc_coverage.get(
                        doc_id,
                        {
                            "coverage_mode": "sampled_retrieval",
                            "is_exhaustive": False,
                            "retrieval_error": None,
                            "retrieved_chunks": 0,
                            "total_chunks": 0,
                        },
                    )

                    if cov.get("retrieval_error"):
                        rel = "insufficient_evidence"
                        claim = f"Retrieval failed for '{aspect_name}' on document '{d_title}'."
                        notes = (
                            f"Retrieval error: {cov['retrieval_error']}. "
                            "Cannot evaluate topic coverage."
                        )
                        conf = 0.0
                    elif cov.get("is_exhaustive", False):
                        rel = "not_addressed"
                        claim = f"Document does not discuss {aspect_name}."
                        notes = (
                            f"Verified absent across all {cov['total_chunks']} chunks of document."
                        )
                        conf = 1.0
                    else:
                        rel = "insufficient_evidence"
                        rc = cov.get("retrieved_chunks", 0)
                        tc = cov.get("total_chunks", 0)
                        claim = (
                            f"Insufficient retrieved evidence for '{aspect_name}': absent from "
                            f"top-{rc} retrieved chunks ({tc} total chunks in document)."
                        )
                        notes = (
                            f"Partial retrieval coverage ({rc}/{tc} chunks examined). "
                            "Unretrieved chunks may discuss this aspect."
                        )
                        conf = 0.5

                    missing_finding = EvidenceFinding(
                        finding_id=f"fnd_gap_{uuid4().hex[:8]}",
                        topic_id=aspect_name,
                        doc_id=doc_id,
                        doc_title=d_title,
                        chunk_id="none",
                        page_number=None,
                        claim=claim,
                        supporting_passage="",
                        relationship=cast(EvidenceRelationship, rel),
                        confidence=conf,
                        methodology_notes=notes,
                    )
                    findings_for_aspect.append(missing_finding)

            # Update relationship counts
            for f in findings_for_aspect:
                total_findings += 1
                rel_counts[f.relationship] = rel_counts.get(f.relationship, 0) + 1

            # Synthesize agreements, disagreements, and gaps
            agreements, disagreements, gaps = self._synthesize_topic_relationships(
                topic_name=aspect_name,
                findings=findings_for_aspect,
                doc_records=doc_records,
            )

            topic = ComparisonTopic(
                topic_id=f"top_{uuid4().hex[:8]}",
                topic_name=aspect_name,
                description=f"Evidence comparison for aspect '{aspect_name}'.",
                findings=findings_for_aspect,
                agreements_summary=agreements,
                disagreements_summary=disagreements,
                evidence_gaps=gaps,
            )
            topics.append(topic)

        # 4. Generate executive summary and methodology synthesis
        exec_summary = self._generate_executive_summary(
            query=inv.query,
            doc_summaries=doc_summaries,
            topics=topics,
            rel_counts=rel_counts,
        )

        methodology_synthesis = self._generate_methodology_synthesis(topics)

        now_utc = datetime.now(timezone.utc).isoformat()

        return InvestigationReport(
            investigation_id=inv.investigation_id,
            title=inv.title,
            query=inv.query,
            comparison_aspects=list(inv.comparison_aspects),
            document_ids=list(inv.document_ids),
            document_summaries=doc_summaries,
            topics=topics,
            total_findings=total_findings,
            relationship_counts=rel_counts,
            executive_summary=exec_summary,
            methodology_synthesis=methodology_synthesis,
            created_at=inv.created_at,
            completed_at=now_utc,
            status="completed",
        )

    def _retrieve_document_evidence(
        self, doc_id: str, query: str, aspects: list[str]
    ) -> tuple[list[Chunk], str | None, bool]:
        """Retrieve relevant evidence chunks specifically for a given document."""
        # 1. First retrieve all chunks from repository as ground truth
        all_chunks = self.document_repository.get_chunks(doc_id)
        if not all_chunks:
            return [], None, True

        if not self.query_service or len(all_chunks) <= 15:
            # Small document or query service absent: include all chunks for exhaustive coverage
            return all_chunks, None, True

        # For larger documents, query BM25 and dense indices to surface top relevant chunks
        selected_chunk_ids: set[str] = set()
        retrieved_chunks: list[Chunk] = []
        retrieval_error: str | None = None

        # Query with overall research query
        try:
            _, trace = self.query_service.query(
                query_text=query,
                selected_doc_ids=[doc_id],
                strategy="hybrid",
                top_k=8,
            )
            for c in trace.evidence_chunks:
                if c.chunk_id not in selected_chunk_ids:
                    selected_chunk_ids.add(c.chunk_id)
                    retrieved_chunks.append(
                        Chunk(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            content=c.content,
                            page_number=c.page_number,
                            char_start=c.char_start,
                            char_end=c.char_end,
                        )
                    )
        except Exception as exc:
            logger.warning("Query retrieval failed for doc '%s': %s", doc_id, exc)
            retrieval_error = str(exc)

        # Ensure first chunk (often title / abstract) is always included
        if all_chunks and all_chunks[0].chunk_id not in selected_chunk_ids:
            retrieved_chunks.insert(0, all_chunks[0])
            selected_chunk_ids.add(all_chunks[0].chunk_id)

        # Fallback if retrieval returned nothing
        if not retrieved_chunks:
            retrieved_chunks = all_chunks[:10]

        is_exhaustive = len(retrieved_chunks) >= len(all_chunks)
        return retrieved_chunks, retrieval_error, is_exhaustive

    def _extract_findings_for_document(
        self,
        doc_id: str,
        doc_title: str,
        query: str,
        aspects: list[str],
        chunks: list[Chunk],
        is_exhaustive: bool = True,
    ) -> list[EvidenceFinding]:
        """Use LLM with strict JSON schema to extract structured findings from chunks."""
        if not chunks:
            return []

        chunk_lookup = {c.chunk_id: c for c in chunks}

        # Format chunk context with explicit chunk identifiers and page numbers
        formatted_passages: list[str] = []
        for c in chunks:
            page_info = f"Page {c.page_number}" if c.page_number is not None else "Page N/A"
            formatted_passages.append(
                f"--- [CHUNK_ID: {c.chunk_id}] ({page_info}) ---\n{c.content}"
            )

        context_text = "\n\n".join(formatted_passages)

        aspects_prompt = "\n".join(f"- {a}" for a in aspects)

        prompt_messages: list[LLMMessage] = [
            LLMMessage(
                role="system",
                content=(
                    "You are an expert evidence analyst for the STING Workbench. "
                    "Extract structured, verifiable findings from the supplied source text. "
                    "CRITICAL PROVENANCE RULES:\n"
                    "1. Extract ONLY statements directly supported by the text.\n"
                    "2. You MUST provide the exact CHUNK_ID where the evidence appears.\n"
                    "3. You MUST provide the EXACT verbatim supporting passage from that chunk.\n"
                    "4. Classify each finding's relationship:\n"
                    "   - 'supports': Affirms, provides positive evidence or corroborates.\n"
                    "   - 'contradicts': Opposes, refutes, or challenges the aspect.\n"
                    "   - 'qualifies': Provides conditional or bounded evidence.\n"
                    "   - 'insufficient_evidence': Touches the topic without adequate proof.\n"
                    "   - 'not_addressed': The source does not mention or cover this aspect.\n"
                    "5. Output valid JSON matching the schema. Never hallucinate quotes or IDs."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f'Investigation Topic / Query: "{query}"\n\n'
                    f'Source Document: "{doc_title}" (ID: {doc_id})\n\n'
                    f"Comparison Aspects to evaluate:\n{aspects_prompt}\n\n"
                    f"Available Document Passages:\n{context_text}\n\n"
                    "Produce a JSON object with this exact structure:\n"
                    "{\n"
                    '  "findings": [\n'
                    "    {\n"
                    '      "topic_name": "<exact or closest aspect name>",\n'
                    '      "claim": "<concise factual statement of what the document claims>",\n'
                    '      "chunk_id": "<exact CHUNK_ID from passage>",\n'
                    '      "page_number": <integer or null>,\n'
                    '      "supporting_passage": "<verbatim excerpt from the chunk>",\n'
                    '      "relationship": "supports" | "contradicts" | "qualifies" '
                    '| "insufficient_evidence" | "not_addressed",\n'
                    '      "confidence": <float between 0.0 and 1.0>,\n'
                    '      "methodology_notes": "<optional notes on dataset or setup>"\n'
                    "    }\n"
                    "  ]\n"
                    "}"
                ),
            ),
        ]

        extracted_findings: list[EvidenceFinding] = []

        try:
            res_json = self.llm_client.generate_json(
                prompt=prompt_messages,
                config=LLMGenerationConfig(temperature=0.0, max_output_tokens=2048),
            )
            raw_findings = res_json.get("findings", [])
            if not isinstance(raw_findings, list):
                raw_findings = []
        except Exception as exc:
            logger.warning("Structured extraction failed for doc '%s': %s", doc_id, exc)
            return self._create_fallback_findings(doc_id, doc_title, aspects, chunks, is_exhaustive)

        for item in raw_findings:
            if not isinstance(item, dict):
                continue

            cid = str(item.get("chunk_id", "")).strip()
            claimed_passage = str(item.get("supporting_passage", "")).strip()
            rel = str(item.get("relationship", "insufficient_evidence")).lower().strip()
            if rel not in VALID_RELATIONSHIPS:
                rel = "insufficient_evidence"

            claim_text = str(item.get("claim", "")).strip()
            topic_name = str(item.get("topic_name", "")).strip() or aspects[0]
            raw_notes = item.get("methodology_notes")
            methodology_notes = str(raw_notes).strip() if raw_notes else None
            page_num = item.get("page_number")

            try:
                conf = float(item.get("confidence", 0.8))
            except (ValueError, TypeError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            norm_pass = _normalize_passage(claimed_passage)

            # Strict Provenance & Fail-Closed Grounding
            if rel in ("supports", "contradicts", "qualifies"):
                if not claimed_passage or not norm_pass:
                    # Empty supporting passage cannot substantiate a claim -> FAIL CLOSED
                    rel = "insufficient_evidence"
                    conf = 0.0
                    claimed_passage = "[PROVENANCE FAILED: Empty supporting passage]"
                    note = "Substantive claim failed provenance: supporting passage is empty."
                    methodology_notes = (
                        f"{methodology_notes} | {note}" if methodology_notes else note
                    )
                    if cid in chunk_lookup:
                        page_num = chunk_lookup[cid].page_number
                    else:
                        cid = "invalid_provenance"
                        page_num = None

                elif not cid or cid.lower() in ("none", "n/a", "unknown", ""):
                    # Missing chunk ID -> search if passage exists in any chunk of the document
                    found_chunk = None
                    for chk in chunks:
                        if norm_pass in _normalize_passage(chk.content):
                            found_chunk = chk
                            break
                    if found_chunk:
                        cid = found_chunk.chunk_id
                        page_num = found_chunk.page_number
                        note = f"Re-anchored to verified chunk {cid} based on passage match."
                        methodology_notes = (
                            f"{methodology_notes} | {note}" if methodology_notes else note
                        )
                    else:
                        rel = "insufficient_evidence"
                        conf = 0.0
                        claimed_passage = (
                            "[PROVENANCE FAILED: Missing chunk ID and unverified passage]"
                        )
                        cid = "invalid_provenance"
                        page_num = None

                else:
                    target_chunk = chunk_lookup.get(cid)
                    if target_chunk is not None:
                        if norm_pass in _normalize_passage(target_chunk.content):
                            # Passage verified in referenced chunk
                            canonical_page = target_chunk.page_number
                            if page_num is None:
                                page_num = canonical_page
                            elif canonical_page is not None and int(page_num) != int(
                                canonical_page
                            ):
                                note = (
                                    f"Canonical chunk page {canonical_page} overrides "
                                    f"reported page {page_num}."
                                )
                                methodology_notes = (
                                    f"{methodology_notes} | {note}" if methodology_notes else note
                                )
                                page_num = canonical_page
                        else:
                            # Passage NOT in referenced chunk -> check other chunks
                            found_other = None
                            for chk in chunks:
                                if chk.chunk_id != cid and norm_pass in _normalize_passage(
                                    chk.content
                                ):
                                    found_other = chk
                                    break
                            if found_other:
                                note = (
                                    f"Passage not in referenced chunk {cid}; re-anchored to "
                                    f"verified chunk {found_other.chunk_id}."
                                )
                                methodology_notes = (
                                    f"{methodology_notes} | {note}" if methodology_notes else note
                                )
                                cid = found_other.chunk_id
                                page_num = found_other.page_number
                                conf = min(conf, 0.7)
                            else:
                                # FAIL CLOSED: Fabricated passage referencing a valid chunk ID
                                rel = "insufficient_evidence"
                                conf = 0.0
                                claimed_passage = (
                                    f"[PROVENANCE FAILED: Passage not found in chunk {cid}]"
                                )
                                note = f"Provenance check failed: passage absent from chunk {cid}."
                                methodology_notes = (
                                    f"{methodology_notes} | {note}" if methodology_notes else note
                                )

                    else:
                        # cid not in document chunks -> search if passage exists in any chunk
                        found_chk = None
                        for chk in chunks:
                            if norm_pass in _normalize_passage(chk.content):
                                found_chk = chk
                                break
                        if found_chk:
                            note = (
                                f"Hallucinated chunk ID '{cid}' re-anchored to verified "
                                f"chunk {found_chk.chunk_id} based on passage."
                            )
                            methodology_notes = (
                                f"{methodology_notes} | {note}" if methodology_notes else note
                            )
                            cid = found_chk.chunk_id
                            page_num = found_chk.page_number
                            conf = min(conf, 0.7)
                        else:
                            # FAIL CLOSED: Fabricated chunk ID and unverified passage
                            rel = "insufficient_evidence"
                            conf = 0.0
                            claimed_passage = (
                                f"[PROVENANCE FAILED: Fabricated chunk ID '{cid}' "
                                "and unverified passage]"
                            )
                            note = f"Provenance check failed: chunk ID '{cid}' does not exist."
                            methodology_notes = (
                                f"{methodology_notes} | {note}" if methodology_notes else note
                            )
                            cid = "invalid_chunk_provenance"
                            page_num = None

            elif rel == "not_addressed":
                if not is_exhaustive:
                    rel = "insufficient_evidence"
                    conf = min(conf, 0.5)
                    note = "Downgraded from not_addressed: only partial chunks examined."
                    methodology_notes = (
                        f"{methodology_notes} | {note}" if methodology_notes else note
                    )
                cid = "none"
                page_num = None
                claimed_passage = ""

            elif rel == "insufficient_evidence":
                if cid in chunk_lookup:
                    page_num = chunk_lookup[cid].page_number
                else:
                    cid = "none" if not cid or cid.lower() in ("none", "n/a", "") else cid
                    page_num = None

            finding = EvidenceFinding(
                finding_id=f"fnd_{uuid4().hex[:12]}",
                topic_id=topic_name,
                doc_id=doc_id,
                doc_title=doc_title,
                chunk_id=cid,
                page_number=int(page_num) if page_num is not None else None,
                claim=claim_text or "No specific claim stated.",
                supporting_passage=claimed_passage,
                relationship=rel,  # type: ignore[arg-type]
                confidence=conf,
                methodology_notes=methodology_notes,
            )
            extracted_findings.append(finding)

        return extracted_findings

    def _create_fallback_findings(
        self,
        doc_id: str,
        doc_title: str,
        aspects: list[str],
        chunks: list[Chunk],
        is_exhaustive: bool = True,
    ) -> list[EvidenceFinding]:
        """Create conservative, grounded findings when LLM extraction fails or is unavailable."""
        findings: list[EvidenceFinding] = []
        first_chunk = chunks[0] if chunks else None

        for aspect in aspects:
            if first_chunk:
                snippet = first_chunk.content[:200].strip()
                findings.append(
                    EvidenceFinding(
                        finding_id=f"fnd_{uuid4().hex[:12]}",
                        topic_id=aspect,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        chunk_id=first_chunk.chunk_id,
                        page_number=first_chunk.page_number,
                        claim=f"Discusses relevant domain concepts for {aspect}.",
                        supporting_passage=snippet,
                        relationship="insufficient_evidence",
                        confidence=0.3,
                        methodology_notes="Fallback finding created due to extraction failure.",
                    )
                )
            else:
                rel = "not_addressed" if is_exhaustive else "insufficient_evidence"
                conf = 1.0 if is_exhaustive else 0.5
                findings.append(
                    EvidenceFinding(
                        finding_id=f"fnd_{uuid4().hex[:12]}",
                        topic_id=aspect,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        chunk_id="none",
                        page_number=None,
                        claim=f"No evidence retrieved for {aspect}.",
                        supporting_passage="",
                        relationship=cast(EvidenceRelationship, rel),
                        confidence=conf,
                    )
                )
        return findings

    def _match_topic_name(self, raw_name: str, valid_aspects: list[str]) -> str:
        """Fuzzy match a model-emitted topic name to the canonical comparison aspects."""
        raw_clean = raw_name.strip().lower()
        for aspect in valid_aspects:
            if aspect.lower() in raw_clean or raw_clean in aspect.lower():
                return aspect
        return valid_aspects[0]

    def _synthesize_topic_relationships(
        self,
        topic_name: str,
        findings: list[EvidenceFinding],
        doc_records: dict[str, DocumentRecord],
    ) -> tuple[str, str, list[str]]:
        """Synthesize cross-source agreements, disagreements, and evidence gaps."""
        supports_findings = [
            f
            for f in findings
            if f.relationship == "supports"
            and not f.supporting_passage.startswith("[PROVENANCE FAILED")
        ]
        contradicts_findings = [
            f
            for f in findings
            if f.relationship == "contradicts"
            and not f.supporting_passage.startswith("[PROVENANCE FAILED")
        ]
        qualifies_findings = [
            f
            for f in findings
            if f.relationship == "qualifies"
            and not f.supporting_passage.startswith("[PROVENANCE FAILED")
        ]
        na_findings = [f for f in findings if f.relationship == "not_addressed"]
        insuff_findings = [f for f in findings if f.relationship == "insufficient_evidence"]

        # Helper to extract environment/methodology indicators
        def _get_env_markers(f: EvidenceFinding) -> set[str]:
            text = f"{f.claim} {f.supporting_passage} {f.methodology_notes or ''}".lower()
            markers = set()
            for kw in (
                "gpu",
                "cpu",
                "tpu",
                "a100",
                "v100",
                "h100",
                "arm",
                "raspberry",
                "mobile",
                "fp16",
                "int4",
                "int8",
            ):
                if kw in text:
                    markers.add(kw)
            return markers

        # 1. Agreements synthesis
        if len(supports_findings) >= 2:
            env_map = {f.doc_title: _get_env_markers(f) for f in supports_findings}
            distinct_envs = False
            docs_list = list(env_map.keys())
            for i in range(len(docs_list)):
                for j in range(i + 1, len(docs_list)):
                    e1, e2 = env_map[docs_list[i]], env_map[docs_list[j]]
                    if e1 and e2 and not (e1 & e2):
                        distinct_envs = True
                        break

            if distinct_envs:
                doc_notes = [
                    f"{f.doc_title}: {f.methodology_notes or 'custom setup'}"
                    for f in supports_findings[:2]
                ]
                sources_str = ", ".join(s.doc_title for s in supports_findings[:3])
                notes_str = "; ".join(doc_notes)
                agreements = (
                    f"Sources ({sources_str}) both support '{topic_name}', "
                    f"but evaluate under distinct experimental conditions ({notes_str}). "
                    "Findings are complementary rather than an identical consensus."
                )
            else:
                agreements = (
                    f"Verified consensus across {len(supports_findings)} sources "
                    f"({', '.join(s.doc_title for s in supports_findings[:3])}): "
                    f"Corroborates claims regarding '{topic_name}' with verified passages."
                )
        elif len(supports_findings) == 1:
            sf = supports_findings[0]
            agreements = (
                f"Supported primarily by {sf.doc_title}. "
                "Other examined sources provide complementary or unstated evidence."
            )
        else:
            agreements = f"No cross-source consensus established for '{topic_name}'."

        # 2. Disagreements synthesis
        if contradicts_findings:
            has_methodological_variance = False
            for cf in contradicts_findings:
                cf_env = _get_env_markers(cf)
                notes_lower = (cf.methodology_notes or "").lower()
                for sf in supports_findings:
                    sf_env = _get_env_markers(sf)
                    if cf_env and sf_env and not (cf_env & sf_env):
                        has_methodological_variance = True
                        break
                if any(
                    w in notes_lower
                    for w in (
                        "baseline",
                        "distinct",
                        "different hardware",
                        "cpu vs gpu",
                        "raspberry",
                    )
                ):
                    has_methodological_variance = True

            contra_doc_names = list({f.doc_title for f in contradicts_findings})
            c_docs_str = ", ".join(contra_doc_names)

            if has_methodological_variance:
                disagreements = (
                    f"Conflicting or opposing evidence identified in {c_docs_str}, "
                    "though divergence reflects distinct experimental setups or baseline "
                    "conditions rather than a direct empirical conflict."
                )
            else:
                disagreements = (
                    f"Conflicting or opposing evidence identified in {c_docs_str}. "
                    "Findings indicate differing results or divergent claims under "
                    "comparable evaluation scopes."
                )
        elif qualifies_findings:
            q_docs = list({f.doc_title for f in qualifies_findings})
            disagreements = (
                f"Conditional variance identified: {', '.join(q_docs)} "
                "qualifies findings based on specific parameters or setups."
            )
        else:
            disagreements = "No direct contradictions detected among the retrieved evidence."

        # 3. Gaps synthesis
        evidence_gaps: list[str] = []
        if na_findings:
            na_docs = list({f.doc_title for f in na_findings})
            na_str = ", ".join(na_docs)
            evidence_gaps.append(
                f"Verified absence of coverage in {len(na_docs)} source(s): {na_str}."
            )
        if insuff_findings:
            insuff_docs = list({f.doc_title for f in insuff_findings})
            ins_str = ", ".join(insuff_docs)
            evidence_gaps.append(
                f"Insufficient retrieved evidence in {len(insuff_docs)} source(s): {ins_str}."
            )
        if not supports_findings and not contradicts_findings and not qualifies_findings:
            evidence_gaps.append(
                f"Overall evidence is weak or unstated across documents for '{topic_name}'."
            )

        return agreements, disagreements, evidence_gaps

    def _generate_executive_summary(
        self,
        query: str,
        doc_summaries: list[dict[str, Any]],
        topics: list[ComparisonTopic],
        rel_counts: dict[str, int],
    ) -> str:
        """Produce an executive overview of the multi-document investigation."""
        total_docs = len(doc_summaries)
        supports = rel_counts.get("supports", 0)
        contradicts = rel_counts.get("contradicts", 0)
        qualifies = rel_counts.get("qualifies", 0)

        overview = (
            f"This investigation evaluated '{query}' across {total_docs} document sources, "
            f"examining {len(topics)} comparative dimensions. "
            f"Analysis surfaced {supports} supporting findings, "
            f"{contradicts} contradictory or opposing findings, "
            f"and {qualifies} qualified or condition-dependent statements. "
        )

        if contradicts > 0:
            overview += (
                "Key disagreements were identified between sources; users should inspect "
                "the comparison matrix for methodological differences. "
            )
        else:
            overview += (
                "Sources show strong mutual corroboration with minimal direct contradiction. "
            )

        return overview

    def _generate_methodology_synthesis(self, topics: list[ComparisonTopic]) -> str:
        """Summarize methodological observations and potential confounding factors."""
        notes: list[str] = []
        for t in topics:
            for f in t.findings:
                if f.methodology_notes and f.methodology_notes not in notes:
                    notes.append(f"{f.doc_title}: {f.methodology_notes}")

        if not notes:
            return (
                "Methodological details were not explicitly extracted "
                "or differ primarily by experimental dataset."
            )

        return "Notable methodological contexts:\n- " + "\n- ".join(notes[:8])
