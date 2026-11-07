"""Core application service orchestrating cross-document evidence investigations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence
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


class InvestigationService:
    """Orchestrates multi-document retrieval, evidence extraction, and cross-source synthesis."""

    def __init__(
        self,
        investigation_repository: InvestigationRepository,
        document_repository: DocumentRepository,
        query_service: DocumentQueryService,
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
    ) -> InvestigationRecord:
        """Validate documents and create a persistent investigation record.

        Raises:
            ValueError: If query is empty, document_ids is empty, or any document is invalid.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Investigation query must be a non-empty string.")

        clean_doc_ids = [d.strip() for d in document_ids if isinstance(d, str) and d.strip()]
        if not clean_doc_ids:
            raise ValueError("At least one valid document ID must be selected for investigation.")

        # Validate document existence and readiness
        for doc_id in clean_doc_ids:
            doc = self.document_repository.get_document(doc_id)
            if doc is None:
                raise ValueError(f"Document '{doc_id}' not found in repository.")
            if doc.status == DocumentStatus.FAILED:
                raise ValueError(
                    f"Document '{doc_id}' is in FAILED state and cannot be investigated."
                )
            if doc.chunk_count == 0:
                raise ValueError(f"Document '{doc_id}' has 0 chunks and cannot be investigated.")

        aspects = (
            [a.strip() for a in comparison_aspects if isinstance(a, str) and a.strip()]
            if comparison_aspects
            else list(DEFAULT_COMPARISON_ASPECTS)
        )
        if not aspects:
            aspects = list(DEFAULT_COMPARISON_ASPECTS)

        effective_title = (title.strip() if title and title.strip() else query.strip()[:80])

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

    def _run_investigation_pipeline(
        self, inv: InvestigationRecord
    ) -> InvestigationReport:
        """Run retrieval, structured extraction, cross-document comparison, and report compilation."""
        doc_summaries: list[dict[str, Any]] = []
        doc_chunks_map: dict[str, list[Chunk]] = {}
        doc_records: dict[str, DocumentRecord] = {}

        # 1. Gather document records and evidence chunks per document
        for doc_id in inv.document_ids:
            doc = self.document_repository.get_document(doc_id)
            if doc is None:
                continue
            doc_records[doc_id] = doc

            # Retrieve top relevant chunks for this document
            chunks = self._retrieve_document_evidence(
                doc_id=doc_id,
                query=inv.query,
                aspects=inv.comparison_aspects,
            )
            doc_chunks_map[doc_id] = chunks

        # 2. Extract structured findings from each document
        extracted_findings_by_topic: dict[str, list[EvidenceFinding]] = {
            aspect: [] for aspect in inv.comparison_aspects
        }

        doc_findings_count: dict[str, int] = {doc_id: 0 for doc_id in inv.document_ids}

        for doc_id in inv.document_ids:
            chunks = doc_chunks_map.get(doc_id, [])
            doc_record = doc_records.get(doc_id)
            doc_title = doc_record.filename if doc_record else doc_id

            findings = self._extract_findings_for_document(
                doc_id=doc_id,
                doc_title=doc_title,
                query=inv.query,
                aspects=inv.comparison_aspects,
                chunks=chunks,
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
                    "chunk_count": len(chunks),
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

            # Ensure all documents are represented (add not_addressed if document has no findings)
            covered_doc_ids = {f.doc_id for f in findings_for_aspect}
            for doc_id in inv.document_ids:
                if doc_id not in covered_doc_ids:
                    doc_rec = doc_records.get(doc_id)
                    d_title = doc_rec.filename if doc_rec else doc_id
                    missing_finding = EvidenceFinding(
                        finding_id=f"fnd_na_{uuid4().hex[:8]}",
                        topic_id=aspect_name,
                        doc_id=doc_id,
                        doc_title=d_title,
                        chunk_id="none",
                        page_number=None,
                        claim=f"Document does not discuss {aspect_name}.",
                        supporting_passage="",
                        relationship="not_addressed",
                        confidence=1.0,
                        methodology_notes="No evidence found in retrieved chunks.",
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
    ) -> list[Chunk]:
        """Retrieve relevant evidence chunks specifically for a given document."""
        # 1. First retrieve all chunks from repository as ground truth
        all_chunks = self.document_repository.get_chunks(doc_id)
        if not all_chunks:
            return []

        if len(all_chunks) <= 15:
            # Small document: include all chunks to ensure exhaustive evidence coverage
            return all_chunks

        # For larger documents, query BM25 and dense indices to surface top relevant chunks
        selected_chunk_ids: set[str] = set()
        retrieved_chunks: list[Chunk] = []

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

        # Ensure first chunk (often title / abstract) is always included
        if all_chunks and all_chunks[0].chunk_id not in selected_chunk_ids:
            retrieved_chunks.insert(0, all_chunks[0])
            selected_chunk_ids.add(all_chunks[0].chunk_id)

        # Fallback if retrieval returned nothing
        if not retrieved_chunks:
            retrieved_chunks = all_chunks[:10]

        return retrieved_chunks

    def _extract_findings_for_document(
        self,
        doc_id: str,
        doc_title: str,
        query: str,
        aspects: list[str],
        chunks: list[Chunk],
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
                    "You are an expert evidence analyst for the STING Evidence Intelligence Workbench. "
                    "Your mission is to extract structured, verifiable findings from the supplied source text. "
                    "CRITICAL PROVENANCE RULES:\n"
                    "1. Extract ONLY statements directly supported by the text.\n"
                    "2. You MUST provide the exact CHUNK_ID where the evidence appears.\n"
                    "3. You MUST provide the EXACT verbatim supporting passage from that chunk.\n"
                    "4. Classify each finding's relationship:\n"
                    "   - 'supports': Affirms, provides positive evidence or corroborates the aspect.\n"
                    "   - 'contradicts': Opposes, refutes, presents conflicting data or challenges the aspect.\n"
                    "   - 'qualifies': Provides conditional, situational or bounded evidence (e.g. only works under X).\n"
                    "   - 'insufficient_evidence': Passage touches the topic but does not provide adequate proof.\n"
                    "   - 'not_addressed': The source does not mention or cover this aspect.\n"
                    "5. Output valid JSON matching the required schema. Never hallucinate fake quotes or IDs."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Investigation Topic / Query: \"{query}\"\n\n"
                    f"Source Document: \"{doc_title}\" (ID: {doc_id})\n\n"
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
                    '      "relationship": "supports" | "contradicts" | "qualifies" | "insufficient_evidence" | "not_addressed",\n'
                    '      "confidence": <float between 0.0 and 1.0>,\n'
                    '      "methodology_notes": "<optional notes on dataset, setup or conditions>"\n'
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
            return self._create_fallback_findings(doc_id, doc_title, aspects, chunks)

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

            # Fail-closed provenance validation
            target_chunk = chunk_lookup.get(cid)
            page_num = item.get("page_number")
            conf = float(item.get("confidence", 0.8))

            if target_chunk is not None:
                if page_num is None:
                    page_num = target_chunk.page_number
                # Verify that supporting passage is grounded in chunk content
                if claimed_passage and claimed_passage.lower() not in target_chunk.content.lower():
                    # Quote mismatch or hallucinated passage -> downgrade relationship
                    rel = "insufficient_evidence"
                    conf = min(conf, 0.4)
            elif cid.lower() in ("none", "n/a", ""):
                # Document did not find evidence
                rel = "not_addressed" if not claimed_passage else "insufficient_evidence"
                conf = 1.0
            else:
                # Hallucinated chunk ID -> search if passage exists in any chunk
                found_chunk = None
                if claimed_passage:
                    for chk in chunks:
                        if claimed_passage.lower() in chk.content.lower():
                            found_chunk = chk
                            break
                if found_chunk:
                    cid = found_chunk.chunk_id
                    page_num = found_chunk.page_number
                else:
                    cid = "unverified"
                    rel = "insufficient_evidence"
                    conf = 0.2

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
                methodology_notes=item.get("methodology_notes"),
            )
            extracted_findings.append(finding)

        if not extracted_findings:
            return self._create_fallback_findings(doc_id, doc_title, aspects, chunks)

        return extracted_findings

    def _create_fallback_findings(
        self,
        doc_id: str,
        doc_title: str,
        aspects: list[str],
        chunks: list[Chunk],
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
                        relationship="qualifies",
                        confidence=0.5,
                        methodology_notes="Extracted via fallback heuristic.",
                    )
                )
            else:
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
                        relationship="insufficient_evidence",
                        confidence=1.0,
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
        supports_docs: list[str] = []
        contradicts_docs: list[str] = []
        qualifies_docs: list[str] = []
        gaps_docs: list[str] = []

        for f in findings:
            if f.relationship == "supports":
                supports_docs.append(f.doc_title)
            elif f.relationship == "contradicts":
                contradicts_docs.append(f.doc_title)
            elif f.relationship == "qualifies":
                qualifies_docs.append(f.doc_title)
            elif f.relationship in ("not_addressed", "insufficient_evidence"):
                gaps_docs.append(f.doc_title)

        # 1. Agreements synthesis
        if len(supports_docs) >= 2:
            agreements = (
                f"Consensus across {len(supports_docs)} sources ({', '.join(supports_docs[:3])}): "
                f"Directly supports claims regarding '{topic_name}' with verified passages."
            )
        elif len(supports_docs) == 1:
            agreements = (
                f"Supported primarily by {supports_docs[0]}. "
                "Other examined sources provide complementary or unstated evidence."
            )
        else:
            agreements = f"No cross-source consensus established for '{topic_name}'."

        # 2. Disagreements synthesis
        if contradicts_docs:
            disagreements = (
                f"Conflicting or opposing evidence identified in {', '.join(contradicts_docs)}. "
                "Findings indicate differing results or divergent claims on this aspect."
            )
        elif qualifies_docs:
            disagreements = (
                f"Conditional variance identified: {', '.join(qualifies_docs)} "
                "qualifies findings based on specific methodological parameters, setups, or constraints."
            )
        else:
            disagreements = "No direct contradictions detected among the retrieved evidence."

        # 3. Gaps
        evidence_gaps: list[str] = []
        if gaps_docs:
            evidence_gaps.append(
                f"Topic not addressed or insufficient evidence in: {', '.join(gaps_docs)}."
            )
        if not supports_docs and not contradicts_docs:
            evidence_gaps.append(
                f"Overall evidence is weak or unstated across all examined documents for '{topic_name}'."
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
            f"Analysis surfaced {supports} supporting findings, {contradicts} contradictory or opposing findings, "
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
                "Methodological details were not explicitly extracted or differ primarily by experimental dataset."
            )

        return "Notable methodological contexts:\n- " + "\n- ".join(notes[:8])
