"""Evidence-first query orchestrator and structured trace generator (STING V2-003)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

from reliability_lab.core.models import Chunk, RetrievalResult
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.indexing.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from reliability_lab.indexing.faiss_index import (
    IncompatibleEmbeddingIdentityError,
)
from reliability_lab.indexing.hybrid import HybridFusionConfig, HybridRetriever
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import BaseLLMClient, OllamaLLMClient
from reliability_lab.llm.models import LLMMessage
from reliability_lab.pipeline.prompt import build_rag_prompt
from reliability_lab.query.citation import CitationAliasMap, CitationResolver
from reliability_lab.query.models import (
    DenseCandidateTrace,
    FusedCandidateTrace,
    QueryTrace,
    SparseCandidateTrace,
)

logger = logging.getLogger(__name__)

STRUCTURED_ABSTENTION_DIRECTIVE: str = (
    "\n\nSTRUCTURED ABSTENTION DIRECTIVE:\n"
    "If the retrieved evidence does not contain sufficient facts to answer the question, "
    "output: [ABSTAIN: INSUFFICIENT_EVIDENCE] followed by a brief statement "
    "explaining what information is missing."
)


class DocumentNotFoundError(KeyError):
    """Raised when one or more selected documents do not exist."""


class UnprocessableDocumentError(ValueError):
    """Raised when a selected document cannot be queried (skipped, empty, or incompatible)."""


class DocumentConflictError(ValueError):
    """Raised when a selected document is currently processing or failed."""


class DocumentQueryService:
    """Orchestrates document-scoped retrieval, citation aliasing, abstention, and tracing."""

    def __init__(
        self,
        repository: DocumentRepository,
        ingestion_service: DocumentIngestionService,
        embedder: BaseEmbedder | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.repository = repository
        self.ingestion_service = ingestion_service
        self._embedder = embedder
        self._llm_client = llm_client

    @property
    def embedder(self) -> BaseEmbedder:
        """Return active embedding adapter, lazily initializing default if needed."""
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(local_files_only=True)
        return self._embedder

    @property
    def llm_client(self) -> BaseLLMClient:
        """Return active LLM client adapter, defaulting to OllamaLLMClient."""
        if self._llm_client is None:
            self._llm_client = OllamaLLMClient()
        return self._llm_client

    def _validate_lifecycle_and_get_documents(
        self,
        selected_doc_ids: Sequence[str],
        requested_strategy: str,
    ) -> tuple[dict[str, DocumentRecord], bool]:
        """Validate all selected documents against the document lifecycle matrix.

        Returns:
            Tuple of (doc_map, has_partial_document).

        Raises:
            DocumentNotFoundError: If any document ID is not in repository.
            UnprocessableDocumentError: If document is SKIPPED or PARTIAL under dense.
            DocumentConflictError: If document is still processing or FAILED.
        """
        doc_map: dict[str, DocumentRecord] = {}
        has_partial = False

        for doc_id in selected_doc_ids:
            doc = self.repository.get_document(doc_id)
            if doc is None:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found.")

            doc_map[doc_id] = doc

            # 1. Processing states -> HTTP 409 Conflict
            if (
                doc.status
                in (
                    DocumentStatus.UPLOADED,
                    DocumentStatus.EXTRACTING,
                    DocumentStatus.CHUNKING,
                    DocumentStatus.INDEXING,
                )
                or doc.indexing_status == IndexingStatus.PENDING
            ):
                raise DocumentConflictError(
                    f"Document '{doc_id}' is currently processing "
                    f"(status={doc.status}, indexing_status={doc.indexing_status})."
                )

            # 2. Failed states -> HTTP 409 Conflict
            if doc.status == DocumentStatus.FAILED or doc.indexing_status == IndexingStatus.FAILED:
                raise DocumentConflictError(f"Document '{doc_id}' processing failed.")

            # 3. Skipped state (empty content) -> HTTP 422 Unprocessable Entity
            if doc.status == DocumentStatus.READY and doc.indexing_status == IndexingStatus.SKIPPED:
                raise UnprocessableDocumentError(
                    f"Document '{doc_id}' was skipped during extraction (empty content). "
                    "Remove from scope."
                )

            # 4. Partial state (BM25 only, dense failed or missing)
            if doc.status == DocumentStatus.READY and doc.indexing_status == IndexingStatus.PARTIAL:
                if requested_strategy == "dense":
                    raise UnprocessableDocumentError(
                        f"Document '{doc_id}' only has lexical index available "
                        "(dense index missing)."
                    )
                has_partial = True

        return doc_map, has_partial

    def query(
        self,
        query_text: str,
        selected_doc_ids: Sequence[str],
        strategy: Literal["bm25", "dense", "hybrid"] | str = "hybrid",
        top_k: int = 5,
        candidate_depth: int = 60,
        min_dense_score: float | None = None,
        prompt_version: str = "v1",
    ) -> tuple[str, QueryTrace]:
        """Execute a document-grounded query, returning the generated answer and immutable trace.

        Args:
            query_text: User question.
            selected_doc_ids: Non-empty sequence of document IDs to query.
            strategy: Requested retrieval strategy ('bm25', 'dense', or 'hybrid').
            top_k: Number of final evidence chunks to supply to generation.
            candidate_depth: Number of intermediate candidates to retrieve per retriever.
            min_dense_score: Optional threshold for dense retrieval abstention.
            prompt_version: Template identifier for prompt construction.

        Returns:
            Tuple of (answer_text, query_trace).
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")
        if not selected_doc_ids:
            raise ValueError("selected_doc_ids must contain at least one document ID.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k}")
        if (
            isinstance(candidate_depth, bool)
            or not isinstance(candidate_depth, int)
            or candidate_depth < 1
        ):
            raise ValueError(
                f"candidate_depth must be a positive integer >= 1, got {candidate_depth}"
            )

        clean_query = query_text.strip()
        normalized_query = " ".join(clean_query.lower().split())
        requested_strategy = str(strategy).strip().lower()
        if requested_strategy not in ("bm25", "dense", "hybrid"):
            raise ValueError(
                f"Invalid retrieval strategy '{strategy}'. Must be 'bm25', 'dense', or 'hybrid'."
            )

        # Lifecycle validation
        doc_map, has_partial = self._validate_lifecycle_and_get_documents(
            selected_doc_ids, requested_strategy
        )

        effective_strategy = requested_strategy
        fallback_reason: str | None = None
        if requested_strategy == "hybrid" and has_partial:
            effective_strategy = "bm25"
            fallback_reason = "selected_documents_partially_indexed"

        trace_id = f"trc_{uuid4().hex}"
        filename_map = {doc_id: doc.filename for doc_id, doc in doc_map.items()}

        t_retrieval_start = time.perf_counter()

        # 1. Lexical retrieval across union corpus
        raw_sparse: list[RetrievalResult] = []
        if effective_strategy in ("bm25", "hybrid"):
            all_chunks: list[Chunk] = []
            for doc_id in selected_doc_ids:
                all_chunks.extend(self.repository.get_chunks(doc_id))
            if all_chunks:
                scoped_bm25 = BM25Retriever()
                scoped_bm25.index(all_chunks)
                raw_sparse = scoped_bm25.retrieve(clean_query, top_k=candidate_depth)

        # 2. Dense retrieval across selected documents
        raw_dense: list[RetrievalResult] = []
        dense_search_mode = "flat_exact"
        if effective_strategy in ("dense", "hybrid"):
            pooled_dense: list[RetrievalResult] = []
            has_hnsw = False
            dense_failed = False

            for doc_id in selected_doc_ids:
                try:
                    retriever = self.ingestion_service.get_faiss_retriever(doc_id, self.embedder)
                    if retriever._index_type == "hnsw":
                        has_hnsw = True
                    fetch_depth = candidate_depth * 2 if has_hnsw else candidate_depth
                    results = retriever.retrieve(clean_query, top_k=fetch_depth)
                    pooled_dense.extend(results)
                except IncompatibleEmbeddingIdentityError as exc:
                    if effective_strategy == "hybrid":
                        logger.warning(
                            "Hybrid fallback to bm25 due to incompatible embedding identity "
                            "on %s: %s",
                            doc_id,
                            exc,
                        )
                        effective_strategy = "bm25"
                        fallback_reason = "unverifiable_or_mismatched_embedding_identity"
                        dense_failed = True
                        pooled_dense = []
                        break
                    raise

            if not dense_failed and pooled_dense:
                # Global sort by score DESC, tie-break chunk_id ASC
                pooled_dense.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
                # Deduplicate by chunk_id
                seen_chunk_ids: set[str] = set()
                deduped_dense: list[RetrievalResult] = []
                for res in pooled_dense:
                    if res.chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(res.chunk.chunk_id)
                        deduped_dense.append(res)

                # Re-assign global contiguous ranks 1..N
                ranked_dense: list[RetrievalResult] = []
                for global_rank, res in enumerate(deduped_dense[:candidate_depth], start=1):
                    ranked_dense.append(
                        RetrievalResult(
                            chunk=res.chunk,
                            score=res.score,
                            rank=global_rank,
                            retriever_name=res.retriever_name,
                            metadata=res.metadata,
                        )
                    )
                raw_dense = ranked_dense
                dense_search_mode = "approximate_pooled" if has_hnsw else "flat_exact"

        # 3. Hybrid RRF fusion
        raw_fused: list[RetrievalResult] = []
        if effective_strategy == "hybrid":
            cfg = HybridFusionConfig(
                rrf_k=60,
                candidate_depth=candidate_depth,
                sparse_name="bm25",
                dense_name="faiss_vector",
                hybrid_name="hybrid_rrf",
            )
            raw_fused = HybridRetriever.fuse_candidates(
                sparse_candidates=raw_sparse,
                dense_candidates=raw_dense,
                top_k=top_k,
                config=cfg,
            )

        retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

        # Build candidate traces
        sparse_traces = tuple(
            SparseCandidateTrace(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                content=r.chunk.content,
                page_number=r.chunk.page_number,
                char_start=r.chunk.char_start,
                char_end=r.chunk.char_end,
                score=float(r.score),
                rank=r.rank,
            )
            for r in raw_sparse
        )
        dense_traces = tuple(
            DenseCandidateTrace(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                content=r.chunk.content,
                page_number=r.chunk.page_number,
                char_start=r.chunk.char_start,
                char_end=r.chunk.char_end,
                score=float(r.score),
                rank=r.rank,
            )
            for r in raw_dense
        )
        fused_traces = tuple(
            FusedCandidateTrace(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                content=r.chunk.content,
                page_number=r.chunk.page_number,
                char_start=r.chunk.char_start,
                char_end=r.chunk.char_end,
                fused_score=float(r.score),
                final_rank=r.rank,
                sparse_rank=r.metadata.get("sparse_rank"),
                sparse_score=r.metadata.get("sparse_score"),
                dense_rank=r.metadata.get("dense_rank"),
                dense_score=r.metadata.get("dense_score"),
                rrf_contributions=tuple(sorted(r.metadata.get("rrf_contributions", {}).items())),
            )
            for r in raw_fused
        )

        # Tier-1 Retrieval Abstention Evaluation
        retrieval_abstained = False
        retrieval_abstention_reason: str | None = None

        if effective_strategy == "bm25":
            if len(raw_sparse) == 0:
                retrieval_abstained = True
                retrieval_abstention_reason = "no_sparse_candidates_found"
            elif all(c.score <= 0.0 for c in raw_sparse):
                retrieval_abstained = True
                retrieval_abstention_reason = "no_positive_bm25_candidates"

        elif effective_strategy == "dense":
            if len(raw_dense) == 0:
                retrieval_abstained = True
                retrieval_abstention_reason = "no_dense_candidates_found"
            elif min_dense_score is not None and raw_dense[0].score < min_dense_score:
                retrieval_abstained = True
                retrieval_abstention_reason = "dense_score_below_threshold"

        elif effective_strategy == "hybrid":
            if len(raw_fused) == 0:
                retrieval_abstained = True
                retrieval_abstention_reason = "no_fused_candidates_found"

        if retrieval_abstained:
            # Skip LLM generation completely
            answer = (
                "I am unable to answer because no relevant evidence was retrieved from "
                "the selected documents."
            )
            total_latency_ms = retrieval_latency_ms
            trace = QueryTrace(
                trace_id=trace_id,
                query_text=clean_query,
                normalized_query=normalized_query,
                selected_doc_ids=tuple(selected_doc_ids),
                requested_strategy=requested_strategy,
                effective_strategy=effective_strategy,
                fallback_reason=fallback_reason,
                sparse_candidates=sparse_traces,
                dense_candidates=dense_traces,
                fused_candidates=fused_traces,
                evidence_chunks=(),
                resolved_citations=(),
                unresolved_citations=(),
                malformed_citations=(),
                abstained=True,
                retrieval_abstained=True,
                retrieval_abstention_reason=retrieval_abstention_reason,
                generation_abstained=False,
                generation_abstention_reason=None,
                provider=self.llm_client.provider,
                model_name=self.llm_client.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                dense_search_mode=dense_search_mode,
            )
            return answer, trace

        # Select evidence chunks based on effective strategy
        if effective_strategy == "hybrid":
            evidence_results = raw_fused[:top_k]
        elif effective_strategy == "dense":
            evidence_results = raw_dense[:top_k]
        else:
            evidence_results = raw_sparse[:top_k]

        evidence_chunks_list = [r.chunk for r in evidence_results]
        alias_map, evidence_traces_list = CitationAliasMap.from_chunks(
            evidence_chunks_list, filename_map
        )
        evidence_traces = tuple(evidence_traces_list)

        # Prompt construction
        prompt_evidence = alias_map.to_prompt_evidence()
        sys_prompt, user_prompt = build_rag_prompt(
            query_text=clean_query,
            evidence=prompt_evidence,
            prompt_version=prompt_version,
        )
        sys_prompt_augmented = sys_prompt + STRUCTURED_ABSTENTION_DIRECTIVE

        messages = [
            LLMMessage(role="system", content=sys_prompt_augmented),
            LLMMessage(role="user", content=user_prompt),
        ]

        # LLM Generation
        t_gen_start = time.perf_counter()
        llm_response = self.llm_client.generate(messages)
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
        total_latency_ms = retrieval_latency_ms + generation_latency_ms

        answer_text = llm_response.text

        # Tier-2 Generation Abstention Evaluation
        gen_abstain_match = re.search(r"\[ABSTAIN:\s*([^\]]+)\]", answer_text, re.IGNORECASE)
        if gen_abstain_match:
            generation_abstained = True
            generation_abstention_reason = gen_abstain_match.group(1).strip().lower()
            abstained = True
        else:
            generation_abstained = False
            generation_abstention_reason = None
            abstained = False

        # Citation Resolution
        resolver = CitationResolver(alias_map)
        resolved_cits, unresolved_cits, malformed_cits = resolver.resolve(answer_text)

        prompt_tokens = llm_response.usage.prompt_tokens if llm_response.usage else None
        completion_tokens = llm_response.usage.completion_tokens if llm_response.usage else None

        trace = QueryTrace(
            trace_id=trace_id,
            query_text=clean_query,
            normalized_query=normalized_query,
            selected_doc_ids=tuple(selected_doc_ids),
            requested_strategy=requested_strategy,
            effective_strategy=effective_strategy,
            fallback_reason=fallback_reason,
            sparse_candidates=sparse_traces,
            dense_candidates=dense_traces,
            fused_candidates=fused_traces,
            evidence_chunks=evidence_traces,
            resolved_citations=resolved_cits,
            unresolved_citations=unresolved_cits,
            malformed_citations=malformed_cits,
            abstained=abstained,
            retrieval_abstained=False,
            retrieval_abstention_reason=None,
            generation_abstained=generation_abstained,
            generation_abstention_reason=generation_abstention_reason,
            provider=self.llm_client.provider,
            model_name=self.llm_client.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            dense_search_mode=dense_search_mode,
        )

        return answer_text, trace
