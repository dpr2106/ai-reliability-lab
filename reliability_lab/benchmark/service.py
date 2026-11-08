"""Service coordinating RAG Evaluation Dataset creation, validation, import, and export."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from reliability_lab.benchmark.models import BenchmarkDataset, BenchmarkDatasetRecord
from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
from reliability_lab.core.models import Chunk, Query
from reliability_lab.ingestion.repository import DocumentRepository

logger = logging.getLogger(__name__)


def _dedup_list(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class DatasetValidationError(ValueError):
    """Raised when evaluation dataset or benchmark queries fail validation."""


class BenchmarkDatasetService:
    """Business logic and validation service for RAG evaluation datasets."""

    def __init__(
        self,
        dataset_repository: BenchmarkDatasetRepository,
        document_repository: DocumentRepository | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.document_repository = document_repository

    def _get_corpus_chunks_for_scope(self, document_scope: list[str]) -> list[Chunk]:
        """Fetch all chunks for scoped document IDs."""
        if not self.document_repository or not document_scope:
            return []
        chunks: list[Chunk] = []
        for doc_id in document_scope:
            doc_chunks = self.document_repository.get_chunks(doc_id)
            chunks.extend(doc_chunks)
        return chunks

    def _validate_queries_against_scope(
        self,
        queries: Sequence[Query],
        document_scope: list[str],
        validate_chunk_existence: bool = True,
    ) -> None:
        """Enforce strict validation of query references against scoped documents."""
        seen_qids: set[str] = set()

        if self.document_repository and document_scope:
            # Verify document IDs exist and are in READY state
            for doc_id in document_scope:
                doc = self.document_repository.get_document(doc_id)
                if doc is None:
                    raise DatasetValidationError(
                        f"Referenced document '{doc_id}' does not exist in repository."
                    )
                status_val = getattr(doc, "status", None)
                status_str = (
                    status_val.value
                    if (status_val is not None and hasattr(status_val, "value"))
                    else str(status_val or "")
                )
                if status_str.upper() != "READY":
                    raise DatasetValidationError(
                        f"Referenced document '{doc_id}' is not in ready state "
                        f"(current status: '{status_str}')."
                    )

            if validate_chunk_existence:
                valid_chunks = self._get_corpus_chunks_for_scope(document_scope)
                valid_chunk_ids = {c.chunk_id for c in valid_chunks}

                for q in queries:
                    for cid in q.relevant_chunk_ids:
                        if cid not in valid_chunk_ids:
                            raise DatasetValidationError(
                                f"Query '{q.query_id}' references unknown chunk '{cid}' "
                                f"not found in document scope."
                            )
                    for cid in q.relevance_grades:
                        if cid not in valid_chunk_ids:
                            raise DatasetValidationError(
                                f"Query '{q.query_id}' has relevance grade for unknown "
                                f"chunk '{cid}' not found in document scope."
                            )
                    for cit in q.required_citations:
                        # Check mapped chunk ID or citation ID itself
                        mapped_cid = q.metadata.get("citation_to_chunk_id", {}).get(cit, cit)
                        if mapped_cid not in valid_chunk_ids and cit not in valid_chunk_ids:
                            # If explicitly mapped or matches chunk ID
                            if cit in q.metadata.get("citation_to_chunk_id", {}):
                                raise DatasetValidationError(
                                    f"Query '{q.query_id}' requires citation '{cit}' targeting "
                                    f"unknown chunk '{mapped_cid}' not found in document scope."
                                )

        for q in queries:
            if not isinstance(q.query_id, str) or not q.query_id.strip():
                raise DatasetValidationError("Query ID must be a non-empty string.")
            if q.query_id in seen_qids:
                raise DatasetValidationError(f"Duplicate query ID '{q.query_id}' detected.")
            seen_qids.add(q.query_id)

            if not isinstance(q.text, str) or not q.text.strip():
                raise DatasetValidationError(
                    f"Query text must be non-empty for query '{q.query_id}'."
                )

            if q.metadata is not None and not isinstance(q.metadata, dict):
                raise DatasetValidationError(
                    f"Metadata must be a dictionary in query '{q.query_id}'."
                )

            if q.required_citations is not None:
                if not isinstance(q.required_citations, (list, tuple)):
                    raise DatasetValidationError(
                        f"required_citations must be a list in query '{q.query_id}'."
                    )
                seen_cits: set[str] = set()
                for cit in q.required_citations:
                    if not isinstance(cit, str) or not cit.strip():
                        raise DatasetValidationError(
                            f"Citation identifier must be non-empty string in query '{q.query_id}'."
                        )
                    if cit in seen_cits:
                        raise DatasetValidationError(
                            f"Duplicate citation identifier '{cit}' in query '{q.query_id}'."
                        )
                    seen_cits.add(cit)

            for cid, grade in q.relevance_grades.items():
                if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
                    raise DatasetValidationError(
                        f"Relevance grade for chunk '{cid}' in query '{q.query_id}' "
                        f"must be a non-negative integer."
                    )

    def create_dataset(
        self,
        name: str,
        description: str | None = None,
        document_scope: list[str] | None = None,
        is_synthetic: bool = False,
        metadata: dict[str, Any] | None = None,
        queries: Sequence[Query] | None = None,
        dataset_id: str | None = None,
        validate_chunk_existence: bool = True,
    ) -> BenchmarkDataset:
        """Create and validate a new benchmark dataset."""
        scope = _dedup_list(document_scope or [])
        q_list = list(queries or [])

        self._validate_queries_against_scope(
            q_list, scope, validate_chunk_existence=validate_chunk_existence
        )

        return self.dataset_repository.create_dataset(
            name=name,
            description=description,
            document_scope=scope,
            is_synthetic=is_synthetic,
            metadata=metadata,
            queries=q_list,
            dataset_id=dataset_id,
        )

    def get_dataset(self, dataset_id: str) -> BenchmarkDataset:
        """Fetch dataset by ID or raise DatasetValidationError."""
        ds = self.dataset_repository.get_dataset(dataset_id)
        if ds is None:
            raise DatasetValidationError(f"Benchmark dataset '{dataset_id}' not found.")
        return ds

    def list_datasets(self, limit: int = 50, offset: int = 0) -> list[BenchmarkDatasetRecord]:
        """List benchmark datasets."""
        # Check if sample seed is needed
        self.dataset_repository.seed_sample_dataset_if_empty()
        return self.dataset_repository.list_datasets(limit=limit, offset=offset)

    def update_dataset(
        self,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        document_scope: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkDatasetRecord:
        """Update dataset metadata."""
        deduped_scope: list[str] | None = None
        if document_scope is not None:
            deduped_scope = _dedup_list(document_scope)
            if self.document_repository:
                for doc_id in deduped_scope:
                    doc = self.document_repository.get_document(doc_id)
                    if doc is None:
                        raise DatasetValidationError(
                            f"Referenced document '{doc_id}' does not exist in repository."
                        )
                    status_val = getattr(doc, "status", None)
                    status_str = (
                        status_val.value
                        if (status_val is not None and hasattr(status_val, "value"))
                        else str(status_val or "")
                    )
                    if status_str.upper() != "READY":
                        raise DatasetValidationError(
                            f"Referenced document '{doc_id}' is not in ready state "
                            f"(current status: '{status_str}')."
                        )

        return self.dataset_repository.update_dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            document_scope=deduped_scope,
            metadata=metadata,
        )

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its queries."""
        return self.dataset_repository.delete_dataset(dataset_id)

    def add_query(
        self,
        dataset_id: str,
        query: Query,
        validate_chunk_existence: bool = True,
    ) -> Query:
        """Add a single query to a dataset with reference validation."""
        ds_rec = self.dataset_repository.get_dataset_record(dataset_id)
        if not ds_rec:
            raise DatasetValidationError(f"Benchmark dataset '{dataset_id}' not found.")

        self._validate_queries_against_scope(
            [query], ds_rec.document_scope, validate_chunk_existence=validate_chunk_existence
        )
        return self.dataset_repository.add_query(dataset_id, query)

    def update_query(
        self,
        dataset_id: str,
        query: Query,
        validate_chunk_existence: bool = True,
    ) -> Query:
        """Update a query with reference validation."""
        ds_rec = self.dataset_repository.get_dataset_record(dataset_id)
        if not ds_rec:
            raise DatasetValidationError(f"Benchmark dataset '{dataset_id}' not found.")

        self._validate_queries_against_scope(
            [query], ds_rec.document_scope, validate_chunk_existence=validate_chunk_existence
        )
        return self.dataset_repository.update_query(dataset_id, query)

    def delete_query(self, dataset_id: str, query_id: str) -> bool:
        """Delete a query from a dataset."""
        return self.dataset_repository.delete_query(dataset_id, query_id)

    def import_jsonl(
        self,
        name: str,
        jsonl_content: str,
        description: str | None = None,
        document_scope: list[str] | None = None,
        is_synthetic: bool = False,
        validate_chunk_existence: bool = True,
    ) -> BenchmarkDataset:
        """Parse and import benchmark queries from a JSONL string."""
        if not jsonl_content.strip():
            raise DatasetValidationError("Import content cannot be empty.")

        scope_raw = list(document_scope or [])
        scope = _dedup_list(scope_raw)

        if self.document_repository and scope:
            for doc_id in scope:
                doc = self.document_repository.get_document(doc_id)
                if doc is None:
                    raise DatasetValidationError(
                        f"Referenced document '{doc_id}' does not exist in repository."
                    )
                status_val = getattr(doc, "status", None)
                status_str = (
                    status_val.value
                    if (status_val is not None and hasattr(status_val, "value"))
                    else str(status_val or "")
                )
                if status_str.upper() != "READY":
                    raise DatasetValidationError(
                        f"Referenced document '{doc_id}' is not in ready state "
                        f"(current status: '{status_str}')."
                    )

        valid_chunks = (
            self._get_corpus_chunks_for_scope(scope)
            if (self.document_repository and scope and validate_chunk_existence)
            else None
        )

        valid_chunk_ids = {c.chunk_id for c in valid_chunks} if valid_chunks is not None else None

        queries: list[Query] = []
        seen_qids: set[str] = set()

        for line_num, line in enumerate(jsonl_content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Malformed JSON on line {line_num}: {exc}") from exc

            if not isinstance(record, dict):
                raise DatasetValidationError(f"Line {line_num} must be a JSON object.")

            qid = record.get("query_id")
            if not isinstance(qid, str) or not qid.strip():
                raise DatasetValidationError(f"Missing query_id on line {line_num}.")
            if qid in seen_qids:
                raise DatasetValidationError(f"Duplicate query_id '{qid}' on line {line_num}.")
            seen_qids.add(qid)

            text = record.get("query") or record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise DatasetValidationError(f"Query text missing or empty on line {line_num}.")

            # Parse relevance judgments
            rel_grades: dict[str, int] = {}
            rel_chunk_ids: list[str] = []

            if "relevance_judgments" in record and record["relevance_judgments"] is not None:
                if not isinstance(record["relevance_judgments"], list):
                    raise DatasetValidationError(
                        f"relevance_judgments must be a list in query '{qid}' on line {line_num}."
                    )
                for j in record["relevance_judgments"]:
                    if not isinstance(j, dict) or "chunk_id" not in j or "grade" not in j:
                        raise DatasetValidationError(
                            f"Malformed relevance judgment in query '{qid}' on line {line_num}."
                        )
                    cid = str(j["chunk_id"]).strip()
                    if not cid:
                        raise DatasetValidationError(
                            f"Empty chunk_id in relevance_judgments for query '{qid}' "
                            f"on line {line_num}."
                        )
                    grade = j["grade"]
                    if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
                        raise DatasetValidationError(
                            f"Grade must be non-negative int for '{cid}' in query '{qid}'."
                        )
                    if valid_chunk_ids is not None and cid not in valid_chunk_ids:
                        raise DatasetValidationError(
                            f"Query '{qid}' references unknown chunk '{cid}' not found in scope."
                        )
                    rel_grades[cid] = grade
                    if grade > 0 and cid not in rel_chunk_ids:
                        rel_chunk_ids.append(cid)

            elif "relevance_grades" in record and record["relevance_grades"] is not None:
                if not isinstance(record["relevance_grades"], dict):
                    raise DatasetValidationError(
                        f"relevance_grades must be a dictionary in query '{qid}' "
                        f"on line {line_num}."
                    )
                for cid_raw, g_raw in record["relevance_grades"].items():
                    cid = str(cid_raw).strip()
                    if not cid:
                        raise DatasetValidationError(
                            f"Empty chunk_id in relevance_grades for query '{qid}' "
                            f"on line {line_num}."
                        )
                    if isinstance(g_raw, bool) or not isinstance(g_raw, int) or g_raw < 0:
                        raise DatasetValidationError(
                            f"Grade for '{cid}' must be non-negative int in query '{qid}'."
                        )
                    if valid_chunk_ids is not None and cid not in valid_chunk_ids:
                        raise DatasetValidationError(
                            f"Query '{qid}' references unknown chunk '{cid}' not found in scope."
                        )
                    rel_grades[cid] = g_raw
                    if g_raw > 0 and cid not in rel_chunk_ids:
                        rel_chunk_ids.append(cid)

            if "relevant_chunk_ids" in record and record["relevant_chunk_ids"] is not None:
                if not isinstance(record["relevant_chunk_ids"], list):
                    raise DatasetValidationError(
                        f"relevant_chunk_ids must be a list in query '{qid}' on line {line_num}."
                    )
                for cid in record["relevant_chunk_ids"]:
                    cid_str = str(cid).strip()
                    if not cid_str:
                        raise DatasetValidationError(
                            f"Empty chunk_id in relevant_chunk_ids for query '{qid}' "
                            f"on line {line_num}."
                        )
                    if valid_chunk_ids is not None and cid_str not in valid_chunk_ids:
                        raise DatasetValidationError(
                            f"Query '{qid}' references unknown chunk '{cid_str}' not in scope."
                        )
                    if cid_str not in rel_chunk_ids:
                        rel_chunk_ids.append(cid_str)
                    if cid_str not in rel_grades:
                        rel_grades[cid_str] = 1

            required_citations: list[str] = []
            citation_records: list[dict[str, Any]] = []
            citation_to_chunk_id: dict[str, str] = {}
            if "required_citations" in record and record["required_citations"] is not None:
                if not isinstance(record["required_citations"], list):
                    raise DatasetValidationError(
                        f"required_citations must be a list in query '{qid}' on line {line_num}."
                    )
                seen_cit_ids: set[str] = set()
                for item in record["required_citations"]:
                    if isinstance(item, dict):
                        cit_id = str(item.get("citation_id") or item.get("chunk_id") or "").strip()
                        if not cit_id:
                            raise DatasetValidationError(
                                f"Empty citation_id in query '{qid}' on line {line_num}."
                            )
                        if cit_id in seen_cit_ids:
                            raise DatasetValidationError(
                                f"Duplicate citation identifier '{cit_id}' in query '{qid}' "
                                f"on line {line_num}."
                            )
                        seen_cit_ids.add(cit_id)
                        required_citations.append(cit_id)
                        citation_records.append(dict(item))
                        c_id = str(item.get("chunk_id", "")).strip()
                        if c_id:
                            if valid_chunk_ids is not None and c_id not in valid_chunk_ids:
                                raise DatasetValidationError(
                                    f"Required citation '{cit_id}' in query '{qid}' maps to "
                                    f"unknown chunk '{c_id}' not found in scope on line {line_num}."
                                )
                            citation_to_chunk_id[cit_id] = c_id
                    elif isinstance(item, str):
                        cit_id = item.strip()
                        if not cit_id:
                            raise DatasetValidationError(
                                f"Empty citation identifier in query '{qid}' on line {line_num}."
                            )
                        if cit_id in seen_cit_ids:
                            raise DatasetValidationError(
                                f"Duplicate citation identifier '{cit_id}' in query '{qid}' "
                                f"on line {line_num}."
                            )
                        seen_cit_ids.add(cit_id)
                        required_citations.append(cit_id)
                        if valid_chunk_ids is not None and cit_id in valid_chunk_ids:
                            citation_to_chunk_id[cit_id] = cit_id
                    else:
                        raise DatasetValidationError(
                            f"Invalid citation entry '{item}' in query '{qid}' on line {line_num}."
                        )

            if "metadata" in record and record["metadata"] is not None:
                if not isinstance(record["metadata"], dict):
                    raise DatasetValidationError(
                        f"Invalid metadata type on line {line_num}: expected JSON object."
                    )
                q_meta = dict(record["metadata"])
            else:
                q_meta = {}

            if citation_records:
                q_meta["citation_records"] = citation_records
            if citation_to_chunk_id:
                q_meta["citation_to_chunk_id"] = citation_to_chunk_id

            queries.append(
                Query(
                    query_id=qid,
                    text=text,
                    relevant_chunk_ids=rel_chunk_ids,
                    relevance_grades=rel_grades,
                    ground_truth_answer=record.get("ground_truth_answer"),
                    required_citations=required_citations,
                    metadata=q_meta,
                )
            )

        if not queries:
            raise DatasetValidationError("Import file contained zero valid query records.")

        return self.dataset_repository.create_dataset(
            name=name,
            description=description,
            document_scope=scope,
            is_synthetic=is_synthetic,
            metadata={"imported": True},
            queries=queries,
        )

    def export_jsonl(self, dataset_id: str) -> str:
        """Export dataset to JSONL string."""
        ds = self.get_dataset(dataset_id)
        return ds.to_jsonl()
