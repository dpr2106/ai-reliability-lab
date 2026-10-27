"""Comprehensive unit tests for core domain models and protocol interfaces."""

import json
from dataclasses import FrozenInstanceError

import pytest

from reliability_lab.core.interfaces import BaseChunker, BaseMetric, BaseRetriever
from reliability_lab.core.models import (
    Chunk,
    Document,
    EvaluationReport,
    Query,
    RetrievalResult,
)


class TestDocument:
    """Tests for the Document domain model."""

    def test_valid_construction(self):
        doc = Document(
            doc_id="doc_001",
            content="Title: Revenue Growth\nRevenue grew by 15%.",
            metadata={"source": "q3.pdf", "author": "Finance Team"},
            page_count=3,
        )
        assert doc.doc_id == "doc_001"
        assert "Revenue grew" in doc.content
        assert doc.metadata["source"] == "q3.pdf"
        assert doc.page_count == 3

    def test_default_metadata_and_page_count(self):
        doc = Document(doc_id="doc_minimal", content="Minimal content")
        assert doc.metadata == {}
        assert doc.page_count is None

    @pytest.mark.parametrize("invalid_id", ["", "   ", None, 123])
    def test_invalid_doc_id(self, invalid_id):
        with pytest.raises((ValueError, TypeError)):
            Document(doc_id=invalid_id, content="Content")

    def test_invalid_content_type(self):
        with pytest.raises(TypeError):
            Document(doc_id="doc_1", content=None)  # type: ignore

    @pytest.mark.parametrize("invalid_page_count", [0, -1, "two"])
    def test_invalid_page_count(self, invalid_page_count):
        with pytest.raises((ValueError, TypeError)):
            Document(doc_id="doc_1", content="Content", page_count=invalid_page_count)  # type: ignore

    def test_dict_serialization_round_trip(self):
        original = Document(
            doc_id="doc_42",
            content="Sample text",
            metadata={"priority": "high"},
            page_count=10,
        )
        data = original.to_dict()
        assert data["doc_id"] == "doc_42"
        reconstructed = Document.from_dict(data)
        assert reconstructed == original

    def test_json_serialization_round_trip(self):
        original = Document(doc_id="doc_json", content="JSON content", page_count=2)
        json_str = original.to_json()
        assert json.loads(json_str)["doc_id"] == "doc_json"
        reconstructed = Document.from_json(json_str)
        assert reconstructed == original


class TestChunk:
    """Tests for the Chunk domain model."""

    def test_valid_construction(self):
        chunk = Chunk(
            chunk_id="chunk_01",
            doc_id="doc_01",
            content="Quarterly revenue was $4.2B.",
            page_number=2,
            char_start=150,
            char_end=178,
            metadata={"token_count": 5},
        )
        assert chunk.chunk_id == "chunk_01"
        assert chunk.doc_id == "doc_01"
        assert chunk.page_number == 2
        assert chunk.char_start == 150
        assert chunk.char_end == 178

    def test_immutability(self):
        chunk = Chunk(chunk_id="c_1", doc_id="d_1", content="Immutable content")
        with pytest.raises(FrozenInstanceError):
            chunk.content = "New content"  # type: ignore

    @pytest.mark.parametrize("invalid_id", ["", "  ", None])
    def test_invalid_identifiers(self, invalid_id):
        with pytest.raises((ValueError, TypeError)):
            Chunk(chunk_id=invalid_id, doc_id="d_1", content="Text")
        with pytest.raises((ValueError, TypeError)):
            Chunk(chunk_id="c_1", doc_id=invalid_id, content="Text")

    @pytest.mark.parametrize("invalid_content", ["", "   ", None])
    def test_invalid_content(self, invalid_content):
        with pytest.raises((ValueError, TypeError)):
            Chunk(chunk_id="c_1", doc_id="d_1", content=invalid_content)

    @pytest.mark.parametrize("invalid_page", [0, -5, "1"])
    def test_invalid_page_number(self, invalid_page):
        with pytest.raises((ValueError, TypeError)):
            Chunk(chunk_id="c_1", doc_id="d_1", content="Text", page_number=invalid_page)  # type: ignore

    def test_char_start_greater_than_char_end(self):
        with pytest.raises(ValueError, match=r"char_end .* must be >= char_start"):
            Chunk(
                chunk_id="c_1",
                doc_id="d_1",
                content="Text",
                char_start=200,
                char_end=150,
            )

    @pytest.mark.parametrize("neg_offset", [-1, -100])
    def test_negative_char_offsets(self, neg_offset):
        with pytest.raises(ValueError):
            Chunk(chunk_id="c_1", doc_id="d_1", content="Text", char_start=neg_offset)
        with pytest.raises(ValueError):
            Chunk(chunk_id="c_1", doc_id="d_1", content="Text", char_end=neg_offset)

    def test_serialization_round_trip(self):
        chunk = Chunk(
            chunk_id="c_round",
            doc_id="d_round",
            content="Evidence text",
            page_number=5,
            char_start=10,
            char_end=23,
            metadata={"score": 0.95},
        )
        json_str = chunk.to_json()
        reconstructed = Chunk.from_json(json_str)
        assert reconstructed == chunk


class TestQuery:
    """Tests for the Query domain model."""

    def test_valid_construction(self):
        query = Query(
            query_id="q_100",
            text="What was Acme's 2024 revenue?",
            relevant_chunk_ids=["chunk_01", "chunk_02"],
            relevance_grades={"chunk_01": 2, "chunk_02": 1},
            ground_truth_answer="Acme revenue was $4.2B.",
            required_citations=["acme.pdf#page=2"],
        )
        assert query.query_id == "q_100"
        assert len(query.relevant_chunk_ids) == 2
        assert query.get_grade("chunk_01") == 2
        assert query.get_grade("chunk_02") == 1
        assert query.get_grade("chunk_unknown") == 0

    def test_get_grade_fallback_to_relevant_chunk_ids(self):
        query = Query(
            query_id="q_fallback",
            text="Any text?",
            relevant_chunk_ids=["chunk_unranked"],
        )
        assert query.get_grade("chunk_unranked") == 1
        assert query.get_grade("not_present") == 0

    @pytest.mark.parametrize("invalid_id", ["", "  ", None])
    def test_invalid_query_id(self, invalid_id):
        with pytest.raises((ValueError, TypeError)):
            Query(query_id=invalid_id, text="Query text")

    @pytest.mark.parametrize("invalid_text", ["", "   ", None])
    def test_invalid_query_text(self, invalid_text):
        with pytest.raises((ValueError, TypeError)):
            Query(query_id="q_1", text=invalid_text)

    def test_negative_relevance_grade(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            Query(query_id="q_1", text="Text", relevance_grades={"c1": -1})

    def test_serialization_round_trip(self):
        query = Query(
            query_id="q_ser",
            text="Revenue query",
            relevant_chunk_ids=["c_1"],
            relevance_grades={"c_1": 2},
            ground_truth_answer="Answer",
            required_citations=["p.1"],
            metadata={"domain": "finance"},
        )
        json_str = query.to_json()
        reconstructed = Query.from_json(json_str)
        assert reconstructed == query


class TestRetrievalResult:
    """Tests for the RetrievalResult domain model and nested Chunk reconstruction."""

    @pytest.fixture
    def sample_chunk(self) -> Chunk:
        return Chunk(chunk_id="c_res", doc_id="d_res", content="Result content", page_number=1)

    def test_valid_construction(self, sample_chunk):
        result = RetrievalResult(
            chunk=sample_chunk,
            score=0.885,
            rank=1,
            retriever_name="bm25_okapi",
            metadata={"latency_ms": 12.4},
        )
        assert result.chunk == sample_chunk
        assert result.score == 0.885
        assert result.rank == 1
        assert result.retriever_name == "bm25_okapi"

    def test_immutability(self, sample_chunk):
        result = RetrievalResult(chunk=sample_chunk, score=0.5, rank=1, retriever_name="bm25")
        with pytest.raises(FrozenInstanceError):
            result.rank = 2  # type: ignore

    @pytest.mark.parametrize("invalid_rank", [0, -1, -100, "1"])
    def test_invalid_rank(self, sample_chunk, invalid_rank):
        with pytest.raises((ValueError, TypeError)):
            RetrievalResult(
                chunk=sample_chunk,
                score=0.5,
                rank=invalid_rank,  # type: ignore
                retriever_name="bm25",
            )

    @pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores(self, sample_chunk, invalid_score):
        with pytest.raises(ValueError, match="finite float"):
            RetrievalResult(
                chunk=sample_chunk,
                score=invalid_score,
                rank=1,
                retriever_name="bm25",
            )

    def test_invalid_chunk_type(self):
        with pytest.raises(TypeError, match="instance of Chunk"):
            RetrievalResult(
                chunk={"content": "not a chunk"},  # type: ignore
                score=0.5,
                rank=1,
                retriever_name="bm25",
            )

    def test_nested_serialization_round_trip(self, sample_chunk):
        result = RetrievalResult(
            chunk=sample_chunk,
            score=14.25,
            rank=3,
            retriever_name="faiss_dense",
            metadata={"model": "all-MiniLM-L6-v2"},
        )
        data = result.to_dict()
        assert isinstance(data["chunk"], dict)
        assert data["chunk"]["chunk_id"] == "c_res"

        json_str = result.to_json()
        reconstructed = RetrievalResult.from_json(json_str)
        assert reconstructed == result
        assert isinstance(reconstructed.chunk, Chunk)


class TestEvaluationReport:
    """Tests for the EvaluationReport domain model."""

    def test_valid_construction(self):
        report = EvaluationReport(
            experiment_name="baseline_bm25_vs_golden",
            metrics={"recall@5": 0.80, "mrr@5": 0.725, "ndcg@5": 0.78},
            per_query_results=[{"query_id": "q1", "hit": True}],
            config={"k": 5, "index": "bm25"},
        )
        assert report.experiment_name == "baseline_bm25_vs_golden"
        assert report.metrics["recall@5"] == 0.80
        assert len(report.report_id) > 0
        assert len(report.timestamp) > 0

    @pytest.mark.parametrize("invalid_metric_val", [float("nan"), float("inf")])
    def test_invalid_metric_values(self, invalid_metric_val):
        with pytest.raises(ValueError, match="finite numerical value"):
            EvaluationReport(
                experiment_name="test",
                metrics={"mrr": invalid_metric_val},
            )

    @pytest.mark.parametrize("invalid_name", ["", "   ", None])
    def test_invalid_experiment_name(self, invalid_name):
        with pytest.raises((ValueError, TypeError)):
            EvaluationReport(experiment_name=invalid_name, metrics={"mrr": 0.5})

    def test_serialization_round_trip(self):
        report = EvaluationReport(
            report_id="rep_12345",
            experiment_name="hybrid_test",
            metrics={"ndcg@10": 0.91},
            per_query_results=[{"q": "q_1", "score": 1.0}],
            timestamp="2026-10-27T18:00:00+00:00",
            config={"alpha": 0.5},
        )
        json_str = report.to_json()
        reconstructed = EvaluationReport.from_json(json_str)
        assert reconstructed == report


class TestAbstractInterfaces:
    """Tests asserting abstract class contracts and representative implementations."""

    def test_base_retriever_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore

    def test_concrete_retriever_implementation(self):
        class MockRetriever(BaseRetriever):
            def __init__(self):
                self._chunks = []

            @property
            def name(self) -> str:
                return "mock_retriever"

            def index(self, chunks):
                self._chunks.extend(chunks)

            def retrieve(self, query: str, top_k: int = 5):
                results = []
                for idx, c in enumerate(self._chunks[:top_k], start=1):
                    results.append(
                        RetrievalResult(
                            chunk=c,
                            score=1.0 / idx,
                            rank=idx,
                            retriever_name=self.name,
                        )
                    )
                return results

        retriever = MockRetriever()
        assert retriever.name == "mock_retriever"
        chunk = Chunk(chunk_id="c_mock", doc_id="d_mock", content="Mock chunk")
        retriever.index([chunk])
        results = retriever.retrieve("query", top_k=1)
        assert len(results) == 1
        assert results[0].chunk == chunk
        assert results[0].rank == 1

    def test_base_chunker_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseChunker()  # type: ignore

    def test_concrete_chunker_implementation(self):
        class MockChunker(BaseChunker):
            @property
            def name(self) -> str:
                return "mock_chunker"

            def chunk(self, document: Document):
                return [
                    Chunk(
                        chunk_id=f"{document.doc_id}_c0",
                        doc_id=document.doc_id,
                        content=document.content,
                        page_number=1,
                    )
                ]

        chunker = MockChunker()
        assert chunker.name == "mock_chunker"
        doc = Document(doc_id="d_test", content="Full doc content")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "d_test_c0"

    def test_base_metric_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseMetric()  # type: ignore

    def test_concrete_metric_implementation(self):
        class MockHitRate(BaseMetric):
            @property
            def name(self) -> str:
                return "hit_rate@1"

            def compute(self, retrieved, query) -> float:
                if not retrieved:
                    return 0.0
                top_chunk_id = retrieved[0].chunk.chunk_id
                return 1.0 if top_chunk_id in query.relevant_chunk_ids else 0.0

        metric = MockHitRate()
        assert metric.name == "hit_rate@1"
        chunk = Chunk(chunk_id="c_hit", doc_id="d1", content="Evidence")
        res = RetrievalResult(chunk=chunk, score=0.9, rank=1, retriever_name="bm25")
        q_match = Query(query_id="q1", text="text", relevant_chunk_ids=["c_hit"])
        q_miss = Query(query_id="q2", text="text", relevant_chunk_ids=["c_other"])

        assert metric.compute([res], q_match) == 1.0
        assert metric.compute([res], q_miss) == 0.0
        assert metric.compute([], q_match) == 0.0
