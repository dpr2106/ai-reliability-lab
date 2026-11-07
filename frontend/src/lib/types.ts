/**
 * Domain types and API response models for AI Reliability Lab frontend.
 */

export interface HealthResponse {
  status: 'ok' | 'degraded';
  database: string;
  database_exists: boolean;
  total_runs: number;
}

export interface RunSummary {
  run_id: string;
  experiment_name: string;
  run_timestamp: string;
  config_hash: string;
  dataset_reference: string | null;
  created_at: string;
  metric_count: number;
  query_count: number;
  metrics: Record<string, number>;
}

export interface RunListResponse {
  total: number;
  limit: number;
  offset: number;
  runs: RunSummary[];
}

export interface RunDetail {
  run_id: string;
  experiment_name: string;
  run_timestamp: string;
  config_hash: string;
  config: Record<string, unknown>;
  dataset_reference: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  metrics: Record<string, number>;
  query_count: number;
  per_query_results: QueryResult[];
}

export interface QueryResult {
  query_id: string;
  text?: string;
  latency_ms?: number;
  metrics?: Record<string, number>;
  candidates?: CandidateRecord[];
  retrieved?: CandidateRecord[];
  failure_trace?: FailureTrace | null;
  answer?: string;
  generated_answer?: string;
  citation_mapping?: Record<string, string>;
  required_citations?: string[];
  [key: string]: unknown;
}

export interface CandidateRecord {
  chunk_id: string;
  score: number;
  rank: number;
  content?: string;
  [key: string]: unknown;
}

export interface FailureTrace {
  unretrieved_relevant_ids?: string[];
  retrieved_irrelevant_ids?: string[];
  reason?: string;
  [key: string]: unknown;
}

export type RegressionStatusType = 'PASS' | 'REGRESSION' | 'INCOMPLETE';

export interface MetricDelta {
  metric_name: string;
  baseline_score: number;
  candidate_score: number;
  delta: number;
  degradation: number;
  tolerance: number;
  direction: 'HIGHER_IS_BETTER' | 'LOWER_IS_BETTER';
  is_regressed: boolean;
}

export interface QueryMetricDelta {
  query_id: string;
  metric_name: string;
  baseline_score: number;
  candidate_score: number;
  delta: number;
  degradation: number;
  tolerance: number;
  is_regressed: boolean;
}

export interface QueryRegression {
  query_id: string;
  query_text: string | null;
  metrics: Record<string, QueryMetricDelta>;
  regressed_metrics: string[];
  max_degradation: number;
  has_regression: boolean;
  failure_trace: Record<string, unknown> | null;
}

export interface RegressionResult {
  status: RegressionStatusType;
  baseline_run_id: string;
  candidate_run_id: string;
  baseline_config_hash: string | null;
  candidate_config_hash: string | null;
  baseline_dataset: string | null;
  candidate_dataset: string | null;
  aggregate_deltas: Record<string, MetricDelta>;
  query_regressions: QueryRegression[];
  regressed_aggregate_metrics: string[];
  regressed_query_ids: string[];
  total_queries_compared: number;
  baseline_only_query_ids: string[];
  candidate_only_query_ids: string[];
  worst_degradations: QueryMetricDelta[];
  warnings: string[];
  is_complete: boolean;
  has_regression: boolean;
}

export interface CompareRequest {
  baseline_run_id: string;
  candidate_run_id: string;
  tolerance?: number;
  metric_tolerances?: Record<string, number> | null;
}

export interface CitationAuditQueryItem {
  query_id: string;
  text: string | null;
  has_answer: boolean;
  answer_text: string | null;
  citation_precision: number;
  citation_recall: number;
  valid_citations: string[];
  invalid_citations: string[];
  unresolved_citations: string[];
  missing_evidence_citations: string[];
  duplicate_citations: string[];
  unreferenced_required_citations: string[];
  malformed_citations: string[];
  retrieved_chunk_ids: string[];
  citation_to_chunk_map: Record<string, string>;
}

export interface CitationAuditResponse {
  run_id: string;
  experiment_name: string;
  total_queries: number;
  audited_queries_count: number;
  mean_citation_precision: number;
  mean_citation_recall: number;
  queries: CitationAuditQueryItem[];
  limitations_note: string;
}

// -----------------------------------------------------------------------------
// Document Types
// -----------------------------------------------------------------------------

export interface DocumentSummary {
  doc_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  status: string;
  chunk_count: number;
  page_count: number | null;
  created_at: string;
  processed_at: string | null;
  error_message: string | null;
}

export interface DocumentListResponse {
  total: number;
  limit: number;
  offset: number;
  items: DocumentSummary[];
}

// -----------------------------------------------------------------------------
// Cross-Document Evidence Intelligence Types (STING V2-004)
// -----------------------------------------------------------------------------

export type EvidenceRelationship =
  | 'supports'
  | 'contradicts'
  | 'qualifies'
  | 'not_addressed'
  | 'insufficient_evidence';

export interface EvidenceFinding {
  finding_id: string;
  doc_id: string;
  doc_title: string;
  chunk_id: string;
  page_number: number | null;
  claim: string;
  supporting_passage: string;
  relationship: EvidenceRelationship;
  confidence: number;
  topic_id?: string;
  topic_name?: string | null;
  investigation_id?: string | null;
  methodology_notes?: string | null;
  created_at?: string;
}

export interface ComparisonTopic {
  topic_id: string;
  topic_name: string;
  description: string;
  findings: EvidenceFinding[];
  agreements_summary?: string;
  disagreements_summary?: string;
  evidence_gaps?: string[];
}

export interface DocumentSummaryInReport {
  doc_id: string;
  title?: string;
  filename?: string;
  chunk_count?: number;
  findings_count?: number;
  status?: string;
}

export interface InvestigationReport {
  investigation_id: string;
  title: string;
  query: string;
  comparison_aspects: string[];
  document_ids: string[];
  document_summaries: DocumentSummaryInReport[];
  topics: ComparisonTopic[];
  total_findings: number;
  relationship_counts: Record<string, number>;
  executive_summary: string;
  methodology_synthesis?: string;
  created_at: string;
  completed_at?: string | null;
  status: string;
  error_message?: string | null;
}

export interface InvestigationSummary {
  investigation_id: string;
  title: string;
  query: string;
  document_count: number;
  aspect_count: number;
  status: 'created' | 'running' | 'completed' | 'failed';
  total_findings: number;
  created_at: string;
  completed_at: string | null;
  relationship_counts?: Record<string, number>;
  error_message?: string | null;
}

export interface InvestigationDetail {
  investigation_id: string;
  title: string;
  query: string;
  comparison_aspects: string[];
  document_ids: string[];
  status: 'created' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at: string | null;
  error_message?: string | null;
  report?: InvestigationReport | null;
}

export interface CreateInvestigationRequest {
  query: string;
  document_ids: string[];
  title?: string | null;
  comparison_aspects?: string[] | null;
  aspects?: string[] | null;
  run_immediately?: boolean;
}

export interface InvestigationListResponse {
  total: number;
  limit: number;
  offset: number;
  investigations: InvestigationSummary[];
}

