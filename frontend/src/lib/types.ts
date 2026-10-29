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

