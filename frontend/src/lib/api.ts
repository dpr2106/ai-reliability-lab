/**
 * Typed HTTP client for the AI Reliability Lab FastAPI adapter.
 */

import type {
  BenchmarkQuery,
  CitationAuditResponse,
  CompareRequest,
  CreateDatasetRequest,
  CreateInvestigationRequest,
  CreateQueryRequest,
  CreateSessionRequest,
  DatasetDetail,
  DatasetListResponse,
  DatasetSummary,
  DocumentChunkListResponse,
  DocumentDetailResponse,
  DocumentListResponse,
  DocumentUploadResponse,
  EvaluationJob,
  EvaluationJobListResponse,
  HealthResponse,
  ImportDatasetRequest,
  InvestigationDetail,
  InvestigationListResponse,
  LaunchEvaluationRequest,
  MetricsMetaResponse,
  QueryRequest,
  QueryResponse,
  QueryResult,
  RegressionResult,
  RunDetail,
  RunListResponse,
  RunReproducibilityResponse,
  SessionDetailResponse,
  SessionListResponse,
  SessionQueryRequest,
  SessionQueryResponse
} from './types';

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public originalError?: unknown
  ) {
    super(`API Error ${status}: ${detail}`);
    this.name = 'ApiError';
  }
}

export class ApiClient {
  private baseUrl: string;
  private fetchFn: typeof fetch;

  constructor(baseUrl?: string, customFetch?: typeof fetch) {
    // Default to localhost:8000 where FastAPI adapter runs
    this.baseUrl = baseUrl || (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || 'http://127.0.0.1:8000';
    this.fetchFn = customFetch || (typeof window !== 'undefined' ? window.fetch.bind(window) : fetch);
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options.headers
    };

    try {
      const response = await this.fetchFn(url, { ...options, headers });
      if (!response.ok) {
        let detail = response.statusText;
        try {
          const errBody = await response.json();
          if (errBody && typeof errBody.detail === 'string') {
            detail = errBody.detail;
          } else if (errBody && errBody.detail) {
            detail = JSON.stringify(errBody.detail);
          }
        } catch {
          // Response body was not JSON
        }
        throw new ApiError(response.status, detail);
      }
      return (await response.json()) as T;
    } catch (err) {
      if (err instanceof ApiError) {
        throw err;
      }
      throw new ApiError(0, (err as Error).message || 'Network request failed', err);
    }
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  async listRuns(params?: {
    limit?: number;
    offset?: number;
    experimentName?: string;
  }): Promise<RunListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));
    if (params?.experimentName) searchParams.set('experiment_name', params.experimentName);

    const query = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return this.request<RunListResponse>(`/api/runs${query}`);
  }

  async getRun(runId: string): Promise<RunDetail> {
    return this.request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
  }

  async getRunQueries(
    runId: string,
    params?: {
      limit?: number;
      offset?: number;
      queryId?: string;
      failed_only?: boolean;
      failure_type?: string;
      sort_by?: string;
    }
  ): Promise<{ run_id: string; total: number; limit: number; offset: number; queries: QueryResult[] }> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));
    if (params?.queryId) searchParams.set('query_id', params.queryId);
    if (params?.failed_only !== undefined) searchParams.set('failed_only', String(params.failed_only));
    if (params?.failure_type) searchParams.set('failure_type', params.failure_type);
    if (params?.sort_by) searchParams.set('sort_by', params.sort_by);

    const query = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return this.request<{ run_id: string; total: number; limit: number; offset: number; queries: QueryResult[] }>(
      `/api/runs/${encodeURIComponent(runId)}/queries${query}`
    );
  }

  async compareRuns(req: CompareRequest): Promise<RegressionResult> {
    return this.request<RegressionResult>('/api/runs/compare', {
      method: 'POST',
      body: JSON.stringify(req)
    });
  }

  async auditCitations(runId: string): Promise<CitationAuditResponse> {
    return this.request<CitationAuditResponse>(`/api/runs/${encodeURIComponent(runId)}/citation-audit`);
  }

  private async requestText(endpoint: string, options: RequestInit = {}): Promise<string> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      ...options.headers
    };

    try {
      const response = await this.fetchFn(url, { ...options, headers });
      if (!response.ok) {
        let detail = response.statusText;
        try {
          const errBody = await response.json();
          if (errBody && typeof errBody.detail === 'string') {
            detail = errBody.detail;
          }
        } catch {
          // Response body was not JSON
        }
        throw new ApiError(response.status, detail);
      }
      return await response.text();
    } catch (err) {
      if (err instanceof ApiError) {
        throw err;
      }
      throw new ApiError(0, (err as Error).message || 'Network request failed', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Documents API
  // ---------------------------------------------------------------------------

  async listDocuments(params?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<DocumentListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));
    if (params?.status) searchParams.set('status', params.status);

    const query = searchParams.toString() ? `?${searchParams.toString()}` : '';
    const res = await this.request<DocumentListResponse>(`/api/documents${query}`);
    const docs = res.documents ?? res.items ?? [];
    return {
      ...res,
      total: res.total ?? docs.length,
      documents: docs,
      items: docs
    };
  }

  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const url = `${this.baseUrl}/api/documents/upload`;
    const response = await this.fetchFn(url, {
      method: 'POST',
      body: formData,
      headers: {
        Accept: 'application/json'
      }
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const errBody = await response.json();
        if (errBody && typeof errBody.detail === 'string') {
          detail = errBody.detail;
        } else if (errBody && errBody.detail) {
          detail = JSON.stringify(errBody.detail);
        }
      } catch {
        // Non-JSON response
      }
      throw new ApiError(response.status, detail);
    }
    return (await response.json()) as DocumentUploadResponse;
  }

  async getDocument(docId: string): Promise<DocumentDetailResponse> {
    return this.request<DocumentDetailResponse>(`/api/documents/${encodeURIComponent(docId)}`);
  }

  async getDocumentChunks(
    docId: string,
    params?: { limit?: number; offset?: number }
  ): Promise<DocumentChunkListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));
    const qs = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return this.request<DocumentChunkListResponse>(
      `/api/documents/${encodeURIComponent(docId)}/chunks${qs}`
    );
  }

  async deleteDocument(docId: string): Promise<{ status: string; doc_id: string }> {
    return this.request<{ status: string; doc_id: string }>(
      `/api/documents/${encodeURIComponent(docId)}`,
      { method: 'DELETE' }
    );
  }

  // ---------------------------------------------------------------------------
  // Document Query & Chat API
  // ---------------------------------------------------------------------------

  async executeQuery(req: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify(req)
    });
  }

  async createSession(req: CreateSessionRequest): Promise<SessionDetailResponse> {
    return this.request<SessionDetailResponse>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify(req)
    });
  }

  async listSessions(params?: { limit?: number; offset?: number }): Promise<SessionListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));
    const qs = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return this.request<SessionListResponse>(`/api/sessions${qs}`);
  }

  async getSession(sessionId: string): Promise<SessionDetailResponse> {
    return this.request<SessionDetailResponse>(`/api/sessions/${encodeURIComponent(sessionId)}`);
  }

  async querySession(sessionId: string, req: SessionQueryRequest): Promise<SessionQueryResponse> {
    return this.request<SessionQueryResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/query`,
      {
        method: 'POST',
        body: JSON.stringify(req)
      }
    );
  }

  async deleteSession(sessionId: string): Promise<{ deleted: boolean; session_id: string }> {
    return this.request<{ deleted: boolean; session_id: string }>(
      `/api/sessions/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE' }
    );
  }

  // ---------------------------------------------------------------------------
  // Cross-Document Evidence Investigations API (STING V2-004)
  // ---------------------------------------------------------------------------

  async listInvestigations(params?: {
    limit?: number;
    offset?: number;
  }): Promise<InvestigationListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));

    const query = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return this.request<InvestigationListResponse>(`/api/investigations${query}`);
  }

  async getInvestigation(investigationId: string): Promise<InvestigationDetail> {
    return this.request<InvestigationDetail>(`/api/investigations/${encodeURIComponent(investigationId)}`);
  }

  async createInvestigation(req: CreateInvestigationRequest): Promise<InvestigationDetail> {
    return this.request<InvestigationDetail>('/api/investigations', {
      method: 'POST',
      body: JSON.stringify(req)
    });
  }

  async runInvestigation(investigationId: string): Promise<InvestigationDetail> {
    return this.request<InvestigationDetail>(
      `/api/investigations/${encodeURIComponent(investigationId)}/run`,
      { method: 'POST' }
    );
  }

  async deleteInvestigation(investigationId: string): Promise<{ investigation_id: string; deleted: boolean }> {
    return this.request<{ investigation_id: string; deleted: boolean }>(
      `/api/investigations/${encodeURIComponent(investigationId)}`,
      { method: 'DELETE' }
    );
  }

  async exportInvestigationMarkdown(investigationId: string): Promise<string> {
    return this.requestText(`/api/investigations/${encodeURIComponent(investigationId)}/export`, {
      headers: { Accept: 'text/markdown' }
    });
  }

  // ========================================================================
  // V2-005: RAG Evaluation Studio Endpoints
  // ========================================================================

  async listDatasets(options: { limit?: number; offset?: number } = {}): Promise<DatasetListResponse> {
    const params = new URLSearchParams();
    if (options.limit !== undefined) params.set('limit', String(options.limit));
    if (options.offset !== undefined) params.set('offset', String(options.offset));
    const qs = params.toString() ? `?${params.toString()}` : '';
    return this.request<DatasetListResponse>(`/api/datasets${qs}`);
  }

  async getDataset(datasetId: string): Promise<DatasetDetail> {
    return this.request<DatasetDetail>(`/api/datasets/${encodeURIComponent(datasetId)}`);
  }

  async createDataset(payload: CreateDatasetRequest): Promise<DatasetDetail> {
    return this.request<DatasetDetail>('/api/datasets', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  async updateDataset(datasetId: string, payload: Partial<CreateDatasetRequest>): Promise<DatasetSummary> {
    return this.request<DatasetSummary>(`/api/datasets/${encodeURIComponent(datasetId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload)
    });
  }

  async deleteDataset(datasetId: string): Promise<{ deleted: boolean; dataset_id: string }> {
    return this.request<{ deleted: boolean; dataset_id: string }>(`/api/datasets/${encodeURIComponent(datasetId)}`, {
      method: 'DELETE'
    });
  }

  async addQuery(datasetId: string, query: CreateQueryRequest): Promise<BenchmarkQuery> {
    return this.request<BenchmarkQuery>(`/api/datasets/${encodeURIComponent(datasetId)}/queries`, {
      method: 'POST',
      body: JSON.stringify(query)
    });
  }

  async updateQuery(
    datasetId: string,
    queryId: string,
    query: Partial<CreateQueryRequest>
  ): Promise<BenchmarkQuery> {
    return this.request<BenchmarkQuery>(
      `/api/datasets/${encodeURIComponent(datasetId)}/queries/${encodeURIComponent(queryId)}`,
      {
        method: 'PUT',
        body: JSON.stringify(query)
      }
    );
  }

  async deleteQuery(datasetId: string, queryId: string): Promise<{ deleted: boolean; query_id: string }> {
    return this.request<{ deleted: boolean; query_id: string }>(
      `/api/datasets/${encodeURIComponent(datasetId)}/queries/${encodeURIComponent(queryId)}`,
      {
        method: 'DELETE'
      }
    );
  }

  async importDataset(payload: ImportDatasetRequest): Promise<DatasetDetail> {
    return this.request<DatasetDetail>('/api/datasets/import', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  async exportDataset(datasetId: string): Promise<string> {
    return this.requestText(`/api/datasets/${encodeURIComponent(datasetId)}/export`, {
      headers: { Accept: 'application/x-jsonlines' }
    });
  }

  async launchEvaluation(payload: LaunchEvaluationRequest): Promise<EvaluationJob> {
    return this.request<EvaluationJob>('/api/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  async listEvaluationJobs(): Promise<EvaluationJobListResponse> {
    return this.request<EvaluationJobListResponse>('/api/evaluation/jobs');
  }

  async getEvaluationJob(jobId: string): Promise<EvaluationJob> {
    return this.request<EvaluationJob>(`/api/evaluation/jobs/${encodeURIComponent(jobId)}`);
  }

  async getMetricsMeta(): Promise<MetricsMetaResponse> {
    return this.request<MetricsMetaResponse>('/api/evaluation/metrics-meta');
  }

  async getRunReproducibility(runId: string): Promise<RunReproducibilityResponse> {
    return this.request<RunReproducibilityResponse>(`/api/runs/${encodeURIComponent(runId)}/reproducibility`);
  }

  async setRunBaseline(runId: string, isBaseline = true): Promise<{ run_id: string; is_baseline: boolean }> {
    return this.request<{ run_id: string; is_baseline: boolean }>(
      `/api/runs/${encodeURIComponent(runId)}/set-baseline?is_baseline=${isBaseline}`,
      { method: 'POST' }
    );
  }
}

export const api = new ApiClient();

