/**
 * Typed HTTP client for the AI Reliability Lab FastAPI adapter.
 */

import type {
  CitationAuditResponse,
  CompareRequest,
  HealthResponse,
  RegressionResult,
  RunDetail,
  RunListResponse
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
    params?: { limit?: number; offset?: number; queryId?: string }
  ): Promise<{ run_id: string; total: number; limit: number; offset: number; queries: unknown[] }> {
    const searchParams = new URLSearchParams();
    if (params?.limit !== undefined) searchParams.set('limit', String(params.limit));
    if (params?.offset !== undefined) searchParams.set('offset', String(params.offset));
    if (params?.queryId) searchParams.set('query_id', params.queryId);

    const query = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return this.request(`/api/runs/${encodeURIComponent(runId)}/queries${query}`);
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
}

export const api = new ApiClient();

