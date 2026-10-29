import { describe, it, expect, vi } from 'vitest';
import { ApiClient, ApiError } from './api';

describe('ApiClient', () => {
  it('constructs with custom base URL', () => {
    const client = new ApiClient('http://test-server:9000');
    expect(client).toBeDefined();
  });

  it('fetches health successfully', async () => {
    const mockHealth = {
      status: 'ok',
      database: 'test.db',
      database_exists: true,
      total_runs: 5
    };

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth
    });

    const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
    const result = await client.getHealth();

    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-api/api/health',
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        })
      })
    );
    expect(result.status).toBe('ok');
    expect(result.total_runs).toBe(5);
  });

  it('handles ApiError on non-200 responses with detail payload', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'Run not found with id: run-xyz' })
    });

    const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);

    await expect(client.getRun('run-xyz')).rejects.toThrow(ApiError);
    await expect(client.getRun('run-xyz')).rejects.toMatchObject({
      status: 404,
      detail: 'Run not found with id: run-xyz'
    });
  });

  it('serializes listRuns query parameters correctly', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ total: 1, limit: 10, offset: 0, runs: [] })
    });

    const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
    await client.listRuns({ limit: 10, offset: 20, experimentName: 'bm25-v1' });

    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-api/api/runs?limit=10&offset=20&experiment_name=bm25-v1',
      expect.anything()
    );
  });

  it('sends POST request to /api/runs/compare', async () => {
    const mockCompareResponse = {
      status: 'PASS',
      baseline_run_id: 'r1',
      candidate_run_id: 'r2',
      aggregate_deltas: {},
      query_regressions: [],
      regressed_aggregate_metrics: [],
      regressed_query_ids: [],
      total_queries_compared: 10,
      baseline_only_query_ids: [],
      candidate_only_query_ids: [],
      worst_degradations: [],
      warnings: [],
      is_complete: true,
      has_regression: false
    };

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockCompareResponse
    });

    const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
    const result = await client.compareRuns({
      baseline_run_id: 'r1',
      candidate_run_id: 'r2',
      tolerance: 0.05
    });

    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-api/api/runs/compare',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          baseline_run_id: 'r1',
          candidate_run_id: 'r2',
          tolerance: 0.05
        })
      })
    );
    expect(result.status).toBe('PASS');
  });

  it('audits citations for a given run ID', async () => {
    const mockAudit = {
      run_id: 'r100',
      experiment_name: 'test',
      total_queries: 1,
      audited_queries_count: 1,
      mean_citation_precision: 1.0,
      mean_citation_recall: 1.0,
      queries: [],
      limitations_note: 'Scope note'
    };

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockAudit
    });

    const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
    const result = await client.auditCitations('r100');

    expect(mockFetch).toHaveBeenCalledWith(
      'http://test-api/api/runs/r100/citation-audit',
      expect.anything()
    );
    expect(result.mean_citation_precision).toBe(1.0);
  });
});

