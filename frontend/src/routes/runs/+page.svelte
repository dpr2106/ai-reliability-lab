<script lang="ts">
  import { onMount } from 'svelte';
  import { api, ApiError } from '#lib/api';
  import type { RunSummary } from '#lib/types';

  let runs = $state<RunSummary[]>([]);
  let total = $state<number>(0);
  let isLoading = $state<boolean>(true);
  let errorMessage = $state<string | null>(null);

  let searchExperiment = $state<string>('');
  let limit = $state<number>(20);
  let offset = $state<number>(0);

  let selectedBaseline = $state<string | null>(null);
  let selectedCandidate = $state<string | null>(null);

  async function loadRuns() {
    isLoading = true;
    errorMessage = null;
    try {
      const res = await api.listRuns({
        limit,
        offset,
        experimentName: searchExperiment.trim() || undefined
      });
      runs = res.runs;
      total = res.total;
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
      runs = [];
      total = 0;
    } finally {
      isLoading = false;
    }
  }

  function handleSearch(e: Event) {
    e.preventDefault();
    offset = 0;
    loadRuns();
  }

  function formatDate(iso: string): string {
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return iso;
    }
  }

  function formatMetric(val: number | undefined): string {
    if (val === undefined || val === null) return '—';
    return val.toFixed(4);
  }

  function toggleBaseline(runId: string) {
    if (selectedBaseline === runId) {
      selectedBaseline = null;
    } else {
      selectedBaseline = runId;
      if (selectedCandidate === runId) selectedCandidate = null;
    }
  }

  function toggleCandidate(runId: string) {
    if (selectedCandidate === runId) {
      selectedCandidate = null;
    } else {
      selectedCandidate = runId;
      if (selectedBaseline === runId) selectedBaseline = null;
    }
  }

  onMount(() => {
    loadRuns();
  });
</script>

<div class="explorer-container">
  <div class="header-row">
    <div>
      <h1 class="page-title">Run Explorer</h1>
      <p class="page-subtitle">
        Inspect recorded benchmark runs, failure traces, and select experiments for regression gating.
      </p>
    </div>
    <div class="header-actions">
      <button class="btn btn-secondary" onclick={loadRuns} disabled={isLoading}>
        {isLoading ? 'Refreshing...' : 'Refresh'}
      </button>
    </div>
  </div>

  {#if selectedBaseline || selectedCandidate}
    <div class="comparison-bar">
      <div class="comparison-bar-info">
        <span class="comparison-label">Selected for Comparison:</span>
        <div class="selected-pill baseline">
          <strong>Baseline:</strong> {selectedBaseline ? selectedBaseline.slice(0, 16) + '...' : 'Not selected'}
        </div>
        <div class="selected-pill candidate">
          <strong>Candidate:</strong> {selectedCandidate ? selectedCandidate.slice(0, 16) + '...' : 'Not selected'}
        </div>
      </div>
      <div class="comparison-bar-actions">
        {#if selectedBaseline && selectedCandidate}
          <a
            href="/compare?baseline={encodeURIComponent(selectedBaseline)}&candidate={encodeURIComponent(selectedCandidate)}"
            class="btn btn-primary"
          >
            Launch Comparison Diff &rarr;
          </a>
        {/if}
        <button
          class="btn btn-ghost btn-sm"
          onclick={() => {
            selectedBaseline = null;
            selectedCandidate = null;
          }}
        >
          Clear Selection
        </button>
      </div>
    </div>
  {/if}

  <div class="filter-bar">
    <form class="search-form" onsubmit={handleSearch}>
      <input
        type="text"
        class="input search-input"
        placeholder="Filter by experiment name..."
        bind:value={searchExperiment}
      />
      <button type="submit" class="btn btn-secondary">Filter</button>
      {#if searchExperiment}
        <button
          type="button"
          class="btn btn-ghost"
          onclick={() => {
            searchExperiment = '';
            offset = 0;
            loadRuns();
          }}
        >
          Reset
        </button>
      {/if}
    </form>

    <div class="count-display">
      {#if !isLoading}
        <span>Showing {runs.length} of {total} runs</span>
      {/if}
    </div>
  </div>

  {#if errorMessage}
    <div class="banner banner-error">
      <strong>Failed to load runs:</strong> {errorMessage}
      <button class="btn btn-sm btn-secondary" onclick={loadRuns} style="margin-left: 1rem;">Retry</button>
    </div>
  {/if}

  {#if isLoading}
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading experiment runs from SQLite database...</p>
    </div>
  {:else if runs.length === 0}
    <div class="empty-state">
      <div class="empty-icon">&#128269;</div>
      <h3>No experiment runs found</h3>
      <p>
        {#if searchExperiment}
          No runs matched filter "<strong>{searchExperiment}</strong>". Try clearing the filter.
        {:else}
          The database is currently empty. Execute a benchmark using the headless CLI:
        {/if}
      </p>
      {#if !searchExperiment}
        <pre class="cli-hint"><code>reliability-lab run-benchmark --dataset corpus.json --name baseline-test</code></pre>
      {/if}
    </div>
  {:else}
    <div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th>Run ID & Name</th>
            <th>Date & Time</th>
            <th>Queries</th>
            <th>MRR@5</th>
            <th>Recall@5</th>
            <th>NDCG@5</th>
            <th>Compare Selector</th>
            <th class="actions-col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each runs as run (run.run_id)}
            <tr class:row-baseline={selectedBaseline === run.run_id} class:row-candidate={selectedCandidate === run.run_id}>
              <td>
                <div class="run-primary">
                  <a href="/runs/{encodeURIComponent(run.run_id)}" class="run-name-link">
                    {run.experiment_name}
                  </a>
                  <span class="run-id-mono" title={run.run_id}>
                    {run.run_id.slice(0, 12)}...
                  </span>
                </div>
              </td>
              <td class="date-col">
                {formatDate(run.run_timestamp)}
              </td>
              <td>
                <span class="badge badge-neutral">{run.query_count}</span>
              </td>
              <td class="metric-num">
                {formatMetric(run.metrics['mrr@5'] ?? run.metrics['mrr'])}
              </td>
              <td class="metric-num">
                {formatMetric(run.metrics['recall@5'] ?? run.metrics['recall'])}
              </td>
              <td class="metric-num">
                {formatMetric(run.metrics['ndcg@5'] ?? run.metrics['ndcg'])}
              </td>
              <td>
                <div class="compare-buttons">
                  <button
                    class="btn btn-sm {selectedBaseline === run.run_id ? 'btn-primary' : 'btn-secondary'}"
                    onclick={() => toggleBaseline(run.run_id)}
                    title="Set as baseline for regression diffing"
                  >
                    {selectedBaseline === run.run_id ? '✓ Baseline' : 'Set Baseline'}
                  </button>
                  <button
                    class="btn btn-sm {selectedCandidate === run.run_id ? 'btn-primary' : 'btn-secondary'}"
                    onclick={() => toggleCandidate(run.run_id)}
                    title="Set as candidate for regression diffing"
                  >
                    {selectedCandidate === run.run_id ? '✓ Candidate' : 'Set Candidate'}
                  </button>
                </div>
              </td>
              <td class="actions-col">
                <div class="action-links">
                  <a href="/runs/{encodeURIComponent(run.run_id)}" class="btn btn-sm btn-ghost">
                    Diagnostics
                  </a>
                  <a href="/citation-audit?run_id={encodeURIComponent(run.run_id)}" class="btn btn-sm btn-ghost">
                    Citations
                  </a>
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if total > limit}
      <div class="pagination-bar">
        <button
          class="btn btn-secondary btn-sm"
          disabled={offset === 0 || isLoading}
          onclick={() => {
            offset = Math.max(0, offset - limit);
            loadRuns();
          }}
        >
          &larr; Previous
        </button>
        <span class="pagination-info">
          Page {Math.floor(offset / limit) + 1} of {Math.ceil(total / limit)}
        </span>
        <button
          class="btn btn-secondary btn-sm"
          disabled={offset + limit >= total || isLoading}
          onclick={() => {
            offset += limit;
            loadRuns();
          }}
        >
          Next &rarr;
        </button>
      </div>
    {/if}
  {/if}
</div>

<style>
  .explorer-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .page-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
  }

  .page-subtitle {
    margin-top: 0.25rem;
    color: var(--text-secondary);
    font-size: 0.95rem;
  }

  .comparison-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid var(--accent-primary);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .comparison-bar-info {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.875rem;
  }

  .comparison-label {
    font-weight: 600;
    color: var(--text-primary);
  }

  .selected-pill {
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
  }

  .selected-pill.baseline {
    border-left: 3px solid #38bdf8;
  }

  .selected-pill.candidate {
    border-left: 3px solid #a855f7;
  }

  .comparison-bar-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .filter-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .search-form {
    display: flex;
    gap: 0.5rem;
    flex: 1;
    max-width: 480px;
  }

  .search-input {
    flex: 1;
  }

  .count-display {
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  .run-primary {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .run-name-link {
    font-weight: 600;
    color: var(--text-primary);
    text-decoration: none;
  }

  .run-name-link:hover {
    color: var(--accent-primary);
    text-decoration: underline;
  }

  .run-id-mono {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .date-col {
    font-size: 0.8rem;
    color: var(--text-secondary);
  }

  .metric-num {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .compare-buttons {
    display: flex;
    gap: 0.35rem;
  }

  .actions-col {
    text-align: right;
  }

  .action-links {
    display: flex;
    gap: 0.35rem;
    justify-content: flex-end;
  }

  .row-baseline {
    background-color: rgba(56, 189, 248, 0.05);
  }

  .row-candidate {
    background-color: rgba(168, 85, 247, 0.05);
  }

  .loading-state,
  .empty-state {
    text-align: center;
    padding: 4rem 1rem;
    background: var(--bg-surface);
    border: 1px dashed var(--border-subtle);
    border-radius: var(--radius-md);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-subtle);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 1rem;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
  }

  .cli-hint {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.65rem 1.25rem;
    background: var(--bg-canvas);
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--accent-primary);
  }

  .pagination-bar {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
    margin-top: 1rem;
  }

  .pagination-info {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .banner-error {
    background: rgba(225, 29, 72, 0.1);
    border: 1px solid rgba(225, 29, 72, 0.3);
    color: #fda4af;
    padding: 0.75rem 1.25rem;
    border-radius: var(--radius-md);
    font-size: 0.875rem;
    display: flex;
    align-items: center;
  }
</style>
