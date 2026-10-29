<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { api, ApiError } from '#lib/api';
  import MetricCard from '#lib/components/MetricCard.svelte';
  import type { RunDetail, QueryResult } from '#lib/types';

  let runId = $derived(page.params.run_id);
  let run = $state<RunDetail | null>(null);
  let isLoading = $state<boolean>(true);
  let errorMessage = $state<string | null>(null);

  let filterQueryText = $state<string>('');
  let onlyFailures = $state<boolean>(false);
  let expandedQueryId = $state<string | null>(null);
  let showConfigModal = $state<boolean>(false);

  async function loadRun() {
    if (!runId) return;
    isLoading = true;
    errorMessage = null;
    try {
      run = await api.getRun(runId);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
      run = null;
    } finally {
      isLoading = false;
    }
  }

  const filteredQueries = $derived.by(() => {
    if (!run || !run.per_query_results) return [];
    return run.per_query_results.filter((q: QueryResult) => {
      const matchText =
        !filterQueryText ||
        q.query_id.toLowerCase().includes(filterQueryText.toLowerCase()) ||
        (q.text && q.text.toLowerCase().includes(filterQueryText.toLowerCase()));

      if (!matchText) return false;

      if (onlyFailures) {
        const hasTrace =
          q.failure_trace &&
          ((q.failure_trace.unretrieved_relevant_ids &&
            q.failure_trace.unretrieved_relevant_ids.length > 0) ||
            Boolean(q.failure_trace.reason));
        const zeroRecall =
          q.metrics && (q.metrics['recall@5'] === 0 || q.metrics['recall'] === 0);
        return hasTrace || zeroRecall;
      }

      return true;
    });
  });

  function toggleQueryDetails(id: string) {
    expandedQueryId = expandedQueryId === id ? null : id;
  }

  function formatDate(iso: string): string {
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  function formatQueryMetric(
    m: Record<string, number> | undefined,
    key: string,
    fallbackKey?: string
  ): string {
    if (!m) return '—';
    const val = m[key] ?? (fallbackKey ? m[fallbackKey] : undefined);
    return val !== undefined ? val.toFixed(2) : '—';
  }

  onMount(() => {
    loadRun();
  });
</script>

<div class="run-details-page">
  <div class="back-nav">
    <a href="/" class="back-link">&larr; Back to Run Explorer</a>
  </div>

  {#if isLoading}
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading run details and diagnostics for {runId}...</p>
    </div>
  {:else if errorMessage}
    <div class="banner banner-error">
      <strong>Error loading run:</strong> {errorMessage}
      <button class="btn btn-sm btn-secondary" onclick={loadRun} style="margin-left: 1rem;">Retry</button>
    </div>
  {:else if run}
    <div class="run-header-card">
      <div class="header-main">
        <div>
          <span class="badge badge-neutral">Experiment Run</span>
          <h1 class="run-title">{run.experiment_name}</h1>
          <div class="metadata-row">
            <span class="meta-item"><strong>ID:</strong> <code class="mono-code">{run.run_id}</code></span>
            <span class="meta-item"><strong>Recorded:</strong> {formatDate(run.run_timestamp)}</span>
            {#if run.dataset_reference}
              <span class="meta-item"><strong>Dataset:</strong> {run.dataset_reference}</span>
            {/if}
            <span class="meta-item"><strong>Config Hash:</strong> <code class="mono-code">{run.config_hash.slice(0, 10)}</code></span>
          </div>
        </div>

        <div class="header-actions">
          <button class="btn btn-secondary btn-sm" onclick={() => (showConfigModal = true)}>
            View Config JSON
          </button>
          <a
            href="/compare?baseline={encodeURIComponent(run.run_id)}"
            class="btn btn-secondary btn-sm"
          >
            Compare Against...
          </a>
          <a
            href="/citation-audit?run_id={encodeURIComponent(run.run_id)}"
            class="btn btn-primary btn-sm"
          >
            Audit Citations &rarr;
          </a>
        </div>
      </div>
    </div>

    <!-- Aggregate Metrics Overview -->
    <section class="section">
      <h2 class="section-title">Aggregate Retrieval Metrics</h2>
      <div class="metric-grid">
        <MetricCard label="MRR@5" value={run.metrics['mrr@5'] ?? run.metrics['mrr']} />
        <MetricCard label="Recall@5" value={run.metrics['recall@5'] ?? run.metrics['recall']} />
        <MetricCard label="NDCG@5" value={run.metrics['ndcg@5'] ?? run.metrics['ndcg']} />
        <MetricCard label="Precision@5" value={run.metrics['precision@5'] ?? run.metrics['precision']} />
        <MetricCard label="Hit Rate@5" value={run.metrics['hit_rate@5'] ?? run.metrics['hit_rate']} />
        <MetricCard label="Evaluated Queries" value={run.query_count} format="raw" />
      </div>
    </section>

    <!-- Per-Query Diagnostic Studio -->
    <section class="section">
      <div class="section-header-bar">
        <div>
          <h2 class="section-title">Failure Diagnostics Studio</h2>
          <p class="section-subtitle">
            Inspect individual query rankings, latency, retrieved chunks, and failure traces.
          </p>
        </div>

        <div class="query-filters">
          <input
            type="text"
            class="input search-input-sm"
            placeholder="Search query ID or text..."
            bind:value={filterQueryText}
          />
          <label class="toggle-label">
            <input type="checkbox" bind:checked={onlyFailures} />
            Show only failures & zero-recall
          </label>
        </div>
      </div>

      {#if filteredQueries.length === 0}
        <div class="empty-panel">
          <p>No queries matched your filter criteria.</p>
        </div>
      {:else}
        <div class="query-list">
          {#each filteredQueries as q (q.query_id)}
            {@const isExpanded = expandedQueryId === q.query_id}
            {@const isZeroRecall = q.metrics && (q.metrics['recall@5'] === 0 || q.metrics['recall'] === 0)}
            <div class="query-card" class:card-failed={isZeroRecall}>
              <div
                class="query-card-summary"
                role="button"
                tabindex="0"
                onclick={() => toggleQueryDetails(q.query_id)}
                onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleQueryDetails(q.query_id); }}
              >
                <div class="query-info">
                  <div class="query-id-row">
                    <span class="mono-id">{q.query_id}</span>
                    {#if isZeroRecall}
                      <span class="badge badge-fail badge-sm">Zero Recall</span>
                    {/if}
                    {#if q.latency_ms !== undefined}
                      <span class="latency-pill">{q.latency_ms.toFixed(1)} ms</span>
                    {/if}
                  </div>
                  <div class="query-text">
                    {q.text || 'No query text recorded'}
                  </div>
                </div>

                <div class="query-metrics-mini">
                  <div class="mini-metric">
                    <span class="mini-label">R@5:</span>
                    <span class="mini-val mono-code">
                      {formatQueryMetric(q.metrics, 'recall@5', 'recall')}
                    </span>
                  </div>
                  <div class="mini-metric">
                    <span class="mini-label">MRR:</span>
                    <span class="mini-val mono-code">
                      {formatQueryMetric(q.metrics, 'mrr@5', 'mrr')}
                    </span>
                  </div>
                  <span class="expand-icon">{isExpanded ? '▲' : '▼'}</span>
                </div>
              </div>

              {#if isExpanded}
                <div class="query-details-expanded">
                  {#if q.failure_trace}
                    <div class="trace-box">
                      <div class="trace-title">Failure Trace Analysis:</div>
                      {#if q.failure_trace.reason}
                        <div class="trace-reason">{q.failure_trace.reason}</div>
                      {/if}
                      {#if q.failure_trace.unretrieved_relevant_ids && q.failure_trace.unretrieved_relevant_ids.length > 0}
                        <div class="trace-group">
                          <span class="trace-group-title">Missed Relevant Chunks:</span>
                          <div class="chunk-tags">
                            {#each q.failure_trace.unretrieved_relevant_ids as cid}
                              <span class="chunk-tag tag-missed">{cid}</span>
                            {/each}
                          </div>
                        </div>
                      {/if}
                      {#if q.failure_trace.retrieved_irrelevant_ids && q.failure_trace.retrieved_irrelevant_ids.length > 0}
                        <div class="trace-group">
                          <span class="trace-group-title">Retrieved Non-Relevant Chunks:</span>
                          <div class="chunk-tags">
                            {#each q.failure_trace.retrieved_irrelevant_ids as cid}
                              <span class="chunk-tag tag-irrelevant">{cid}</span>
                            {/each}
                          </div>
                        </div>
                      {/if}
                    </div>
                  {/if}

                  <!-- Retrieved Chunks / Candidates -->
                  <div class="candidates-section">
                    <h4 class="subheading">
                      Retrieved Chunks ({q.candidates?.length ?? q.retrieved?.length ?? 0})
                    </h4>
                    {#if (q.candidates && q.candidates.length > 0) || (q.retrieved && q.retrieved.length > 0)}
                      {@const items = q.candidates || q.retrieved || []}
                      <div class="candidates-table-wrap">
                        <table class="data-table">
                          <thead>
                            <tr>
                              <th style="width: 60px;">Rank</th>
                              <th style="width: 90px;">Score</th>
                              <th>Chunk ID</th>
                              <th>Content Preview</th>
                            </tr>
                          </thead>
                          <tbody>
                            {#each items as cand}
                              <tr>
                                <td class="mono-code">#{cand.rank}</td>
                                <td class="mono-code">{cand.score.toFixed(4)}</td>
                                <td class="mono-code">{cand.chunk_id}</td>
                                <td class="chunk-content-cell">
                                  {cand.content || '(no text preview)'}
                                </td>
                              </tr>
                            {/each}
                          </tbody>
                        </table>
                      </div>
                    {:else}
                      <p class="muted-note">No ranked candidate chunks stored for this query.</p>
                    {/if}
                  </div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <!-- Config Modal -->
    {#if showConfigModal}
      <div
        class="modal-backdrop"
        onclick={() => (showConfigModal = false)}
        onkeydown={(e) => { if (e.key === 'Escape') showConfigModal = false; }}
        role="presentation"
      >
        <div
          class="modal-box"
          onclick={(e) => e.stopPropagation()}
          onkeydown={(e) => { if (e.key === 'Escape') showConfigModal = false; }}
          role="dialog"
          aria-modal="true"
          tabindex="-1"
        >
          <div class="modal-header">
            <h3>Configuration JSON</h3>
            <button class="btn btn-ghost btn-sm" onclick={() => (showConfigModal = false)}>✕</button>
          </div>
          <pre class="json-code"><code>{JSON.stringify(run.config, null, 2)}</code></pre>
          <div class="modal-footer">
            <button class="btn btn-secondary btn-sm" onclick={() => navigator.clipboard.writeText(JSON.stringify(run?.config, null, 2))}>
              Copy to Clipboard
            </button>
            <button class="btn btn-primary btn-sm" onclick={() => (showConfigModal = false)}>
              Close
            </button>
          </div>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .run-details-page {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .back-nav {
    margin-bottom: -0.5rem;
  }

  .back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .back-link:hover {
    color: var(--accent-primary);
  }

  .run-header-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
  }

  .header-main {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1.5rem;
  }

  .run-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0.5rem 0 0.75rem;
  }

  .metadata-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .mono-code {
    font-family: var(--font-mono);
    color: var(--text-primary);
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
  }

  .section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
  }

  .section-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 0.25rem 0 0;
  }

  .section-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1rem;
  }

  .query-filters {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .search-input-sm {
    width: 260px;
    padding: 0.4rem 0.75rem;
    font-size: 0.85rem;
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
  }

  .query-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .query-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
    transition: border-color 0.15s;
  }

  .query-card.card-failed {
    border-color: rgba(225, 29, 72, 0.35);
  }

  .query-card-summary {
    padding: 1rem 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    background: transparent;
  }

  .query-card-summary:hover {
    background: var(--bg-hover);
  }

  .query-info {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    flex: 1;
  }

  .query-id-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .mono-id {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent-primary);
  }

  .latency-pill {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .query-text {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-primary);
  }

  .query-metrics-mini {
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }

  .mini-metric {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.85rem;
  }

  .mini-label {
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .expand-icon {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .query-details-expanded {
    padding: 1.25rem;
    border-top: 1px solid var(--border-subtle);
    background: rgba(10, 14, 23, 0.4);
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .trace-box {
    background: rgba(225, 29, 72, 0.05);
    border: 1px solid rgba(225, 29, 72, 0.25);
    border-radius: var(--radius-sm);
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .trace-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #fda4af;
  }

  .trace-reason {
    font-size: 0.9rem;
    color: var(--text-primary);
  }

  .trace-group {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .trace-group-title {
    font-size: 0.75rem;
    color: var(--text-secondary);
  }

  .chunk-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .chunk-tag {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-sm);
  }

  .tag-missed {
    background: rgba(225, 29, 72, 0.15);
    color: #fda4af;
    border: 1px solid rgba(225, 29, 72, 0.3);
  }

  .tag-irrelevant {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  .candidates-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .subheading {
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0;
  }

  .candidates-table-wrap {
    overflow-x: auto;
  }

  .chunk-content-cell {
    font-size: 0.85rem;
    color: var(--text-secondary);
    max-width: 500px;
    white-space: normal;
  }

  .muted-note {
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  .empty-panel {
    text-align: center;
    padding: 2.5rem;
    background: var(--bg-surface);
    border: 1px dashed var(--border-subtle);
    border-radius: var(--radius-md);
    color: var(--text-muted);
  }

  /* Modal */
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.75);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    width: 90%;
    max-width: 650px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    padding: 1.25rem 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .modal-header h3 {
    margin: 0;
    font-size: 1.1rem;
  }

  .json-code {
    padding: 1.5rem;
    margin: 0;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    background: var(--bg-canvas);
    color: #93c5fd;
    line-height: 1.4;
  }

  .modal-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
  }

  .loading-state {
    text-align: center;
    padding: 4rem;
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

  .banner-error {
    background: rgba(225, 29, 72, 0.1);
    border: 1px solid rgba(225, 29, 72, 0.3);
    color: #fda4af;
    padding: 1rem;
    border-radius: var(--radius-md);
  }
</style>
