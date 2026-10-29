<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { api, ApiError } from '#lib/api';
  import MetricCard from '#lib/components/MetricCard.svelte';
  import type { CitationAuditResponse, CitationAuditQueryItem, RunSummary } from '#lib/types';

  let availableRuns = $state<RunSummary[]>([]);
  let selectedRunId = $state<string>('');
  let isAuditing = $state<boolean>(false);
  let auditData = $state<CitationAuditResponse | null>(null);
  let errorMessage = $state<string | null>(null);

  let filterQueryText = $state<string>('');
  let filterOnlyIssues = $state<boolean>(false);

  async function loadRuns() {
    try {
      const res = await api.listRuns({ limit: 100 });
      availableRuns = res.runs;
    } catch {
      // Ignore
    }
  }

  async function runAudit() {
    if (!selectedRunId.trim()) {
      errorMessage = 'Please select or enter a run ID to audit citations.';
      return;
    }

    isAuditing = true;
    errorMessage = null;
    auditData = null;

    try {
      auditData = await api.auditCitations(selectedRunId.trim());
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
    } finally {
      isAuditing = false;
    }
  }

  const filteredQueries = $derived.by(() => {
    if (!auditData || !auditData.queries) return [];
    return auditData.queries.filter((q: CitationAuditQueryItem) => {
      const matchText =
        !filterQueryText ||
        q.query_id.toLowerCase().includes(filterQueryText.toLowerCase()) ||
        (q.text && q.text.toLowerCase().includes(filterQueryText.toLowerCase())) ||
        (q.answer_text && q.answer_text.toLowerCase().includes(filterQueryText.toLowerCase()));

      if (!matchText) return false;

      if (filterOnlyIssues) {
        const hasIssues =
          q.invalid_citations.length > 0 ||
          q.unresolved_citations.length > 0 ||
          q.missing_evidence_citations.length > 0 ||
          q.malformed_citations.length > 0 ||
          q.unreferenced_required_citations.length > 0 ||
          q.citation_precision < 1.0;
        return hasIssues;
      }

      return true;
    });
  });

  onMount(async () => {
    await loadRuns();

    const paramRun = page.url.searchParams.get('run_id');
    if (paramRun) {
      selectedRunId = paramRun;
      runAudit();
    }
  });
</script>

<div class="audit-page">
  <div class="page-header">
    <h1 class="page-title">Citation Audit Studio</h1>
    <p class="page-subtitle">
      Inspect citation resolution, precision, recall, and evidence mapping across generated answers.
    </p>
  </div>

  <!-- Run Selector -->
  <div class="selection-card">
    <div class="selection-row">
      <div class="field-group flex-1">
        <label for="run-select" class="field-label">Target Run for Citation Evaluation</label>
        {#if availableRuns.length > 0}
          <select id="run-select" class="input" bind:value={selectedRunId}>
            <option value="">-- Choose run to audit --</option>
            {#each availableRuns as r (r.run_id)}
              <option value={r.run_id}>
                {r.experiment_name} ({r.run_id.slice(0, 8)})
              </option>
            {/each}
          </select>
        {/if}
        <input
          type="text"
          class="input mono-input"
          placeholder="Or paste Run ID directly..."
          bind:value={selectedRunId}
        />
      </div>

      <div class="button-group">
        <button
          class="btn btn-primary"
          disabled={isAuditing || !selectedRunId.trim()}
          onclick={runAudit}
        >
          {isAuditing ? 'Auditing Citations...' : 'Audit Citations'}
        </button>
      </div>
    </div>
  </div>

  {#if errorMessage}
    <div class="banner banner-error">
      <strong>Citation Audit Error:</strong> {errorMessage}
    </div>
  {/if}

  {#if auditData}
    <!-- Limitations / Architectural Note -->
    <div class="banner banner-info">
      <div class="info-icon">&#9432;</div>
      <div class="info-content">
        <strong class="info-title">Citation Verification Model & Groundedness Scope:</strong>
        <p class="info-text">
          {auditData.limitations_note}
        </p>
      </div>
    </div>

    <!-- Summary Metrics -->
    <section class="section">
      <div class="metric-grid">
        <MetricCard
          label="Mean Citation Precision"
          value={auditData.mean_citation_precision}
          format="percent"
          subtext="Valid citations / Total emitted"
        />
        <MetricCard
          label="Mean Citation Recall"
          value={auditData.mean_citation_recall}
          format="percent"
          subtext="Referenced evidence / Required evidence"
        />
        <MetricCard
          label="Audited Queries"
          value={auditData.audited_queries_count}
          format="raw"
          subtext="Queries with answers and citations"
        />
        <MetricCard
          label="Total Run Queries"
          value={auditData.total_queries}
          format="raw"
        />
      </div>
    </section>

    <!-- Query Drill-Down -->
    <section class="section">
      <div class="section-header-bar">
        <div>
          <h2 class="section-title">Query-by-Query Citation Breakdown</h2>
          <p class="section-subtitle">
            Inspect citations in context, mapped chunk targets, and detected anomalies.
          </p>
        </div>

        <div class="query-filters">
          <input
            type="text"
            class="input search-input-sm"
            placeholder="Search query or answer..."
            bind:value={filterQueryText}
          />
          <label class="toggle-label">
            <input type="checkbox" bind:checked={filterOnlyIssues} />
            Show only citation anomalies
          </label>
        </div>
      </div>

      {#if filteredQueries.length === 0}
        <div class="empty-panel">
          <p>No queries matched your filter criteria.</p>
        </div>
      {:else}
        <div class="audit-list">
          {#each filteredQueries as item (item.query_id)}
            {@const hasIssues = item.invalid_citations.length > 0 || item.unresolved_citations.length > 0 || item.missing_evidence_citations.length > 0 || item.malformed_citations.length > 0}
            <div class="audit-card" class:card-issues={hasIssues}>
              <div class="audit-card-header">
                <div>
                  <span class="mono-id">{item.query_id}</span>
                  {#if item.text}
                    <h3 class="query-text-heading">{item.text}</h3>
                  {/if}
                </div>

                <div class="audit-scores">
                  <div class="score-pill">
                    <span class="score-lbl">Prec:</span>
                    <span class="mono-code font-bold">{(item.citation_precision * 100).toFixed(0)}%</span>
                  </div>
                  <div class="score-pill">
                    <span class="score-lbl">Rec:</span>
                    <span class="mono-code font-bold">{(item.citation_recall * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>

              {#if item.answer_text}
                <div class="answer-box">
                  <div class="answer-label">Generated Answer:</div>
                  <p class="answer-content">{item.answer_text}</p>
                </div>
              {:else}
                <div class="no-answer-notice">
                  <em>No generated answer text stored for this query. Retrieval-only query.</em>
                </div>
              {/if}

              <!-- Citation Tags Breakdown -->
              <div class="tags-container">
                {#if item.valid_citations.length > 0}
                  <div class="tag-row">
                    <span class="tag-row-label">Valid Citations:</span>
                    <div class="tags-list">
                      {#each item.valid_citations as cit}
                        <span class="badge badge-pass badge-sm">{cit}</span>
                      {/each}
                    </div>
                  </div>
                {/if}

                {#if item.unresolved_citations.length > 0}
                  <div class="tag-row">
                    <span class="tag-row-label">Unresolved Citations:</span>
                    <div class="tags-list">
                      {#each item.unresolved_citations as cit}
                        <span class="badge badge-fail badge-sm">Unresolved: {cit}</span>
                      {/each}
                    </div>
                  </div>
                {/if}

                {#if item.missing_evidence_citations.length > 0}
                  <div class="tag-row">
                    <span class="tag-row-label">Missing Evidence:</span>
                    <div class="tags-list">
                      {#each item.missing_evidence_citations as cit}
                        <span class="badge badge-warn badge-sm">Missing Evidence: {cit}</span>
                      {/each}
                    </div>
                  </div>
                {/if}

                {#if item.malformed_citations.length > 0}
                  <div class="tag-row">
                    <span class="tag-row-label">Malformed Markers:</span>
                    <div class="tags-list">
                      {#each item.malformed_citations as cit}
                        <span class="badge badge-fail badge-sm">Malformed: {cit}</span>
                      {/each}
                    </div>
                  </div>
                {/if}

                {#if item.unreferenced_required_citations.length > 0}
                  <div class="tag-row">
                    <span class="tag-row-label">Unreferenced Required Evidence:</span>
                    <div class="tags-list">
                      {#each item.unreferenced_required_citations as cit}
                        <span class="badge badge-warn badge-sm">Required: {cit}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
              </div>

              <!-- Citation to Chunk Mapping -->
              {#if Object.keys(item.citation_to_chunk_map).length > 0}
                <div class="mapping-box">
                  <span class="mapping-title">Citation &rarr; Chunk Resolution Map:</span>
                  <div class="mapping-grid">
                    {#each Object.entries(item.citation_to_chunk_map) as [cit, chunkId]}
                      <div class="mapping-item">
                        <span class="cit-marker">{cit}</span>
                        <span class="map-arrow">&rarr;</span>
                        <code class="chunk-target">{chunkId}</code>
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </section>
  {/if}
</div>

<style>
  .audit-page {
    display: flex;
    flex-direction: column;
    gap: 2rem;
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

  .selection-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
  }

  .selection-row {
    display: flex;
    align-items: flex-end;
    gap: 1.25rem;
  }

  .flex-1 {
    flex: 1;
  }

  .field-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .field-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
  }

  .mono-input {
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }

  .banner-info {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
  }

  .info-icon {
    font-size: 1.25rem;
    color: #38bdf8;
    line-height: 1;
  }

  .info-content {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .info-title {
    color: #38bdf8;
    font-size: 0.9rem;
  }

  .info-text {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
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

  .audit-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .audit-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .audit-card.card-issues {
    border-color: rgba(225, 29, 72, 0.3);
  }

  .audit-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .mono-id {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent-primary);
  }

  .query-text-heading {
    margin: 0.25rem 0 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .audit-scores {
    display: flex;
    gap: 0.75rem;
  }

  .score-pill {
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    display: flex;
    gap: 0.35rem;
    align-items: center;
  }

  .score-lbl {
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .font-bold {
    font-weight: 600;
  }

  .answer-box {
    background: rgba(10, 14, 23, 0.4);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .answer-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .answer-content {
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
    color: var(--text-primary);
  }

  .no-answer-notice {
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  .tags-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .tag-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .tag-row-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    min-width: 140px;
  }

  .tags-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .mapping-box {
    background: var(--bg-canvas);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .mapping-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .mapping-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
  }

  .mapping-item {
    font-size: 0.8rem;
    font-family: var(--font-mono);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-surface);
    padding: 0.2rem 0.5rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
  }

  .cit-marker {
    color: #38bdf8;
  }

  .map-arrow {
    color: var(--text-muted);
  }

  .chunk-target {
    color: #a78bfa;
  }

  .empty-panel {
    text-align: center;
    padding: 2.5rem;
    background: var(--bg-surface);
    border: 1px dashed var(--border-subtle);
    border-radius: var(--radius-md);
    color: var(--text-muted);
  }

  .banner-error {
    background: rgba(225, 29, 72, 0.1);
    border: 1px solid rgba(225, 29, 72, 0.3);
    color: #fda4af;
    padding: 1rem;
    border-radius: var(--radius-md);
  }
</style>
