<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { api, ApiError } from '#lib/api';
  import StatusBadge from '#lib/components/StatusBadge.svelte';
  import type { RegressionResult, RunSummary, MetricDelta, QueryMetricDelta } from '#lib/types';

  let availableRuns = $state<RunSummary[]>([]);
  let baselineRunId = $state<string>('');
  let candidateRunId = $state<string>('');
  let tolerance = $state<number>(0.02);

  let isComparing = $state<boolean>(false);
  let regressionResult = $state<RegressionResult | null>(null);
  let errorMessage = $state<string | null>(null);

  async function loadRuns() {
    try {
      const res = await api.listRuns({ limit: 100 });
      availableRuns = res.runs;
    } catch {
      // Ignore initial list load failures
    }
  }

  async function runComparison() {
    if (!baselineRunId.trim() || !candidateRunId.trim()) {
      errorMessage = 'Please select or enter both a baseline run ID and a candidate run ID.';
      return;
    }

    if (baselineRunId.trim() === candidateRunId.trim()) {
      errorMessage = 'Baseline and candidate run IDs must be distinct.';
      return;
    }

    isComparing = true;
    errorMessage = null;
    regressionResult = null;

    try {
      regressionResult = await api.compareRuns({
        baseline_run_id: baselineRunId.trim(),
        candidate_run_id: candidateRunId.trim(),
        tolerance: Number(tolerance)
      });
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
    } finally {
      isComparing = false;
    }
  }

  onMount(async () => {
    await loadRuns();

    const paramBaseline = page.url.searchParams.get('baseline');
    const paramCandidate = page.url.searchParams.get('candidate');

    if (paramBaseline) baselineRunId = paramBaseline;
    if (paramCandidate) candidateRunId = paramCandidate;

    if (baselineRunId && candidateRunId) {
      runComparison();
    }
  });
</script>

<div class="compare-page">
  <div class="page-header">
    <h1 class="page-title">Regression Comparison & Gating</h1>
    <p class="page-subtitle">
      Diff two benchmark runs with directional metric tolerances to detect regressions and evaluate quality shifts.
    </p>
  </div>

  <!-- Run Selector Bar -->
  <div class="selection-card">
    <div class="selection-grid">
      <div class="field-group">
        <label for="baseline-select" class="field-label">Baseline Run (Reference)</label>
        {#if availableRuns.length > 0}
          <select id="baseline-select" class="input" bind:value={baselineRunId}>
            <option value="">-- Choose baseline run --</option>
            {#each availableRuns as run (run.run_id)}
              <option value={run.run_id}>
                {run.experiment_name} ({run.run_id.slice(0, 8)})
              </option>
            {/each}
          </select>
        {/if}
        <input
          type="text"
          class="input mono-input"
          placeholder="Or paste Baseline Run ID..."
          bind:value={baselineRunId}
        />
      </div>

      <div class="field-group">
        <label for="candidate-select" class="field-label">Candidate Run (New)</label>
        {#if availableRuns.length > 0}
          <select id="candidate-select" class="input" bind:value={candidateRunId}>
            <option value="">-- Choose candidate run --</option>
            {#each availableRuns as run (run.run_id)}
              <option value={run.run_id}>
                {run.experiment_name} ({run.run_id.slice(0, 8)})
              </option>
            {/each}
          </select>
        {/if}
        <input
          type="text"
          class="input mono-input"
          placeholder="Or paste Candidate Run ID..."
          bind:value={candidateRunId}
        />
      </div>

      <div class="field-group tolerance-group">
        <label for="tolerance-input" class="field-label">Allowed Degradation Tolerance</label>
        <div class="tolerance-input-wrap">
          <input
            id="tolerance-input"
            type="number"
            step="0.005"
            min="0"
            max="1"
            class="input"
            bind:value={tolerance}
          />
          <span class="tolerance-hint">e.g. 0.02 = 2% drop allowed</span>
        </div>
      </div>
    </div>

    <div class="action-row">
      <button
        class="btn btn-primary"
        disabled={isComparing || !baselineRunId || !candidateRunId}
        onclick={runComparison}
      >
        {isComparing ? 'Comparing Runs...' : 'Compute Regression Diff'}
      </button>
    </div>
  </div>

  {#if errorMessage}
    <div class="banner banner-error">
      <strong>Comparison Error:</strong> {errorMessage}
    </div>
  {/if}

  {#if regressionResult}
    <!-- Status Result Banner -->
    <div class="result-banner status-{regressionResult.status.toLowerCase()}">
      <div class="result-status-col">
        <span class="status-title">GATE STATUS</span>
        <div class="status-large">
          <StatusBadge status={regressionResult.status} />
        </div>
      </div>
      <div class="result-summary-col">
        <div class="summary-headline">
          {#if regressionResult.status === 'PASS'}
            All aggregate and per-query metrics within defined tolerances.
          {:else if regressionResult.status === 'REGRESSION'}
            Regressions detected exceeding the allowed degradation tolerance!
          {:else}
            Evaluation incomplete or baseline/candidate query discrepancy observed.
          {/if}
        </div>
        <div class="summary-details">
          <span><strong>Queries Compared:</strong> {regressionResult.total_queries_compared}</span>
          <span><strong>Regressed Aggregates:</strong> {regressionResult.regressed_aggregate_metrics.length}</span>
          <span><strong>Regressed Queries:</strong> {regressionResult.regressed_query_ids.length}</span>
        </div>
      </div>
    </div>

    <!-- Warnings if any -->
    {#if regressionResult.warnings && regressionResult.warnings.length > 0}
      <div class="banner banner-warn">
        <strong>Warnings:</strong>
        <ul class="warning-list">
          {#each regressionResult.warnings as warn}
            <li>{warn}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Aggregate Metric Deltas -->
    <section class="section">
      <h2 class="section-title">Aggregate Metric Diff</h2>
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Direction</th>
              <th>Baseline Score</th>
              <th>Candidate Score</th>
              <th>Delta (Δ)</th>
              <th>Degradation</th>
              <th>Tolerance</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {#each Object.keys(regressionResult.aggregate_deltas) as name}
              {@const delta = regressionResult.aggregate_deltas[name] as MetricDelta}
              <tr class:row-regressed={delta.is_regressed}>
                <td class="metric-name-cell">{name}</td>
                <td>
                  <span class="direction-tag">
                    {delta.direction === 'HIGHER_IS_BETTER' ? '▲ Higher' : '▼ Lower'}
                  </span>
                </td>
                <td class="mono-code">{delta.baseline_score.toFixed(4)}</td>
                <td class="mono-code">{delta.candidate_score.toFixed(4)}</td>
                <td class="mono-code" class:val-drop={delta.delta < 0} class:val-gain={delta.delta > 0}>
                  {delta.delta > 0 ? '+' : ''}{delta.delta.toFixed(4)}
                </td>
                <td class="mono-code">{delta.degradation.toFixed(4)}</td>
                <td class="mono-code">{delta.tolerance.toFixed(4)}</td>
                <td>
                  {#if delta.is_regressed}
                    <span class="badge badge-fail badge-sm">REGRESSED</span>
                  {:else}
                    <span class="badge badge-pass badge-sm">PASS</span>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

    <!-- Worst Degradations List -->
    {#if regressionResult.worst_degradations && regressionResult.worst_degradations.length > 0}
      <section class="section">
        <h2 class="section-title">Worst Query Degradations</h2>
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>Query ID</th>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Candidate</th>
                <th>Degradation</th>
                <th>Tolerance</th>
              </tr>
            </thead>
            <tbody>
              {#each regressionResult.worst_degradations as d}
                <tr class="row-regressed">
                  <td class="mono-code font-bold">{d.query_id}</td>
                  <td>{d.metric_name}</td>
                  <td class="mono-code">{d.baseline_score.toFixed(4)}</td>
                  <td class="mono-code">{d.candidate_score.toFixed(4)}</td>
                  <td class="mono-code val-drop">+{d.degradation.toFixed(4)}</td>
                  <td class="mono-code">{d.tolerance.toFixed(4)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}

    <!-- Regressed Queries Detailed Breakdown -->
    {#if regressionResult.query_regressions && regressionResult.query_regressions.length > 0}
      <section class="section">
        <h2 class="section-title">
          Regressed Queries Breakdown ({regressionResult.query_regressions.length})
        </h2>
        <div class="query-regression-cards">
          {#each regressionResult.query_regressions as qr (qr.query_id)}
            <div class="q-regress-card">
              <div class="q-regress-header">
                <div>
                  <span class="mono-code font-bold">{qr.query_id}</span>
                  {#if qr.query_text}
                    <div class="q-text">{qr.query_text}</div>
                  {/if}
                </div>
                <div class="q-max-deg">
                  Max Degradation: <strong>+{qr.max_degradation.toFixed(4)}</strong>
                </div>
              </div>

              <div class="q-regress-metrics">
                {#each Object.keys(qr.metrics) as mName}
                  {@const mDelta = qr.metrics[mName] as QueryMetricDelta}
                  {#if mDelta.is_regressed}
                    <div class="q-regress-pill">
                      <span class="q-metric-name">{mName}:</span>
                      <span class="mono-code">
                        {mDelta.baseline_score.toFixed(2)} &rarr; {mDelta.candidate_score.toFixed(2)}
                      </span>
                      <span class="q-deg-val">(+{mDelta.degradation.toFixed(3)})</span>
                    </div>
                  {/if}
                {/each}
              </div>
            </div>
          {/each}
        </div>
      </section>
    {/if}
  {/if}
</div>

<style>
  .compare-page {
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
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .selection-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 220px;
    gap: 1.25rem;
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

  .tolerance-input-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .tolerance-hint {
    font-size: 0.7rem;
    color: var(--text-muted);
  }

  .action-row {
    display: flex;
    justify-content: flex-end;
  }

  .result-banner {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 1.5rem;
    border-radius: var(--radius-lg);
    border: 1px solid;
  }

  .result-banner.status-pass {
    background: rgba(5, 150, 105, 0.08);
    border-color: rgba(5, 150, 105, 0.4);
  }

  .result-banner.status-regression {
    background: rgba(225, 29, 72, 0.08);
    border-color: rgba(225, 29, 72, 0.4);
  }

  .result-banner.status-incomplete {
    background: rgba(217, 119, 6, 0.08);
    border-color: rgba(217, 119, 6, 0.4);
  }

  .result-status-col {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .status-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }

  .status-large {
    font-size: 1.2rem;
  }

  .result-summary-col {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .summary-headline {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .summary-details {
    display: flex;
    gap: 1.5rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
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

  .metric-name-cell {
    font-weight: 600;
    color: var(--text-primary);
  }

  .direction-tag {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .val-drop {
    color: #fda4af;
  }

  .val-gain {
    color: #6ee7b7;
  }

  .row-regressed {
    background-color: rgba(225, 29, 72, 0.06);
  }

  .font-bold {
    font-weight: 600;
  }

  .query-regression-cards {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .q-regress-card {
    background: var(--bg-surface);
    border: 1px solid rgba(225, 29, 72, 0.3);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .q-regress-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .q-text {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }

  .q-max-deg {
    font-size: 0.85rem;
    color: #fda4af;
  }

  .q-regress-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .q-regress-pill {
    background: rgba(225, 29, 72, 0.12);
    border: 1px solid rgba(225, 29, 72, 0.25);
    padding: 0.35rem 0.65rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .q-metric-name {
    font-weight: 600;
    color: #fda4af;
  }

  .q-deg-val {
    color: #f87171;
    font-weight: 600;
  }

  .banner-warn {
    background: rgba(217, 119, 6, 0.1);
    border: 1px solid rgba(217, 119, 6, 0.3);
    color: #fcd34d;
    padding: 1rem;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
  }

  .warning-list {
    margin: 0.5rem 0 0 1.25rem;
    padding: 0;
  }

  .banner-error {
    background: rgba(225, 29, 72, 0.1);
    border: 1px solid rgba(225, 29, 72, 0.3);
    color: #fda4af;
    padding: 1rem;
    border-radius: var(--radius-md);
  }
</style>
