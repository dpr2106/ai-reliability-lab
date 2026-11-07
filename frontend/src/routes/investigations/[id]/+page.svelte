<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { api, ApiError } from '#lib/api';
  import StatusBadge from '#lib/components/StatusBadge.svelte';
  import type {
    InvestigationDetail,
    ComparisonTopic,
    EvidenceFinding
  } from '#lib/types';

  const investigationId = $derived(page.params.id);

  let investigation = $state<InvestigationDetail | null>(null);
  let isLoading = $state<boolean>(true);
  let isRerunning = $state<boolean>(false);
  let isExporting = $state<boolean>(false);
  let errorMessage = $state<string | null>(null);

  // Evidence Inspector drawer state
  let selectedFinding = $state<EvidenceFinding | null>(null);
  let activeTopicFilter = $state<string | null>(null);

  async function loadInvestigation() {
    if (!investigationId) return;
    isLoading = true;
    errorMessage = null;
    try {
      investigation = await api.getInvestigation(investigationId);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
    } finally {
      isLoading = false;
    }
  }

  async function handleRerun() {
    if (!investigationId) return;
    isRerunning = true;
    errorMessage = null;
    try {
      investigation = await api.runInvestigation(investigationId);
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isRerunning = false;
    }
  }

  async function handleExportMarkdown() {
    if (!investigationId || !investigation) return;
    isExporting = true;
    try {
      const markdown = await api.exportInvestigationMarkdown(investigationId);
      const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const safeFilename = `${investigation.title.replace(/[^a-zA-Z0-9_-]/g, '_')}_investigation.md`;
      link.download = safeFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      isExporting = false;
    }
  }

  async function handleDelete() {
    if (!investigationId) return;
    if (!confirm('Are you sure you want to delete this investigation and its findings report?')) {
      return;
    }
    try {
      await api.deleteInvestigation(investigationId);
      goto('/investigations');
    } catch (err: unknown) {
      alert(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function getFindingsForCell(topic: ComparisonTopic, docId: string): EvidenceFinding[] {
    return topic.findings.filter((f) => f.doc_id === docId);
  }

  function getRelationshipClass(rel: string): string {
    switch (rel) {
      case 'supports':
        return 'rel-supports';
      case 'contradicts':
        return 'rel-contradicts';
      case 'qualifies':
        return 'rel-qualifies';
      case 'not_addressed':
        return 'rel-not-addressed';
      case 'insufficient_evidence':
        return 'rel-insufficient';
      default:
        return 'rel-neutral';
    }
  }

  function formatRelationshipLabel(rel: string): string {
    switch (rel) {
      case 'supports':
        return 'Corroborates / Supports';
      case 'contradicts':
        return 'Contradicts / Opposes';
      case 'qualifies':
        return 'Qualified / Bounded';
      case 'not_addressed':
        return 'Not Addressed';
      case 'insufficient_evidence':
        return 'Insufficient Evidence';
      default:
        return rel.toUpperCase();
    }
  }

  onMount(() => {
    loadInvestigation();
  });
</script>

<svelte:head>
  <title>
    {investigation ? `${investigation.title} | STING Workbench` : 'Evidence Investigation | STING'}
  </title>
</svelte:head>

<div class="workbench-layout">
  <!-- Top Navigation & Controls Header -->
  <header class="workbench-header">
    <div class="header-breadcrumb">
      <a href="/investigations" class="breadcrumb-link">← All Investigations</a>
      <span class="breadcrumb-sep">/</span>
      <span class="breadcrumb-current">Workbench</span>
    </div>

    {#if investigation}
      <div class="header-main-row">
        <div class="header-title-col">
          <div class="title-meta-row">
            <h1 class="workbench-title">{investigation.title}</h1>
            <StatusBadge status={investigation.status} size="sm" />
          </div>
          <p class="workbench-query">"{investigation.query}"</p>
        </div>

        <div class="header-controls">
          <button
            class="btn btn-secondary"
            onclick={handleExportMarkdown}
            disabled={isExporting || !investigation.report}
            title="Download full publication-grade Markdown report"
          >
            {#if isExporting}
              <span class="spinner-sm"></span> Exporting...
            {:else}
              ↓ Export Markdown
            {/if}
          </button>

          <button
            class="btn btn-primary"
            onclick={handleRerun}
            disabled={isRerunning}
            title="Re-run retrieval, evidence extraction, and synthesis"
          >
            {#if isRerunning}
              <span class="spinner-sm"></span> Re-running...
            {:else}
              ↻ Re-run Analysis
            {/if}
          </button>

          <button class="btn btn-danger-icon" onclick={handleDelete} title="Delete Investigation">
            🗑
          </button>
        </div>
      </div>

      <!-- Examined Document Sources Bar -->
      {#if investigation.report}
        <div class="sources-strip">
          <span class="sources-label">Investigated Sources:</span>
          <div class="sources-chips">
            {#each investigation.report.document_summaries as doc (doc.doc_id)}
              <span class="source-chip" title="{doc.title || doc.filename} ({doc.doc_id})">
                <span class="doc-icon">📄</span>
                <span class="doc-name">{doc.title || doc.filename}</span>
                <span class="doc-chunk-count">{doc.chunk_count} chunks</span>
              </span>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </header>

  {#if errorMessage}
    <div class="alert alert-error" role="alert">
      <span>{errorMessage}</span>
      <button class="btn-close" onclick={() => (errorMessage = null)}>✕</button>
    </div>
  {/if}

  {#if isLoading}
    <div class="loading-state">
      <div class="spinner"></div>
      <p>Loading evidence investigation...</p>
    </div>
  {:else if !investigation}
    <div class="empty-state">
      <h3>Investigation Not Found</h3>
      <p>The requested investigation entity could not be retrieved from the persistent store.</p>
      <a href="/investigations" class="btn btn-primary">Return to Investigations</a>
    </div>
  {:else if investigation.status === 'running'}
    <div class="running-state">
      <div class="spinner"></div>
      <h3>Analysis In Progress</h3>
      <p>Extracting structured evidence from documents and evaluating cross-source consistency...</p>
    </div>
  {:else if investigation.status === 'failed'}
    <div class="failed-state">
      <div class="failed-icon">⚠️</div>
      <h3>Investigation Execution Failed</h3>
      <p class="failed-detail">{investigation.error_message || 'An unexpected error occurred during synthesis.'}</p>
      <button class="btn btn-primary" onclick={handleRerun} disabled={isRerunning}>
        Try Running Again
      </button>
    </div>
  {:else if investigation.report}
    {@const report = investigation.report}

    <!-- Executive Summary Card -->
    <section class="summary-card">
      <div class="summary-header">
        <h2 class="summary-title">Executive Evidence Synthesis</h2>
        <span class="synthesis-badge">Deterministic Grounding</span>
      </div>
      <p class="summary-text">{report.executive_summary}</p>

      <!-- Key Breakdown Counters -->
      <div class="evidence-stats-grid">
        <div class="metric-card">
          <span class="metric-val">{report.total_findings}</span>
          <span class="metric-lbl">Total Findings</span>
        </div>
        <div class="metric-card metric-supports">
          <span class="metric-val">{report.relationship_counts.supports || 0}</span>
          <span class="metric-lbl">Corroborating Evidence</span>
        </div>
        <div class="metric-card metric-contradicts">
          <span class="metric-val">{report.relationship_counts.contradicts || 0}</span>
          <span class="metric-lbl">Disagreements / Contradictions</span>
        </div>
        <div class="metric-card metric-qualifies">
          <span class="metric-val">{report.relationship_counts.qualifies || 0}</span>
          <span class="metric-lbl">Qualified Statements</span>
        </div>
        <div class="metric-card metric-gaps">
          <span class="metric-val">{(report.relationship_counts.not_addressed || 0) + (report.relationship_counts.insufficient_evidence || 0)}</span>
          <span class="metric-lbl">Evidence Gaps</span>
        </div>
      </div>
    </section>

    <!-- SIGNATURE WORKBENCH COMPONENT: Cross-Document Comparison Matrix -->
    <section class="matrix-section">
      <div class="matrix-header">
        <div>
          <h2 class="matrix-title">Cross-Document Evidence Comparison Matrix</h2>
          <p class="matrix-desc">
            Aspects (rows) cross-analyzed against document collections (columns). Click any cell to inspect the verbatim source passage, chunk citation, and methodology notes.
          </p>
        </div>
      </div>

      <div class="matrix-table-wrapper">
        <table class="matrix-table">
          <thead>
            <tr>
              <th class="th-aspect">Comparison Dimension / Aspect</th>
              {#each report.document_summaries as doc (doc.doc_id)}
                <th class="th-doc">
                  <div class="th-doc-title">{doc.title || doc.filename}</div>
                  <div class="th-doc-sub">`{doc.doc_id.slice(0, 8)}` • {doc.chunk_count} chunks</div>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each report.topics as topic (topic.topic_id)}
              <tr class="matrix-row">
                <!-- Aspect Header Cell -->
                <td class="td-aspect">
                  <div class="aspect-title">{topic.topic_name}</div>
                  <div class="aspect-desc">{topic.description}</div>
                  {#if topic.agreements_summary}
                    <div class="topic-consensus-chip" title="{topic.agreements_summary}">
                      Consensus noted
                    </div>
                  {/if}
                  {#if topic.disagreements_summary && topic.disagreements_summary.includes('Conflicting')}
                    <div class="topic-conflict-chip" title="{topic.disagreements_summary}">
                      Disagreement detected
                    </div>
                  {/if}
                </td>

                <!-- Document Evidence Cells -->
                {#each report.document_summaries as doc (doc.doc_id)}
                  {@const cellFindings = getFindingsForCell(topic, doc.doc_id)}
                  <td class="td-evidence-cell">
                    {#if cellFindings.length === 0}
                      <div class="empty-cell">
                        <span class="muted-dash">-</span>
                        <span class="empty-lbl">Not addressed</span>
                      </div>
                    {:else}
                      <div class="cell-findings-list">
                        {#each cellFindings as finding (finding.finding_id)}
                          <button
                            type="button"
                            class="finding-card {getRelationshipClass(finding.relationship)}"
                            onclick={() => (selectedFinding = finding)}
                          >
                            <div class="finding-top">
                              <span class="rel-pill {finding.relationship}">
                                {finding.relationship.replace('_', ' ').toUpperCase()}
                              </span>
                              {#if finding.page_number}
                                <span class="page-pill">p. {finding.page_number}</span>
                              {/if}
                            </div>

                            <p class="finding-claim">{finding.claim}</p>

                            <div class="finding-bottom">
                              <span class="chunk-badge">`{finding.chunk_id}`</span>
                              <span class="conf-badge">{(finding.confidence * 100).toFixed(0)}% conf</span>
                            </div>
                          </button>
                        {/each}
                      </div>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

    <!-- Cross-Source Thematic Synthesis Cards -->
    <section class="thematic-cards-grid">
      {#each report.topics as topic (topic.topic_id)}
        <div class="thematic-card">
          <div class="thematic-header">
            <h3 class="thematic-title">{topic.topic_name}</h3>
            <span class="thematic-count">{topic.findings.length} findings</span>
          </div>

          {#if topic.agreements_summary}
            <div class="synthesis-block block-agreements">
              <span class="synthesis-label">✓ Mutual Agreement & Corroboration</span>
              <p class="synthesis-body">{topic.agreements_summary}</p>
            </div>
          {/if}

          {#if topic.disagreements_summary}
            <div class="synthesis-block block-disagreements">
              <span class="synthesis-label">✕ Divergences & Contradictions</span>
              <p class="synthesis-body">{topic.disagreements_summary}</p>
            </div>
          {/if}

          {#if topic.evidence_gaps && topic.evidence_gaps.length > 0}
            <div class="synthesis-block block-gaps">
              <span class="synthesis-label">○ Evidence Gaps & Missing Data</span>
              <ul class="gaps-list">
                {#each topic.evidence_gaps as gap}
                  <li>{gap}</li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
      {/each}
    </section>

    <!-- Methodology & Experimental Nuance -->
    {#if report.methodology_synthesis}
      <section class="methodology-card">
        <h3 class="methodology-title">Methodological Nuances & Experimental Context</h3>
        <p class="methodology-body">{report.methodology_synthesis}</p>
      </section>
    {/if}
  {/if}
</div>

<!-- SLIDE-OVER / MODAL EVIDENCE INSPECTOR DRAWER -->
{#if selectedFinding}
  <div
    class="drawer-backdrop"
    onclick={() => (selectedFinding = null)}
    onkeydown={(e) => e.key === 'Escape' && (selectedFinding = null)}
    role="presentation"
  >
    <div
      class="drawer-card"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="drawer-heading"
      tabindex="-1"
    >
      <div class="drawer-header">
        <div>
          <span class="drawer-pretitle">Verified Source Passage Inspector</span>
          <h2 id="drawer-heading" class="drawer-title">{selectedFinding.doc_title}</h2>
        </div>
        <button class="drawer-close" onclick={() => (selectedFinding = null)}>✕</button>
      </div>

      <div class="drawer-content">
        <!-- Relationship & Confidence Banner -->
        <div class="drawer-banner {getRelationshipClass(selectedFinding.relationship)}">
          <div class="banner-rel-title">
            Status: {formatRelationshipLabel(selectedFinding.relationship)}
          </div>
          <div class="banner-conf">
            Model Confidence: {(selectedFinding.confidence * 100).toFixed(1)}%
          </div>
        </div>

        <!-- Claim Statement -->
        <div class="drawer-section">
          <h4 class="drawer-section-title">Synthesized Finding / Claim</h4>
          <p class="drawer-claim-text">{selectedFinding.claim}</p>
        </div>

        <!-- Verbatim Supporting Evidence Passage -->
        <div class="drawer-section">
          <div class="passage-header">
            <h4 class="drawer-section-title">Verbatim Source Passage</h4>
            <div class="passage-coordinates">
              {#if selectedFinding.page_number}
                <span class="coord-pill">Page {selectedFinding.page_number}</span>
              {/if}
              <span class="coord-pill">Chunk `{selectedFinding.chunk_id}`</span>
            </div>
          </div>

          {#if selectedFinding.supporting_passage}
            <blockquote class="verbatim-quote">
              "{selectedFinding.supporting_passage}"
            </blockquote>
          {:else}
            <div class="empty-passage-note">
              No verbatim passage excerpt was recorded for this aspect.
            </div>
          {/if}
        </div>

        <!-- Methodology & Conditions Notes -->
        {#if selectedFinding.methodology_notes}
          <div class="drawer-section">
            <h4 class="drawer-section-title">Experimental Setup & Methodology Notes</h4>
            <p class="methodology-notes-text">{selectedFinding.methodology_notes}</p>
          </div>
        {/if}

        <!-- Provenance Trace Info -->
        <div class="provenance-info-box">
          <div class="prov-row">
            <span class="prov-lbl">Document ID:</span>
            <span class="prov-val">`{selectedFinding.doc_id}`</span>
          </div>
          <div class="prov-row">
            <span class="prov-lbl">Chunk ID:</span>
            <span class="prov-val">`{selectedFinding.chunk_id}`</span>
          </div>
          <div class="prov-row">
            <span class="prov-lbl">Finding ID:</span>
            <span class="prov-val">`{selectedFinding.finding_id}`</span>
          </div>
          <p class="prov-disclaimer">
            Passage coordinates verified against deterministic SQLite repository chunks. Verification status indicates claim-passage alignment.
          </p>
        </div>
      </div>

      <div class="drawer-footer">
        <button type="button" class="btn btn-secondary" onclick={() => (selectedFinding = null)}>
          Close Inspector
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .workbench-layout {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 5rem;
  }

  .workbench-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-breadcrumb {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    margin-bottom: 1rem;
  }

  .breadcrumb-link {
    color: #38bdf8;
    text-decoration: none;
  }
  .breadcrumb-link:hover {
    text-decoration: underline;
  }

  .breadcrumb-sep {
    color: var(--text-muted);
  }

  .breadcrumb-current {
    color: var(--text-muted);
  }

  .header-main-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 2rem;
  }

  .title-meta-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
  }

  .workbench-title {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
  }

  .workbench-query {
    font-size: 0.95rem;
    color: var(--text-secondary);
    font-style: italic;
    margin: 0;
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .sources-strip {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px dashed rgba(255, 255, 255, 0.08);
  }

  .sources-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .sources-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.25rem 0.6rem;
    font-size: 0.8rem;
    color: var(--text-primary);
  }

  .doc-name {
    font-weight: 500;
  }

  .doc-chunk-count {
    color: var(--text-muted);
    font-size: 0.72rem;
  }

  /* Executive Summary Card */
  .summary-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .summary-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .summary-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .synthesis-badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.2rem 0.5rem;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.25);
  }

  .summary-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-secondary);
    margin: 0 0 1.25rem;
  }

  .evidence-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.75rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border-subtle);
  }

  .metric-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.8rem;
    text-align: center;
  }

  .metric-val {
    display: block;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
  }

  .metric-lbl {
    display: block;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
  }

  .metric-supports .metric-val {
    color: #4ade80;
  }
  .metric-contradicts .metric-val {
    color: #fb7185;
  }
  .metric-qualifies .metric-val {
    color: #fbbf24;
  }
  .metric-gaps .metric-val {
    color: #94a3b8;
  }

  /* Comparison Matrix */
  .matrix-section {
    margin-bottom: 2.5rem;
  }

  .matrix-header {
    margin-bottom: 1rem;
  }

  .matrix-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.35rem;
  }

  .matrix-desc {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 0;
  }

  .matrix-table-wrapper {
    overflow-x: auto;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    background: #090d16;
  }

  .matrix-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    text-align: left;
  }

  .matrix-table th,
  .matrix-table td {
    border: 1px solid var(--border-subtle);
    padding: 1rem;
    vertical-align: top;
  }

  .th-aspect {
    width: 280px;
    min-width: 240px;
    background: #0f172a;
    color: var(--text-primary);
    font-weight: 600;
  }

  .th-doc {
    min-width: 260px;
    background: #0f172a;
  }

  .th-doc-title {
    font-weight: 600;
    color: var(--text-primary);
  }

  .th-doc-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: monospace;
    margin-top: 0.2rem;
  }

  .td-aspect {
    background: rgba(15, 23, 42, 0.6);
  }

  .aspect-title {
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 0.25rem;
  }

  .aspect-desc {
    font-size: 0.75rem;
    color: var(--text-muted);
    line-height: 1.4;
    margin-bottom: 0.5rem;
  }

  .topic-consensus-chip {
    display: inline-block;
    font-size: 0.68rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
    margin-right: 0.35rem;
  }

  .topic-conflict-chip {
    display: inline-block;
    font-size: 0.68rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.3);
  }

  .td-evidence-cell {
    background: rgba(255, 255, 255, 0.01);
  }

  .empty-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1rem 0;
    color: var(--text-muted);
    font-size: 0.75rem;
  }

  .cell-findings-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .finding-card {
    display: block;
    width: 100%;
    text-align: left;
    border-radius: var(--radius-sm);
    padding: 0.75rem;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid var(--border-subtle);
    background: #0f172a;
  }
  .finding-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .finding-card.rel-supports {
    border-left: 3px solid #22c55e;
  }
  .finding-card.rel-contradicts {
    border-left: 3px solid #f43f5e;
  }
  .finding-card.rel-qualifies {
    border-left: 3px solid #f59e0b;
  }
  .finding-card.rel-not-addressed {
    border-left: 3px solid #64748b;
    opacity: 0.7;
  }

  .finding-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
  }

  .rel-pill {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
  }
  .rel-pill.supports {
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
  }
  .rel-pill.contradicts {
    background: rgba(244, 63, 94, 0.2);
    color: #fb7185;
  }
  .rel-pill.qualifies {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
  }
  .rel-pill.not_addressed {
    background: rgba(100, 116, 139, 0.2);
    color: #94a3b8;
  }

  .page-pill {
    font-size: 0.7rem;
    color: var(--text-muted);
  }

  .finding-claim {
    font-size: 0.82rem;
    line-height: 1.4;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
  }

  .finding-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.68rem;
    color: var(--text-muted);
  }

  .chunk-badge {
    font-family: monospace;
    color: #38bdf8;
  }

  /* Thematic Deep Dive Cards */
  .thematic-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2.5rem;
  }

  .thematic-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.25rem;
  }

  .thematic-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .thematic-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .thematic-count {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .synthesis-block {
    margin-bottom: 0.75rem;
    padding: 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    line-height: 1.45;
  }

  .synthesis-label {
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
  }

  .block-agreements {
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.2);
    color: #dcfce7;
  }
  .block-agreements .synthesis-label {
    color: #4ade80;
  }

  .block-disagreements {
    background: rgba(244, 63, 94, 0.06);
    border: 1px solid rgba(244, 63, 94, 0.2);
    color: #ffe4e6;
  }
  .block-disagreements .synthesis-label {
    color: #fb7185;
  }

  .block-gaps {
    background: rgba(148, 163, 184, 0.06);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: var(--text-secondary);
  }
  .block-gaps .synthesis-label {
    color: #94a3b8;
  }

  .gaps-list {
    margin: 0;
    padding-left: 1.25rem;
  }

  .methodology-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
  }

  .methodology-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
  }

  .methodology-body {
    font-size: 0.875rem;
    line-height: 1.5;
    color: var(--text-secondary);
    margin: 0;
    white-space: pre-line;
  }

  /* Slide-Over Drawer */
  .drawer-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(3px);
    display: flex;
    justify-content: flex-end;
    z-index: 1000;
  }

  .drawer-card {
    background: #0f172a;
    border-left: 1px solid var(--border-subtle);
    width: 100%;
    max-width: 540px;
    height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: -8px 0 25px rgba(0, 0, 0, 0.6);
  }

  .drawer-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
  }

  .drawer-pretitle {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38bdf8;
  }

  .drawer-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.25rem 0 0;
  }

  .drawer-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1.25rem;
    cursor: pointer;
    padding: 0.25rem;
  }
  .drawer-close:hover {
    color: var(--text-primary);
  }

  .drawer-content {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .drawer-banner {
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid;
  }
  .drawer-banner.rel-supports {
    background: rgba(34, 197, 94, 0.1);
    border-color: rgba(34, 197, 94, 0.3);
    color: #4ade80;
  }
  .drawer-banner.rel-contradicts {
    background: rgba(244, 63, 94, 0.1);
    border-color: rgba(244, 63, 94, 0.3);
    color: #fb7185;
  }
  .drawer-banner.rel-qualifies {
    background: rgba(245, 158, 11, 0.1);
    border-color: rgba(245, 158, 11, 0.3);
    color: #fbbf24;
  }
  .drawer-banner.rel-not-addressed {
    background: rgba(100, 116, 139, 0.1);
    border-color: rgba(100, 116, 139, 0.3);
    color: #94a3b8;
  }

  .banner-rel-title {
    font-weight: 700;
    font-size: 0.85rem;
  }
  .banner-conf {
    font-size: 0.78rem;
  }

  .drawer-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin: 0 0 0.5rem;
  }

  .drawer-claim-text {
    font-size: 0.95rem;
    line-height: 1.5;
    color: var(--text-primary);
    margin: 0;
  }

  .passage-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .passage-coordinates {
    display: flex;
    gap: 0.4rem;
  }

  .coord-pill {
    font-size: 0.72rem;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    color: #38bdf8;
    font-family: monospace;
  }

  .verbatim-quote {
    margin: 0;
    padding: 1rem 1.25rem;
    background: rgba(0, 0, 0, 0.3);
    border-left: 3px solid #38bdf8;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-style: italic;
    color: #f1f5f9;
    line-height: 1.6;
    font-size: 0.9rem;
  }

  .methodology-notes-text {
    font-size: 0.875rem;
    line-height: 1.5;
    color: var(--text-secondary);
    margin: 0;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.75rem;
  }

  .provenance-info-box {
    background: rgba(0, 0, 0, 0.25);
    border: 1px dashed var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.85rem;
  }

  .prov-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    margin-bottom: 0.35rem;
  }

  .prov-lbl {
    color: var(--text-muted);
  }

  .prov-val {
    color: var(--text-primary);
    font-family: monospace;
  }

  .prov-disclaimer {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-style: italic;
    margin: 0.5rem 0 0;
    line-height: 1.35;
  }

  .drawer-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: flex-end;
  }

  /* General elements */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    border: none;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-primary {
    background: #0284c7;
    color: #fff;
  }
  .btn-primary:hover:not(:disabled) {
    background: #0369a1;
  }

  .btn-secondary {
    background: var(--bg-surface);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
  }
  .btn-secondary:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .btn-danger-icon {
    background: none;
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
    font-size: 1rem;
    cursor: pointer;
    padding: 0.45rem 0.65rem;
    border-radius: var(--radius-sm);
  }
  .btn-danger-icon:hover {
    color: #f43f5e;
    border-color: rgba(244, 63, 94, 0.4);
    background: rgba(244, 63, 94, 0.1);
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .loading-state,
  .running-state,
  .empty-state,
  .failed-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
  }

  .spinner-sm {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .alert {
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .alert-error {
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.3);
    color: #fda4af;
  }
  .btn-close {
    background: none;
    border: none;
    color: currentColor;
    cursor: pointer;
  }
</style>
