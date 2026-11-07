<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api, ApiError } from '#lib/api';
  import StatusBadge from '#lib/components/StatusBadge.svelte';
  import type {
    DocumentSummary,
    InvestigationSummary,
    CreateInvestigationRequest
  } from '#lib/types';

  let investigations = $state<InvestigationSummary[]>([]);
  let availableDocs = $state<DocumentSummary[]>([]);
  let isLoading = $state<boolean>(true);
  let isCreating = $state<boolean>(false);
  let errorMessage = $state<string | null>(null);

  // Creator form state
  let showCreateModal = $state<boolean>(false);
  let newTitle = $state<string>('');
  let newQuery = $state<string>('');
  let selectedDocIds = $state<string[]>([]);
  let customAspectInput = $state<string>('');
  let aspects = $state<string[]>([
    'Core Claims & Architecture',
    'Performance Metrics & Results',
    'Methodology & Experimental Setup',
    'Agreements & Common Findings',
    'Contradictions & Opposing Results',
    'Limitations & Open Questions'
  ]);

  async function loadData() {
    isLoading = true;
    errorMessage = null;
    try {
      const [invRes, docRes] = await Promise.all([
        api.listInvestigations({ limit: 50 }),
        api.listDocuments({ limit: 100 })
      ]);
      investigations = invRes.investigations;
      availableDocs = docRes.items.filter((d) => d.status === 'ready' && d.chunk_count > 0);
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    } finally {
      isLoading = false;
    }
  }

  function toggleDoc(docId: string) {
    if (selectedDocIds.includes(docId)) {
      selectedDocIds = selectedDocIds.filter((id) => id !== docId);
    } else {
      selectedDocIds = [...selectedDocIds, docId];
    }
  }

  function selectAllDocs() {
    if (selectedDocIds.length === availableDocs.length) {
      selectedDocIds = [];
    } else {
      selectedDocIds = availableDocs.map((d) => d.doc_id);
    }
  }

  function addAspect() {
    const trimmed = customAspectInput.trim();
    if (trimmed && !aspects.includes(trimmed)) {
      aspects = [...aspects, trimmed];
      customAspectInput = '';
    }
  }

  function removeAspect(aspectToRemove: string) {
    aspects = aspects.filter((a) => a !== aspectToRemove);
  }

  async function handleCreateInvestigation() {
    if (!newQuery.trim()) {
      errorMessage = 'Please specify a research query or topic to investigate.';
      return;
    }
    if (selectedDocIds.length === 0) {
      errorMessage = 'Please select at least one indexed document.';
      return;
    }

    isCreating = true;
    errorMessage = null;

    try {
      const payload: CreateInvestigationRequest = {
        title: newTitle.trim() || undefined,
        query: newQuery.trim(),
        document_ids: selectedDocIds,
        comparison_aspects: aspects.length > 0 ? aspects : undefined,
        run_immediately: true
      };

      const result = await api.createInvestigation(payload);
      showCreateModal = false;
      // Navigate straight to the investigation workbench
      goto(`/investigations/${result.investigation_id}`);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
    } finally {
      isCreating = false;
    }
  }

  async function handleDelete(id: string, e: MouseEvent) {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this investigation and its evidence report?')) {
      return;
    }
    try {
      await api.deleteInvestigation(id);
      investigations = investigations.filter((inv) => inv.investigation_id !== id);
    } catch (err: unknown) {
      alert(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  onMount(() => {
    loadData();
  });
</script>

<svelte:head>
  <title>Cross-Document Investigations | STING Evidence Intelligence Workbench</title>
</svelte:head>

<div class="investigations-hub">
  <header class="hub-header">
    <div class="header-left">
      <div class="badge-line">
        <span class="hub-pill">STING Signature Capability</span>
      </div>
      <h1 class="hub-title">Evidence Intelligence Workbench</h1>
      <p class="hub-subtitle">
        Deep cross-document analysis: extract verified empirical findings, identify agreements and
        contradictions, map evidence gaps, and inspect verbatim chunk citations across sources.
      </p>
    </div>
    <div class="header-actions">
      <button class="btn btn-primary" onclick={() => (showCreateModal = true)}>
        <span class="icon">+</span>
        New Investigation
      </button>
    </div>
  </header>

  {#if errorMessage}
    <div class="alert alert-error" role="alert">
      <span>{errorMessage}</span>
      <button class="btn-close" onclick={() => (errorMessage = null)}>✕</button>
    </div>
  {/if}

  {#if showCreateModal}
    <div
      class="modal-backdrop"
      onclick={() => !isCreating && (showCreateModal = false)}
      onkeydown={(e) => e.key === 'Escape' && !isCreating && (showCreateModal = false)}
      role="presentation"
    >
      <div
        class="modal-card"
        onclick={(e) => e.stopPropagation()}
        onkeydown={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        tabindex="-1"
      >
        <div class="modal-header">
          <div>
            <h2 class="modal-title">New Cross-Document Investigation</h2>
            <p class="modal-desc">
              Select documents and specify comparison aspects. Findings are strictly grounded in deterministic chunk citations.
            </p>
          </div>
          <button class="modal-close" onclick={() => (showCreateModal = false)} disabled={isCreating}>✕</button>
        </div>

        <form onsubmit={(e) => { e.preventDefault(); handleCreateInvestigation(); }} class="modal-form">
          <div class="form-group">
            <label for="inv-query" class="form-label">
              Research Query / Topic <span class="required">*</span>
            </label>
            <input
              id="inv-query"
              type="text"
              class="form-input"
              placeholder="e.g. Compare throughput, memory consumption, and hardware scaling across attention implementations"
              bind:value={newQuery}
              required
              disabled={isCreating}
            />
          </div>

          <div class="form-group">
            <label for="inv-title" class="form-label">
              Investigation Title <span class="optional">(Optional)</span>
            </label>
            <input
              id="inv-title"
              type="text"
              class="form-input"
              placeholder="e.g. FlashAttention vs Standard Transformer Benchmark Analysis"
              bind:value={newTitle}
              disabled={isCreating}
            />
          </div>

          <div class="form-group">
            <div class="doc-selection-header">
              <span class="form-label">
                Select Source Documents ({selectedDocIds.length} of {availableDocs.length}) <span class="required">*</span>
              </span>
              {#if availableDocs.length > 0}
                <button type="button" class="btn-text-link" onclick={selectAllDocs} disabled={isCreating}>
                  {selectedDocIds.length === availableDocs.length ? 'Deselect All' : 'Select All'}
                </button>
              {/if}
            </div>

            {#if availableDocs.length === 0}
              <div class="empty-docs-warning">
                No indexed documents available in STING. Please upload and index documents first.
              </div>
            {:else}
              <div class="docs-picker-list">
                {#each availableDocs as doc (doc.doc_id)}
                  <label class="doc-picker-item {selectedDocIds.includes(doc.doc_id) ? 'selected' : ''}">
                    <input
                      type="checkbox"
                      checked={selectedDocIds.includes(doc.doc_id)}
                      onchange={() => toggleDoc(doc.doc_id)}
                      disabled={isCreating}
                    />
                    <div class="doc-picker-info">
                      <span class="doc-picker-name">{doc.filename}</span>
                      <span class="doc-picker-meta">
                        {doc.chunk_count} chunks • {doc.file_type.toUpperCase()} • {(doc.file_size_bytes / 1024).toFixed(0)} KB
                      </span>
                    </div>
                  </label>
                {/each}
              </div>
            {/if}
          </div>

          <div class="form-group">
            <span class="form-label">Comparison Dimensions / Aspects</span>
            <p class="form-sublabel">STING will extract structured claims and synthesize consensus/contradictions for each aspect:</p>

            <div class="aspects-tag-cloud">
              {#each aspects as aspect (aspect)}
                <span class="aspect-chip">
                  {aspect}
                  {#if !isCreating}
                    <button type="button" class="chip-remove" onclick={() => removeAspect(aspect)}>✕</button>
                  {/if}
                </span>
              {/each}
            </div>

            <div class="aspect-input-row">
              <input
                type="text"
                class="form-input form-input-sm"
                placeholder="Add custom aspect (e.g. Hardware Requirements)..."
                bind:value={customAspectInput}
                onkeydown={(e) => e.key === 'Enter' && (e.preventDefault(), addAspect())}
                disabled={isCreating}
              />
              <button type="button" class="btn btn-secondary btn-sm" onclick={addAspect} disabled={isCreating}>
                Add Aspect
              </button>
            </div>
          </div>

          <div class="modal-footer">
            <button
              type="button"
              class="btn btn-secondary"
              onclick={() => (showCreateModal = false)}
              disabled={isCreating}
            >
              Cancel
            </button>
            <button
              type="submit"
              class="btn btn-primary"
              disabled={isCreating || availableDocs.length === 0 || !newQuery.trim() || selectedDocIds.length === 0}
            >
              {#if isCreating}
                <span class="spinner-sm"></span> Extracting & Synthesizing Evidence...
              {:else}
                Launch Investigation
              {/if}
            </button>
          </div>
        </form>
      </div>
    </div>
  {/if}

  <section class="investigations-list-section">
    <div class="section-header">
      <h2 class="section-title">Past Investigations</h2>
      <span class="count-badge">{investigations.length}</span>
    </div>

    {#if isLoading}
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Loading investigations...</p>
      </div>
    {:else if investigations.length === 0}
      <div class="empty-hub-card">
        <div class="empty-icon">🔬</div>
        <h3>No Investigations Conducted Yet</h3>
        <p>
          STING evolves document search into an Evidence Intelligence Workbench. Compare findings across multiple papers or reports with full passage traceability.
        </p>
        <button class="btn btn-primary" onclick={() => (showCreateModal = true)}>
          Start Your First Investigation
        </button>
      </div>
    {:else}
      <div class="investigations-grid">
        {#each investigations as inv (inv.investigation_id)}
          <div
            class="inv-card"
            onclick={() => goto(`/investigations/${inv.investigation_id}`)}
            role="link"
            tabindex="0"
            onkeydown={(e) => e.key === 'Enter' && goto(`/investigations/${inv.investigation_id}`)}
          >
            <div class="inv-card-top">
              <div class="inv-card-badges">
                <StatusBadge status={inv.status} size="sm" />
                <span class="inv-id">`{inv.investigation_id}`</span>
              </div>
              <button
                class="btn-icon-danger"
                title="Delete investigation"
                onclick={(e) => handleDelete(inv.investigation_id, e)}
              >
                🗑
              </button>
            </div>

            <h3 class="inv-card-title">{inv.title}</h3>
            <p class="inv-card-query">"{inv.query}"</p>

            <div class="inv-card-stats">
              <div class="stat-pill">
                <span class="stat-val">{inv.document_count}</span>
                <span class="stat-lbl">Documents</span>
              </div>
              <div class="stat-pill">
                <span class="stat-val">{inv.aspect_count}</span>
                <span class="stat-lbl">Aspects</span>
              </div>
              <div class="stat-pill">
                <span class="stat-val">{inv.total_findings}</span>
                <span class="stat-lbl">Findings</span>
              </div>
            </div>

            {#if inv.relationship_counts}
              <div class="relationship-mini-bar">
                {#if inv.relationship_counts.supports}
                  <span class="badge-supports" title="Supporting findings">
                    ✓ {inv.relationship_counts.supports}
                  </span>
                {/if}
                {#if inv.relationship_counts.contradicts}
                  <span class="badge-contradicts" title="Contradictions / Disagreements">
                    ✕ {inv.relationship_counts.contradicts}
                  </span>
                {/if}
                {#if inv.relationship_counts.qualifies}
                  <span class="badge-qualifies" title="Qualified / Conditioned">
                    ~ {inv.relationship_counts.qualifies}
                  </span>
                {/if}
              </div>
            {/if}

            <div class="inv-card-footer">
              <span class="inv-date">Created {new Date(inv.created_at).toLocaleDateString()}</span>
              <span class="inv-link-action">Inspect Matrix →</span>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </section>
</div>

<style>
  .investigations-hub {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
  }

  .hub-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 2rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
  }

  .badge-line {
    margin-bottom: 0.5rem;
  }

  .hub-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
  }

  .hub-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
  }

  .hub-subtitle {
    font-size: 0.95rem;
    line-height: 1.5;
    color: var(--text-secondary);
    max-width: 780px;
    margin: 0;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.55rem 1.15rem;
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
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
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

  .btn-sm {
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
  }

  .btn-text-link {
    background: none;
    border: none;
    color: #38bdf8;
    font-size: 0.8rem;
    cursor: pointer;
    padding: 0;
  }
  .btn-text-link:hover {
    text-decoration: underline;
  }

  .btn-icon-danger {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1rem;
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 4px;
  }
  .btn-icon-danger:hover {
    color: #f43f5e;
    background: rgba(244, 63, 94, 0.1);
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }

  .section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: var(--text-primary);
  }

  .count-badge {
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .investigations-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.25rem;
  }

  .inv-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    cursor: pointer;
    transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
  }
  .inv-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }

  .inv-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .inv-card-badges {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .inv-id {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: monospace;
  }

  .inv-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 0.4rem;
    color: var(--text-primary);
  }

  .inv-card-query {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-style: italic;
    margin: 0 0 1rem;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .inv-card-stats {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }

  .stat-pill {
    flex: 1;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.5rem;
    text-align: center;
  }

  .stat-val {
    display: block;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
  }

  .stat-lbl {
    display: block;
    font-size: 0.68rem;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.05em;
  }

  .relationship-mini-bar {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }

  .badge-supports {
    font-size: 0.72rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.25);
  }

  .badge-contradicts {
    font-size: 0.72rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(244, 63, 94, 0.12);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.25);
  }

  .badge-qualifies {
    font-size: 0.72rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.25);
  }

  .inv-card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid var(--border-subtle);
    padding-top: 0.75rem;
    margin-top: auto;
    font-size: 0.78rem;
  }

  .inv-date {
    color: var(--text-muted);
  }

  .inv-link-action {
    color: #38bdf8;
    font-weight: 500;
  }

  /* Modal */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 1.5rem;
  }

  .modal-card {
    background: #0f172a;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    width: 100%;
    max-width: 680px;
    max-height: 90vh;
    overflow-y: auto;
    padding: 1.75rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
  }

  .modal-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.35rem;
  }

  .modal-desc {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.4;
  }

  .modal-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1.25rem;
    cursor: pointer;
    padding: 0.25rem;
  }
  .modal-close:hover {
    color: var(--text-primary);
  }

  .form-group {
    margin-bottom: 1.25rem;
  }

  .form-label {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
  }

  .form-sublabel {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0 0 0.5rem;
  }

  .required {
    color: #f43f5e;
  }
  .optional {
    font-weight: normal;
    color: var(--text-muted);
    font-size: 0.78rem;
  }

  .form-input {
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    padding: 0.55rem 0.8rem;
    font-size: 0.875rem;
    box-sizing: border-box;
  }
  .form-input:focus {
    outline: none;
    border-color: #38bdf8;
  }

  .doc-selection-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
  }

  .docs-picker-list {
    max-height: 180px;
    overflow-y: auto;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    background: rgba(0, 0, 0, 0.2);
    padding: 0.4rem;
  }

  .doc-picker-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.45rem 0.6rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.1s;
  }
  .doc-picker-item:hover {
    background: var(--bg-hover);
  }
  .doc-picker-item.selected {
    background: rgba(56, 189, 248, 0.08);
  }

  .doc-picker-info {
    display: flex;
    flex-direction: column;
  }

  .doc-picker-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-primary);
  }

  .doc-picker-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
  }

  .empty-docs-warning {
    padding: 1rem;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    color: #fbbf24;
  }

  .aspects-tag-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
  }

  .aspect-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 9999px;
    padding: 0.25rem 0.6rem;
    font-size: 0.78rem;
    color: var(--text-primary);
  }

  .chip-remove {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 0.75rem;
    cursor: pointer;
    padding: 0;
  }
  .chip-remove:hover {
    color: #f43f5e;
  }

  .aspect-input-row {
    display: flex;
    gap: 0.5rem;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.75rem;
    margin-top: 1.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border-subtle);
  }

  .empty-hub-card {
    text-align: center;
    background: var(--bg-surface);
    border: 1px dashed var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 4rem 2rem;
    max-width: 600px;
    margin: 2rem auto;
  }

  .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .empty-hub-card h3 {
    font-size: 1.25rem;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
  }

  .empty-hub-card p {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0 0 1.5rem;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4rem;
    color: var(--text-muted);
    gap: 1rem;
  }

  .spinner {
    width: 28px;
    height: 28px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
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
    font-size: 0.875rem;
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
