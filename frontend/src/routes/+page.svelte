<script lang="ts">
  import { onMount } from 'svelte';
  import { api, ApiError } from '#lib/api';
  import { devMode } from '#lib/stores/devMode';
  import DocumentProcessor from '#lib/components/DocumentProcessor.svelte';
  import DocumentChat from '#lib/components/DocumentChat.svelte';
  import type { DocumentSummary, DocumentDetailResponse } from '#lib/types';

  // Workspace View State
  type ViewState = 'landing' | 'chat';
  let viewState = $state<ViewState>('landing');

  // Documents State
  let documents = $state<DocumentSummary[]>([]);
  let totalDocs = $state<number>(0);
  let isLoading = $state<boolean>(true);
  let errorMessage = $state<string | null>(null);

  // Selected Document for Chat Workspace
  let selectedDoc = $state<DocumentSummary | DocumentDetailResponse | null>(null);

  // Active Upload & Processing State
  let isDragging = $state<boolean>(false);
  let isUploading = $state<boolean>(false);
  let activeProcessingDocId = $state<string | null>(null);
  let activeProcessingFilename = $state<string>('');
  let activeProcessingFileSize = $state<number>(0);
  let fileInputRef = $state<HTMLInputElement | null>(null);

  async function loadDocuments() {
    isLoading = true;
    errorMessage = null;
    try {
      const res = await api.listDocuments({ limit: 100 });
      const docs = res.documents ?? res.items ?? [];
      documents = docs;
      totalDocs = res.total ?? docs.length;
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        errorMessage = err.detail;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      } else {
        errorMessage = String(err);
      }
      documents = [];
      totalDocs = 0;
    } finally {
      isLoading = false;
    }
  }

  async function handleFileUpload(file: File) {
    if (!file) return;

    // Client-side format pre-check based on actual backend support (.pdf, .txt, .md, .markdown)
    const validExtensions = ['.pdf', '.txt', '.md', '.markdown'];
    const lowerName = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => lowerName.endsWith(ext));
    if (!isValid) {
      errorMessage = `Unsupported file format '${file.name}'. STING supports PDF (.pdf), Markdown (.md), and Plain Text (.txt) documents.`;
      return;
    }

    // 50MB limit check
    if (file.size > 50 * 1024 * 1024) {
      errorMessage = `File size exceeds the 50 MB upload limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`;
      return;
    }

    errorMessage = null;
    isUploading = true;
    activeProcessingFilename = file.name;
    activeProcessingFileSize = file.size;

    try {
      const uploadRes = await api.uploadDocument(file);
      activeProcessingDocId = uploadRes.doc_id;
      // Refresh documents list
      await loadDocuments();
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
      activeProcessingDocId = null;
    } finally {
      isUploading = false;
    }
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    isDragging = false;
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleFileUpload(file);
    }
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    isDragging = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    isDragging = false;
  }

  function handleFileInputChange(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      handleFileUpload(file);
      input.value = ''; // Reset input
    }
  }

  function openDocumentWorkspace(doc: DocumentSummary | DocumentDetailResponse) {
    selectedDoc = doc;
    viewState = 'chat';
  }

  function handleBackToDocuments() {
    selectedDoc = null;
    viewState = 'landing';
    loadDocuments();
  }

  async function handleDeleteDocument(docId: string, filename: string) {
    if (!confirm(`Are you sure you want to delete '${filename}'? This will remove its chunks and indexes.`)) {
      return;
    }
    try {
      await api.deleteDocument(docId);
      if (selectedDoc && selectedDoc.doc_id === docId) {
        selectedDoc = null;
        viewState = 'landing';
      }
      await loadDocuments();
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err);
    }
  }

  function formatBytes(bytes?: number): string {
    if (!bytes || bytes <= 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  }

  function formatDate(iso: string): string {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch {
      return iso;
    }
  }

  onMount(() => {
    loadDocuments();
  });
</script>

<svelte:head>
  <title>STING — Your documents. Understood.</title>
  <meta
    name="description"
    content="Document intelligence with verified citations and evidence-first RAG reliability."
  />
</svelte:head>

<div class="user-workspace-root">
  <!-- Developer Mode Ribbon (Only visible when developer mode is toggled ON) -->
  {#if $devMode}
    <div class="dev-mode-banner">
      <div class="dmb-left">
        <span class="dmb-badge">DEVELOPER WORKBENCH ACTIVE</span>
        <span class="dmb-text">
          Diagnostic tools are unlocked in navigation: Evaluation Studio, Investigations, Run Explorer, and Citation Audit.
        </span>
      </div>
      <div class="dmb-actions">
        <a href="/evaluation" class="btn-sm btn-ghost dmb-link">Evaluation Studio &rarr;</a>
        <a href="/runs" class="btn-sm btn-ghost dmb-link">Run Explorer &rarr;</a>
      </div>
    </div>
  {/if}

  {#if viewState === 'chat' && selectedDoc}
    <!-- Document Chat Workspace View -->
    <DocumentChat document={selectedDoc} onBack={handleBackToDocuments} />
  {:else}
    <!-- Document Hub / Landing View -->
    <div class="landing-container">
      <!-- Hero Header -->
      <section class="hero-section">
        <div class="hero-badge">
          <span class="badge-dot"></span>
          <span>Evidence-First Document Intelligence</span>
        </div>
        <h1 class="hero-headline">Your documents. Understood.</h1>
        <p class="hero-description">
          Upload contracts, technical specifications, research papers, or manuals. STING extracts
          knowledge, builds high-precision retrieval indexes, and grounds every answer with verified
          source citations.
        </p>
      </section>

      <!-- Active Processing Card -->
      {#if activeProcessingDocId}
        <section class="active-processing-section">
          <DocumentProcessor
            docId={activeProcessingDocId}
            initialFilename={activeProcessingFilename}
            initialFileSize={activeProcessingFileSize}
            onReady={(doc) => {
              loadDocuments();
              openDocumentWorkspace(doc);
              activeProcessingDocId = null;
            }}
            onCancel={() => {
              activeProcessingDocId = null;
              loadDocuments();
            }}
          />
        </section>
      {/if}

      <!-- Upload Zone -->
      <section class="upload-section">
        <div
          class="dropzone-card"
          class:is-active-drag={isDragging}
          class:is-uploading-card={isUploading}
          ondrop={handleDrop}
          ondragover={handleDragOver}
          ondragleave={handleDragLeave}
          role="region"
          aria-label="Document upload dropzone"
        >
          <div class="dropzone-glow"></div>
          <div class="dropzone-content">
            <div class="upload-icon-circle">
              {#if isUploading}
                <span class="upload-spinner"></span>
              {:else}
                <svg
                  class="upload-arrow-svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
              {/if}
            </div>

            <div class="drop-text-group">
              <h3 class="drop-title">
                {isUploading ? 'Uploading document...' : 'Drag and drop your document here'}
              </h3>
              <p class="drop-subtitle">or choose a file from your computer</p>
            </div>

            <div class="upload-action-row">
              <input
                type="file"
                bind:this={fileInputRef}
                class="hidden-file-input"
                accept=".pdf,.txt,.md,.markdown"
                aria-label="Upload document file"
                onchange={handleFileInputChange}
              />
              <button
                type="button"
                class="btn-select-file"
                onclick={() => fileInputRef?.click()}
                disabled={isUploading}
              >
                Choose Document
              </button>
            </div>

            <div class="format-support-note">
              <span>Supported formats: <strong>PDF</strong>, <strong>Markdown (.md)</strong>, <strong>Plain Text (.txt)</strong></span>
              <span class="dot-sep">•</span>
              <span>Max file size: <strong>50 MB</strong></span>
            </div>
          </div>
        </div>
      </section>

      {#if errorMessage}
        <div class="error-banner">
          <span class="error-icon">⚠️</span>
          <span>{errorMessage}</span>
          <button class="btn-close-error" onclick={() => (errorMessage = null)} aria-label="Dismiss error message">✕</button>
        </div>
      {/if}

      <!-- Documents Library Section -->
      <section class="library-section">
        <div class="library-header">
          <div class="lh-left">
            <h2 class="library-title">Recently Uploaded Documents</h2>
            <span class="library-count-pill">{totalDocs} {totalDocs === 1 ? 'document' : 'documents'}</span>
          </div>
          <button class="btn-refresh" onclick={loadDocuments} disabled={isLoading}>
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        {#if isLoading && documents.length === 0}
          <div class="loading-state-card">
            <div class="loading-spinner"></div>
            <p>Loading document collection...</p>
          </div>
        {:else if documents.length === 0}
          <div class="empty-docs-card">
            <div class="empty-icon-wrap">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="12" y1="18" x2="12" y2="12"></line>
                <line x1="9" y1="15" x2="15" y2="15"></line>
              </svg>
            </div>
            <h3 class="empty-docs-title">No documents uploaded yet</h3>
            <p class="empty-docs-desc">
              Drop a PDF, Markdown, or text document into the upload zone above to begin asking questions with verified citations.
            </p>
          </div>
        {:else}
          <div class="documents-grid">
            {#each documents as doc (doc.doc_id)}
              {@const isReady = (doc.status || '').toUpperCase() === 'READY'}
              <div class="doc-card" class:is-ready-card={isReady}>
                <div class="doc-card-top">
                  <div class="doc-format-badge format-{doc.file_type.toLowerCase()}">
                    {doc.file_type.toUpperCase()}
                  </div>
                  <div class="doc-actions-top">
                    {#if isReady}
                      <span class="status-pill status-ready">READY</span>
                    {:else if (doc.status || '').toUpperCase() === 'FAILED'}
                      <span class="status-pill status-failed">FAILED</span>
                    {:else}
                      <span class="status-pill status-processing">{doc.status.toUpperCase()}</span>
                    {/if}
                  </div>
                </div>

                <div class="doc-card-main">
                  <h3 class="card-doc-filename" title={doc.filename}>{doc.filename}</h3>
                  <div class="card-meta-row">
                    <span>{formatBytes(doc.file_size || doc.file_size_bytes)}</span>
                    <span class="meta-dot">•</span>
                    <span>{doc.chunk_count} {doc.chunk_count === 1 ? 'chunk' : 'chunks'}</span>
                    {#if doc.page_count}
                      <span class="meta-dot">•</span>
                      <span>{doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'}</span>
                    {/if}
                  </div>
                  <div class="card-date-row">
                    <span>Added {formatDate(doc.created_at)}</span>
                  </div>
                </div>

                <div class="doc-card-footer">
                  {#if isReady}
                    <button
                      type="button"
                      class="btn-open-chat"
                      onclick={() => openDocumentWorkspace(doc)}
                    >
                      <span>Chat with Document</span>
                      <span class="btn-arrow">&rarr;</span>
                    </button>
                  {:else}
                    <button
                      type="button"
                      class="btn-open-chat btn-disabled"
                      disabled
                    >
                      <span>Processing...</span>
                    </button>
                  {/if}

                  <button
                    type="button"
                    class="btn-delete-doc"
                    onclick={() => handleDeleteDocument(doc.doc_id, doc.filename)}
                    aria-label="Delete document {doc.filename}"
                    title="Delete document and index"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </section>
    </div>
  {/if}
</div>

<style>
  .user-workspace-root {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  /* Dev Mode Banner */
  .dev-mode-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: var(--radius-md, 6px);
    font-size: 0.85rem;
    color: #e0f2fe;
    gap: 1rem;
  }

  .dmb-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .dmb-badge {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
    white-space: nowrap;
  }

  .dmb-text {
    color: #bae6fd;
  }

  .dmb-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .dmb-link {
    color: #38bdf8;
    text-decoration: none;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.1);
    transition: background 0.15s;
    white-space: nowrap;
  }

  .dmb-link:hover {
    background: rgba(56, 189, 248, 0.2);
    text-decoration: none;
  }

  /* Landing Container */
  .landing-container {
    max-width: 1100px;
    margin: 0 auto;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 3rem;
    padding: 1rem 0 3rem;
  }

  /* Hero Section */
  .hero-section {
    text-align: center;
    max-width: 760px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 1.5rem;
  }

  .badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 9999px;
    background: #38bdf8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.8);
  }

  .hero-headline {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 1rem;
    background: linear-gradient(180deg, #ffffff 30%, #cbd5e1 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .hero-description {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #94a3b8;
    max-width: 640px;
  }

  /* Active Processing Section */
  .active-processing-section {
    max-width: 820px;
    margin: 0 auto;
    width: 100%;
  }

  /* Upload Section */
  .upload-section {
    max-width: 820px;
    margin: 0 auto;
    width: 100%;
  }

  .dropzone-card {
    position: relative;
    background: rgba(17, 24, 39, 0.8);
    border: 2px dashed rgba(255, 255, 255, 0.15);
    border-radius: var(--radius-lg, 10px);
    padding: 3rem 2rem;
    text-align: center;
    cursor: default;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    overflow: hidden;
  }

  .dropzone-card:hover,
  .dropzone-card.is-active-drag {
    border-color: #38bdf8;
    background: rgba(14, 165, 233, 0.05);
    box-shadow: 0 0 32px rgba(14, 165, 233, 0.15);
  }

  .dropzone-glow {
    position: absolute;
    top: -50%;
    left: 20%;
    right: 20%;
    height: 100%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, transparent 70%);
    pointer-events: none;
  }

  .dropzone-content {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.25rem;
  }

  .upload-icon-circle {
    width: 58px;
    height: 58px;
    border-radius: 9999px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #38bdf8;
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.2);
  }

  .upload-arrow-svg {
    width: 26px;
    height: 26px;
  }

  .upload-spinner {
    width: 24px;
    height: 24px;
    border: 3px solid rgba(56, 189, 248, 0.25);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .drop-text-group {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .drop-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
  }

  .drop-subtitle {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
  }

  .upload-action-row {
    margin-top: 0.5rem;
  }

  .hidden-file-input {
    display: none;
  }

  .btn-select-file {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem 1.4rem;
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: 600;
    border: none;
    border-radius: var(--radius-md, 6px);
    cursor: pointer;
    box-shadow: 0 0 16px rgba(14, 165, 233, 0.35);
    transition: all 0.15s ease;
  }

  .btn-select-file:hover:not(:disabled) {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    box-shadow: 0 0 22px rgba(14, 165, 233, 0.5);
    transform: translateY(-1px);
  }

  .format-support-note {
    font-size: 0.78rem;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .format-support-note strong {
    color: #94a3b8;
  }

  .dot-sep {
    color: #475569;
  }

  /* Library Section */
  .library-section {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .library-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .lh-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .library-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
  }

  .library-count-pill {
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    padding: 0.15rem 0.55rem;
    border-radius: 9999px;
  }

  .btn-refresh {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #94a3b8;
    font-size: 0.8rem;
    padding: 0.35rem 0.75rem;
    border-radius: var(--radius-sm, 4px);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-refresh:hover:not(:disabled) {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.1);
  }

  /* Documents Grid */
  .documents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.25rem;
  }

  .doc-card {
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg, 10px);
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 1.25rem;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .doc-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    background: rgba(22, 27, 34, 0.95);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
  }

  .doc-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .doc-format-badge {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
  }

  .format-pdf {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .format-md,
  .format-markdown {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
  }

  .format-txt {
    background: rgba(148, 163, 184, 0.15);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.3);
  }

  .status-pill {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.12rem 0.45rem;
    border-radius: 9999px;
  }

  .status-ready {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  .status-failed {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .status-processing {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
  }

  .doc-card-main {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .card-doc-filename {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f3f4f6;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .card-meta-row {
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }

  .meta-dot {
    color: #475569;
  }

  .card-date-row {
    font-size: 0.72rem;
    color: #64748b;
  }

  .doc-card-footer {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
  }

  .btn-open-chat {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.85rem;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: var(--radius-md, 6px);
    color: #38bdf8;
    font-size: 0.825rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-open-chat:hover:not(:disabled) {
    background: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
  }

  .btn-disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-arrow {
    font-size: 1rem;
    transition: transform 0.15s;
  }

  .btn-open-chat:hover .btn-arrow {
    transform: translateX(2px);
  }

  .btn-delete-doc {
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-md, 6px);
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }

  .btn-delete-doc svg {
    width: 15px;
    height: 15px;
  }

  .btn-delete-doc:hover {
    color: #f87171;
    border-color: rgba(239, 68, 68, 0.35);
    background: rgba(239, 68, 68, 0.1);
  }

  /* Empty and Loading States */
  .empty-docs-card {
    text-align: center;
    padding: 3.5rem 2rem;
    background: rgba(17, 24, 39, 0.5);
    border: 1px dashed rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-lg, 10px);
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .empty-icon-wrap {
    width: 52px;
    height: 52px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #64748b;
    margin-bottom: 1rem;
  }

  .empty-icon-wrap svg {
    width: 24px;
    height: 24px;
  }

  .empty-docs-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.4rem;
  }

  .empty-docs-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    max-width: 440px;
    line-height: 1.5;
  }

  .loading-state-card {
    text-align: center;
    padding: 3rem;
    color: #94a3b8;
  }

  .loading-spinner {
    width: 28px;
    height: 28px;
    border: 3px solid rgba(56, 189, 248, 0.25);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 0.75rem;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.25rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: var(--radius-md, 6px);
    color: #fca5a5;
    font-size: 0.85rem;
  }

  .btn-close-error {
    margin-left: auto;
    background: none;
    border: none;
    color: #fca5a5;
    cursor: pointer;
  }

  @media (max-width: 768px) {
    .hero-headline {
      font-size: 2.25rem;
    }

    .documents-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
