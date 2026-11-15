<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api } from '#lib/api';
  import type { DocumentDetailResponse, DocumentSummary } from '#lib/types';

  interface Props {
    docId: string;
    initialFilename?: string;
    initialFileSize?: number;
    onReady?: (doc: DocumentDetailResponse) => void;
    onCancel?: () => void;
    onError?: (error: string) => void;
  }

  let { docId, initialFilename = '', initialFileSize = 0, onReady, onCancel, onError }: Props = $props();

  type ProcessingStage = 'upload' | 'extraction' | 'chunking' | 'indexing' | 'ready';

  const STAGES: { key: ProcessingStage; label: string; desc: string }[] = [
    { key: 'upload', label: 'UPLOAD', desc: 'File received & validated' },
    { key: 'extraction', label: 'EXTRACTION', desc: 'Parsing structure & text' },
    { key: 'chunking', label: 'CHUNKING', desc: 'Citation-aware chunk boundaries' },
    { key: 'indexing', label: 'INDEXING', desc: 'Dense vectors & BM25 generation' },
    { key: 'ready', label: 'READY', desc: 'Index verified & ready' }
  ];

  let currentDoc = $state<DocumentDetailResponse | null>(null);
  let status = $state<string>('UPLOADED');
  let errorMessage = $state<string | null>(null);
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let isCompleted = $state<boolean>(false);

  // Derived current stage index (0 to 4)
  let currentStageIndex = $derived.by(() => {
    const s = (status || '').toUpperCase();
    if (s === 'READY') return 4;
    if (s === 'INDEXING') return 3;
    if (s === 'CHUNKING') return 2;
    if (s === 'EXTRACTING') return 1;
    return 0; // UPLOADED
  });

  // Honest progress percentage based on verified stage transitions
  let progressPercent = $derived.by(() => {
    switch (currentStageIndex) {
      case 0:
        return 20;
      case 1:
        return 42;
      case 2:
        return 68;
      case 3:
        return 88;
      case 4:
        return 100;
      default:
        return 20;
    }
  });

  function formatBytes(bytes: number): string {
    if (!bytes || bytes <= 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  }

  async function checkDocumentStatus() {
    try {
      const doc = await api.getDocument(docId);
      currentDoc = doc;
      status = doc.status;

      const upperStatus = doc.status.toUpperCase();
      if (upperStatus === 'READY') {
        if (pollTimer) clearInterval(pollTimer);
        isCompleted = true;
        if (onReady) {
          onReady(doc);
        }
      } else if (upperStatus === 'FAILED') {
        if (pollTimer) clearInterval(pollTimer);
        errorMessage = doc.error_message || 'Document processing failed during ingestion.';
        if (onError) onError(errorMessage);
      }
    } catch (err: unknown) {
      // Temporary network jitter, continue polling
    }
  }

  onMount(() => {
    checkDocumentStatus();
    pollTimer = setInterval(checkDocumentStatus, 1000);
  });

  onDestroy(() => {
    if (pollTimer) clearInterval(pollTimer);
  });
</script>

<div class="processor-card">
  <div class="processor-header">
    <div class="doc-icon-wrap">
      <svg class="file-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
        <polyline points="10 9 9 9 8 9"></polyline>
      </svg>
    </div>

    <div class="doc-title-info">
      <h3 class="doc-name">{currentDoc?.filename || initialFilename || 'Processing Document'}</h3>
      <div class="doc-sub">
        <span>{formatBytes(currentDoc?.file_size || initialFileSize)}</span>
        <span class="sep">•</span>
        <span class="status-text" class:status-failed={status === 'FAILED'} class:status-ready={isCompleted}>
          {#if errorMessage}
            Failed
          {:else if isCompleted}
            Ready ({currentDoc?.chunk_count ?? 0} chunks indexed)
          {:else}
            Stage {currentStageIndex + 1} of 5: {STAGES[currentStageIndex]?.label}
          {/if}
        </span>
      </div>
    </div>

    {#if !isCompleted && !errorMessage && onCancel}
      <button class="btn-cancel" onclick={onCancel} title="Dismiss progress card">✕</button>
    {/if}
  </div>

  {#if errorMessage}
    <div class="error-banner">
      <span class="error-icon">⚠️</span>
      <div class="error-body">
        <strong>Ingestion Failed</strong>
        <p>{errorMessage}</p>
      </div>
      {#if onCancel}
        <button class="btn-sm btn-secondary" onclick={onCancel}>Dismiss</button>
      {/if}
    </div>
  {:else}
    <!-- Stage Progression Bar -->
    <div class="stages-track">
      <div class="progress-bar-glow" style="width: {progressPercent}%"></div>

      <div class="stages-row">
        {#each STAGES as stage, idx}
          {@const isPassed = currentStageIndex > idx || isCompleted}
          {@const isCurrent = currentStageIndex === idx && !isCompleted}
          <div
            class="stage-node-wrap"
            class:node-passed={isPassed}
            class:node-active={isCurrent}
            class:node-pending={currentStageIndex < idx && !isCompleted}
          >
            <div class="node-circle">
              {#if isPassed}
                <svg class="check-icon" viewBox="0 0 16 16" fill="currentColor">
                  <path fill-rule="evenodd" d="M12.416 3.376a.75.75 0 0 1 .208 1.04l-5 7.5a.75.75 0 0 1-1.154.114l-3-3a.75.75 0 0 1 1.06-1.06l2.353 2.353 4.493-6.74a.75.75 0 0 1 1.04-.207Z" clip-rule="evenodd" />
                </svg>
              {:else if isCurrent}
                <span class="pulse-dot"></span>
              {:else}
                <span class="step-num">{idx + 1}</span>
              {/if}
            </div>
            <div class="stage-info">
              <span class="stage-name">{stage.label}</span>
              <span class="stage-sub">{stage.desc}</span>
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Active Stage Diagnostics Banner -->
    <div class="stage-diagnostic-box">
      {#if isCompleted}
        <div class="diag-ready">
          <span class="ready-badge">✓ READY</span>
          <span>Corpus ingestion complete. Dense vectors and lexical BM25 index verified.</span>
        </div>
        {#if onReady && currentDoc}
          <button class="btn-open-workspace" onclick={() => onReady(currentDoc!)}>
            Open Document Workspace &rarr;
          </button>
        {/if}
      {:else}
        <div class="diag-live">
          <span class="diag-spinner"></span>
          <span>
            {#if currentStageIndex === 0}
              Saving stream and computing cryptographic content digest...
            {:else if currentStageIndex === 1}
              Extracting textual contents and page geometries...
            {:else if currentStageIndex === 2}
              Generating citation-anchored chunk spans...
            {:else if currentStageIndex === 3}
              Embedding vectors with FastEmbed &amp; building BM25 index...
            {/if}
          </span>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .processor-card {
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: var(--radius-lg, 10px);
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(14, 165, 233, 0.08);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
  }

  .processor-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.75rem;
  }

  .doc-icon-wrap {
    width: 44px;
    height: 44px;
    border-radius: 8px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #38bdf8;
    flex-shrink: 0;
  }

  .file-svg {
    width: 22px;
    height: 22px;
  }

  .doc-title-info {
    flex: 1;
    min-width: 0;
  }

  .doc-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f3f4f6;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .doc-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.2rem;
  }

  .sep {
    color: #4b5563;
  }

  .status-text {
    color: #38bdf8;
    font-weight: 500;
  }

  .status-ready {
    color: #34d399;
  }

  .status-failed {
    color: #f87171;
  }

  .btn-cancel {
    background: transparent;
    border: none;
    color: #9ca3af;
    font-size: 1.1rem;
    cursor: pointer;
    padding: 0.4rem;
    border-radius: 4px;
    transition: color 0.15s;
  }

  .btn-cancel:hover {
    color: #ffffff;
  }

  /* Track and Progression */
  .stages-track {
    position: relative;
    padding: 1rem 0 1.25rem;
  }

  .progress-bar-glow {
    position: absolute;
    top: 24px;
    left: 0;
    height: 3px;
    background: linear-gradient(90deg, #0284c7, #38bdf8);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
    border-radius: 2px;
    transition: width 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    z-index: 1;
  }

  .stages-track::before {
    content: '';
    position: absolute;
    top: 24px;
    left: 0;
    right: 0;
    height: 3px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 2px;
    z-index: 0;
  }

  .stages-row {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
  }

  .stage-node-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    flex: 1;
    text-align: center;
  }

  .node-circle {
    width: 28px;
    height: 28px;
    border-radius: 9999px;
    background: #111827;
    border: 2px solid rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #9ca3af;
    transition: all 0.25s ease;
  }

  .node-passed .node-circle {
    background: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
  }

  .node-active .node-circle {
    background: #0f172a;
    border-color: #38bdf8;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.7);
  }

  .pulse-dot {
    width: 10px;
    height: 10px;
    border-radius: 9999px;
    background: #38bdf8;
    animation: radar-pulse 1.4s ease-out infinite;
  }

  @keyframes radar-pulse {
    0% {
      transform: scale(0.8);
      box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
    }
    70% {
      transform: scale(1.1);
      box-shadow: 0 0 0 8px rgba(56, 189, 248, 0);
    }
    100% {
      transform: scale(0.8);
      box-shadow: 0 0 0 0 rgba(56, 189, 248, 0);
    }
  }

  .check-icon {
    width: 14px;
    height: 14px;
  }

  .step-num {
    font-size: 0.7rem;
  }

  .stage-info {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }

  .stage-name {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #94a3b8;
  }

  .node-active .stage-name {
    color: #38bdf8;
  }

  .node-passed .stage-name {
    color: #f3f4f6;
  }

  .stage-sub {
    font-size: 0.68rem;
    color: #64748b;
    max-width: 110px;
    line-height: 1.25;
  }

  .stage-diagnostic-box {
    margin-top: 1.5rem;
    padding: 0.85rem 1.15rem;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-md, 6px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    font-size: 0.85rem;
    color: #94a3b8;
  }

  .diag-live {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    color: #e0f2fe;
  }

  .diag-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(56, 189, 248, 0.25);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .diag-ready {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    color: #d1fae5;
  }

  .ready-badge {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
  }

  .btn-open-workspace {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 1rem;
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 600;
    border: none;
    border-radius: var(--radius-md, 6px);
    cursor: pointer;
    box-shadow: 0 0 16px rgba(14, 165, 233, 0.35);
    transition: all 0.15s ease;
    flex-shrink: 0;
  }

  .btn-open-workspace:hover {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.5);
    transform: translateY(-1px);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: var(--radius-md, 6px);
  }

  .error-icon {
    font-size: 1.5rem;
  }

  .error-body {
    flex: 1;
    font-size: 0.85rem;
  }

  .error-body strong {
    color: #fca5a5;
    display: block;
    margin-bottom: 0.2rem;
  }

  .error-body p {
    color: #fecaca;
    margin: 0;
  }
</style>

