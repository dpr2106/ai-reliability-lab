<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { api } from '#lib/api';
  import { devMode } from '#lib/stores/devMode';
  import type {
    DocumentSummary,
    DocumentDetailResponse,
    QueryResponse,
    ResolvedCitation
  } from '#lib/types';

  interface Props {
    document: DocumentSummary | DocumentDetailResponse;
    onBack: () => void;
  }

  let { document, onBack }: Props = $props();

  interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    response?: QueryResponse;
    selectedCitationMarker?: string | null;
  }

  let messages = $state<ChatMessage[]>([]);
  let inputQuery = $state<string>('');
  let isQuerying = $state<boolean>(false);
  let queryError = $state<string | null>(null);
  let showSourcesDrawer = $state<boolean>(false);
  let activeTabSources = $state<'citations' | 'trace'>('citations');

  let chatBottomRef: HTMLDivElement | null = null;
  let inputRef: HTMLTextAreaElement | null = null;

  let latestAssistantMsg = $derived(
    messages.filter((m) => m.role === 'assistant' && m.response).slice(-1)[0]
  );

  const starterQuestions = [
    'What are the primary conclusions and key takeaways?',
    'What architecture or system constraints are documented?',
    'What failures, incidents, or risks are identified?',
    'Summarize the core technical specifications.'
  ];

  async function scrollToBottom() {
    await tick();
    if (chatBottomRef) {
      chatBottomRef.scrollIntoView({ behavior: 'smooth' });
    }
  }

  async function handleSubmit(queryToRun?: string) {
    const text = (queryToRun || inputQuery).trim();
    if (!text || isQuerying) return;

    queryError = null;
    const userMsgId = `usr_${Date.now()}`;
    messages = [
      ...messages,
      {
        id: userMsgId,
        role: 'user',
        content: text,
        timestamp: new Date()
      }
    ];

    inputQuery = '';
    isQuerying = true;
    scrollToBottom();

    try {
      const res = await api.executeQuery({
        query: text,
        document_ids: [document.doc_id],
        strategy: 'hybrid',
        top_k: 5
      });

      const assistantMsgId = `ast_${Date.now()}`;
      messages = [
        ...messages,
        {
          id: assistantMsgId,
          role: 'assistant',
          content: res.answer,
          timestamp: new Date(),
          response: res,
          selectedCitationMarker: res.resolved_citations.length > 0 ? res.resolved_citations[0].marker : null
        }
      ];
    } catch (err: unknown) {
      queryError = err instanceof Error ? err.message : String(err);
      messages = [
        ...messages,
        {
          id: `err_${Date.now()}`,
          role: 'assistant',
          content: `Unable to complete query: ${queryError}`,
          timestamp: new Date()
        }
      ];
    } finally {
      isQuerying = false;
      scrollToBottom();
    }
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function toggleCitation(msgIndex: number, marker: string) {
    const msg = messages[msgIndex];
    if (msg) {
      if (msg.selectedCitationMarker === marker) {
        msg.selectedCitationMarker = null;
      } else {
        msg.selectedCitationMarker = marker;
      }
    }
  }

  // Format bytes for header
  function formatBytes(bytes?: number): string {
    if (!bytes || bytes <= 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  }

  // Helper to split answer text into segments and citation tokens
  function parseAnswerWithCitations(answer: string): { type: 'text' | 'citation'; content: string }[] {
    if (!answer) return [];
    // Match [CIT-X]
    const regex = /(\[CIT-[A-Za-z0-9_-]+\])/g;
    const parts: { type: 'text' | 'citation'; content: string }[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(answer)) !== null) {
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: answer.substring(lastIndex, match.index)
        });
      }
      parts.push({
        type: 'citation',
        content: match[0]
      });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < answer.length) {
      parts.push({
        type: 'text',
        content: answer.substring(lastIndex)
      });
    }

    return parts;
  }

  onMount(() => {
    if (inputRef) {
      inputRef.focus();
    }
  });
</script>

<div class="workspace-container">
  <!-- Document Context Header -->
  <header class="doc-context-bar">
    <div class="dcb-left">
      <button class="btn-back" onclick={onBack} title="Return to Document Library">
        <svg class="back-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="19" y1="12" x2="5" y2="12"></line>
          <polyline points="12 19 5 12 12 5"></polyline>
        </svg>
        <span>All Documents</span>
      </button>

      <div class="doc-identity">
        <div class="file-type-pill">{document.file_type.toUpperCase()}</div>
        <h2 class="current-doc-title" title={document.filename}>{document.filename}</h2>
        <span class="doc-meta-badge">{document.chunk_count} chunks</span>
        {#if document.page_count}
          <span class="doc-meta-badge">{document.page_count} {document.page_count === 1 ? 'page' : 'pages'}</span>
        {/if}
        <span class="badge-ready">READY</span>
      </div>
    </div>

    <div class="dcb-right">
      <button
        class="btn-sources-toggle"
        class:is-active={showSourcesDrawer}
        onclick={() => (showSourcesDrawer = !showSourcesDrawer)}
        title="Toggle retrieved evidence sources"
      >
        <svg class="sources-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
        </svg>
        <span>Evidence Sources</span>
      </button>
    </div>
  </header>

  <!-- Main Chat & Sources Layout -->
  <div class="workspace-body" class:drawer-open={showSourcesDrawer}>
    <main class="chat-main">
      <!-- Conversation Message Stream -->
      <div class="messages-scroll">
        {#if messages.length === 0}
          <div class="welcome-hero">
            <div class="welcome-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
            </div>
            <h3 class="welcome-title">Ask anything about <em>{document.filename}</em></h3>
            <p class="welcome-sub">
              Every factual answer generated by STING is directly anchored to verified passages in your document with interactive citations.
            </p>

            <div class="starter-chips">
              <span class="chips-label">Try asking:</span>
              <div class="chips-grid">
                {#each starterQuestions as question}
                  <button
                    type="button"
                    class="starter-chip"
                    onclick={() => handleSubmit(question)}
                    disabled={isQuerying}
                  >
                    <span>{question}</span>
                    <span class="chip-arrow">&rarr;</span>
                  </button>
                {/each}
              </div>
            </div>
          </div>
        {:else}
          {#each messages as msg, msgIdx (msg.id)}
            <div class="message-row" class:is-user={msg.role === 'user'} class:is-assistant={msg.role === 'assistant'}>
              <div class="msg-avatar">
                {#if msg.role === 'user'}
                  <span class="user-avatar-mark">U</span>
                {:else}
                  <span class="ai-avatar-mark">S</span>
                {/if}
              </div>

              <div class="msg-content-box">
                <div class="msg-header">
                  <span class="sender-name">{msg.role === 'user' ? 'You' : 'STING Document Intelligence'}</span>
                  <span class="msg-time">{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  {#if msg.response?.latency_ms}
                    <span class="latency-tag">{(msg.response.latency_ms).toFixed(0)} ms</span>
                  {/if}
                </div>

                <div class="msg-text">
                  {#if msg.response?.abstained}
                    <div class="abstention-callout">
                      <div class="abs-header">
                        <span class="abs-icon">🛡️</span>
                        <strong>Safely Abstained</strong>
                      </div>
                      <p>
                        {msg.response.abstention_reason ||
                          'No sufficiently grounded evidence passages were retrieved to answer this query with verified citations.'}
                      </p>
                    </div>
                  {/if}

                  <!-- Render text with interactive citations -->
                  <div class="prose-body">
                    {#each parseAnswerWithCitations(msg.content) as segment}
                      {#if segment.type === 'text'}
                        <span>{segment.content}</span>
                      {:else}
                        {@const isSelected = msg.selectedCitationMarker === segment.content}
                        <button
                          type="button"
                          class="inline-cit-pill"
                          class:cit-selected={isSelected}
                          onclick={() => toggleCitation(msgIdx, segment.content)}
                          title="Click to view verified evidence passage"
                        >
                          {segment.content}
                        </button>
                      {/if}
                    {/each}
                  </div>
                </div>

                <!-- Expandable Citation Passages Drawer for this Assistant message -->
                {#if msg.response && msg.response.resolved_citations.length > 0}
                  <div class="citations-shelf">
                    <div class="shelf-label">
                      <span class="shelf-title">Verified Evidence Citations ({msg.response.resolved_citations.length})</span>
                      <span class="shelf-hint">Click a citation tag to inspect its grounded passage</span>
                    </div>

                    <div class="citations-cards-row">
                      {#each msg.response.resolved_citations as cit}
                        {@const isExpanded = msg.selectedCitationMarker === cit.marker}
                        <div class="evidence-quote-card" class:is-focused={isExpanded}>
                          <div class="eqc-header">
                            <span class="eqc-marker">{cit.marker}</span>
                            <span class="eqc-source">
                              {cit.filename}
                              {#if cit.page_number}
                                • Page {cit.page_number}
                              {/if}
                            </span>
                            <span class="eqc-badge">✓ Verified Grounded</span>
                          </div>

                          <blockquote class="eqc-passage">
                            "{cit.snippet}"
                          </blockquote>

                          <div class="eqc-footer">
                            <code class="chunk-ref">{cit.chunk_id}</code>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Developer Mode Diagnostics (Only shown if devMode is enabled) -->
                {#if $devMode && msg.response}
                  <details class="dev-trace-details">
                    <summary class="dev-trace-summary">
                      <span class="dev-dot"></span>
                      <span>Developer Diagnostics &bull; Strategy: {msg.response.strategy} ({msg.response.effective_strategy}) &bull; Latency: {msg.response.latency_ms.toFixed(1)}ms</span>
                    </summary>
                    <div class="dev-trace-body">
                      <pre><code>{JSON.stringify(msg.response.trace, null, 2)}</code></pre>
                    </div>
                  </details>
                {/if}
              </div>
            </div>
          {/each}

          {#if isQuerying}
            <div class="message-row is-assistant thinking-row">
              <div class="msg-avatar">
                <span class="ai-avatar-mark">S</span>
              </div>
              <div class="msg-content-box">
                <div class="thinking-indicator">
                  <span class="thinking-spinner"></span>
                  <span>Retrieving candidate chunks &amp; validating citations...</span>
                </div>
              </div>
            </div>
          {/if}
        {/if}

        <div bind:this={chatBottomRef}></div>
      </div>

      <!-- Input Action Bar -->
      <div class="chat-input-bar">
        {#if queryError}
          <div class="input-error-alert">
            <span>{queryError}</span>
            <button class="btn-clear-err" onclick={() => (queryError = null)}>✕</button>
          </div>
        {/if}

        <div class="input-composer">
          <textarea
            bind:this={inputRef}
            class="composer-textarea"
            placeholder="Ask a question about {document.filename}... (Press Enter to send)"
            bind:value={inputQuery}
            onkeydown={handleKeyDown}
            disabled={isQuerying}
            rows="2"
          ></textarea>

          <button
            class="btn-send"
            onclick={() => handleSubmit()}
            disabled={isQuerying || !inputQuery.trim()}
            title="Send query"
          >
            {#if isQuerying}
              <span class="send-spinner"></span>
            {:else}
              <svg class="send-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            {/if}
          </button>
        </div>

        <div class="input-footer-note">
          <span>STING cites ground-truth evidence passages from {document.filename}.</span>
        </div>
      </div>
    </main>

    <!-- Slide-over Evidence Sources Drawer -->
    {#if showSourcesDrawer}
      <aside class="sources-drawer">
        <div class="drawer-header">
          <div class="drawer-title-row">
            <h3 class="drawer-title">Document Evidence Sources</h3>
            <button class="btn-close-drawer" onclick={() => (showSourcesDrawer = false)}>✕</button>
          </div>
          <p class="drawer-sub">
            Inspecting grounded knowledge sources for <em>{document.filename}</em>.
          </p>
        </div>

        <div class="drawer-content">
          {#if latestAssistantMsg?.response}
            <div class="drawer-section">
              <h4 class="section-heading">Active Query Citations</h4>
              {#if latestAssistantMsg.response.resolved_citations.length === 0}
                <div class="empty-citations-box">
                  No citations were required or resolved for the latest query.
                </div>
              {:else}
                <div class="drawer-citations-list">
                  {#each latestAssistantMsg.response.resolved_citations as cit}
                    <div class="drawer-cit-item">
                      <div class="dci-header">
                        <span class="dci-marker">{cit.marker}</span>
                        {#if cit.page_number}
                          <span class="dci-page">Page {cit.page_number}</span>
                        {/if}
                      </div>
                      <p class="dci-passage">"{cit.snippet}"</p>
                      <div class="dci-footer">
                        <code>{cit.chunk_id}</code>
                      </div>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>

            <div class="drawer-section mt-4">
              <h4 class="section-heading">Retrieval Pipeline Metrics</h4>
              <div class="metrics-mini-grid">
                <div class="mini-stat">
                  <span class="stat-label">Effective Strategy</span>
                  <span class="stat-value">{latestAssistantMsg.response.effective_strategy}</span>
                </div>
                <div class="mini-stat">
                  <span class="stat-label">Latency</span>
                  <span class="stat-value">{latestAssistantMsg.response.latency_ms.toFixed(0)} ms</span>
                </div>
                <div class="mini-stat">
                  <span class="stat-label">Citations Resolved</span>
                  <span class="stat-value">{latestAssistantMsg.response.resolved_citations.length}</span>
                </div>
                <div class="mini-stat">
                  <span class="stat-label">Abstention</span>
                  <span class="stat-value">{latestAssistantMsg.response.abstained ? 'Abstained' : 'Answered'}</span>
                </div>
              </div>
            </div>
          {:else}
            <div class="drawer-empty-state">
              <span class="drawer-empty-icon">📖</span>
              <p>Ask a question in the workspace to inspect the verified candidate chunks and citations retrieved by STING.</p>
            </div>
          {/if}
        </div>
      </aside>
    {/if}
  </div>
</div>

<style>
  .workspace-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 6.5rem);
    background: rgba(13, 17, 23, 0.7);
    border: 1px solid var(--border-subtle, #30363d);
    border-radius: var(--radius-lg, 10px);
    overflow: hidden;
  }

  /* Header Bar */
  .doc-context-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    background: rgba(22, 27, 34, 0.85);
    border-bottom: 1px solid var(--border-subtle, #30363d);
    backdrop-filter: blur(8px);
    gap: 1rem;
  }

  .dcb-left {
    display: flex;
    align-items: center;
    gap: 1rem;
    min-width: 0;
  }

  .btn-back {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.65rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md, 6px);
    color: var(--text-secondary, #94a3b8);
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .btn-back:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
  }

  .back-svg {
    width: 14px;
    height: 14px;
  }

  .doc-identity {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    min-width: 0;
  }

  .file-type-pill {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    letter-spacing: 0.05em;
  }

  .current-doc-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary, #f3f4f6);
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .doc-meta-badge {
    font-size: 0.72rem;
    color: var(--text-muted, #9ca3af);
    background: rgba(255, 255, 255, 0.06);
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    white-space: nowrap;
  }

  .badge-ready {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.12rem 0.45rem;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
  }

  .btn-sources-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.8rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md, 6px);
    color: var(--text-secondary, #94a3b8);
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-sources-toggle:hover,
  .btn-sources-toggle.is-active {
    color: #38bdf8;
    background: rgba(14, 165, 233, 0.12);
    border-color: rgba(56, 189, 248, 0.35);
  }

  .sources-icon {
    width: 14px;
    height: 14px;
  }

  /* Body Layout */
  .workspace-body {
    display: flex;
    flex: 1;
    overflow: hidden;
    position: relative;
  }

  .chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    height: 100%;
  }

  .messages-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem 1.5rem 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  /* Welcome Hero */
  .welcome-hero {
    max-width: 680px;
    margin: 2rem auto;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
  }

  .welcome-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.25rem;
  }

  .welcome-icon svg {
    width: 26px;
    height: 26px;
  }

  .welcome-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.5rem;
  }

  .welcome-title em {
    color: #38bdf8;
    font-style: normal;
  }

  .welcome-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 2rem;
    max-width: 540px;
  }

  .starter-chips {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
  }

  .chips-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .chips-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    width: 100%;
  }

  .starter-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(22, 27, 34, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-md, 6px);
    color: #e2e8f0;
    font-size: 0.825rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .starter-chip:hover {
    background: rgba(14, 165, 233, 0.1);
    border-color: rgba(56, 189, 248, 0.35);
    color: #ffffff;
    transform: translateY(-1px);
  }

  .chip-arrow {
    color: #38bdf8;
    font-size: 0.95rem;
  }

  /* Message Rows */
  .message-row {
    display: flex;
    gap: 1rem;
    max-width: 900px;
    width: 100%;
  }

  .message-row.is-user {
    align-self: flex-end;
    flex-direction: row-reverse;
  }

  .msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 9999px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
  }

  .user-avatar-mark {
    background: #334155;
    color: #ffffff;
    width: 100%;
    height: 100%;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .ai-avatar-mark {
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    color: #ffffff;
    width: 100%;
    height: 100%;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 10px rgba(14, 165, 233, 0.4);
  }

  .msg-content-box {
    flex: 1;
    min-width: 0;
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg, 10px);
    padding: 1rem 1.25rem;
  }

  .is-user .msg-content-box {
    background: rgba(30, 41, 59, 0.7);
    border-color: rgba(56, 189, 248, 0.25);
  }

  .msg-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.45rem;
    font-size: 0.75rem;
  }

  .sender-name {
    font-weight: 600;
    color: #f3f4f6;
  }

  .msg-time {
    color: #64748b;
  }

  .latency-tag {
    font-size: 0.65rem;
    font-family: var(--font-mono);
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
  }

  .prose-body {
    font-size: 0.9rem;
    line-height: 1.65;
    color: #e2e8f0;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Inline Citation Badge */
  .inline-cit-pill {
    display: inline-flex;
    align-items: center;
    margin: 0 0.2rem;
    padding: 0.1rem 0.45rem;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: var(--font-mono);
    border-radius: 4px;
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    cursor: pointer;
    vertical-align: baseline;
    transition: all 0.15s ease;
  }

  .inline-cit-pill:hover,
  .inline-cit-pill.cit-selected {
    background: #0284c7;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
  }

  /* Citation Shelf */
  .citations-shelf {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
  }

  .shelf-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
  }

  .shelf-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #38bdf8;
    text-transform: uppercase;
  }

  .shelf-hint {
    font-size: 0.7rem;
    color: #64748b;
  }

  .citations-cards-row {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
  }

  .evidence-quote-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 3px solid #38bdf8;
    border-radius: var(--radius-sm, 4px);
    padding: 0.75rem 1rem;
    transition: all 0.2s ease;
  }

  .evidence-quote-card.is-focused {
    background: rgba(14, 165, 233, 0.1);
    border-color: rgba(56, 189, 248, 0.45);
    border-left: 3px solid #38bdf8;
    box-shadow: 0 0 16px rgba(14, 165, 233, 0.2);
  }

  .eqc-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.4rem;
    font-size: 0.75rem;
  }

  .eqc-marker {
    font-family: var(--font-mono);
    font-weight: 700;
    color: #38bdf8;
  }

  .eqc-source {
    color: #94a3b8;
  }

  .eqc-badge {
    margin-left: auto;
    font-size: 0.65rem;
    font-weight: 600;
    color: #34d399;
    background: rgba(16, 185, 129, 0.15);
    padding: 0.1rem 0.45rem;
    border-radius: 3px;
  }

  .eqc-passage {
    margin: 0 0 0.4rem 0;
    font-size: 0.825rem;
    font-style: italic;
    color: #cbd5e1;
    line-height: 1.5;
  }

  .chunk-ref {
    font-size: 0.68rem;
    color: #64748b;
  }

  /* Abstention */
  .abstention-callout {
    margin-bottom: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: var(--radius-md, 6px);
    font-size: 0.85rem;
    color: #fde68a;
  }

  .abs-header {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.25rem;
  }

  /* Dev Trace */
  .dev-trace-details {
    margin-top: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-sm, 4px);
    background: rgba(0, 0, 0, 0.35);
    font-size: 0.75rem;
  }

  .dev-trace-summary {
    padding: 0.45rem 0.75rem;
    cursor: pointer;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }

  .dev-dot {
    width: 6px;
    height: 6px;
    border-radius: 9999px;
    background: #c084fc;
  }

  .dev-trace-body {
    padding: 0.75rem;
    overflow-x: auto;
  }

  .dev-trace-body pre {
    margin: 0;
    color: #cbd5e1;
    font-family: var(--font-mono);
  }

  /* Thinking Indicator */
  .thinking-indicator {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    color: #94a3b8;
    font-size: 0.85rem;
    padding: 0.35rem 0;
  }

  .thinking-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(56, 189, 248, 0.25);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Composer */
  .chat-input-bar {
    padding: 1rem 1.5rem 1.25rem;
    background: rgba(22, 27, 34, 0.85);
    border-top: 1px solid var(--border-subtle, #30363d);
  }

  .input-error-alert {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.85rem;
    margin-bottom: 0.75rem;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: var(--radius-sm, 4px);
    color: #fca5a5;
    font-size: 0.8rem;
  }

  .btn-clear-err {
    background: none;
    border: none;
    color: #fca5a5;
    cursor: pointer;
  }

  .input-composer {
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: var(--radius-md, 6px);
    padding: 0.65rem 0.85rem;
    transition: border-color 0.15s;
  }

  .input-composer:focus-within {
    border-color: #38bdf8;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
  }

  .composer-textarea {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: #ffffff;
    font-size: 0.9rem;
    font-family: inherit;
    resize: none;
    line-height: 1.45;
  }

  .composer-textarea::placeholder {
    color: #64748b;
  }

  .btn-send {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm, 4px);
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    border: none;
    color: #ffffff;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.15s;
  }

  .btn-send:hover:not(:disabled) {
    background: linear-gradient(135deg, #0369a1, #0284c7);
    box-shadow: 0 0 12px rgba(14, 165, 233, 0.4);
  }

  .btn-send:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .send-svg {
    width: 16px;
    height: 16px;
  }

  .send-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #ffffff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .input-footer-note {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.4rem;
    text-align: center;
  }

  /* Slide-over Sources Drawer */
  .sources-drawer {
    width: 380px;
    background: rgba(17, 24, 39, 0.95);
    border-left: 1px solid var(--border-subtle, #30363d);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    backdrop-filter: blur(12px);
    animation: slideInRight 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes slideInRight {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
  }

  .drawer-header {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }

  .drawer-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .drawer-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
  }

  .btn-close-drawer {
    background: none;
    border: none;
    color: #94a3b8;
    font-size: 1.1rem;
    cursor: pointer;
  }

  .drawer-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
  }

  .drawer-content {
    padding: 1.25rem;
    flex: 1;
  }

  .section-heading {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 0.75rem;
  }

  .drawer-citations-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .drawer-cit-item {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-sm, 4px);
    padding: 0.75rem;
  }

  .dci-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
  }

  .dci-marker {
    font-family: var(--font-mono);
    font-weight: 700;
    color: #38bdf8;
    font-size: 0.75rem;
  }

  .dci-page {
    font-size: 0.7rem;
    color: #94a3b8;
  }

  .dci-passage {
    font-size: 0.8rem;
    color: #cbd5e1;
    font-style: italic;
    margin: 0 0 0.35rem 0;
    line-height: 1.45;
  }

  .dci-footer code {
    font-size: 0.65rem;
    color: #64748b;
  }

  .metrics-mini-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
  }

  .mini-stat {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 0.5rem 0.65rem;
    border-radius: var(--radius-sm, 4px);
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .stat-label {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
  }

  .stat-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: #f1f5f9;
    font-family: var(--font-mono);
  }

  .drawer-empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #64748b;
    font-size: 0.85rem;
  }

  .drawer-empty-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 0.5rem;
  }

  .empty-citations-box {
    padding: 1rem;
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    background: rgba(15, 23, 42, 0.4);
    border-radius: 4px;
  }

  @media (max-width: 768px) {
    .sources-drawer {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: 100%;
      z-index: 10;
    }

    .chips-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
