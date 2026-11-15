<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { api } from '#lib/api';
  import { devMode } from '#lib/stores/devMode';
  import type { HealthResponse } from '#lib/types';

  let health = $state<HealthResponse | null>(null);
  let isChecking = $state<boolean>(true);
  let checkError = $state<string | null>(null);

  async function checkHealth() {
    isChecking = true;
    checkError = null;
    try {
      health = await api.getHealth();
    } catch (err: unknown) {
      checkError = err instanceof Error ? err.message : String(err);
      health = null;
    } finally {
      isChecking = false;
    }
  }

  onMount(() => {
    checkHealth();
  });
</script>

<header class="app-nav">
  <div class="nav-container">
    <div class="nav-left">
      <a href="/" class="brand-link" title="STING — Document Intelligence & RAG Reliability">
        <div class="brand-logo-mark">
          <svg class="brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
        </div>
        <div class="brand-text">
          <div class="brand-title-row">
            <span class="brand-title">STING</span>
            {#if $devMode}
              <span class="dev-mode-chip">DEV</span>
            {/if}
          </div>
          <span class="brand-subtitle">{$devMode ? 'Reliability Lab' : 'Document Intelligence'}</span>
        </div>
      </a>

      <nav class="nav-links">
        <a href="/" class="nav-link" class:active={page.url.pathname === '/'}>
          Workspace
        </a>

        {#if $devMode}
          <a
            href="/evaluation"
            class="nav-link eval-link"
            class:active={page.url.pathname.startsWith('/evaluation')}
          >
            <span class="eval-dot"></span>
            Evaluation Studio
          </a>
          <a
            href="/investigations"
            class="nav-link highlight-link"
            class:active={page.url.pathname.startsWith('/investigations')}
          >
            <span class="workbench-dot"></span>
            Investigations
          </a>
          <a
            href="/runs"
            class="nav-link"
            class:active={page.url.pathname.startsWith('/runs')}
          >
            Run Explorer
          </a>
          <a
            href="/compare"
            class="nav-link"
            class:active={page.url.pathname.startsWith('/compare')}
          >
            Run Comparison
          </a>
          <a
            href="/citation-audit"
            class="nav-link"
            class:active={page.url.pathname.startsWith('/citation-audit')}
          >
            Citation Audit
          </a>
        {/if}
      </nav>
    </div>

    <div class="nav-right">
      {#if $devMode}
        {#if isChecking}
          <span class="health-pill health-checking">
            <span class="status-indicator status-amber"></span>
            Checking API...
          </span>
        {:else if health && health.status === 'ok'}
          <span class="health-pill health-ok" title="Database: {health.database}">
            <span class="status-indicator status-green"></span>
            API Connected ({health.total_runs} {health.total_runs === 1 ? 'run' : 'runs'})
          </span>
        {:else}
          <button class="health-pill health-err" onclick={checkHealth} title={checkError ?? 'API unreachable'}>
            <span class="status-indicator status-red"></span>
            API Disconnected (retry)
          </button>
        {/if}
      {/if}

      <!-- Mode Switch Toggle -->
      <div class="mode-switch-wrapper">
        <button
          type="button"
          class="mode-toggle-btn"
          class:is-dev={$devMode}
          onclick={() => devMode.toggle()}
          role="switch"
          aria-checked={$devMode}
          title={$devMode ? 'Switch to User Mode (Default)' : 'Switch to Developer Mode (Diagnostic Workbench)'}
        >
          <span class="toggle-track">
            <span class="toggle-thumb"></span>
          </span>
          <span class="toggle-label">Developer Mode</span>
          <span class="toggle-status-tag" class:tag-active={$devMode}>
            {$devMode ? 'ACTIVE' : 'OFF'}
          </span>
        </button>
      </div>
    </div>
  </div>
</header>

<style>
  .app-nav {
    border-bottom: 1px solid var(--border-subtle);
    background: rgba(13, 17, 23, 0.92);
    backdrop-filter: blur(16px);
    position: sticky;
    top: 0;
    z-index: 50;
  }

  .nav-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 1.5rem;
    height: 4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
  }

  .nav-left {
    display: flex;
    align-items: center;
    gap: 2.25rem;
  }

  .brand-link {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none;
    color: var(--text-primary, #f3f4f6);
    transition: opacity 0.15s;
  }

  .brand-link:hover {
    opacity: 0.95;
    text-decoration: none;
  }

  .brand-logo-mark {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(2, 132, 199, 0.4));
    border: 1px solid rgba(56, 189, 248, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #38bdf8;
    box-shadow: 0 0 16px rgba(14, 165, 233, 0.25);
  }

  .brand-icon {
    width: 18px;
    height: 18px;
  }

  .brand-text {
    display: flex;
    flex-direction: column;
  }

  .brand-title-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }

  .brand-title {
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    background: linear-gradient(180deg, #ffffff 30%, #94a3b8 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .dev-mode-chip {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    background: rgba(168, 85, 247, 0.2);
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #c084fc;
  }

  .brand-subtitle {
    font-size: 0.68rem;
    color: var(--text-muted, #9ca3af);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }

  .nav-links {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .nav-link {
    color: var(--text-secondary, #94a3b8);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.45rem 0.8rem;
    border-radius: var(--radius-sm, 4px);
    transition: all 0.15s ease;
    border: 1px solid transparent;
  }

  .nav-link:hover {
    color: var(--text-primary, #ffffff);
    background: rgba(255, 255, 255, 0.05);
    text-decoration: none;
  }

  .nav-link.active {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.1);
  }

  .nav-link.eval-link {
    color: #c084fc;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }

  .nav-link.eval-link:hover,
  .nav-link.eval-link.active {
    color: #e9d5ff;
    background: rgba(192, 132, 252, 0.1);
  }

  .eval-dot {
    width: 6px;
    height: 6px;
    border-radius: 9999px;
    background: #c084fc;
    box-shadow: 0 0 8px rgba(192, 132, 252, 0.6);
  }

  .nav-link.highlight-link {
    color: #38bdf8;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }

  .nav-link.highlight-link:hover,
  .nav-link.highlight-link.active {
    color: #bae6fd;
    background: rgba(56, 189, 248, 0.1);
  }

  .workbench-dot {
    width: 6px;
    height: 6px;
    border-radius: 9999px;
    background: #38bdf8;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.6);
  }

  .nav-right {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .health-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid var(--border-subtle, #30363d);
    color: var(--text-secondary, #94a3b8);
  }

  .health-pill.health-ok {
    border-color: rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
  }

  .health-pill.health-err {
    border-color: rgba(225, 29, 72, 0.35);
    color: #fda4af;
    cursor: pointer;
  }

  .health-pill.health-checking {
    border-color: rgba(217, 119, 6, 0.35);
    color: #fcd34d;
  }

  .status-indicator {
    width: 6px;
    height: 6px;
    border-radius: 9999px;
  }

  .status-green {
    background: #10b981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.7);
  }

  .status-red {
    background: #f43f5e;
    box-shadow: 0 0 6px rgba(244, 63, 94, 0.7);
  }

  .status-amber {
    background: #f59e0b;
    box-shadow: 0 0 6px rgba(245, 158, 11, 0.7);
  }

  /* Mode switch control */
  .mode-switch-wrapper {
    display: flex;
    align-items: center;
  }

  .mode-toggle-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.35rem 0.65rem 0.35rem 0.45rem;
    background: rgba(22, 27, 34, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 9999px;
    cursor: pointer;
    color: var(--text-secondary, #94a3b8);
    font-size: 0.78rem;
    font-weight: 500;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .mode-toggle-btn:hover {
    border-color: rgba(255, 255, 255, 0.25);
    color: #f3f4f6;
  }

  .mode-toggle-btn.is-dev {
    background: rgba(14, 165, 233, 0.12);
    border-color: rgba(56, 189, 248, 0.45);
    color: #e0f2fe;
    box-shadow: 0 0 14px rgba(14, 165, 233, 0.15);
  }

  .toggle-track {
    width: 28px;
    height: 16px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 9999px;
    position: relative;
    transition: background-color 0.2s;
  }

  .mode-toggle-btn.is-dev .toggle-track {
    background: #0284c7;
  }

  .toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 12px;
    height: 12px;
    border-radius: 9999px;
    background: #ffffff;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  }

  .mode-toggle-btn.is-dev .toggle-thumb {
    transform: translateX(12px);
  }

  .toggle-label {
    user-select: none;
    font-weight: 600;
  }

  .toggle-status-tag {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.1rem 0.4rem;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-muted, #9ca3af);
  }

  .toggle-status-tag.tag-active {
    background: rgba(56, 189, 248, 0.25);
    color: #38bdf8;
  }
</style>
