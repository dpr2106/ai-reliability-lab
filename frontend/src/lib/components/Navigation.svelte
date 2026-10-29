<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '#lib/api';
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
      <a href="/" class="brand-link">
        <span class="brand-badge">RL</span>
        <div class="brand-text">
          <span class="brand-title">AI Reliability Lab</span>
          <span class="brand-subtitle">Diagnostics Studio</span>
        </div>
      </a>

      <nav class="nav-links">
        <a href="/" class="nav-link">Run Explorer</a>
        <a href="/compare" class="nav-link">Run Comparison</a>
        <a href="/citation-audit" class="nav-link">Citation Audit</a>
      </nav>
    </div>

    <div class="nav-right">
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
    </div>
  </div>
</header>

<style>
  .app-nav {
    border-bottom: 1px solid var(--border-subtle);
    background: rgba(18, 22, 31, 0.85);
    backdrop-filter: blur(12px);
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
  }

  .nav-left {
    display: flex;
    align-items: center;
    gap: 2.5rem;
  }

  .brand-link {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none;
    color: var(--text-primary);
  }

  .brand-badge {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-hover));
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.35rem 0.55rem;
    border-radius: var(--radius-sm);
    letter-spacing: 0.05em;
  }

  .brand-text {
    display: flex;
    flex-direction: column;
  }

  .brand-title {
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: -0.01em;
  }

  .brand-subtitle {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .nav-links {
    display: flex;
    gap: 1.25rem;
  }

  .nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.5rem 0.75rem;
    border-radius: var(--radius-sm);
    transition: color 0.15s, background 0.15s;
  }

  .nav-link:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .health-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .health-pill.health-ok {
    border-color: rgba(5, 150, 105, 0.3);
    color: #6ee7b7;
  }

  .health-pill.health-err {
    border-color: rgba(225, 29, 72, 0.3);
    color: #fda4af;
    cursor: pointer;
  }

  .health-pill.health-checking {
    border-color: rgba(217, 119, 6, 0.3);
    color: #fcd34d;
  }
</style>
