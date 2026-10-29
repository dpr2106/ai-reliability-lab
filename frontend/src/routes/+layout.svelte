<script lang="ts">
  import '../app.css';
  import Navigation from '#lib/components/Navigation.svelte';
  import type { LayoutProps } from './$types';

  let { children }: LayoutProps = $props();
</script>

<svelte:head>
  <title>AI Reliability Lab — Diagnostics Studio</title>
  <meta name="description" content="Experiment tracking, regression diagnostics, and citation audit studio" />
</svelte:head>

<div class="app-layout">
  <Navigation />
  <main class="main-content">
    {@render children()}
  </main>
</div>

<style>
  .app-layout {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background-color: var(--bg-canvas);
    color: var(--text-primary);
  }

  .main-content {
    flex: 1;
    max-width: 1400px;
    width: 100%;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
  }
</style>
