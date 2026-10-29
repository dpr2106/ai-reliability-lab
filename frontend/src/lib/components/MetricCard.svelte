<script lang="ts">
  interface Props {
    label: string;
    value: number | string | null | undefined;
    format?: 'number' | 'percent' | 'raw';
    delta?: number | null;
    isRegressed?: boolean;
    subtext?: string;
  }

  let {
    label,
    value,
    format = 'number',
    delta = null,
    isRegressed = false,
    subtext = undefined
  }: Props = $props();

  const formattedValue = $derived.by(() => {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'string') return value;
    if (format === 'percent') {
      return `${(value * 100).toFixed(1)}%`;
    }
    return value.toFixed(4);
  });

  const deltaFormatted = $derived.by(() => {
    if (delta === null || delta === undefined) return null;
    const sign = delta > 0 ? '+' : '';
    return `${sign}${delta.toFixed(4)}`;
  });
</script>

<div class="metric-card" class:regressed={isRegressed}>
  <div class="metric-label">{label}</div>
  <div class="metric-val-row">
    <span class="metric-value">{formattedValue}</span>
    {#if deltaFormatted !== null}
      <span
        class="delta-badge"
        class:delta-pos={delta !== null && delta > 0 && !isRegressed}
        class:delta-neg={isRegressed || (delta !== null && delta < 0)}
      >
        {deltaFormatted}
      </span>
    {/if}
  </div>
  {#if subtext}
    <div class="metric-subtext">{subtext}</div>
  {/if}
</div>

<style>
  .metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    transition: border-color 0.15s;
  }

  .metric-card.regressed {
    border-color: rgba(225, 29, 72, 0.4);
    background: linear-gradient(to bottom, rgba(225, 29, 72, 0.05), var(--bg-surface));
  }

  .metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .metric-val-row {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
  }

  .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }

  .delta-badge {
    font-size: 0.8rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    padding: 0.15rem 0.45rem;
    border-radius: var(--radius-sm);
  }

  .delta-pos {
    background: rgba(5, 150, 105, 0.15);
    color: #6ee7b7;
  }

  .delta-neg {
    background: rgba(225, 29, 72, 0.15);
    color: #fda4af;
  }

  .metric-subtext {
    font-size: 0.75rem;
    color: var(--text-muted);
  }
</style>

