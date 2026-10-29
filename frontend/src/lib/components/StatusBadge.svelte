<script lang="ts">
  interface Props {
    status: string;
    size?: 'sm' | 'md';
  }

  let { status, size = 'md' }: Props = $props();

  const normalized = $derived(status ? status.toUpperCase() : 'UNKNOWN');

  const badgeClass = $derived.by(() => {
    switch (normalized) {
      case 'PASS':
      case 'OK':
      case 'VALID':
        return 'badge-pass';
      case 'REGRESSION':
      case 'FAIL':
      case 'ERROR':
      case 'INVALID':
        return 'badge-fail';
      case 'INCOMPLETE':
      case 'DEGRADED':
      case 'WARNING':
      case 'MISSING_EVIDENCE':
        return 'badge-warn';
      default:
        return 'badge-neutral';
    }
  });
</script>

<span class="badge {badgeClass} {size === 'sm' ? 'badge-sm' : ''}">
  <span class="dot"></span>
  {status}
</span>

<style>
  .badge-sm {
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: currentColor;
    display: inline-block;
  }
</style>

