import { writable } from 'svelte/store';

const STORAGE_KEY = 'sting_dev_mode';

function createDevModeStore() {
  const isBrowser = typeof window !== 'undefined';
  let initial = false;

  if (isBrowser) {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      initial = stored === 'true';
    } catch {
      initial = false;
    }
  }

  const { subscribe, set, update } = writable<boolean>(initial);

  return {
    subscribe,
    toggle: () => {
      update((curr) => {
        const next = !curr;
        if (typeof window !== 'undefined') {
          try {
            localStorage.setItem(STORAGE_KEY, String(next));
          } catch {
            // ignore localStorage quota errors
          }
        }
        return next;
      });
    },
    set: (val: boolean) => {
      if (typeof window !== 'undefined') {
        try {
          localStorage.setItem(STORAGE_KEY, String(val));
        } catch {
          // ignore
        }
      }
      set(val);
    }
  };
}

export const devMode = createDevModeStore();

