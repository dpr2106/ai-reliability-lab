import { writable } from 'svelte/store';

const STORAGE_KEY = 'sting_dev_mode';

function getStorage(): Storage | null {
  try {
    if (typeof localStorage !== 'undefined') return localStorage;
    if (typeof window !== 'undefined' && window.localStorage) return window.localStorage;
  } catch {
    return null;
  }
  return null;
}

function createDevModeStore() {
  const storage = getStorage();
  let initial = false;

  if (storage) {
    try {
      const stored = storage.getItem(STORAGE_KEY);
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
        const s = getStorage();
        if (s) {
          try {
            s.setItem(STORAGE_KEY, String(next));
          } catch {
            // ignore localStorage quota errors
          }
        }
        return next;
      });
    },
    set: (val: boolean) => {
      const s = getStorage();
      if (s) {
        try {
          s.setItem(STORAGE_KEY, String(val));
        } catch {
          // ignore
        }
      }
      set(val);
    }
  };
}

export const devMode = createDevModeStore();
export { createDevModeStore, STORAGE_KEY };

