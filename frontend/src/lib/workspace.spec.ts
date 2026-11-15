import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createDevModeStore, STORAGE_KEY } from './stores/devMode';
import { ApiClient, ApiError } from './api';
import { getStageIndex, getProgressPercent, PROCESSING_STAGES } from './processing';
import { parseAnswerWithCitations, toggleCitationSelection } from './citations';
import type { DocumentListResponse } from './types';

describe('V2-007: User-First Workspace & Developer Mode Tests', () => {
  // ---------------------------------------------------------------------------
  // 1. Developer Mode: Default, Persistence, and Storage Error Safety
  // ---------------------------------------------------------------------------
  describe('Developer Mode Store', () => {
    let mockStorage: Record<string, string> = {};

    beforeEach(() => {
      mockStorage = {};
      vi.stubGlobal('localStorage', {
        getItem: vi.fn((key: string) => mockStorage[key] ?? null),
        setItem: vi.fn((key: string, val: string) => {
          mockStorage[key] = val;
        }),
        removeItem: vi.fn((key: string) => {
          delete mockStorage[key];
        }),
        clear: vi.fn(() => {
          mockStorage = {};
        })
      });
    });

    it('defaults to OFF (false) when localStorage has no saved preference', () => {
      const store = createDevModeStore();
      let value: boolean | undefined;
      const unsubscribe = store.subscribe((v) => {
        value = v;
      });
      expect(value).toBe(false);
      unsubscribe();
    });

    it('restores ON (true) when localStorage has "true"', () => {
      mockStorage[STORAGE_KEY] = 'true';
      const store = createDevModeStore();
      let value: boolean | undefined;
      const unsubscribe = store.subscribe((v) => {
        value = v;
      });
      expect(value).toBe(true);
      unsubscribe();
    });

    it('restores OFF (false) when localStorage has "false"', () => {
      mockStorage[STORAGE_KEY] = 'false';
      const store = createDevModeStore();
      let value: boolean | undefined;
      const unsubscribe = store.subscribe((v) => {
        value = v;
      });
      expect(value).toBe(false);
      unsubscribe();
    });

    it('toggles value and persists new state to localStorage', () => {
      const store = createDevModeStore();
      let value: boolean | undefined;
      const unsubscribe = store.subscribe((v) => {
        value = v;
      });
      expect(value).toBe(false);

      store.toggle();
      expect(value).toBe(true);
      expect(mockStorage[STORAGE_KEY]).toBe('true');

      store.toggle();
      expect(value).toBe(false);
      expect(mockStorage[STORAGE_KEY]).toBe('false');

      unsubscribe();
    });

    it('sets value explicitly and persists to localStorage', () => {
      const store = createDevModeStore();
      let value: boolean | undefined;
      const unsubscribe = store.subscribe((v) => {
        value = v;
      });

      store.set(true);
      expect(value).toBe(true);
      expect(mockStorage[STORAGE_KEY]).toBe('true');

      store.set(false);
      expect(value).toBe(false);
      expect(mockStorage[STORAGE_KEY]).toBe('false');

      unsubscribe();
    });

    it('handles localStorage throwing exceptions gracefully without crashing', () => {
      vi.stubGlobal('localStorage', {
        getItem: vi.fn(() => {
          throw new Error('SecurityError: Access Denied');
        }),
        setItem: vi.fn(() => {
          throw new Error('QuotaExceededError');
        })
      });

      const store = createDevModeStore();
      let value: boolean | undefined;
      const unsubscribe = store.subscribe((v) => {
        value = v;
      });
      expect(value).toBe(false);

      expect(() => store.toggle()).not.toThrow();
      expect(value).toBe(true);

      expect(() => store.set(false)).not.toThrow();
      expect(value).toBe(false);

      unsubscribe();
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Document API: Response Normalization & Error Handling
  // ---------------------------------------------------------------------------
  describe('Document API Response Normalization', () => {
    it('normalizes backend response containing "documents" into both documents and items arrays', async () => {
      const mockBackendPayload = {
        total: 1,
        limit: 10,
        offset: 0,
        documents: [
          {
            doc_id: 'doc_101',
            filename: 'architecture.pdf',
            file_type: 'pdf',
            file_size: 1048576,
            chunk_count: 42,
            page_count: 8,
            status: 'READY',
            created_at: '2026-11-15T00:00:00Z'
          }
        ]
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockBackendPayload
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const res: DocumentListResponse = await client.listDocuments();

      expect(res.total).toBe(1);
      expect(res.documents).toHaveLength(1);
      expect(res.items).toHaveLength(1);
      expect(res.documents[0].filename).toBe('architecture.pdf');
      expect(res.items?.[0].doc_id).toBe('doc_101');
    });

    it('normalizes legacy backend response containing "items" into both documents and items arrays', async () => {
      const mockLegacyPayload = {
        total: 1,
        limit: 10,
        offset: 0,
        items: [
          {
            doc_id: 'doc_102',
            filename: 'notes.md',
            file_type: 'md',
            file_size: 2048,
            chunk_count: 5,
            status: 'READY',
            created_at: '2026-11-15T01:00:00Z'
          }
        ]
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockLegacyPayload
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const res: DocumentListResponse = await client.listDocuments();

      expect(res.total).toBe(1);
      expect(res.documents).toHaveLength(1);
      expect(res.items).toHaveLength(1);
      expect(res.documents[0].filename).toBe('notes.md');
    });

    it('handles empty document collection safely without undefined errors', async () => {
      const mockEmptyPayload = {
        total: 0,
        limit: 50,
        offset: 0,
        documents: []
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockEmptyPayload
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const res: DocumentListResponse = await client.listDocuments();

      expect(res.total).toBe(0);
      expect(res.documents).toEqual([]);
      expect(res.items).toEqual([]);
      // Ensure .filter does not throw on empty collection
      expect(res.documents.filter((d) => d.status === 'READY')).toEqual([]);
    });

    it('handles missing total field by defaulting to document array length', async () => {
      const mockPayloadWithoutTotal = {
        documents: [
          {
            doc_id: 'doc_103',
            filename: 'specs.txt',
            file_type: 'txt',
            file_size: 512,
            chunk_count: 2,
            status: 'READY',
            created_at: '2026-11-15T02:00:00Z'
          }
        ]
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockPayloadWithoutTotal
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const res = await client.listDocuments();

      expect(res.total).toBe(1);
      expect(res.documents).toHaveLength(1);
    });

    it('throws ApiError with formatted backend detail message on HTTP failure', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({ detail: 'Unsupported document format .exe' })
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      await expect(client.listDocuments()).rejects.toThrow(ApiError);
      await expect(client.listDocuments()).rejects.toMatchObject({
        status: 400,
        detail: 'Unsupported document format .exe'
      });
    });

    it('uploads a document via multipart FormData', async () => {
      const mockUploadResponse = {
        doc_id: 'doc_new_999',
        filename: 'manual.pdf',
        file_type: 'pdf',
        status: 'UPLOADED',
        chunk_count: 0
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockUploadResponse
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const dummyFile = new File(['content'], 'manual.pdf', { type: 'application/pdf' });
      const result = await client.uploadDocument(dummyFile);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-api/api/documents/upload',
        expect.objectContaining({
          method: 'POST'
        })
      );
      expect(result.doc_id).toBe('doc_new_999');
      expect(result.status).toBe('UPLOADED');
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Document Processing: Lifecycle Stages & Progress Honesty
  // ---------------------------------------------------------------------------
  describe('Document Processing Lifecycle', () => {
    it('defines all 5 canonical stages in strict sequence', () => {
      expect(PROCESSING_STAGES).toHaveLength(5);
      expect(PROCESSING_STAGES.map((s) => s.label)).toEqual([
        'UPLOAD',
        'EXTRACTION',
        'CHUNKING',
        'INDEXING',
        'READY'
      ]);
    });

    it('computes honest stage indices from backend status strings', () => {
      expect(getStageIndex('UPLOADED')).toBe(0);
      expect(getStageIndex('EXTRACTING')).toBe(1);
      expect(getStageIndex('CHUNKING')).toBe(2);
      expect(getStageIndex('INDEXING')).toBe(3);
      expect(getStageIndex('READY')).toBe(4);
      // Case insensitivity & fallback
      expect(getStageIndex('ready')).toBe(4);
      expect(getStageIndex('indexing')).toBe(3);
      expect(getStageIndex('UNKNOWN_STATUS')).toBe(0);
    });

    it('provides strictly monotonic progress percentages across stage progression', () => {
      const p0 = getProgressPercent(0);
      const p1 = getProgressPercent(1);
      const p2 = getProgressPercent(2);
      const p3 = getProgressPercent(3);
      const p4 = getProgressPercent(4);

      expect(p0).toBe(20);
      expect(p1).toBe(42);
      expect(p2).toBe(68);
      expect(p3).toBe(88);
      expect(p4).toBe(100);

      expect(p0).toBeLessThan(p1);
      expect(p1).toBeLessThan(p2);
      expect(p2).toBeLessThan(p3);
      expect(p3).toBeLessThan(p4);
    });

    it('handles document polling completion when READY', async () => {
      const mockDoc = {
        doc_id: 'doc_ready_1',
        filename: 'guide.pdf',
        file_type: 'pdf',
        file_size: 1024,
        chunk_count: 12,
        status: 'READY',
        created_at: '2026-11-15T00:00:00Z',
        indexed_at: '2026-11-15T00:01:00Z'
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockDoc
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const res = await client.getDocument('doc_ready_1');

      expect(res.status).toBe('READY');
      expect(res.chunk_count).toBe(12);
      expect(getStageIndex(res.status)).toBe(4);
      expect(getProgressPercent(getStageIndex(res.status))).toBe(100);
    });

    it('handles document failure state with error details', async () => {
      const mockFailedDoc = {
        doc_id: 'doc_fail_1',
        filename: 'corrupted.pdf',
        file_type: 'pdf',
        file_size: 512,
        chunk_count: 0,
        status: 'FAILED',
        error_message: 'PDF stream decryption failed: damaged trailer dictionary',
        created_at: '2026-11-15T00:00:00Z'
      };

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockFailedDoc
      });

      const client = new ApiClient('http://test-api', mockFetch as unknown as typeof fetch);
      const res = await client.getDocument('doc_fail_1');

      expect(res.status).toBe('FAILED');
      expect(res.error_message).toContain('PDF stream decryption failed');
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Citation Parsing & Evidence Drawer Interaction
  // ---------------------------------------------------------------------------
  describe('Citation Parsing and Evidence Drawer Logic', () => {
    it('parses mixed text with single and multiple citation tokens', () => {
      const answer = 'STING enforces baseline-first evaluation [CIT-1] and honest metrics [CIT-2].';
      const segments = parseAnswerWithCitations(answer);

      expect(segments).toEqual([
        { type: 'text', content: 'STING enforces baseline-first evaluation ' },
        { type: 'citation', content: '[CIT-1]' },
        { type: 'text', content: ' and honest metrics ' },
        { type: 'citation', content: '[CIT-2]' },
        { type: 'text', content: '.' }
      ]);
    });

    it('parses answers with no citations as a single text segment', () => {
      const answer = 'This is a general summary without specific citation anchors.';
      const segments = parseAnswerWithCitations(answer);

      expect(segments).toEqual([
        {
          type: 'text',
          content: 'This is a general summary without specific citation anchors.'
        }
      ]);
    });

    it('handles empty or blank answer strings safely', () => {
      expect(parseAnswerWithCitations('')).toEqual([]);
    });

    it('toggles citation selection marker correctly', () => {
      // Initially no citation selected
      let selected: string | null = null;

      // Select CIT-1
      selected = toggleCitationSelection(selected, '[CIT-1]');
      expect(selected).toBe('[CIT-1]');

      // Clicking CIT-1 again deselects it
      selected = toggleCitationSelection(selected, '[CIT-1]');
      expect(selected).toBeNull();

      // Clicking CIT-2 selects CIT-2
      selected = toggleCitationSelection(selected, '[CIT-2]');
      expect(selected).toBe('[CIT-2]');

      // Clicking CIT-1 switches selection to CIT-1
      selected = toggleCitationSelection(selected, '[CIT-1]');
      expect(selected).toBe('[CIT-1]');
    });
  });
});
