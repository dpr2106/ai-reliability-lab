/**
 * Document processing lifecycle stages and progress calculations.
 */

export type ProcessingStage = 'upload' | 'extraction' | 'chunking' | 'indexing' | 'ready';

export interface StageDefinition {
  key: ProcessingStage;
  label: string;
  desc: string;
}

export const PROCESSING_STAGES: StageDefinition[] = [
  { key: 'upload', label: 'UPLOAD', desc: 'File received & validated' },
  { key: 'extraction', label: 'EXTRACTION', desc: 'Parsing structure & text' },
  { key: 'chunking', label: 'CHUNKING', desc: 'Citation-aware chunk boundaries' },
  { key: 'indexing', label: 'INDEXING', desc: 'Dense vectors & BM25 generation' },
  { key: 'ready', label: 'READY', desc: 'Index verified & ready' }
];

/**
 * Returns the 0-based stage index (0 to 4) for a given document status string.
 */
export function getStageIndex(status: string): number {
  const s = (status || '').toUpperCase();
  if (s === 'READY') return 4;
  if (s === 'INDEXING') return 3;
  if (s === 'CHUNKING') return 2;
  if (s === 'EXTRACTING') return 1;
  return 0; // UPLOADED or default
}

/**
 * Returns an honest stage-based progress percentage (20% to 100%).
 */
export function getProgressPercent(stageIndex: number): number {
  switch (stageIndex) {
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
}
