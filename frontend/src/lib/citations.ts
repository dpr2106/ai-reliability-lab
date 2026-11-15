/**
 * Utilities for parsing and managing citation tokens within document chat responses.
 */

export interface AnswerSegment {
  type: 'text' | 'citation';
  content: string;
}

/**
 * Splits an answer string into plaintext segments and [CIT-...] citation tokens.
 */
export function parseAnswerWithCitations(answer: string): AnswerSegment[] {
  if (!answer) return [];
  const regex = /(\[CIT-[A-Za-z0-9_-]+\])/g;
  const parts: AnswerSegment[] = [];
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

/**
 * Toggles citation selection state: if already selected, deselects; otherwise selects the new marker.
 */
export function toggleCitationSelection(
  currentMarker: string | null | undefined,
  targetMarker: string
): string | null {
  return currentMarker === targetMarker ? null : targetMarker;
}
