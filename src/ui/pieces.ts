import type { PieceCode } from '../game/types';

export const PIECE_CODES: readonly PieceCode[] = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK'];

/**
 * Vite asset URLs of the vendored cburnett SVGs (src/assets/pieces/cburnett/<code>.svg).
 * A glob rather than 12 static imports, so a missing file degrades to the overlay's glyph fallback instead of
 * breaking the build.
 */
const URLS = import.meta.glob<string>('../assets/pieces/cburnett/*.svg', { query: '?url', import: 'default', eager: true });

/** Asset URL of a piece icon, or null when that SVG is not vendored. */
export function pieceUrl(code: PieceCode): string | null {
  return URLS[`../assets/pieces/cburnett/${code}.svg`] ?? null;
}

const images = new Map<PieceCode, HTMLImageElement>();

/** Starts loading every icon (once); returns how many icons are vendored. */
export function preloadPieces(): number {
  for (const code of PIECE_CODES) {
    const url = pieceUrl(code);
    if (!url || images.has(code)) continue;
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    images.set(code, img);
  }
  return images.size;
}

/** The loaded icon of a piece, or null (missing, still loading or broken). */
export function pieceImage(code: PieceCode): HTMLImageElement | null {
  const img = images.get(code);
  return img && img.complete && img.naturalWidth > 0 ? img : null;
}
