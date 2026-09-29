import { Chess, type Move } from 'chess.js';
import { OCC_BLACK, OCC_EMPTY, OCC_WHITE } from '../worker/protocol';
import type { PieceCode, SquareIndex } from './types';

/** Standard starting position. */
export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

/** 'e4' -> 28 (file + 8 * rank, a1 = 0). */
export function sqIndex(name: string): SquareIndex {
  return name.charCodeAt(0) - 97 + 8 * (name.charCodeAt(1) - 49);
}

/** 28 -> 'e4'. */
export function sqName(sq: SquareIndex): string {
  return String.fromCharCode(97 + (sq & 7), 49 + (sq >> 3));
}

/** Occupancy grid (OCC_* per square, a1 = 0) of a chess.js position. */
export function gridOf(chess: Chess): Uint8Array {
  const g = new Uint8Array(64);
  const rows = chess.board();
  for (let r = 0; r < 8; r++) {
    const row = rows[r]!;
    for (let f = 0; f < 8; f++) {
      const p = row[f];
      if (p) g[f + 8 * (7 - r)] = p.color === 'w' ? OCC_WHITE : OCC_BLACK;
    }
  }
  return g;
}

/** Pieces (PieceCode or null per square, a1 = 0) of a chess.js position. */
export function piecesOf(chess: Chess): Array<PieceCode | null> {
  const out: Array<PieceCode | null> = new Array(64).fill(null);
  const rows = chess.board();
  for (let r = 0; r < 8; r++) {
    const row = rows[r]!;
    for (let f = 0; f < 8; f++) {
      const p = row[f];
      if (p) out[f + 8 * (7 - r)] = `${p.color}${p.type.toUpperCase()}` as PieceCode;
    }
  }
  return out;
}

/** Occupancy of the starting position. */
export const START_GRID: Uint8Array = gridOf(new Chess());

/** 'e7e8q' -> chess.js move object. */
export function uciMove(uci: string): { from: string; to: string; promotion?: string } {
  return uci.length > 4 ? { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] } : { from: uci.slice(0, 2), to: uci.slice(2, 4) };
}

/** A legal move with everything the lattice needs, precomputed once per position (see `legalMoves`). */
export interface MoveSpec {
  /** Long algebraic without separators, e.g. 'e2e4', 'e7e8q'. Identity of the move inside a position. */
  uci: string;
  san: string;
  /** Moving piece type ('p', 'n', 'b', 'r', 'q', 'k'). */
  piece: string;
  from: SquareIndex;
  to: SquareIndex;
  promotion?: string;
  /** FEN after the move (chess.js computes it with the verbose move list anyway). */
  after: string;
  /**
   * Occupancy edits as (square, class) pairs, applied in order: origin emptied, destination takes the mover's colour,
   * plus the en-passant victim or the castling rook.
   */
  edits: Int8Array;
}

function specOf(m: Move): MoveSpec {
  const from = sqIndex(m.from);
  const to = sqIndex(m.to);
  const cls = m.color === 'w' ? OCC_WHITE : OCC_BLACK;
  const e: number[] = [from, OCC_EMPTY, to, cls];
  if (m.isEnPassant()) e.push((to & 7) + (from & ~7), OCC_EMPTY);
  else if (m.isKingsideCastle()) e.push(from + 3, OCC_EMPTY, from + 1, cls);
  else if (m.isQueensideCastle()) e.push(from - 4, OCC_EMPTY, from - 1, cls);
  const spec: MoveSpec = { uci: m.lan, san: m.san, piece: m.piece, from, to, after: m.after, edits: Int8Array.from(e) };
  if (m.promotion) spec.promotion = m.promotion;
  return spec;
}

/** Applies a move's occupancy edits to a copy of `grid`. */
export function applyEdits(grid: Uint8Array, spec: MoveSpec): Uint8Array {
  const g = grid.slice();
  const e = spec.edits;
  for (let i = 0; i < e.length; i += 2) g[e[i]!] = e[i + 1]!;
  return g;
}

const MOVE_CACHE_MAX = 4096;
const moveCache = new Map<string, MoveSpec[]>();
const scratch = new Chess();

/**
 * Legal moves of a FEN, memoised: the lattice re-expands the same positions after every commit and re-rooting, and
 * chess.js's verbose generation (which also builds each move's FEN) costs close to a millisecond per position.
 */
export function legalMoves(fen: string): MoveSpec[] {
  let r = moveCache.get(fen);
  if (!r) {
    scratch.load(fen);
    r = scratch.moves({ verbose: true }).map(specOf);
    if (moveCache.size >= MOVE_CACHE_MAX) moveCache.clear();
    moveCache.set(fen, r);
  }
  return r;
}
