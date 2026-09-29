import type { GameSnapshot, PendingChange, PieceCode, PlyInfo } from '../game/types';

/** How a sidebar ply relates to the pending change (optimistic display of the tracker's current favourite). */
export type PlyMark = 'committed' | 'pending' | 'dropped';

export interface DisplayPly {
  ply: PlyInfo;
  mark: PlyMark;
}

/**
 * The plies the sidebar shows: the committed ones, with those from `pending.fromPly` on replaced by the pending line.
 * Committed plies the pending line would remove (a takeback, or a shorter revision) stay listed as 'dropped' after
 * it, so the correction is visible before it commits.
 */
export function displayPlies(plies: readonly PlyInfo[], pending: PendingChange | null): DisplayPly[] {
  if (!pending) return plies.map((ply) => ({ ply, mark: 'committed' }));
  const from = Math.max(0, Math.min(plies.length, pending.fromPly));
  const out: DisplayPly[] = plies.slice(0, from).map((ply) => ({ ply, mark: 'committed' }));
  for (const ply of pending.plies) out.push({ ply, mark: 'pending' });
  for (let i = from + pending.plies.length; i < plies.length; i++) out.push({ ply: plies[i]!, mark: 'dropped' });
  return out;
}

/** Identity of a pending change for re-rendering: kind and line, not progress (progress is a cheap style update). */
export function pendingKey(pending: PendingChange | null): string {
  return pending ? `${pending.kind}:${pending.fromPly}:${pending.plies.map((p) => `${p.san}${p.promotion ?? ''}`).join(' ')}` : '';
}

/** Pending progress clamped to 0..1 (0 when absent or not finite). */
export function pendingProgress(pending: PendingChange | null): number {
  const p = pending?.progress;
  return p !== undefined && Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0;
}

/** The board the pending change would lead to, and the squares it touches (for the overlay's ghost pieces). */
export interface PendingBoard {
  /** Pieces in chess square order after the pending change. */
  pieces: Array<PieceCode | null>;
  /** From / to squares of the pending line's plies (tinted faintly). */
  squares: number[];
}

/** Rook squares [from, to] of a castling king move, or null. */
function castleRook(code: PieceCode, p: PlyInfo): [number, number] | null {
  if (code[1] !== 'K' || Math.abs((p.to % 8) - (p.from % 8)) !== 2) return null;
  const rank = p.from - (p.from % 8);
  return p.to > p.from ? [rank + 7, rank + 5] : [rank, rank + 3];
}

/** Applies a from -> to move to `board` (promotion and castling rook included; en passant is not worth it here). */
function applyPly(board: Array<PieceCode | null>, p: PlyInfo): void {
  const code = board[p.from];
  if (!code) return;
  board[p.to] = p.promotion ? (`${code[0]}${p.promotion.toUpperCase()}` as PieceCode) : code;
  board[p.from] = null;
  const rook = castleRook(code, p);
  if (rook && board[rook[0]]?.[1] === 'R') {
    board[rook[1]] = board[rook[0]]!;
    board[rook[0]] = null;
  }
}

/** Takes a ply back on `board` (the captured piece, unknown here, is not restored). */
function unapplyPly(board: Array<PieceCode | null>, p: PlyInfo): void {
  const code = board[p.to];
  if (!code) return;
  board[p.from] = p.promotion ? (`${code[0]}P` as PieceCode) : code;
  board[p.to] = null;
  const rook = castleRook(code, p);
  if (rook && board[rook[1]]?.[1] === 'R' && !board[rook[0]]) {
    board[rook[0]] = board[rook[1]]!;
    board[rook[1]] = null;
  }
}

/**
 * Approximate position after the pending change, from the current `pieces`: the committed plies it replaces are taken
 * back (newest first), then its own plies are played. Null without a pending change or with malformed pieces.
 */
export function pendingBoard(s: Pick<GameSnapshot, 'pieces' | 'plies' | 'pending'>): PendingBoard | null {
  const pending = s.pending;
  if (!pending || s.pieces.length !== 64) return null;
  const board = s.pieces.slice();
  const from = Math.max(0, Math.min(s.plies.length, pending.fromPly));
  for (let i = s.plies.length - 1; i >= from; i--) unapplyPly(board, s.plies[i]!);
  const squares: number[] = [];
  for (const p of pending.plies) {
    applyPly(board, p);
    squares.push(p.from, p.to);
  }
  return { pieces: board, squares };
}
