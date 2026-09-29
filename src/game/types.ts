import type { Observation, SquareIndex } from '../worker/protocol';

export type { Observation, SquareIndex };

/** Piece code as used by the icon set: colour ('w' | 'b') + upper-case type, e.g. 'wP', 'bK'. */
export type PieceCode = `${'w' | 'b'}${'P' | 'N' | 'B' | 'R' | 'Q' | 'K'}`;

/** waiting: no game yet (looking for the start position); playing; desync: the board matches no hypothesis; over: mate / draw. */
export type GameState = 'waiting' | 'playing' | 'desync' | 'over';

export interface PlyInfo {
  san: string;
  from: SquareIndex;
  to: SquareIndex;
  /** Inside the revision window: may still be corrected by later evidence. */
  tentative: boolean;
  /** Time (ms, observation clock) of the last correction of this ply, if it was ever corrected. */
  correctedAt?: number;
  /** Promotion piece ('q' | 'r' | 'b' | 'n') when the move promotes. */
  promotion?: string;
}

export interface GameSnapshot {
  state: GameState;
  fen: string;
  turn: 'w' | 'b';
  check: boolean;
  /** PGN result token: '1-0', '0-1', '1/2-1/2' or '*'. */
  result: string;
  plies: PlyInfo[];
  /** Pieces in chess square order (a1 = 0), null for empty squares. */
  pieces: Array<PieceCode | null>;
  lastMove: { from: SquareIndex; to: SquareIndex } | null;
  /** Expected occupancy in chess square order (OCC_EMPTY / OCC_WHITE / OCC_BLACK), null while waiting. */
  grid: Uint8Array | null;
  /**
   * The change the tracker is leaning towards but has not committed yet (the current challenger, while it leads the
   * incumbent), for optimistic display; null when the incumbent is best. `progress` (0..1) is how close it is to
   * committing (max of lead/threshold and dwell/hold).
   */
  pending: PendingChange | null;
  /** Timing of the last committed change (observation clock), for the debug latency timeline; null before any. */
  lastTiming: ChangeTiming | null;
  /** Debug: best hypotheses (move line from the anchor, e.g. 'e4 e5', '' for the anchor itself) and scores. */
  top: Array<{ line: string; score: number }>;
}

export type GameEvent =
  | { type: 'started' }
  | { type: 'move'; ply: PlyInfo }
  /** Tentative plies from index `fromPly` on were replaced (a correction, or a takeback when `plies` is shorter). */
  | { type: 'revised'; fromPly: number; plies: PlyInfo[] }
  | { type: 'undone' }
  | { type: 'desync' }
  | { type: 'resynced' }
  /** A confident start position after moves: the previous game was archived and a new one started. */
  | { type: 'newGame'; previousPgn: string };

export interface PendingChange {
  kind: 'advance' | 'revise' | 'takeback';
  /** Plies of the line the challenger would put in place of the tentative ones (from `fromPly` on). */
  fromPly: number;
  plies: PlyInfo[];
  progress: number;
}

export interface ChangeTiming {
  kind: 'advance' | 'revise' | 'takeback';
  /** Time of the first stable frame after the last unstable one (hand gone and settled), or null if none preceded. */
  stableAt: number | null;
  /** Time of the last unstable frame before the commit (≈ hand leaving), or null. */
  unstableEndAt: number | null;
  /** When the committed challenger became (and stayed) the favourite. */
  favouriteAt: number;
  committedAt: number;
}
