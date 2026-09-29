export type DebugView = 'none' | 'gradient' | 'edges' | 'lines' | 'rectified' | 'verify' | 'occupancy';
export const DEBUG_VIEWS: readonly DebugView[] = ['none', 'gradient', 'edges', 'lines', 'rectified', 'verify', 'occupancy'];

/** Slider values registered in the debug panel, keyed by parameter name. */
export type Params = Record<string, number>;

import type { BoardProfile } from '../vision/profile';

export type Corner = [number, number];
export type Corners = [Corner, Corner, Corner, Corner];

/** Main thread -> worker. Provide either `bitmap` (transferred) or `imageData`. */
export interface FrameMessage {
  type: 'frame';
  id: number;
  bitmap?: ImageBitmap;
  imageData?: ImageData;
  width: number;
  height: number;
  params: Params;
  debugView: DebugView;
}

/** Drops the locked board profile (the "Reset board" button). */
export interface ResetProfileMessage {
  type: 'resetProfile';
}

/** Source changed: drop the tracked board. */
export interface ResetMessage {
  type: 'reset';
}

/**
 * Game -> vision feedback (milestone 9). `grid` is the game's current occupancy (64 entries in CHESS square order,
 * see Observation), or null when no game is being played (waiting for the start position, desync, game layer off).
 * While a grid is set, vision uses it as its committed occupancy (occluder prior, illumination / learning labels)
 * instead of its own per-cell filter, and uses it to re-orient the board after a long loss.
 */
export interface PositionHintMessage {
  type: 'positionHint';
  grid: Uint8Array | null;
}

export type MainToWorker = FrameMessage | ResetProfileMessage | ResetMessage | PositionHintMessage;

export interface ReadyMessage {
  type: 'ready';
}

export interface ErrorMessage {
  type: 'error';
  message: string;
  /** Set when the error belongs to a frame; the frame is then considered finished. */
  id?: number;
}

/** Worker -> main. `corners` are in coordinates of the processed frame (width x height). */
export interface ResultMessage {
  type: 'result';
  id: number;
  width: number;
  height: number;
  corners: Corners | null;
  confidence: number;
  mode: 'full' | 'tracking';
  /** Milliseconds per stage, plus 'total'. */
  timings: Record<string, number>;
  /** Board profile locked in the worker after this frame, or null while auto-detecting. */
  profile?: BoardProfile | null;
  /** Frame-sized RGBA image for the selected debug view (transferred). */
  debugImage?: ImageBitmap;
  /**
   * Committed square occupancy (64 entries, row-major in board cell coordinates: cell (i, j) at index j * 8 + i,
   * where corners[0] is board point (0,0), corners[1] (8,0), corners[2] (8,8), corners[3] (0,8)).
   * Values: OCC_EMPTY 0, OCC_WHITE 1, OCC_BLACK 2. Null when this frame's occupancy was dropped (low confidence,
   * hands, no board); the main thread then keeps showing the last grid.
   */
  occupancy?: Uint8Array | null;
  /**
   * Per cell (same indexing as `occupancy`), this frame's calibrated probability of the cell's committed class
   * (exp of the classifier log-likelihood). Sent whenever the tracker is calibrated and a board is present, even on
   * frames where `occupancy` is dropped (so low-confidence cells are visible). Null otherwise; the main thread then
   * keeps the last array with its grid.
   */
  occupancyProb?: Float32Array | null;
  occupancyStats?: OccupancyStats;
  /** Vision -> game hook (milestone 9): this frame's occupancy evidence, whenever a board is present and occupancy ran. */
  observation?: Observation | null;
  /**
   * Board orientation found by vision: orientation[sq] = board cell index (j * 8 + i, same indexing as `occupancy`)
   * of chess square sq (a1 = 0, b1 = 1, ..., h8 = 63). Null while unknown. Lets the UI draw game pieces on the quad.
   */
  orientation?: Uint8Array | null;
}

/**
 * Chess square index: file + 8 * rank, a1 = 0, h1 = 7, a8 = 56, h8 = 63. Every array that the game layer sees is in
 * this order; vision never exposes board-cell order to the game once oriented.
 */
export type SquareIndex = number;

/** One frame of occupancy evidence pushed from vision to the game layer. */
export interface Observation {
  /** Frame time (ms, worker performance.now()). */
  t: number;
  /**
   * Per-square class log-probabilities, square sq's (empty, white, black) at sq * 3 + (OCC_EMPTY, OCC_WHITE,
   * OCC_BLACK), normalised per square. In chess square order when `oriented`, otherwise in board cell order (then
   * the game must not use it except to detect the starting position's presence).
   */
  logLik: Float32Array;
  /** Per-square visible fraction of the sampling footprint (0..1), same order as logLik. */
  vis: Float32Array;
  /** Whether vision knows which cell is a1 (the arrays are then in chess square order). */
  oriented: boolean;
  /** Occupancy calibration state (OccupancyStats.state); the game only locks in on 'calibrated' (from the start position). */
  calibration: OccupancyStats['state'];
  /** False on frames that must not count as evidence: dropped, frozen (hand / mass change), or just after a board loss. */
  stable: boolean;
}

export const OCC_EMPTY = 0;
export const OCC_WHITE = 1;
export const OCC_BLACK = 2;

export interface OccupancyStats {
  /** 'start' = waiting for the starting position to calibrate; 'calibrated' = models learned from it; 'fallback' = unsupervised. */
  state: 'start' | 'calibrated' | 'fallback';
  empty: number;
  white: number;
  black: number;
  /** Cells whose confidence was below the per-cell minimum on this frame (not updated). */
  lowCells: number;
  /** Whether this frame's occupancy evidence was dropped as a whole. */
  dropped: boolean;
  /** Whether updates are frozen (mass change, e.g. a hand over the board). */
  frozen: boolean;
}

export type WorkerToMain = ReadyMessage | ErrorMessage | ResultMessage;
