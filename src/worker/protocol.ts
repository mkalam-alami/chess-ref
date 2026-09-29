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

export type MainToWorker = FrameMessage | ResetProfileMessage | ResetMessage;

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
