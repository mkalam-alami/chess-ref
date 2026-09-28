export type DebugView = 'raw' | 'gradient' | 'edges' | 'lines' | 'rectified' | 'verify';
export const DEBUG_VIEWS: readonly DebugView[] = ['raw', 'gradient', 'edges', 'lines', 'rectified', 'verify'];

/** Slider values registered in the debug panel, keyed by parameter name. */
export type Params = Record<string, number>;

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

export type MainToWorker = FrameMessage;

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
  /** Frame-sized RGBA image for the selected debug view (transferred). */
  debugImage?: ImageBitmap;
}

export type WorkerToMain = ReadyMessage | ErrorMessage | ResultMessage;
