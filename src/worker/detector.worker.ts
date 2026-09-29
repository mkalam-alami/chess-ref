/// <reference lib="webworker" />
import opencv from '@techstark/opencv-js';
import type { CV } from '../vision/preprocess';
import { Detector } from '../vision/detector';
import { drawLines, drawVerify, rectifiedView } from '../vision/debugViews';
import { ProfileLock } from '../vision/profile';
import { drawOccupancy, OCCUPANCY_PARAMS, OccupancyTracker } from '../vision/occupancy';
import { param } from '../vision/preprocess';
import { TrackingSession } from './tracker';
import type { DebugView, FrameMessage, MainToWorker, Observation, OccupancyStats, ResultMessage, WorkerToMain } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;

function post(msg: WorkerToMain, transfer: Transferable[] = []): void {
  scope.postMessage(msg, transfer);
}

interface Runtime {
  Mat?: unknown;
  then?: unknown;
  onRuntimeInitialized?: () => void;
}

/**
 * @techstark/opencv-js exports either a ready module, a promise, or a module that fires
 * onRuntimeInitialized. Statically imported (a dynamic import of the promise-valued CJS
 * export would try to unwrap it as a thenable); the result is boxed so returning it from
 * an async function never unwraps a thenable module by accident.
 */
async function loadOpenCV(): Promise<{ cv: CV }> {
  const exported = opencv as unknown as Runtime;
  if (exported.Mat) return { cv: exported as unknown as CV };
  if (typeof exported.then === 'function') {
    try {
      const resolved = (await (exported as unknown as Promise<Runtime>)) as Runtime;
      if (resolved?.Mat) return { cv: resolved as unknown as CV };
    } catch (e) {
      console.warn('opencv promise unwrap failed, waiting for runtime callback', e);
    }
  }
  if (!exported.Mat) {
    await new Promise<void>((resolve) => {
      const prev = exported.onRuntimeInitialized;
      exported.onRuntimeInitialized = () => {
        prev?.();
        resolve();
      };
    });
  }
  return { cv: exported as unknown as CV };
}

interface State {
  cv: CV;
  detector: Detector;
  session: TrackingSession;
  rgba: InstanceType<CV['Mat']> | null;
  rgbaOut: InstanceType<CV['Mat']>;
}

// --- board profile lock (agent B) ---
const profileLock = new ProfileLock();
// --- square occupancy (milestone 8): models and temporal filter live next to the profile lock ---
// The class models and the orientation survive board losses (the orientation is only re-verified after one); only
// the reset / resetProfile messages drop them. The game's position hint is kept across both.
const occupancy = new OccupancyTracker();
let state: State | null = null;
let canvas: OffscreenCanvas | null = null;

function frameToImageData(msg: FrameMessage): ImageData {
  if (msg.imageData) return msg.imageData;
  if (!msg.bitmap) throw new Error('frame has neither bitmap nor imageData');
  if (!canvas || canvas.width !== msg.width || canvas.height !== msg.height) {
    canvas = new OffscreenCanvas(msg.width, msg.height);
  }
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(msg.bitmap, 0, 0, msg.width, msg.height);
  return ctx.getImageData(0, 0, msg.width, msg.height);
}

async function toBitmap(data: Uint8ClampedArray<ArrayBuffer>, w: number, h: number): Promise<ImageBitmap> {
  return createImageBitmap(new ImageData(data, w, h));
}

async function process(msg: FrameMessage): Promise<void> {
  const s = state!;
  const cv = s.cv;
  const total = performance.now();
  const timings: Record<string, number> = {};
  const { width, height } = msg;

  let t = performance.now();
  const img = frameToImageData(msg);
  msg.bitmap?.close();
  if (!s.rgba || s.rgba.cols !== width || s.rgba.rows !== height) {
    s.rgba?.delete();
    s.rgba = new cv.Mat(height, width, cv.CV_8UC4);
  }
  s.rgba.data.set(img.data);
  timings.decode = performance.now() - t;

  const view: DebugView = msg.debugView;
  const wantDebug = view === 'lines' || view === 'rectified' || view === 'verify';
  // Debug views need the full pipeline's intermediate images, so they disable tracking.
  const params = view === 'none' ? msg.params : { ...msg.params, tracking: 0 };
  const det = s.session.process(s.rgba, width, height, params, { debug: wantDebug, profile: profileLock.profile });
  const profile = profileLock.update(det, performance.now(), msg.params);
  Object.assign(timings, det.timings);
  const pre = s.detector.lastPre!;

  // Occupancy samples the raw frame (not the CLAHE / L-only preprocessor output).
  t = performance.now();
  let grid: Uint8Array | null = null;
  let occProb: Float32Array | null = null;
  let occStats: OccupancyStats | undefined;
  let observation: Observation | null = null;
  let orientation: Uint8Array | null = null;
  if (param(msg.params, OCCUPANCY_PARAMS, 'occupancy') > 0) {
    if (det.hb) {
      const occ = occupancy.update(img, det.hb, msg.params, performance.now());
      grid = occ.grid;
      occProb = occ.committedProb;
      occStats = occ.stats;
      observation = occ.observation;
      orientation = occ.orientation;
    } else occStats = occupancy.noBoard(performance.now());
  }
  timings.occupancy = performance.now() - t;

  t = performance.now();
  let debugImage: ImageBitmap | undefined;
  if (view === 'gradient' || view === 'edges') {
    cv.cvtColor(view === 'gradient' ? pre.gradient : pre.edges, s.rgbaOut, cv.COLOR_GRAY2RGBA);
    debugImage = await toBitmap(new Uint8ClampedArray(s.rgbaOut.data) as Uint8ClampedArray<ArrayBuffer>, width, height);
  } else if (view === 'occupancy') {
    const c = new OffscreenCanvas(width, height);
    const ctx = c.getContext('2d')!;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(img.data), width, height), 0, 0);
    if (det.hb && occupancy.debug) drawOccupancy(ctx, occupancy.debug);
    debugImage = await createImageBitmap(c);
  } else if (wantDebug && det.debug) {
    const c = new OffscreenCanvas(width, height);
    const ctx = c.getContext('2d')!;
    if (view === 'rectified') {
      const rv = rectifiedView(det.debug, width, height);
      if (rv) {
        const hm = cv.matFromArray(3, 3, cv.CV_64F, rv.h);
        try {
          cv.warpPerspective(pre.l, s.rgbaOut, hm, new cv.Size(width, height), cv.INTER_LINEAR);
          cv.cvtColor(s.rgbaOut, s.rgbaOut, cv.COLOR_GRAY2RGBA);
        } finally {
          hm.delete();
        }
        ctx.putImageData(new ImageData(new Uint8ClampedArray(s.rgbaOut.data), width, height), 0, 0);
        ctx.strokeStyle = 'rgba(255,60,60,0.8)';
        ctx.lineWidth = 1;
        for (const [a, b] of rv.lines) {
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
      }
    } else {
      ctx.putImageData(new ImageData(new Uint8ClampedArray(img.data), width, height), 0, 0);
      if (view === 'lines') drawLines(ctx, det.debug);
      else drawVerify(ctx, det.debug);
    }
    debugImage = await createImageBitmap(c);
  }
  timings.debug = performance.now() - t;
  timings.total = performance.now() - total;

  const result: ResultMessage = {
    type: 'result',
    id: msg.id,
    width,
    height,
    corners: det.corners,
    confidence: det.confidence,
    mode: det.mode,
    timings,
    profile,
    debugImage,
    occupancy: grid,
    occupancyProb: occProb,
    occupancyStats: occStats,
    observation,
    orientation,
  };
  post(result, debugImage ? [debugImage] : []);
}

let queue: Promise<void> = Promise.resolve();

scope.onmessage = (ev: MessageEvent<MainToWorker>) => {
  const msg = ev.data;
  if (msg.type === 'reset') {
    state?.session.reset();
    occupancy.reset();
    return;
  }
  if (msg.type === 'resetProfile') {
    profileLock.reset();
    occupancy.reset();
    return;
  }
  if (msg.type === 'positionHint') {
    // Applied from the next processed frame on (one frame of latency, see docs/PLAN-game.md).
    occupancy.setPosition(msg.grid);
    return;
  }
  if (msg.type !== 'frame' || !state) return;
  queue = queue.then(() =>
    process(msg).catch((e: unknown) => {
      msg.bitmap?.close();
      post({ type: 'error', id: msg.id, message: e instanceof Error ? e.message : String(e) });
    }),
  );
};

loadOpenCV().then(
  ({ cv }) => {
    const detector = new Detector(cv);
    state = {
      cv,
      detector,
      session: new TrackingSession(detector),
      rgba: null,
      rgbaOut: new cv.Mat(),
    };
    post({ type: 'ready' });
  },
  (e: unknown) => post({ type: 'error', message: `OpenCV failed to load: ${e instanceof Error ? e.message : String(e)}` }),
);
