/// <reference lib="webworker" />
import opencv from '@techstark/opencv-js';
import type { CV } from '../vision/preprocess';
import { Preprocessor } from '../vision/preprocess';
import type { DebugView, FrameMessage, ResultMessage, WorkerToMain } from './protocol';

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
  pre: Preprocessor;
  rgba: InstanceType<CV['Mat']> | null;
  rgbaOut: InstanceType<CV['Mat']>;
}

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

  const pre = s.pre.run(s.rgba, msg.params);
  Object.assign(timings, pre.timings);

  t = performance.now();
  let debugImage: ImageBitmap | undefined;
  const view: DebugView = msg.debugView;
  if (view === 'raw') {
    debugImage = await toBitmap(new Uint8ClampedArray(img.data) as Uint8ClampedArray<ArrayBuffer>, width, height);
  } else if (view === 'gradient' || view === 'edges') {
    cv.cvtColor(view === 'gradient' ? pre.gradient : pre.edges, s.rgbaOut, cv.COLOR_GRAY2RGBA);
    debugImage = await toBitmap(new Uint8ClampedArray(s.rgbaOut.data) as Uint8ClampedArray<ArrayBuffer>, width, height);
  }
  timings.debug = performance.now() - t;
  timings.total = performance.now() - total;

  const result: ResultMessage = {
    type: 'result',
    id: msg.id,
    width,
    height,
    corners: null,
    confidence: 0,
    mode: 'full',
    timings,
    debugImage,
  };
  post(result, debugImage ? [debugImage] : []);
}

let queue: Promise<void> = Promise.resolve();

scope.onmessage = (ev: MessageEvent<FrameMessage>) => {
  const msg = ev.data;
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
    state = {
      cv,
      pre: new Preprocessor(cv),
      rgba: null,
      rgbaOut: new cv.Mat(),
    };
    post({ type: 'ready' });
  },
  (e: unknown) => post({ type: 'error', message: `OpenCV failed to load: ${e instanceof Error ? e.message : String(e)}` }),
);
