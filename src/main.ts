import { CameraError, FrameGrabber, loadFileSource, startCamera, type FrameSource } from './camera';
import { DebugPanel, formatOccupancy, getParams, registerParam } from './debug/panel';
import { isSimpleQuad } from './geom/cornerOrder';
import { PointsFilter } from './geom/oneEuro';
import { Overlay, FADE_MS, HOLD_MS } from './overlay';
import { ALL_PARAMS } from './vision/detector';
import { describeProfile } from './vision/profile';
import type { FrameMessage, OccupancyStats, ResultMessage, WorkerToMain } from './worker/protocol';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const app = $('app');
const startScreen = $('start');
const message = $('message');
const loading = $('loading');

const setStatus = (s: string) => (app.dataset.status = s);
const setWorker = (s: string) => (app.dataset.worker = s);

for (const p of ALL_PARAMS) registerParam(p.name, p.min, p.max, p.step, p.default);

const overlay = new Overlay($<HTMLCanvasElement>('overlay'));
const panel = new DebugPanel(app);
const grabber = new FrameGrabber(panel.resolution);
panel.onResolutionChange((r) => (grabber.longSide = r));

// Detector worker: starts loading OpenCV immediately so it is warm by the time the user taps Start.
const worker = new Worker(new URL('./worker/detector.worker.ts', import.meta.url), { type: 'module' });
let workerReady = false;
let running = false;
setWorker('loading');

const updateLoading = () => {
  loading.hidden = !running || workerReady;
};

let source: FrameSource | null = null;
let frameId = 0;
let inFlightId = -1;
let lastFrameW = 0;
let lastFrameH = 0;
let lastTimings: Record<string, number> = {};
let confidence = 0;
let mode = 'full';
let profileText = describeProfile(null);
const resetBoardBtn = $<HTMLButtonElement>('resetBoard');
resetBoardBtn.addEventListener('click', () => {
  worker.postMessage({ type: 'resetProfile' });
  resetBoardBtn.hidden = true;
  profileText = describeProfile(null);
});
const modeCounts = { full: 0, tracking: 0 };

// The worker keeps corners[k] = board corner k across frames (orientation is stabilised in TrackingSession), so
// the quad is only smoothed here, never reordered: reordering would misalign the occupancy grid.
const tracker = { filter: new PointsFilter(4, 1.0, 0.02), grid: null as Uint8Array | null };
let occStats: OccupancyStats | undefined;
/** (time, dropped) per result carrying occupancy stats, over the last ~2 s. */
const occDrops: Array<[number, boolean]> = [];
const OCC_DROP_WINDOW_MS = 2000;
const detTimes: number[] = [];
let fps = 0;
let frames = 0;
let fpsStart = performance.now();

worker.onmessage = (ev: MessageEvent<WorkerToMain>) => {
  const msg = ev.data;
  if (msg.type === 'ready') {
    workerReady = true;
    setWorker('ready');
    updateLoading();
  } else if (msg.type === 'error') {
    console.error('worker:', msg.message);
    if (msg.id === undefined) {
      setWorker('error');
      showError(msg.message);
    } else {
      grabber.release();
    }
  } else {
    handleResult(msg);
  }
};
worker.onerror = (e) => {
  setWorker('error');
  showError(`Worker error: ${e.message}`);
};

function handleResult(r: ResultMessage): void {
  if (r.id === inFlightId) grabber.release();
  const now = performance.now();
  detTimes.push(now);
  lastTimings = r.timings;
  confidence = r.confidence;
  mode = r.mode;
  resetBoardBtn.hidden = !r.profile;
  profileText = describeProfile(r.profile ?? null);
  modeCounts[r.mode]++;
  lastFrameW = r.width;
  lastFrameH = r.height;
  if (r.occupancyStats) {
    occStats = r.occupancyStats;
    occDrops.push([now, r.occupancyStats.dropped]);
  }
  // A debug image is only ever shown while the panel is open and a real view is selected; otherwise it would
  // be a lagging copy of the video drawn over the live stream (ghosting).
  if (r.debugImage && panel.visible && panel.view !== 'none') overlay.setDebugImage(r.debugImage);
  else {
    r.debugImage?.close();
    overlay.setDebugImage(null);
  }
  if (r.corners) {
    // Reset smoothing after the quad has fully faded so stale state does not drag the new one.
    if (overlay.msSinceQuad(now) > HOLD_MS + FADE_MS) {
      tracker.filter.reset();
      tracker.grid = null;
    }
    // Never draw a bow-tie: drop the result (the previous quad and grid are held / fade).
    if (isSimpleQuad(r.corners)) {
      // A null occupancy (dropped frame) keeps showing the last committed grid.
      if (r.occupancy) tracker.grid = r.occupancy;
      overlay.setQuad(tracker.filter.filter(r.corners, now / 1000), r.width, r.height, now, tracker.grid);
    }
  }
}

function occupancyText(): string {
  const rate = occDrops.length ? occDrops.filter(([, d]) => d).length / occDrops.length : null;
  return formatOccupancy(occStats, rate);
}

function showError(text: string): void {
  stopSource();
  message.textContent = text;
  startScreen.hidden = false;
  running = false;
  updateLoading();
  setStatus('error');
}

function stopSource(): void {
  source?.stop();
  source = null;
}

async function enterFullscreen(): Promise<void> {
  try {
    await document.documentElement.requestFullscreen();
  } catch {
    /* not available or refused */
  }
  try {
    await (screen.orientation as ScreenOrientation & { lock(o: string): Promise<void> }).lock('landscape');
  } catch {
    /* unsupported outside fullscreen or on desktop */
  }
}

async function begin(open: () => Promise<FrameSource>, fullscreen: boolean): Promise<void> {
  message.textContent = '';
  setStatus('starting');
  if (fullscreen) await enterFullscreen();
  try {
    stopSource();
    source = await open();
  } catch (e) {
    showError(e instanceof CameraError ? e.message : `Could not start: ${e instanceof Error ? e.message : String(e)}`);
    return;
  }
  app.prepend(source.element);
  startScreen.hidden = true;
  running = true;
  overlay.setDebugImage(null);
  worker.postMessage({ type: 'reset' });
  modeCounts.full = modeCounts.tracking = 0;
  occStats = undefined;
  occDrops.length = 0;
  updateLoading();
  setStatus('running');
}

$('startBtn').addEventListener('click', () => void begin(startCamera, true));
$<HTMLInputElement>('fileInput').addEventListener('change', (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0];
  if (file) void begin(() => loadFileSource(file), false);
});

async function pump(): Promise<void> {
  if (!source || !workerReady || grabber.isBusy) return;
  const frame = await grabber.grab(source);
  if (!frame) return;
  const msg: FrameMessage = {
    type: 'frame',
    id: ++frameId,
    bitmap: frame.bitmap,
    width: frame.width,
    height: frame.height,
    params: getParams(),
    debugView: panel.visible ? panel.view : 'none',
  };
  inFlightId = msg.id;
  worker.postMessage(msg, [frame.bitmap]);
}

function loop(now: number): void {
  frames++;
  if (now - fpsStart >= 1000) {
    fps = (frames * 1000) / (now - fpsStart);
    frames = 0;
    fpsStart = now;
  }
  while (detTimes.length && now - detTimes[0]! > 1000) detTimes.shift();
  while (occDrops.length && now - occDrops[0]![0] > OCC_DROP_WINDOW_MS) occDrops.shift();

  if (running && source) {
    void pump();
    overlay.draw(now, lastFrameW || source.width, lastFrameH || source.height);
    panel.update({ fps, detectionsPerSec: detTimes.length, timings: lastTimings, confidence, profile: profileText, occupancy: occupancyText(), mode: `${mode} (tracked ${Math.round((100 * modeCounts.tracking) / Math.max(1, modeCounts.full + modeCounts.tracking))}%)` });
  }
  requestAnimationFrame(loop);
}

if (!window.isSecureContext) message.textContent = 'The camera needs a secure connection (HTTPS or localhost).';
setStatus('idle');
requestAnimationFrame(loop);
