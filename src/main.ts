import {
  CameraError,
  FrameGrabber,
  listCameras,
  loadCameraId,
  loadFileSource,
  resolveCameraId,
  saveCameraId,
  startCamera,
  type CameraInfo,
  type FrameSource,
} from './camera';
import { DebugPanel, fillCameraSelect, formatGame, formatOccupancy, formatTiming, getParams, registerParam } from './debug/panel';
import { GameTracker } from './game/game';
import type { GameEvent, GameSnapshot } from './game/types';
import { isSimpleQuad } from './geom/cornerOrder';
import { PointsFilter } from './geom/oneEuro';
import { Overlay, FADE_MS, HOLD_MS, visibleFrameRect } from './overlay';
import { MoveList, statusText } from './ui/moves';
import { pendingBoard, pendingKey, pendingProgress } from './ui/pending';
import { pieceImage, preloadPieces } from './ui/pieces';
import { SoundPlayer } from './ui/sound';
import { archivePgn, loadMuted, loadSavedGame, saveGame, saveMuted, unarchiveLast } from './ui/storage';
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
preloadPieces();
overlay.setPieceImages(pieceImage);
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
const modeCounts = { full: 0, tracking: 0 };

// The worker keeps corners[k] = board corner k across frames (orientation is stabilised in TrackingSession), so
// the quad is only smoothed here, never reordered: reordering would misalign the occupancy grid.
const tracker = {
  filter: new PointsFilter(4, 1.0, 0.02),
  grid: null as Uint8Array | null,
  prob: null as Float32Array | null,
  orientation: null as Uint8Array | null,
};
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
  profileText = describeProfile(r.profile ?? null);
  modeCounts[r.mode]++;
  lastFrameW = r.width;
  lastFrameH = r.height;
  // Test hook (e2e): shape of the last result's occupancy field.
  const occ = r.occupancy;
  app.dataset.occupancy = occ === undefined ? 'missing' : occ === null ? 'null' : occ instanceof Uint8Array ? `u8:${occ.length}` : 'invalid';
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
  if (r.observation) {
    obsTime = r.observation.t;
    framed = r.observation.framed;
    handleGameEvents(game.observe(r.observation));
  }
  if (r.corners) {
    // Reset smoothing after the quad has fully faded so stale state does not drag the new one.
    if (overlay.msSinceQuad(now) > HOLD_MS + FADE_MS) {
      tracker.filter.reset();
      tracker.grid = null;
      tracker.prob = null;
      tracker.orientation = null;
    }
    // Never draw a bow-tie: drop the result (the previous quad and grid are held / fade).
    if (isSimpleQuad(r.corners)) {
      // A null occupancy (dropped frame) keeps showing the last committed grid.
      if (r.occupancy) tracker.grid = r.occupancy;
      // Confidence of the committed classes, also sent on dropped frames; a null keeps the previous one.
      if (r.occupancyProb) tracker.prob = r.occupancyProb;
      // Orientation (square -> cell) is held with the quad like the grid; a null result keeps the previous one.
      if (r.orientation) tracker.orientation = r.orientation;
      overlay.setQuad(tracker.filter.filter(r.corners, now / 1000), r.width, r.height, now, tracker.grid, tracker.grid ? tracker.prob : null, tracker.orientation);
    }
  }
}

// ---- Game (milestone 9): runs on the main thread, fed by the worker's observations. ----

const game = new GameTracker();
/** Latest observation time: the clock of PlyInfo.correctedAt (worker performance.now()). */
let obsTime = -Infinity;
let snapshot: GameSnapshot = game.snapshot();
/** Last game.save() written to storage: the state to restore when an automatic new game is undone. */
let savedGame = '';
/** Last grid sent to the worker as a position hint ('' = none sent yet). */
let hintKey = '';
const gameChip = $('gameChip');
/** Whether the last observation had the whole board in view; while not, the chip asks to keep it in view. */
let framed = true;
const FRAMING_TEXT = 'Keep the whole board in view';

const sound = new SoundPlayer(loadMuted());

const moves = new MoveList($('moves'), {
  copy: () => void copyPgn(),
  undo: () => handleGameEvents(game.undo(), true),
  newGame: () => {
    const prevSave = game.save();
    const prevPgn = game.pgn(pgnHeaders());
    archivePgn(prevPgn);
    game.newGame();
    worker.postMessage({ type: 'resetProfile' });
    hintKey = '';
    profileText = describeProfile(null);
    handleGameEvents([], true);
    offerUndoNewGame(prevSave, prevPgn, 'New game');
  },
  continueAfterDesync: () => {
    game.continueAfterDesync();
    handleGameEvents([], true);
  },
  promote: (ply, piece) => {
    if (!game.setPromotion(ply, piece)) showNotice('That promotion is not possible here.');
    handleGameEvents([], true);
  },
  toggleMute: () => {
    sound.muted = !sound.muted;
    saveMuted(sound.muted);
    // A tap is a user gesture: the moment to (re)create the audio context if Start did not manage to.
    if (!sound.muted) sound.unlock();
    return sound.muted;
  },
});
moves.setMuted(sound.muted);

function pgnHeaders(): Record<string, string> {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return { Date: `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}` };
}

async function copyPgn(): Promise<void> {
  const pgn = game.pgn(pgnHeaders());
  try {
    await navigator.clipboard.writeText(pgn);
    showNotice('PGN copied');
  } catch {
    moves.showPgn(pgn);
    showNotice('Copy the selected PGN');
  }
}

/** Toast after a new game (automatic or from the button) with an Undo that restores the previous game. */
function offerUndoNewGame(prevSave: string, prevPgn: string, text: string): void {
  showNotice(text, 8000, {
    label: 'Undo',
    run: () => {
      if (prevSave && game.load(prevSave)) {
        unarchiveLast(prevPgn);
        handleGameEvents([], true);
      } else {
        showNotice('Could not restore the previous game.');
      }
    },
  });
}

/** Grid hint for the worker: the game's expected occupancy while playing, else none. */
function sendHint(): void {
  const grid = snapshot.state === 'playing' ? snapshot.grid : null;
  const key = grid ? grid.join('') : 'null';
  if (key === hintKey) return;
  hintKey = key;
  worker.postMessage({ type: 'positionHint', grid });
}

/**
 * Applies the effects of game events (or of a UI command when `changed`): refreshes the snapshot, the sidebar, the
 * overlay and the position hint, and persists the game.
 */
function handleGameEvents(events: GameEvent[], changed = false): void {
  const prevState = snapshot.state;
  if (events.length === 0 && !changed) {
    // Nothing committed, but snapshots also carry the pending change (optimistic display) and debug scores: refresh
    // while a game is on or the panel shows them. Only the pending display is updated; the sidebar list is
    // rebuilt only when the pending line changes (its progress is a style update).
    if (panel.visible || snapshot.state !== 'waiting') {
      const prevPending = pendingKey(snapshot.pending);
      const hadPending = snapshot.pending !== null;
      snapshot = game.snapshot();
      if (hadPending || snapshot.pending) {
        if (pendingKey(snapshot.pending) !== prevPending) moves.render(snapshot, obsTime);
        else moves.setPendingProgress(pendingProgress(snapshot.pending));
        renderOverlayGame();
      }
    }
    return;
  }
  const prevSave = savedGame;
  snapshot = game.snapshot();
  sound.playFor(events);
  for (const e of events) {
    if (e.type === 'newGame') {
      archivePgn(e.previousPgn);
      offerUndoNewGame(prevSave, e.previousPgn, 'New game started');
    }
  }
  savedGame = game.save();
  saveGame(savedGame);
  if (snapshot.state !== prevState) console.debug(`game: ${prevState} -> ${snapshot.state}`);
  renderGame();
}

function renderGame(): void {
  app.dataset.game = snapshot.state;
  moves.render(snapshot, obsTime);
  renderOverlayGame();
  renderChip();
  sendHint();
}

function renderOverlayGame(): void {
  const s = snapshot;
  overlay.setGame(s.state === 'waiting' ? null : { pieces: s.pieces, lastMove: s.lastMove, pending: pendingBoard(s), pendingProgress: pendingProgress(s.pending) });
}

/** Status chip text: the framing request while the board is not wholly in view, else the game status. */
function renderChip(): void {
  const text = framed ? statusText(snapshot) : FRAMING_TEXT;
  if (gameChip.textContent !== text) gameChip.textContent = text;
}

{
  const saved = loadSavedGame();
  if (saved && !game.load(saved)) saveGame('');
  snapshot = game.snapshot();
  savedGame = saved && snapshot.state !== 'waiting' ? saved : '';
  renderGame();
}

function occupancyText(): string {
  const rate = occDrops.length ? occDrops.filter(([, d]) => d).length / occDrops.length : null;
  return formatOccupancy(occStats, rate, tracker.grid ? tracker.prob : null);
}

function showError(text: string): void {
  stopSource();
  message.textContent = text;
  startScreen.hidden = false;
  running = false;
  updateLoading();
  setStatus('error');
  moves.root.hidden = true;
  gameChip.hidden = true;
}

let notices = 0;
function showNotice(text: string, ms = 4000, action?: { label: string; run: () => void }): void {
  const n = $('notice');
  n.textContent = text;
  if (action) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = action.label;
    b.addEventListener('click', () => {
      n.hidden = true;
      action.run();
    });
    n.append(b);
  }
  n.hidden = false;
  const k = ++notices;
  setTimeout(() => k === notices && (n.hidden = true), ms);
}

function stopSource(): void {
  source?.stop();
  source = null;
  sourceKind = 'none';
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
  overlay.clearQuad();
  tracker.filter.reset();
  tracker.grid = null;
  tracker.prob = null;
  tracker.orientation = null;
  // Only vision resets on a source change; the game carries on (re-sent as a hint to the fresh tracker).
  worker.postMessage({ type: 'reset' });
  hintKey = '';
  sendHint();
  moves.root.hidden = false;
  framed = true;
  modeCounts.full = modeCounts.tracking = 0;
  occStats = undefined;
  occDrops.length = 0;
  updateLoading();
  setStatus('running');
}

// Camera choice. `cameras` is empty until the browser exposes device ids (after the first permission grant).
let cameras: CameraInfo[] = [];
let activeCameraId: string | null = null;
let sourceKind: 'none' | 'camera' | 'file' = 'none';
let switching = false;
const cameraPick = $('cameraPick');
const cameraSelect = $<HTMLSelectElement>('cameraSelect');

function renderCameraPickers(): void {
  const startId = activeCameraId ?? resolveCameraId(loadCameraId(), cameras);
  fillCameraSelect(cameraSelect, cameras, startId);
  // Nothing chosen yet: say so rather than implying the first listed camera is the one that will open.
  if (!startId || !cameras.some((c) => c.deviceId === startId)) cameraSelect.prepend(new Option('Default (rear)', '', true, true));
  cameraPick.hidden = cameras.length < 2;
  panel.setCameras(cameras, activeCameraId, sourceKind === 'camera');
  app.dataset.cameras = String(cameras.length);
}

async function refreshCameras(): Promise<void> {
  cameras = await listCameras();
  // Drop a stored id that no longer exists (ids can change, e.g. after clearing site data).
  const stored = loadCameraId();
  if (stored && cameras.length > 0 && resolveCameraId(stored, cameras) === null) saveCameraId(null);
  renderCameraPickers();
}

/** Starts the camera `requested` (stored/default when undefined); `explicit` when the user just picked it. */
async function startCameraSource(requested: string | null | undefined, explicit: boolean, fullscreen: boolean): Promise<void> {
  const id = requested === undefined ? resolveCameraId(loadCameraId(), cameras) : requested;
  await begin(async () => {
    const cam = await startCamera(id);
    activeCameraId = cam.deviceId;
    sourceKind = 'camera';
    if (cam.fellBack) {
      saveCameraId(null);
      if (explicit) showNotice('That camera could not be opened; using the default camera.');
    } else if (explicit) {
      saveCameraId(cam.deviceId ?? id);
    }
    return cam;
  }, fullscreen);
  await refreshCameras();
}

$('startBtn').addEventListener('click', () => {
  // The tap is the user gesture mobile browsers require before any audio.
  sound.unlock();
  void startCameraSource(undefined, false, true);
});
// On the start screen the choice only takes effect when Start is tapped.
cameraSelect.addEventListener('change', () => saveCameraId(cameraSelect.value || null));
panel.onCameraChange((id) => {
  if (switching || sourceKind !== 'camera' || id === activeCameraId) return;
  switching = true;
  void startCameraSource(id, true, false).finally(() => (switching = false));
});
navigator.mediaDevices?.addEventListener?.('devicechange', () => void refreshCameras());
void refreshCameras();
$<HTMLInputElement>('fileInput').addEventListener('change', (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0];
  if (!file) return;
  void begin(async () => {
    const src = await loadFileSource(file);
    sourceKind = 'file';
    activeCameraId = null;
    return src;
  }, false).then(renderCameraPickers);
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
    // What the user sees of the frame (object-fit: cover). The sidebar is not subtracted: the board stays visible
    // through it.
    visibleRect: visibleFrameRect(frame.width, frame.height, window.innerWidth, window.innerHeight),
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
    overlay.setPlayMode(!panel.visible);
    // Play mode only: while waiting for the starting position, or (priority) while the board is not wholly in view.
    renderChip();
    gameChip.hidden = panel.visible || (framed && snapshot.state !== 'waiting');
    overlay.draw(now, lastFrameW || source.width, lastFrameH || source.height);
    panel.update({ fps, detectionsPerSec: detTimes.length, timings: lastTimings, confidence, profile: profileText, occupancy: occupancyText(), game: formatGame(snapshot, framed), latency: formatTiming(snapshot.lastTiming), mode: `${mode} (tracked ${Math.round((100 * modeCounts.tracking) / Math.max(1, modeCounts.full + modeCounts.tracking))}%)` });
  }
  requestAnimationFrame(loop);
}

if (!window.isSecureContext) message.textContent = 'The camera needs a secure connection (HTTPS or localhost).';
setStatus('idle');
requestAnimationFrame(loop);
