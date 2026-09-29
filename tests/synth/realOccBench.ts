/*
 * Real-photo occupancy bench: `BENCH_ONLY=realocc npm run bench`.
 *
 * For every starting-position photo in the fixture directories (JPGs + a corners.json in the format of
 * tests/fixtures/real/corners.json: 4 outer corners of the 64 squares, clockwise from the smallest x+y, plus
 * `whiteEdge` k = the white pieces stand along corners[k] -> corners[k+1]) it runs the app pipeline at the worker's
 * resolution (Detector.detect, then OccupancyTracker.update on the same frame at 100 ms steps until it calibrates from
 * the starting position (testStart) or falls back to the unsupervised fit after occFallbackMs, then a few more frames)
 * and reports, per image and per group:
 *   - detection: found / OK (mean corner error < 2% of the board diagonal, best of the 8 corner relabellings, as
 *     bench.ts cornerError) and the mean / max corner error in % of the diagonal;
 *   - boot: start (testStart succeeded, tracker state 'calibrated'), fallback (fitUnsupervised), none (never);
 *   - orient: the tracker's orientation (square -> cell) against the one implied by the GT corners + whiteEdge
 *     (a1 = corners[k+1], h1 = corners[k]): ok / WRONG / none (with the tracker's refusal reason);
 *   - raw per-cell accuracy of the last frame's raw classification (and the committed grid's), and a confusion
 *     matrix truth x predicted over E/W/B, split by square colour (light / dark).
 * Two runs per image: "det" (end-to-end, on the detected homography; cells are aligned with the GT like
 * occBench.alignGt, and a detection that is not a relabelling of the GT board scores as misaligned) and "gt" (the
 * tracker on the homography of the GT corners: isolates occupancy from detection).
 * Images without a GT entry are reported detection/boot only; `neg-*` files (corners: null) as false positives.
 *
 * Groups: tests/fixtures/real is "orig"; other directories take `view` (oblique / topdown) and `split`
 * (tune / holdout; "excluded" images are skipped entirely) from their SOURCES.json ([{file, view, split, ...}], or {images: [...]}, or keyed by file);
 * without SOURCES.json every image is tune with view "unknown".
 *
 * Environment:
 *   REALOCC_DIRS       comma-separated fixture dirs (relative to the repo root or absolute);
 *                      default tests/fixtures/real,tests/fixtures/web (missing dirs / corners.json are reported, not fatal)
 *   REALOCC_HOLDOUT    unset: holdout images are skipped entirely (not even decoded); 1: included; only: holdout only
 *   REALOCC_FILTER     substring of the file name
 *   REALOCC_SIZE       long side (px) the pipeline runs at (default 640, the app's camera long side)
 *   REALOCC_OUT        results JSON (default tests/synth/realocc-out/results.json; gitignored)
 *   REALOCC_BASELINE   a previous results JSON: adds before -> after delta columns per image and per group
 *   REALOCC_PNG_DIR    overlay PNG dir (default tests/synth/realocc-out/png; gitignored); REALOCC_PNG=0 disables
 *   REALOCC_PNG_SIZE   long side of the overlay PNGs (default 1600, capped at the source size)
 *   REALOCC_GRID_ONLY  1: only render the GT annotation overlays (8x8 grid from the GT corners, a1/h1/h8/a8 labels,
 *                      white edge highlighted, corner indices) for annotation review; no detection / occupancy.
 *                      Holdout images ARE included here (it evaluates nothing).
 *   BENCH_PARAMS       JSON of parameter overrides (detector and occupancy), as for the rest of the bench
 *
 * Overlays (named <dir>__<image>.<kind>.png): .det / .gt show the GT quad (green), the detected quad (red), and per
 * cell the raw predicted class (E/W/B) on a green tag when right, or "pred>truth" on a red tag when wrong;
 * .grid is the GT annotation overlay; .fp marks a detection on a negative image.
 *
 * Before/after workflow: run once with REALOCC_OUT=/tmp/before.json, change the code, then run with
 * REALOCC_BASELINE=/tmp/before.json to get per-image and per-group delta columns.
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { applyH, homographyFrom4, invert3, mul3, type Mat3, type Point } from '../../src/geom/homography';
import type { Detector } from '../../src/vision/detector';
import { OCCUPANCY_PARAMS, OccupancyTracker, START_SQUARES, startGrid, type OccupancyResult } from '../../src/vision/occupancy';
import { param, type CV } from '../../src/vision/preprocess';
import type { Params } from '../../src/worker/protocol';
import { ERR_THRESHOLD } from './bench';
import { alignGt } from './occBench';
import { encodePng } from './png';

const REPO = path.resolve(import.meta.dirname, '../..');
const DEFAULT_OUT_DIR = path.join(REPO, 'tests/synth/realocc-out');
const CLS = ['E', 'W', 'B'] as const;
const FRAME_MS = 100;
/** Frames after the tracker calibrated (the existing real-photo test scores after 5 frames in all). */
const AFTER_BOOT_FRAMES = 4;

type Split = 'tune' | 'holdout';
type Boot = 'start' | 'fallback' | 'none';
/** confusion[truth][pred], truth / pred = E, W, B. */
type Confusion = number[][];

interface ImageMeta {
  dir: string;
  file: string;
  key: string;
  group: string;
  view: string;
  split: Split;
}

export interface RunResult {
  /** Detected hb is a relabelling of the GT board (always true for the gt run). */
  aligned: boolean;
  framed: boolean;
  boot: Boot;
  /** Frame (0-based, 100 ms apart) on which the tracker left the 'start' state; -1 if never. */
  bootFrame: number;
  frames: number;
  orient: 'ok' | 'wrong' | 'none' | null;
  orientReason: string;
  /** Raw per-cell accuracy of the last frame (null without GT / not aligned). */
  rawAcc: number | null;
  /** Committed grid accuracy (null when dropped / no GT). */
  gridAcc: number | null;
  lowCells: number;
  conf: { light: Confusion; dark: Confusion } | null;
  /** Raw classes in chess square order a1..h8 (E/W/B), from the GT orientation; null without GT. */
  rawSquares: string | null;
}

export interface ImageResult extends Omit<ImageMeta, 'dir'> {
  dir: string;
  width: number;
  height: number;
  hasGt: boolean;
  negative: boolean;
  whiteEdge: number | null;
  detFound: boolean;
  detOk: boolean | null;
  errMean: number | null;
  errMax: number | null;
  det: RunResult | null;
  gt: RunResult | null;
}

// ---------------------------------------------------------------------------------------------------------------
// Fixture discovery

interface CornerEntry {
  width?: number;
  height?: number;
  corners: [number, number][] | null;
  whiteEdge?: number;
}

interface SourceEntry {
  file: string;
  view?: string;
  split?: string;
}

function readSources(dir: string): Map<string, SourceEntry> | null {
  const f = path.join(dir, 'SOURCES.json');
  if (!fs.existsSync(f)) return null;
  const j = JSON.parse(fs.readFileSync(f, 'utf8')) as unknown;
  const out = new Map<string, SourceEntry>();
  const add = (e: unknown, key?: string) => {
    if (!e || typeof e !== 'object') return;
    const o = e as Record<string, unknown>;
    const file = typeof o.file === 'string' ? o.file : key;
    if (file) out.set(path.basename(file), { ...(o as object), file } as SourceEntry);
  };
  if (Array.isArray(j)) j.forEach((e) => add(e));
  else if (j && typeof j === 'object') {
    const o = j as Record<string, unknown>;
    const arr = Object.values(o).find((v) => Array.isArray(v) && v.some((e) => e && typeof e === 'object' && 'file' in e));
    if (arr) (arr as unknown[]).forEach((e) => add(e));
    else for (const [k, v] of Object.entries(o)) if (!k.startsWith('_')) add(v, k);
  }
  return out;
}

function listImages(dirs: string[], holdout: string | undefined, filter: string | undefined, gridOnly: boolean, log: string[]): { metas: ImageMeta[]; corners: Map<string, Record<string, CornerEntry>> } {
  const metas: ImageMeta[] = [];
  const corners = new Map<string, Record<string, CornerEntry>>();
  for (const d of dirs) {
    const dir = path.resolve(REPO, d);
    const rel = path.relative(REPO, dir);
    if (!fs.existsSync(dir)) {
      log.push(`[skip] ${rel}: directory not found`);
      continue;
    }
    const cf = path.join(dir, 'corners.json');
    if (fs.existsSync(cf)) corners.set(dir, JSON.parse(fs.readFileSync(cf, 'utf8')) as Record<string, CornerEntry>);
    else log.push(`[note] ${rel}: no corners.json yet, images are reported detection / boot only (no GT)`);
    const sources = readSources(dir);
    const orig = path.basename(dir) === 'real' && !sources;
    if (!sources && !orig) log.push(`[note] ${rel}: no SOURCES.json, all images treated as tune (view unknown)`);
    let skippedHoldout = 0;
    let skippedExcluded = 0;
    for (const file of fs.readdirSync(dir).filter((f) => /\.jpe?g$/i.test(f)).sort()) {
      if (filter && !file.includes(filter)) continue;
      const s = sources?.get(file);
      // Unusable images (not a starting position, ...) stay in SOURCES.json with split "excluded": never run.
      if (s?.split === 'excluded') {
        skippedExcluded++;
        continue;
      }
      const split: Split = s?.split === 'holdout' ? 'holdout' : 'tune';
      if (!gridOnly) {
        if (split === 'holdout' && holdout !== '1' && holdout !== 'only') {
          skippedHoldout++;
          continue;
        }
        if (split !== 'holdout' && holdout === 'only') continue;
      }
      const view = orig ? 'orig' : (s?.view ?? 'unknown');
      metas.push({ dir, file, key: `${rel}/${file}`, group: orig ? 'orig' : `${view}/${split}`, view, split });
    }
    if (skippedExcluded) log.push(`[excluded] ${rel}: ${skippedExcluded} image(s) with split "excluded" skipped`);
    if (skippedHoldout) log.push(`[holdout] ${rel}: ${skippedHoldout} holdout image(s) excluded (REALOCC_HOLDOUT=1 to include)`);
  }
  return { metas, corners };
}

// ---------------------------------------------------------------------------------------------------------------
// Geometry

export const BOARD: Point[] = [[0, 0], [8, 0], [8, 8], [0, 8]];

/** Mean and max corner error (fraction of the GT diagonal) over the best of the 8 corner assignments (by mean). */
export function cornerErrors(det: readonly Point[], gt: readonly Point[]): { mean: number; max: number } {
  const diag = Math.max(Math.hypot(gt[0]![0] - gt[2]![0], gt[0]![1] - gt[2]![1]), Math.hypot(gt[1]![0] - gt[3]![0], gt[1]![1] - gt[3]![1]));
  let best = { mean: Infinity, max: Infinity };
  for (const rev of [false, true])
    for (let r = 0; r < 4; r++) {
      let s = 0;
      let m = 0;
      for (let i = 0; i < 4; i++) {
        const j = rev ? (r - i + 8) % 4 : (r + i) % 4;
        const d = Math.hypot(det[j]![0] - gt[i]![0], det[j]![1] - gt[i]![1]);
        s += d;
        m = Math.max(m, d);
      }
      if (s / 4 < best.mean) best = { mean: s / 4, max: m };
    }
  return { mean: best.mean / diag, max: best.max / diag };
}

/** a1, h1, h8, a8 image points from the GT corners and whiteEdge k (camera above the board, image not mirrored: white
 *  sits on edge k looking across the board, a1 on its left = corners[k+1]). */
export function chessCorners(gt: readonly Point[], whiteEdge: number): Point[] {
  const c = (i: number) => gt[(whiteEdge + i) % 4]!;
  return [c(1), c(0), c(3), c(2)];
}

/** orientation[sq] = cell of hb containing chess square sq's centre (null if hb is not a relabelling of that board). */
export function expectedOrientation(gt: readonly Point[], whiteEdge: number, hb: Mat3): Uint8Array | null {
  const ha1 = homographyFrom4(BOARD, chessCorners(gt, whiteEdge));
  const inv = invert3(hb);
  if (!ha1 || !inv) return null;
  const m = mul3(inv, ha1);
  const out = new Uint8Array(64);
  for (let sq = 0; sq < 64; sq++) {
    const [u, v] = applyH(m, [(sq & 7) + 0.5, (sq >> 3) + 0.5]);
    const i = Math.floor(u);
    const j = Math.floor(v);
    if (i < 0 || j < 0 || i > 7 || j > 7 || Math.hypot(u - i - 0.5, v - j - 0.5) > 0.25) return null;
    out[sq] = j * 8 + i;
  }
  return out;
}

const darkSquare = (sq: number) => ((sq & 7) + (sq >> 3)) % 2 === 0;

// ---------------------------------------------------------------------------------------------------------------
// Pipeline run

const emptyConf = (): Confusion => [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

export function runTracker(frame: { data: Uint8ClampedArray; width: number; height: number }, hb: Mat3, params: Params, orient: Uint8Array | null, aligned: boolean): { r: RunResult; out: OccupancyResult; truthCells: Uint8Array | null } {
  const tr = new OccupancyTracker();
  const maxMs = param(params, OCCUPANCY_PARAMS, 'occFallbackMs') + 5 * FRAME_MS;
  let out: OccupancyResult;
  let f = 0;
  let bootFrame = -1;
  let framed = false;
  for (;;) {
    out = tr.update(frame, hb, params, f * FRAME_MS);
    framed ||= out.observation.framed;
    if (bootFrame < 0 && out.stats.state !== 'start') bootFrame = f;
    f++;
    if (bootFrame >= 0 && f > bootFrame + AFTER_BOOT_FRAMES) break;
    if (bootFrame < 0 && f * FRAME_MS > maxMs) break;
  }
  const boot: Boot = out.stats.state === 'calibrated' ? 'start' : out.stats.state === 'fallback' ? 'fallback' : 'none';
  let truthCells: Uint8Array | null = null;
  let rawAcc: number | null = null;
  let gridAcc: number | null = null;
  let conf: RunResult['conf'] = null;
  let rawSquares: string | null = null;
  let orientState: RunResult['orient'] = null;
  if (orient) {
    truthCells = new Uint8Array(64);
    const colourCell = new Uint8Array(64);
    for (let sq = 0; sq < 64; sq++) {
      truthCells[orient[sq]!] = START_SQUARES[sq]!;
      colourCell[orient[sq]!] = darkSquare(sq) ? 1 : 0;
    }
    conf = { light: emptyConf(), dark: emptyConf() };
    let ok = 0;
    for (let c = 0; c < 64; c++) {
      if (out.raw[c] === truthCells[c]) ok++;
      (colourCell[c] ? conf.dark : conf.light)[truthCells[c]!]![out.raw[c]!]!++;
    }
    rawAcc = ok / 64;
    if (out.grid) {
      let g = 0;
      for (let c = 0; c < 64; c++) if (out.grid[c] === truthCells[c]) g++;
      gridAcc = g / 64;
    }
    rawSquares = Array.from({ length: 64 }, (_, sq) => CLS[out.raw[orient[sq]!]!]).join('');
    const o = out.orientation;
    orientState = !o ? 'none' : o.every((v, sq) => v === orient[sq]) ? 'ok' : 'wrong';
  }
  return {
    r: {
      aligned, framed, boot, bootFrame, frames: f, orient: orientState, orientReason: tr.orientReason,
      rawAcc, gridAcc, lowCells: out.stats.lowCells, conf, rawSquares,
    },
    out,
    truthCells,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Drawing

// 5x7 bitmap font (rows top to bottom, 5 bits each, MSB left).
const FONT: Record<string, string> = {
  A: '01110 10001 10001 11111 10001 10001 10001', B: '11110 10001 10001 11110 10001 10001 11110',
  C: '01110 10001 10000 10000 10000 10001 01110', D: '11110 10001 10001 10001 10001 10001 11110',
  E: '11111 10000 10000 11110 10000 10000 11111', F: '11111 10000 10000 11110 10000 10000 10000',
  G: '01110 10001 10000 10111 10001 10001 01111', H: '10001 10001 10001 11111 10001 10001 10001',
  I: '01110 00100 00100 00100 00100 00100 01110', J: '00111 00010 00010 00010 00010 10010 01100',
  K: '10001 10010 10100 11000 10100 10010 10001', L: '10000 10000 10000 10000 10000 10000 11111',
  M: '10001 11011 10101 10101 10001 10001 10001', N: '10001 10001 11001 10101 10011 10001 10001',
  O: '01110 10001 10001 10001 10001 10001 01110', P: '11110 10001 10001 11110 10000 10000 10000',
  Q: '01110 10001 10001 10001 10101 10010 01101', R: '11110 10001 10001 11110 10100 10010 10001',
  S: '01111 10000 10000 01110 00001 00001 11110', T: '11111 00100 00100 00100 00100 00100 00100',
  U: '10001 10001 10001 10001 10001 10001 01110', V: '10001 10001 10001 10001 10001 01010 00100',
  W: '10001 10001 10001 10101 10101 10101 01010', X: '10001 10001 01010 00100 01010 10001 10001',
  Y: '10001 10001 01010 00100 00100 00100 00100', Z: '11111 00001 00010 00100 01000 10000 11111',
  0: '01110 10001 10011 10101 11001 10001 01110', 1: '00100 01100 00100 00100 00100 00100 01110',
  2: '01110 10001 00001 00010 00100 01000 11111', 3: '11111 00010 00100 00010 00001 10001 01110',
  4: '00010 00110 01010 10010 11111 00010 00010', 5: '11111 10000 11110 00001 00001 10001 01110',
  6: '00110 01000 10000 11110 10001 10001 01110', 7: '11111 00001 00010 00100 01000 01000 01000',
  8: '01110 10001 10001 01110 10001 10001 01110', 9: '01110 10001 10001 01111 00001 00010 01100',
  '-': '00000 00000 00000 11111 00000 00000 00000', '.': '00000 00000 00000 00000 00000 01100 01100',
  ':': '00000 01100 01100 00000 01100 01100 00000', '%': '11000 11001 00010 00100 01000 10011 00011',
  '/': '00001 00010 00010 00100 01000 01000 10000', '>': '01000 00100 00010 00001 00010 00100 01000',
  '=': '00000 00000 11111 00000 11111 00000 00000', '_': '00000 00000 00000 00000 00000 00000 11111',
  '(': '00010 00100 01000 01000 01000 00100 00010', ')': '01000 00100 00010 00010 00010 00100 01000',
  '+': '00000 00100 00100 11111 00100 00100 00000', ',': '00000 00000 00000 00000 01100 00100 01000',
  '?': '01110 10001 00001 00010 00100 00000 00100',
};

export type RGB = readonly [number, number, number];

export class Canvas {
  constructor(readonly img: Uint8ClampedArray, readonly w: number, readonly h: number) {}

  px(x: number, y: number, c: RGB, a = 1): void {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    const o = (y * this.w + x) * 4;
    for (let k = 0; k < 3; k++) this.img[o + k] = Math.round(this.img[o + k]! * (1 - a) + c[k]! * a);
  }

  rect(x0: number, y0: number, x1: number, y1: number, c: RGB, a = 1): void {
    for (let y = Math.max(0, Math.round(y0)); y < Math.min(this.h, Math.round(y1)); y++)
      for (let x = Math.max(0, Math.round(x0)); x < Math.min(this.w, Math.round(x1)); x++) this.px(x, y, c, a);
  }

  line(p: Point, q: Point, c: RGB, th = 1, a = 1): void {
    const n = Math.ceil(Math.max(Math.abs(q[0] - p[0]), Math.abs(q[1] - p[1]))) + 1;
    const r0 = Math.floor((th - 1) / 2);
    const seen = new Set<number>();
    for (let i = 0; i <= n; i++) {
      const x = Math.round(p[0] + ((q[0] - p[0]) * i) / n);
      const y = Math.round(p[1] + ((q[1] - p[1]) * i) / n);
      for (let dy = -r0; dy < th - r0; dy++)
        for (let dx = -r0; dx < th - r0; dx++) {
          const k = (y + dy) * this.w + x + dx;
          if (seen.has(k)) continue;
          seen.add(k);
          this.px(x + dx, y + dy, c, a);
        }
    }
  }

  quad(q: readonly Point[], c: RGB, th: number): void {
    for (let i = 0; i < 4; i++) this.line(q[i]!, q[(i + 1) % 4]!, c, th);
  }

  textSize(s: string, scale: number): [number, number] {
    return [s.length * 6 * scale - scale, 7 * scale];
  }

  /** Text with its top-left at (x, y) on a background box. */
  text(s: string, x: number, y: number, scale: number, fg: RGB, bg: RGB | null = [0, 0, 0], bgA = 0.7): void {
    const [tw, th] = this.textSize(s, scale);
    const pad = Math.max(1, scale);
    if (bg) this.rect(x - pad, y - pad, x + tw + pad, y + th + pad, bg, bgA);
    [...s.toUpperCase()].forEach((ch, k) => {
      const rows = FONT[ch]?.split(' ');
      if (!rows) return;
      rows.forEach((row, ry) => {
        for (let rx = 0; rx < 5; rx++) if (row[rx] === '1') this.rect(x + (k * 6 + rx) * scale, y + ry * scale, x + (k * 6 + rx + 1) * scale, y + (ry + 1) * scale, fg);
      });
    });
  }

  centredText(s: string, cx: number, cy: number, scale: number, fg: RGB, bg: RGB | null, bgA = 0.7): void {
    const [tw, th] = this.textSize(s, scale);
    // Kept inside the image (labels of corners near the border).
    const x = Math.max(2 * scale, Math.min(this.w - tw - 2 * scale, Math.round(cx - tw / 2)));
    const y = Math.max(2 * scale, Math.min(this.h - th - 2 * scale, Math.round(cy - th / 2)));
    this.text(s, x, y, scale, fg, bg, bgA);
  }
}

export const GREEN: RGB = [0, 230, 0];
export const RED: RGB = [255, 40, 40];
export const YELLOW: RGB = [255, 220, 0];
export const CYAN: RGB = [0, 230, 255];
export const WHITE: RGB = [255, 255, 255];
export const BLACK: RGB = [0, 0, 0];

interface Decoded {
  data: Uint8Array;
  width: number;
  height: number;
}

export function resizeRgba(cv: CV, img: Decoded, longSide: number): { data: Uint8ClampedArray; width: number; height: number } {
  const scale = Math.min(1, longSide / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  if (w === img.width && h === img.height) return { data: new Uint8ClampedArray(img.data), width: w, height: h };
  const src = new cv.Mat(img.height, img.width, cv.CV_8UC4);
  const dst = new cv.Mat();
  try {
    src.data.set(img.data);
    cv.resize(src, dst, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
    return { data: new Uint8ClampedArray(dst.data), width: w, height: h };
  } finally {
    src.delete();
    dst.delete();
  }
}

const scalePts = (q: readonly Point[], s: number): Point[] => q.map(([x, y]) => [x * s, y * s] as Point);

/** GT annotation overlay: 8x8 grid from the GT corners, a1/h1/h8/a8, the white edge, corner indices. */
export function drawGrid(cv: Canvas, gt: readonly Point[], whiteEdge: number | null, title: string): void {
  const u = Math.max(1, Math.round(Math.max(cv.w, cv.h) / 800));
  const H = homographyFrom4(BOARD, gt)!;
  for (let k = 1; k < 8; k++) {
    cv.line(applyH(H, [k, 0]), applyH(H, [k, 8]), YELLOW, u);
    cv.line(applyH(H, [0, k]), applyH(H, [8, k]), YELLOW, u);
  }
  cv.quad(gt, GREEN, u);
  const centre = applyH(H, [4, 4]);
  const away = (p: Point, d: number): Point => {
    const dx = p[0] - centre[0];
    const dy = p[1] - centre[1];
    const n = Math.hypot(dx, dy) || 1;
    return [p[0] + (dx / n) * d, p[1] + (dy / n) * d];
  };
  if (whiteEdge !== null) {
    cv.line(gt[whiteEdge]!, gt[(whiteEdge + 1) % 4]!, CYAN, 3 * u);
    const mid: Point = [(gt[whiteEdge]![0] + gt[(whiteEdge + 1) % 4]![0]) / 2, (gt[whiteEdge]![1] + gt[(whiteEdge + 1) % 4]![1]) / 2];
    const m = away(mid, 14 * u);
    cv.centredText('WHITE', m[0], m[1], 2 * u, CYAN, BLACK);
    const names = ['A1', 'H1', 'H8', 'A8'];
    chessCorners(gt, whiteEdge).forEach((p, i) => {
      const q = away(p, 26 * u);
      const ci = (whiteEdge + [1, 0, 3, 2][i]!) % 4;
      cv.centredText(`${names[i]} C${ci}`, q[0], q[1], 2 * u, i === 0 ? CYAN : WHITE, BLACK);
    });
    // a1 marker cell outline (must be a dark square).
    const ha1 = homographyFrom4(BOARD, chessCorners(gt, whiteEdge))!;
    cv.quad([[0, 0], [1, 0], [1, 1], [0, 1]].map((p) => applyH(ha1, p as Point)), CYAN, 2 * u);
  }
  else
    gt.forEach((p, i) => {
      const q = away(p, 20 * u);
      cv.centredText(`C${i}`, q[0], q[1], 2 * u, YELLOW, BLACK);
    });
  cv.text(title, 6 * u, cv.h - 20 * u, 2 * u, WHITE);
}

function drawRun(cv: Canvas, s: number, hb: Mat3, gt: readonly Point[] | null, detQuad: readonly Point[] | null, out: OccupancyResult, truthCells: Uint8Array | null, title: string): void {
  const u = Math.max(1, Math.round(Math.max(cv.w, cv.h) / 800));
  if (gt) cv.quad(scalePts(gt, s), GREEN, 2 * u);
  if (detQuad) cv.quad(scalePts(detQuad, s), RED, 2 * u);
  const P = (x: number, y: number): Point => {
    const p = applyH(hb, [x, y]);
    return [p[0] * s, p[1] * s];
  };
  const cellPx = Math.hypot(P(4, 4)[0] - P(5, 5)[0], P(4, 4)[1] - P(5, 5)[1]) / Math.SQRT2;
  const sc = Math.max(1, Math.min(4, Math.round(cellPx / 22)));
  for (let c = 0; c < 64; c++) {
    const [x, y] = P((c & 7) + 0.5, (c >> 3) + 0.5);
    const pred = out.raw[c]!;
    const t = truthCells?.[c];
    const wrong = t !== undefined && t !== pred;
    const label = wrong ? `${CLS[pred]}>${CLS[t]}` : CLS[pred]!;
    const bg: RGB = t === undefined ? [60, 60, 60] : wrong ? RED : [0, 140, 0];
    cv.centredText(label, x, y, sc, pred === 2 ? BLACK : pred === 1 ? WHITE : [255, 255, 160], bg, wrong ? 0.85 : 0.55);
  }
  cv.text(title, 6 * u, 6 * u, 2 * u, WHITE);
  cv.text('GREEN GT  RED DETECTED  TAG=PRED(>TRUTH)', 6 * u, 26 * u, u + 1, WHITE);
}

// ---------------------------------------------------------------------------------------------------------------
// Reporting

const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '-' : (x * 100).toFixed(d));

/** Off-diagonal confusion entries, e.g. "E>W3 B>W5" (truth>pred count). */
function errStr(m: Confusion): string {
  const parts: string[] = [];
  for (let t = 0; t < 3; t++) for (let p = 0; p < 3; p++) if (t !== p && m[t]![p]!) parts.push(`${CLS[t]}>${CLS[p]}${m[t]![p]}`);
  return parts.join(' ') || '.';
}

function addConf(a: Confusion, b: Confusion): void {
  for (let t = 0; t < 3; t++) for (let p = 0; p < 3; p++) a[t]![p]! += b[t]![p]!;
}

function fmtMatrix(m: Confusion, title: string): string[] {
  const rows = [`${title.padEnd(10)}    E    W    B`];
  for (let t = 0; t < 3; t++) rows.push(`       ${CLS[t]}   ${m[t]!.map((v) => String(v).padStart(4)).join(' ')}`);
  return rows;
}

const delta = (a: number | null | undefined, b: number | null | undefined) =>
  a === null || a === undefined || b === null || b === undefined ? '' : `${a - b >= 0 ? '+' : ''}${((a - b) * 100).toFixed(1)}`;

function report(results: ImageResult[], baseline: Map<string, ImageResult> | null): string {
  const L: string[] = [];
  const hdr = 'image                                  grp              det  err%(mean/max)  | det: boot  f  orient  raw%  grid% | gt: boot  f  orient  raw%  grid% | errors (truth>pred) light / dark';
  L.push(hdr + (baseline ? ' | d det raw  d gt raw' : ''));
  L.push('-'.repeat(hdr.length + (baseline ? 22 : 0)));
  const runCol = (r: RunResult | null) =>
    !r ? `${'-'.padStart(8)}  ${'-'.padStart(2)}  ${'-'.padEnd(6)} ${'-'.padStart(5)}  ${'-'.padStart(5)}` :
    `${r.boot.padStart(8)} ${String(r.bootFrame).padStart(2)}  ${(r.orient ?? '-').padEnd(6)} ${(r.aligned ? pct(r.rawAcc) : 'misal').padStart(5)}  ${pct(r.gridAcc).padStart(5)}`;
  for (const r of results) {
    const name = r.file.replace(/\.jpe?g$/i, '').slice(0, 38).padEnd(38);
    const det = r.negative ? (r.detFound ? 'FP ' : 'ok ') : !r.hasGt ? (r.detFound ? 'yes' : 'no ') : !r.detFound ? 'no ' : r.detOk ? 'OK ' : 'bad';
    const err = r.errMean === null ? '-'.padStart(14) : `${pct(r.errMean, 2).padStart(6)} /${pct(r.errMax, 2).padStart(6)} `;
    const e = (run: RunResult | null) => (run?.conf && run.aligned ? `${errStr(run.conf.light)} / ${errStr(run.conf.dark)}` : '-');
    const errs = r.hasGt ? `gt ${e(r.gt)}  det ${e(r.det)}` : '';
    let line = `${name} ${r.group.padEnd(16)} ${det}  ${err} |${runCol(r.det)} |${runCol(r.gt)} | ${errs}`;
    if (baseline) {
      const b = baseline.get(r.key);
      line += b ? ` | ${delta(r.det?.rawAcc, b.det?.rawAcc).padStart(8)}  ${delta(r.gt?.rawAcc, b.gt?.rawAcc).padStart(8)}` : ' | (new)';
    }
    L.push(line);
  }
  L.push('columns: det = detection OK (<2% mean corner err) / bad / no; boot = start (testStart) / fallback (fitUnsupervised) / none, f = frame it booted (100 ms steps);');
  L.push('         orient = tracker orientation vs GT (a1 = corners[whiteEdge+1]); raw% = last-frame raw per-cell accuracy, grid% = committed grid; misal = detection not aligned with GT;');
  L.push('         errors = confusion off-diagonals truth>pred count on light squares / dark squares, for the GT-corner and the detected-corner runs.');

  // Groups
  const groups = new Map<string, ImageResult[]>();
  for (const r of results) {
    if (r.negative) continue;
    const add = (k: string) => groups.set(k, [...(groups.get(k) ?? []), r]);
    add(r.group);
    add('ALL');
    if (r.group !== 'orig') {
      add(`view:${r.view}`);
      add(`split:${r.split}`);
      add('new(all)');
    }
  }
  // Drop a group with the same images as an earlier one (e.g. a single view, or ALL = orig).
  const keys = [...groups.keys()];
  keys.forEach((k, i) => {
    const g = groups.get(k)!;
    if (keys.slice(0, i).some((k2) => groups.get(k2)?.length === g.length && groups.get(k2)!.every((r) => g.includes(r)))) groups.delete(k);
  });
  const mean = (xs: (number | null | undefined)[]) => {
    const v = xs.filter((x): x is number => typeof x === 'number');
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  };
  const detRawOf = (r: ImageResult) => (r.det?.aligned ? r.det.rawAcc : 0);
  const gtRawOf = (r: ImageResult) => r.gt?.rawAcc;
  L.push('');
  L.push('group            n  gt  detOK  err%   | det: start fb none  orientOK  raw%  | gt: start fb none  orientOK  raw%' + (baseline ? '  | baseline (common images) n: det raw%, gt raw%' : ''));
  const rank = (k: string) => (k === 'orig' ? 0 : k === 'ALL' ? 5 : k === 'new(all)' ? 4 : k.startsWith('split:') ? 3 : k.startsWith('view:') ? 2 : 1);
  const ordered = [...groups.keys()].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  for (const k of ordered) {
    const g = groups.get(k)!;
    const withGt = g.filter((r) => r.hasGt);
    const cnt = (sel: (r: ImageResult) => RunResult | null, b: Boot) => g.filter((r) => sel(r)?.boot === b).length;
    const oriented = (sel: (r: ImageResult) => RunResult | null) => `${withGt.filter((r) => sel(r)?.orient === 'ok').length}/${withGt.length}`;
    const boots = (sel: (r: ImageResult) => RunResult | null) => `${String(cnt(sel, 'start')).padStart(5)} ${String(cnt(sel, 'fallback')).padStart(2)} ${String(cnt(sel, 'none')).padStart(4)}`;
    const detRaw = mean(withGt.map(detRawOf));
    const gtRaw = mean(withGt.map(gtRawOf));
    let line = `${k.padEnd(15)} ${String(g.length).padStart(2)} ${String(withGt.length).padStart(3)}  ${`${withGt.filter((r) => r.detOk).length}/${withGt.length}`.padStart(5)}  ${pct(mean(withGt.filter((r) => r.detOk).map((r) => r.errMean)), 2).padStart(5)}  |      ${boots((r) => r.det)}  ${oriented((r) => r.det).padStart(8)}  ${pct(detRaw).padStart(5)}  |     ${boots((r) => r.gt)}  ${oriented((r) => r.gt).padStart(8)}  ${pct(gtRaw).padStart(5)}`;
    if (baseline) {
      const common = withGt.filter((r) => baseline.get(r.key)?.hasGt);
      const cmp = (f: (r: ImageResult) => number | null | undefined) => {
        const now = mean(common.map(f));
        const was = mean(common.map((r) => f(baseline.get(r.key)!)));
        return now === null || was === null ? '-' : `${pct(was)}->${pct(now)} (${delta(now, was)})`;
      };
      line += `  | ${String(common.length).padStart(2)}: ${cmp(detRawOf)}, ${cmp(gtRawOf)}`;
    }
    L.push(line);
  }
  L.push('(det raw% counts an undetected / misaligned board as 0%; err% is over OK detections)');

  // Confusion matrices per group: GT-corner run and detected-corner run, each on light / dark squares.
  for (const k of ordered) {
    const g = groups.get(k)!.filter((r) => r.hasGt);
    const blocks: string[][] = [];
    const titles: string[] = [];
    for (const mode of ['gt', 'det'] as const) {
      const light = emptyConf();
      const dark = emptyConf();
      let n = 0;
      for (const r of g) {
        const run = r[mode];
        if (!run?.conf || !run.aligned) continue;
        n++;
        addConf(light, run.conf.light);
        addConf(dark, run.conf.dark);
      }
      if (!n) continue;
      titles.push(`${mode} corners (${n})`);
      blocks.push(fmtMatrix(light, `${mode} light`), fmtMatrix(dark, `${mode} dark`));
    }
    if (!blocks.length) continue;
    L.push('');
    L.push(`confusion ${k}: ${titles.join(', ')}; rows truth, columns predicted`);
    for (let i = 0; i < blocks[0]!.length; i++) L.push(('  ' + blocks.map((b) => b[i]!.padEnd(32)).join('  ')).trimEnd());
  }
  return L.join('\n');
}

// ---------------------------------------------------------------------------------------------------------------
// Entry

export function realOccBench(cv: CV, det: Detector, params: Params = {}): string {
  const env = process.env;
  const dirs = (env.REALOCC_DIRS ?? 'tests/fixtures/real,tests/fixtures/web').split(',').map((s) => s.trim()).filter(Boolean);
  const gridOnly = env.REALOCC_GRID_ONLY === '1';
  const size = Number(env.REALOCC_SIZE ?? 640);
  const pngSize = Number(env.REALOCC_PNG_SIZE ?? 1600);
  const pngDir = env.REALOCC_PNG === '0' ? null : path.resolve(REPO, env.REALOCC_PNG_DIR ?? path.join(DEFAULT_OUT_DIR, 'png'));
  const outFile = path.resolve(REPO, env.REALOCC_OUT ?? path.join(DEFAULT_OUT_DIR, 'results.json'));
  const log: string[] = [];
  const { metas, corners } = listImages(dirs, env.REALOCC_HOLDOUT, env.REALOCC_FILTER, gridOnly, log);
  const require = createRequire(import.meta.url);
  const jpeg = require('jpeg-js') as { decode(b: Buffer, o: { useTArray: boolean; maxMemoryUsageInMB?: number }): Decoded };
  if (pngDir) fs.mkdirSync(pngDir, { recursive: true });
  const pngName = (m: ImageMeta, suffix: string) => path.join(pngDir!, `${path.basename(m.dir)}__${m.file.replace(/\.jpe?g$/i, '')}.${suffix}.png`);

  const results: ImageResult[] = [];
  let pngs = 0;
  for (const m of metas) {
    const img = jpeg.decode(fs.readFileSync(path.join(m.dir, m.file)), { useTArray: true, maxMemoryUsageInMB: 2048 });
    const entry = corners.get(m.dir)?.[m.file];
    const negative = entry !== undefined && entry.corners === null;
    // GT corners in source pixels (the annotation may be at another resolution than the file).
    const gtSrc: Point[] | null = entry?.corners ? entry.corners.map(([x, y]) => [(x * img.width) / (entry.width ?? img.width), (y * img.height) / (entry.height ?? img.height)] as Point) : null;
    const whiteEdge = entry?.whiteEdge ?? null;
    const vis = pngDir ? resizeRgba(cv, img, pngSize) : null;
    const vs = vis ? vis.width / img.width : 1;
    if (gridOnly) {
      if (!gtSrc || !vis) {
        log.push(`[grid] ${m.key}: ${negative ? 'negative (no board)' : 'no GT corners'}, skipped`);
        continue;
      }
      const c = new Canvas(vis.data, vis.width, vis.height);
      drawGrid(c, scalePts(gtSrc, vs), whiteEdge, `${m.file}  WHITEEDGE ${whiteEdge ?? '?'}`);
      fs.writeFileSync(pngName(m, 'grid'), encodePng(vis.data, vis.width, vis.height));
      pngs++;
      continue;
    }
    const frame = resizeRgba(cv, img, size);
    const s = frame.width / img.width;
    const gt = gtSrc ? scalePts(gtSrc, s) : null;
    const hasGt = gt !== null && whiteEdge !== null;
    const res = det.detect(frame as unknown as ImageData, params);
    const r: ImageResult = {
      ...m, dir: path.relative(REPO, m.dir), width: img.width, height: img.height, hasGt, negative, whiteEdge,
      detFound: !!res.hb, detOk: null, errMean: null, errMax: null, det: null, gt: null,
    };
    if (res.corners && gt) {
      const e = cornerErrors(res.corners, gt);
      r.errMean = e.mean;
      r.errMax = e.max;
      r.detOk = e.mean < ERR_THRESHOLD;
    } else if (gt) r.detOk = false;
    const draws: { suffix: string; hb: Mat3; out: OccupancyResult; truth: Uint8Array | null; title: string }[] = [];
    if (res.hb && !negative) {
      const orient = hasGt ? expectedOrientation(gt!, whiteEdge!, res.hb) : null;
      if (orient) {
        // Cross-check with the orientation-free GT used by tests/occupancy.test.ts.
        const gtStart = [startGrid(0, 0), startGrid(1, 1), startGrid(0, 1), startGrid(1, 0)][whiteEdge!]!;
        const a = alignGt(gtStart, gt!, res.hb);
        const mine = new Uint8Array(64);
        for (let sq = 0; sq < 64; sq++) mine[orient[sq]!] = START_SQUARES[sq]!;
        if (!a || a.some((v, c) => v !== mine[c])) log.push(`[warn] ${m.key}: GT grid disagrees with alignGt(gtStart)`);
      }
      const run = runTracker(frame, res.hb, params, orient, !hasGt || orient !== null);
      r.det = run.r;
      draws.push({ suffix: 'det', hb: res.hb, out: run.out, truth: run.truthCells, title: `${m.file} DET ${r.detOk ? 'OK' : hasGt ? 'BAD' : ''} BOOT ${run.r.boot} RAW ${pct(run.r.rawAcc)}% ORIENT ${run.r.orient ?? '-'}` });
    }
    if (hasGt) {
      const hb = homographyFrom4(BOARD, gt!)!;
      const orient = expectedOrientation(gt!, whiteEdge!, hb)!;
      const run = runTracker(frame, hb, params, orient, true);
      r.gt = run.r;
      draws.push({ suffix: 'gt', hb, out: run.out, truth: run.truthCells, title: `${m.file} GT CORNERS BOOT ${run.r.boot} RAW ${pct(run.r.rawAcc)}% ORIENT ${run.r.orient ?? '-'}` });
    }
    if (vis) {
      for (const d of draws) {
        const buf = new Uint8ClampedArray(vis.data);
        drawRun(new Canvas(buf, vis.width, vis.height), vis.width / frame.width, d.hb, gt, res.corners, d.out, d.truth, d.title);
        fs.writeFileSync(pngName(m, d.suffix), encodePng(buf, vis.width, vis.height));
        pngs++;
      }
      if (negative && res.corners) {
        const buf = new Uint8ClampedArray(vis.data);
        const c = new Canvas(buf, vis.width, vis.height);
        c.quad(scalePts(res.corners, vis.width / frame.width), RED, 3);
        c.text(`${m.file} FALSE POSITIVE`, 6, 6, 2, WHITE);
        fs.writeFileSync(pngName(m, 'fp'), encodePng(buf, vis.width, vis.height));
        pngs++;
      }
    }
    results.push(r);
  }

  const head = [...log, `dirs: ${dirs.join(', ')}  holdout: ${env.REALOCC_HOLDOUT || 'excluded'}  size: ${size}px  images: ${metas.length}`];
  if (gridOnly) return `${head.join('\n')}\nGT grid overlays (${pngs}) in ${pngDir}`;
  let baseline: Map<string, ImageResult> | null = null;
  if (env.REALOCC_BASELINE) {
    const bf = path.resolve(REPO, env.REALOCC_BASELINE);
    const b = JSON.parse(fs.readFileSync(bf, 'utf8')) as { images: ImageResult[] };
    baseline = new Map(b.images.map((r) => [r.key, r]));
    head.push(`baseline: ${path.relative(REPO, bf)}`);
  }
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, JSON.stringify({ generatedAt: new Date().toISOString(), size, params, dirs, holdout: env.REALOCC_HOLDOUT || null, images: results }, null, 1));
  return `${head.join('\n')}\n\n${report(results, baseline)}\n\nresults JSON: ${outFile}${pngDir ? `\noverlay PNGs (${pngs}): ${pngDir}` : ''}`;
}
