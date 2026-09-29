/*
 * Annotation tooling for starting-position fixture photos (Wave 2 of docs/PLAN-multiboard.md). Workflow and command
 * lines: tests/fixtures/web/ANNOTATION.md. Entry: tests/tools/annotate.entry.ts (`npm run annot`, options by env).
 *
 * Steps, each a function here:
 *   ingest   downscale images to <= 1600 px long side (JPEG q88), bake the EXIF orientation, convert PNG to JPEG,
 *            strip metadata, optionally rename to NN-short-name.jpg, and create / complete SOURCES.json;
 *   split    assign split = tune / holdout (~1/3 holdout, seeded, stratified over view x material) to entries whose
 *            split is still null;
 *   propose  run the app's detector (the bench's code path, Detector.detect on an INTER_AREA downscale) at a few
 *            sizes, refine the lattice on the full-resolution image (sub-pixel saddle points of the inner grid +
 *            least-squares homography), guess whiteEdge (the app's OccupancyTracker orientation from the starting
 *            position; a band-brightness heuristic as fallback), and write corners.draft.json;
 *   crops    zoomed PNG crops around each draft corner (crosshair, pixel ticks, projected grid lines), a wider context
 *            crop, and a downscaled full-image preview with the quad / grid / white edge; or crops at given points;
 *   promote  copy draft entries marked "reviewed": true into corners.json;
 *   compare  per-image corner error (% of the diagonal) and whiteEdge agreement between two corner files.
 *
 * Coordinates are image pixels of the fixture file as stored (after ingest), continuous, pixel centres at integer
 * coordinates (OpenCV convention; the half-pixel ambiguity is < 0.1% of a board diagonal anyway). Corner order everywhere: clockwise on screen starting from the corner with the smallest x + y, and
 * whiteEdge k = the white pieces stand along corners[k] -> corners[k + 1] (tests/fixtures/real/corners.json).
 */
import fs from 'node:fs';
import path from 'node:path';
import { applyH, homographyFrom4, solveLinear, type Mat3, type Point } from '../../src/geom/homography';
import type { Detector } from '../../src/vision/detector';
import type { CV } from '../../src/vision/preprocess';
import type { Params } from '../../src/worker/protocol';
import {
  BLACK, BOARD, Canvas, chessCorners, CYAN, drawGrid, expectedOrientation, GREEN, RED, resizeRgba, runTracker, WHITE, YELLOW,
  cornerErrors, type RGB,
} from '../synth/realOccBench';
import { encodePng } from '../synth/png';
import { decodeImage, encodeJpeg, type Rgba } from './imageio';

export const REPO = path.resolve(import.meta.dirname, '../..');
export const DEFAULT_OUT = path.join(REPO, 'tests/tools/annot-out');
const IMG_RE = /\.(jpe?g|png)$/i;
const JPG_RE = /\.jpe?g$/i;

// ---------------------------------------------------------------------------------------------------------------
// JSON files

export interface SourceEntry {
  file: string;
  title: string | null;
  author: string | null;
  sourceUrl: string | null;
  license: string | null;
  view: 'oblique' | 'topdown' | null;
  material: string | null;
  lighting: string | null;
  split: 'tune' | 'holdout' | null;
  notes: string | null;
  [k: string]: unknown;
}

const SOURCE_FIELDS = ['file', 'title', 'author', 'sourceUrl', 'license', 'view', 'material', 'lighting', 'split', 'notes'] as const;

const blankSource = (file: string): SourceEntry => ({
  file, title: null, author: null, sourceUrl: null, license: null, view: null, material: null, lighting: null, split: null, notes: null,
});

const filled = (v: unknown) => v !== null && v !== undefined && v !== '';

type SourcesShape = { kind: 'array' } | { kind: 'wrapped'; key: string; rest: Record<string, unknown> } | { kind: 'keyed'; rest: Record<string, unknown> };

/** Reads SOURCES.json in any of the shapes realOccBench accepts ([...], {images: [...]}, {file: {...}}); keeps the shape. */
export function readSources(dir: string): { entries: SourceEntry[]; shape: SourcesShape } {
  const f = path.join(dir, 'SOURCES.json');
  if (!fs.existsSync(f)) return { entries: [], shape: { kind: 'array' } };
  const j = JSON.parse(fs.readFileSync(f, 'utf8')) as unknown;
  const norm = (e: Record<string, unknown>, file?: string): SourceEntry => {
    const out = { ...blankSource(String(e.file ?? file)), ...e } as SourceEntry;
    out.file = String(e.file ?? file);
    return out;
  };
  if (Array.isArray(j)) return { entries: j.map((e) => norm(e as Record<string, unknown>)), shape: { kind: 'array' } };
  const o = j as Record<string, unknown>;
  const key = Object.keys(o).find((k) => Array.isArray(o[k]) && (o[k] as unknown[]).some((e) => e && typeof e === 'object' && 'file' in e));
  if (key) {
    const rest = { ...o };
    delete rest[key];
    return { entries: (o[key] as Record<string, unknown>[]).map((e) => norm(e)), shape: { kind: 'wrapped', key, rest } };
  }
  const rest: Record<string, unknown> = {};
  const entries: SourceEntry[] = [];
  for (const [k, v] of Object.entries(o)) {
    if (k.startsWith('_') || !v || typeof v !== 'object') rest[k] = v;
    else entries.push(norm(v as Record<string, unknown>, k));
  }
  return { entries, shape: { kind: 'keyed', rest } };
}

export function writeSources(dir: string, entries: SourceEntry[], shape: SourcesShape): void {
  // Canonical field order first, then any extra fields.
  const ordered = entries.map((e) => {
    const o: Record<string, unknown> = {};
    for (const k of SOURCE_FIELDS) o[k] = e[k] ?? null;
    for (const [k, v] of Object.entries(e)) if (!(k in o)) o[k] = v;
    return o;
  });
  let j: unknown;
  if (shape.kind === 'array') j = ordered;
  else if (shape.kind === 'wrapped') j = { ...shape.rest, [shape.key]: ordered };
  else j = { ...shape.rest, ...Object.fromEntries(ordered.map((o) => [o.file as string, o])) };
  fs.writeFileSync(path.join(dir, 'SOURCES.json'), `${JSON.stringify(j, null, 2)}\n`);
}

export interface CornerEntry {
  width: number;
  height: number;
  corners: Point[] | null;
  whiteEdge?: number | null;
}

/** corners.draft.json entry: the corners.json fields plus provenance and review state. */
export interface DraftEntry extends CornerEntry {
  /** detector: proposed by propose(); none: detection failed, annotate by hand; manual: set by an annotator. */
  source: 'detector' | 'none' | 'manual';
  /** Detector confidence of the chosen detection (null when none). */
  confidence: number | null;
  /** Long side (px) the chosen detection ran at. */
  detectSize: number | null;
  /** Max corner disagreement between the detection sizes, % of the diagonal (null with < 2 detections). */
  sizeSpread: number | null;
  /** Lattice refinement: inlier saddle points, residual RMS (px), corner shift vs the raw detection (% diagonal). */
  refine: { used: boolean; inliers: number; rms: number; shift: number } | null;
  /** tracker: orientation of the app's OccupancyTracker; heuristic: band brightness; null: unknown. */
  whiteEdgeMethod: 'tracker' | 'heuristic' | 'manual' | null;
  whiteEdgeTracker: number | null;
  whiteEdgeHeuristic: number | null;
  /** Relative contrast of the band heuristic (larger = more reliable). */
  heuristicMargin: number | null;
  /** Things the annotator must look at (detection failed, sizes disagree, methods disagree, ...). */
  flags: string[];
  reviewed: boolean;
  reviewer: string | null;
  crossChecked: boolean;
  crossChecker: string | null;
  notes: string | null;
}

export function readJson<T>(f: string, fallback: T): T {
  return fs.existsSync(f) ? (JSON.parse(fs.readFileSync(f, 'utf8')) as T) : fallback;
}

/** One entry per line (like tests/fixtures/real/corners.json) for readable diffs; `_`-keys first. */
export function writeCornersJson(f: string, obj: Record<string, unknown>): void {
  const keys = Object.keys(obj).sort((a, b) => Number(!a.startsWith('_')) - Number(!b.startsWith('_')) || (a.startsWith('_') ? 0 : a.localeCompare(b)));
  const lines = keys.map((k) => `  ${JSON.stringify(k)}: ${JSON.stringify(obj[k])}`);
  fs.writeFileSync(f, `{\n${lines.join(',\n')}\n}\n`);
}

// ---------------------------------------------------------------------------------------------------------------
// Corner order

const r1 = (v: number) => Math.round(v * 2) / 2;

/**
 * Reorders a quad clockwise on screen (image y down) starting from the smallest x + y, and remaps whiteEdge (an index
 * into the ORIGINAL order) to the new order. Returns null for a degenerate / non-convex quad.
 */
export function normaliseQuad(q: readonly Point[], whiteEdge: number | null = null): { corners: Point[]; whiteEdge: number | null } | null {
  if (q.length !== 4) return null;
  let area = 0;
  for (let i = 0; i < 4; i++) area += q[i]![0] * q[(i + 1) % 4]![1] - q[(i + 1) % 4]![0] * q[i]![1];
  if (!(Math.abs(area) > 1e-6)) return null;
  // Convexity: all cross products of consecutive edges share the sign of the area.
  for (let i = 0; i < 4; i++) {
    const a = q[i]!;
    const b = q[(i + 1) % 4]!;
    const c = q[(i + 2) % 4]!;
    if (Math.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) !== Math.sign(area)) return null;
  }
  // Positive shoelace area in y-down coordinates = clockwise on screen.
  const rev = area < 0;
  const order = rev ? [0, 3, 2, 1] : [0, 1, 2, 3];
  let start = 0;
  for (let i = 1; i < 4; i++) {
    const p = q[order[i]!]!;
    const s = q[order[start]!]!;
    if (p[0] + p[1] < s[0] + s[1]) start = i;
  }
  const idx = [0, 1, 2, 3].map((i) => order[(start + i) % 4]!);
  let we: number | null = null;
  if (whiteEdge !== null && whiteEdge !== undefined) {
    // Old edge k joins old corners k and k + 1; find the new edge joining the same two corners.
    const a = whiteEdge % 4;
    const b = (whiteEdge + 1) % 4;
    for (let i = 0; i < 4; i++) {
      const u = idx[i]!;
      const v = idx[(i + 1) % 4]!;
      if ((u === a && v === b) || (u === b && v === a)) we = i;
    }
  }
  return { corners: idx.map((i) => [q[i]![0], q[i]![1]] as Point), whiteEdge: we };
}

// ---------------------------------------------------------------------------------------------------------------
// Ingest

export interface IngestOptions {
  maxSide?: number;
  quality?: number;
  rename?: boolean;
  /** Originals are copied here before being replaced (default tests/tools/annot-out/originals/<dir>). */
  backupDir?: string;
}

const slug = (s: string) => s.toLowerCase().replace(/\.[a-z0-9]+$/, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'image';

/** Removes APP1..APP15 (EXIF, XMP, IPTC, ...) and COM segments from a JPEG losslessly; keeps APP0 (JFIF) and APP2 (ICC). */
export function stripJpegMetadata(b: Uint8Array): { out: Uint8Array; removed: number } {
  const parts: Uint8Array[] = [b.subarray(0, 2)];
  let o = 2;
  let removed = 0;
  while (o + 4 <= b.length && b[o] === 0xff) {
    const m = b[o + 1]!;
    if (m === 0xda || m === 0xd9) break;
    const len = (b[o + 2]! << 8) | b[o + 3]!;
    const drop = (m >= 0xe1 && m <= 0xef && m !== 0xe2) || m === 0xfe;
    if (drop) removed++;
    else parts.push(b.subarray(o, o + 2 + len));
    o += 2 + len;
  }
  if (!removed) return { out: b, removed: 0 };
  parts.push(b.subarray(o));
  return { out: Buffer.concat(parts), removed };
}

export function ingest(cv: CV, dir: string, opt: IngestOptions = {}): string[] {
  const maxSide = opt.maxSide ?? 1600;
  const quality = opt.quality ?? 88;
  const log: string[] = [];
  if (!fs.existsSync(dir)) throw new Error(`no such directory: ${dir}`);
  const backup = opt.backupDir ?? path.join(DEFAULT_OUT, 'originals', path.basename(dir));
  const { entries, shape } = readSources(dir);
  const byFile = new Map(entries.map((e) => [e.file, e]));
  const renames = new Map<string, string>();
  const files = fs.readdirSync(dir).filter((f) => IMG_RE.test(f) || /\.(heic|heif|avif)$/i.test(f)).sort();
  const taken = new Set(files.map((f) => f.toLowerCase()));
  let nextNo = 1 + Math.max(0, ...files.map((f) => /^(\d\d)-/.exec(f)?.[1]).filter(Boolean).map(Number));

  for (const file of files) {
    const src = path.join(dir, file);
    const bytes = fs.readFileSync(src);
    let decoded: ReturnType<typeof decodeImage>;
    try {
      decoded = decodeImage(bytes);
    } catch (e) {
      log.push(`[error] ${file}: ${(e as Error).message}; skipped (not added to SOURCES.json)`);
      continue;
    }
    const { img, orientation, format } = decoded;
    const long = Math.max(img.width, img.height);
    const needScale = long > maxSide;
    const reencode = needScale || orientation !== 1 || format === 'png';
    let target = format === 'png' ? file.replace(/\.png$/i, '.jpg') : file;
    if (opt.rename && !/^\d\d-[a-z0-9-]+\.jpg$/.test(target) && !file.startsWith('neg-')) {
      let name: string;
      do name = `${String(nextNo++).padStart(2, '0')}-${slug(file)}.jpg`;
      while (taken.has(name));
      target = name;
    }
    if (target !== file && fs.existsSync(path.join(dir, target))) {
      log.push(`[error] ${file}: target name ${target} already exists; skipped`);
      continue;
    }
    const notes: string[] = [];
    if (reencode || target !== file) {
      fs.mkdirSync(backup, { recursive: true });
      const bk = path.join(backup, file);
      if (!fs.existsSync(bk)) fs.copyFileSync(src, bk);
    }
    if (reencode) {
      let out: Rgba = img;
      if (needScale) {
        const r = resizeRgba(cv, img, maxSide);
        out = { data: new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.byteLength), width: r.width, height: r.height };
      }
      fs.writeFileSync(path.join(dir, target), encodeJpeg(out, quality));
      notes.push(`${img.width}x${img.height}${orientation !== 1 ? ` (EXIF orientation ${orientation} applied)` : ''}${format === 'png' ? ' png' : ''} -> ${out.width}x${out.height} q${quality}`);
    } else {
      const { out, removed } = stripJpegMetadata(bytes);
      if (removed) {
        fs.mkdirSync(backup, { recursive: true });
        if (!fs.existsSync(path.join(backup, file))) fs.copyFileSync(src, path.join(backup, file));
        fs.writeFileSync(path.join(dir, target), out);
        notes.push(`${img.width}x${img.height} kept, ${removed} metadata segment(s) stripped`);
      } else if (target !== file) fs.copyFileSync(src, path.join(dir, target));
      else notes.push(`${img.width}x${img.height} unchanged`);
    }
    if (target !== file) {
      fs.rmSync(src);
      renames.set(file, target);
      taken.add(target.toLowerCase());
      notes.push(`renamed from ${file}`);
    }
    log.push(`[ingest] ${target}: ${notes.join('; ')}`);

    // SOURCES.json: never overwrite a filled field.
    const prev = byFile.get(file) ?? byFile.get(target);
    if (prev) {
      prev.file = target;
      for (const k of SOURCE_FIELDS) if (!(k in prev)) (prev as Record<string, unknown>)[k] = null;
      if (target !== file && !filled(prev.notes)) prev.notes = `original file name: ${file}`;
    } else {
      const e = blankSource(target);
      if (target !== file) e.notes = `original file name: ${file}`;
      entries.push(e);
      byFile.set(target, e);
    }
  }
  // Renames also apply to corner files keyed by file name.
  if (renames.size)
    for (const cf of ['corners.json', 'corners.draft.json']) {
      const f = path.join(dir, cf);
      if (!fs.existsSync(f)) continue;
      const j = readJson<Record<string, unknown>>(f, {});
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(j)) out[renames.get(k) ?? k] = v;
      writeCornersJson(f, out);
      log.push(`[rename] keys updated in ${cf}`);
    }
  const present = new Set(fs.readdirSync(dir));
  for (const e of entries) if (!present.has(e.file)) log.push(`[warn] SOURCES.json entry ${e.file}: file not found`);
  entries.sort((a, b) => a.file.localeCompare(b.file));
  writeSources(dir, entries, shape);
  const missing = entries.filter((e) => present.has(e.file) && SOURCE_FIELDS.some((k) => k !== 'notes' && !filled(e[k]))).length;
  log.push(`SOURCES.json: ${entries.length} entries (${missing} with unfilled fields)${fs.existsSync(backup) ? `; originals backed up in ${path.relative(REPO, backup)}` : ''}`);
  return log;
}

// ---------------------------------------------------------------------------------------------------------------
// Split

/** Mulberry32. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const hashStr = (s: string) => [...s].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0, 2166136261);

/**
 * Assigns split to entries whose split is null: ~1/3 holdout overall (round(n / 3), counting existing holdouts),
 * distributed over the view x material groups in proportion to their size (largest remainder, seeded ties), with a
 * seeded choice inside each group. Existing split values are never changed. Negative images (neg-*) are skipped.
 */
export function assignSplit(entries: SourceEntry[], seed = 1): string[] {
  const log: string[] = [];
  const pool = entries.filter((e) => !e.file.startsWith('neg-'));
  const n = pool.length;
  const H = Math.round(n / 3);
  const groups = new Map<string, SourceEntry[]>();
  for (const e of pool) {
    const k = `${e.view ?? '?'}|${e.material ?? '?'}`;
    groups.set(k, [...(groups.get(k) ?? []), e]);
  }
  const keys = [...groups.keys()].sort();
  const r = rng(seed);
  const desired = keys.map((k) => (groups.get(k)!.length * H) / Math.max(1, n));
  const target = desired.map(Math.floor);
  const order = keys.map((_, i) => i).map((i) => ({ i, frac: desired[i]! - target[i]!, tie: r() })).sort((a, b) => b.frac - a.frac || a.tie - b.tie);
  let rest = H - target.reduce((a, b) => a + b, 0);
  for (const o of order) if (rest-- > 0) target[o.i]!++;
  keys.forEach((k, gi) => {
    const g = groups.get(k)!;
    const have = g.filter((e) => e.split === 'holdout').length;
    const free = g.filter((e) => e.split !== 'tune' && e.split !== 'holdout');
    const gr = rng(seed ^ hashStr(k));
    const shuffled = free.map((e) => ({ e, t: gr() })).sort((a, b) => a.t - b.t || a.e.file.localeCompare(b.e.file)).map((x) => x.e);
    const add = Math.max(0, Math.min(free.length, target[gi]! - have));
    shuffled.forEach((e, i) => {
      e.split = i < add ? 'holdout' : 'tune';
      log.push(`[split] ${e.file}: ${e.split} (group ${k})`);
    });
  });
  const hold = pool.filter((e) => e.split === 'holdout').length;
  log.push(`split: ${hold}/${n} holdout (target ${H}), groups: ${keys.map((k) => `${k} ${groups.get(k)!.filter((e) => e.split === 'holdout').length}/${groups.get(k)!.length}`).join(', ')}`);
  return log;
}

// ---------------------------------------------------------------------------------------------------------------
// Proposal: detection, lattice refinement, whiteEdge

const toFrame = (cv: CV, img: Rgba, size: number) => resizeRgba(cv, img, size);

function grayOf(img: Rgba): Float32Array {
  const g = new Float32Array(img.width * img.height);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * img.data[i * 4]! + 0.587 * img.data[i * 4 + 1]! + 0.114 * img.data[i * 4 + 2]!;
  return g;
}

function sampleGray(g: Float32Array, w: number, h: number, x: number, y: number): number {
  const xi = Math.max(0, Math.min(w - 2, Math.floor(x)));
  const yi = Math.max(0, Math.min(h - 2, Math.floor(y)));
  const fx = Math.max(0, Math.min(1, x - xi));
  const fy = Math.max(0, Math.min(1, y - yi));
  const o = yi * w + xi;
  return (g[o]! * (1 - fx) + g[o + 1]! * fx) * (1 - fy) + (g[o + w]! * (1 - fx) + g[o + w + 1]! * fx) * fy;
}

/** Saddle (X-corner) sub-pixel refinement, as OpenCV cornerSubPix: argmin_p sum_q (g_q . (q - p))^2 over a window. */
function subPix(g: Float32Array, w: number, h: number, p0: Point, rad: number): Point | null {
  let [px, py] = p0;
  const sig2 = 2 * (rad * 0.6) ** 2;
  for (let it = 0; it < 30; it++) {
    let a = 0;
    let b = 0;
    let c = 0;
    let bx = 0;
    let by = 0;
    const cx = Math.round(px);
    const cy = Math.round(py);
    if (cx - rad < 1 || cy - rad < 1 || cx + rad >= w - 1 || cy + rad >= h - 1) return null;
    for (let dy = -rad; dy <= rad; dy++)
      for (let dx = -rad; dx <= rad; dx++) {
        if (dx * dx + dy * dy <= 1) continue; // the centre's gradient is ill-defined at a saddle
        const x = cx + dx;
        const y = cy + dy;
        const o = y * w + x;
        const gx = (g[o + 1]! - g[o - 1]!) / 2;
        const gy = (g[o + w]! - g[o - w]!) / 2;
        const wt = Math.exp(-(dx * dx + dy * dy) / sig2);
        a += wt * gx * gx;
        b += wt * gx * gy;
        c += wt * gy * gy;
        bx += wt * (gx * gx * x + gx * gy * y);
        by += wt * (gx * gy * x + gy * gy * y);
      }
    const det = a * c - b * b;
    if (!(det > 1e-9)) return null;
    const nx = (c * bx - b * by) / det;
    const ny = (a * by - b * bx) / det;
    const d = Math.hypot(nx - px, ny - py);
    px = nx;
    py = ny;
    if (Math.hypot(px - p0[0], py - p0[1]) > rad) return null;
    if (d < 0.01) break;
  }
  return [px, py];
}

/** Least-squares homography board -> image from >= 4 correspondences (normalised DLT with h33 = 1). */
export function fitHomography(src: readonly Point[], dst: readonly Point[]): Mat3 | null {
  if (src.length < 4) return null;
  const mx = dst.reduce((s, p) => s + p[0], 0) / dst.length;
  const my = dst.reduce((s, p) => s + p[1], 0) / dst.length;
  const sc = Math.sqrt(dst.reduce((s, p) => s + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0) / dst.length) || 1;
  const ata = Array.from({ length: 8 }, () => new Array<number>(8).fill(0));
  const atb = new Array<number>(8).fill(0);
  const acc = (row: number[], v: number) => {
    for (let i = 0; i < 8; i++) {
      atb[i]! += row[i]! * v;
      for (let j = 0; j < 8; j++) ata[i]![j]! += row[i]! * row[j]!;
    }
  };
  src.forEach(([x0, y0], k) => {
    const x = (x0 - 4) / 4;
    const y = (y0 - 4) / 4;
    const u = (dst[k]![0] - mx) / sc;
    const v = (dst[k]![1] - my) / sc;
    acc([x, y, 1, 0, 0, 0, -u * x, -u * y], u);
    acc([0, 0, 0, x, y, 1, -v * x, -v * y], v);
  });
  const s = solveLinear(ata, atb);
  if (!s) return null;
  const hn: Mat3 = [s[0]!, s[1]!, s[2]!, s[3]!, s[4]!, s[5]!, s[6]!, s[7]!, 1];
  // H = T_img^-1 * Hn * T_board, T_board: x -> (x - 4) / 4, T_img: u -> (u - m) / sc.
  const tb: Mat3 = [0.25, 0, -1, 0, 0.25, -1, 0, 0, 1];
  const ti: Mat3 = [sc, 0, mx, 0, sc, my, 0, 0, 1];
  const m = (p: Mat3, q: Mat3) => {
    const r = new Array<number>(9).fill(0);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i * 3 + j]! += p[i * 3 + k]! * q[k * 3 + j]!;
    return r as Mat3;
  };
  const H = m(ti, m(hn, tb));
  return H.map((v) => v / H[8]) as Mat3;
}

export interface LatticeRefine {
  hb: Mat3;
  inliers: number;
  rms: number;
  points: { board: Point; img: Point; ok: boolean }[];
}

/**
 * Refines a board homography (board cell units -> image px) on the full-resolution grey image: every inner lattice
 * point (i, j in 1..7) is moved to the nearest saddle point (subPix) and kept when it is a clear X-corner (diagonal
 * quadrants alike, adjacent quadrants different) that moved less than 0.25 cell; then a least-squares homography is
 * fit and outliers (> max(1.5 px, 3 x median residual)) are dropped over a few rounds.
 */
export function refineLattice(g: Float32Array, w: number, h: number, hb0: Mat3): LatticeRefine | null {
  let hb = hb0;
  let result: LatticeRefine | null = null;
  for (let pass = 0; pass < 2; pass++) {
    const pts: { board: Point; img: Point; ok: boolean }[] = [];
    for (let j = 1; j <= 7; j++)
      for (let i = 1; i <= 7; i++) {
        const p = applyH(hb, [i, j]);
        const cell = Math.min(
          Math.hypot(...(sub(applyH(hb, [i + 1, j]), applyH(hb, [i - 1, j])) as [number, number])) / 2,
          Math.hypot(...(sub(applyH(hb, [i, j + 1]), applyH(hb, [i, j - 1])) as [number, number])) / 2,
        );
        const rad = Math.max(3, Math.min(25, Math.round(cell * 0.3)));
        const q = subPix(g, w, h, p, rad);
        if (!q || Math.hypot(q[0] - p[0], q[1] - p[1]) > 0.25 * cell) {
          pts.push({ board: [i, j], img: p, ok: false });
          continue;
        }
        // X-corner check on quadrant means around q (offsets along the local board axes).
        const d = [q[0] - p[0], q[1] - p[1]];
        const quad = (si: number, sj: number) => {
          let s = 0;
          let n = 0;
          for (const f of [0.12, 0.2, 0.28])
            for (const e of [0.12, 0.2, 0.28]) {
              const r = applyH(hb, [i + si * f, j + sj * e]);
              s += sampleGray(g, w, h, r[0] + d[0]!, r[1] + d[1]!);
              n++;
            }
          return s / n;
        };
        const A = quad(1, 1);
        const C = quad(-1, -1);
        const B = quad(1, -1);
        const D = quad(-1, 1);
        const score = Math.abs((A + C) / 2 - (B + D) / 2) - (Math.abs(A - C) + Math.abs(B - D)) / 2;
        pts.push({ board: [i, j], img: q, ok: score > 12 });
      }
    let use = pts.filter((p) => p.ok);
    let H: Mat3 | null = null;
    let rms = 0;
    for (let round = 0; round < 4; round++) {
      if (use.length < 8) return result;
      H = fitHomography(use.map((p) => p.board), use.map((p) => p.img));
      if (!H) return result;
      const res = use.map((p) => Math.hypot(...(sub(applyH(H!, p.board), p.img) as [number, number])));
      const med = [...res].sort((a, b) => a - b)[Math.floor(res.length / 2)]!;
      const thr = Math.max(1.5, 3 * med);
      const keep = use.filter((_, k) => res[k]! <= thr);
      rms = Math.sqrt(res.reduce((s, r) => s + r * r, 0) / res.length);
      if (keep.length === use.length) break;
      use = keep;
    }
    // Spread: need at least 3 distinct lattice rows and columns.
    if (new Set(use.map((p) => p.board[0])).size < 3 || new Set(use.map((p) => p.board[1])).size < 3) return result;
    for (const p of pts) p.ok = use.includes(p);
    hb = H!;
    result = { hb, inliers: use.length, rms, points: pts };
  }
  return result;
}

const sub = (a: Point, b: Point) => [a[0] - b[0], a[1] - b[1]];

const diagOf = (q: readonly Point[]) => Math.max(Math.hypot(q[0]![0] - q[2]![0], q[0]![1] - q[2]![1]), Math.hypot(q[1]![0] - q[3]![0], q[1]![1] - q[3]![1]));

/**
 * Band-brightness whiteEdge heuristic (fallback when the tracker does not orient): for each of the two rank axes,
 * how much more textured the two outer bands (2 lines each) are than the middle 4 lines tells which axis carries the
 * pieces; on that axis the band whose cells are lighter (relative to the middle cells of the same square colour) is
 * white. Returns k (edge index for BOARD-ordered corners) and a margin (relative contrast).
 */
export function bandHeuristic(frame: { data: Uint8ClampedArray; width: number; height: number }, hb: Mat3): { k: number; margin: number } {
  const g = grayOf({ data: frame.data as unknown as Uint8Array, width: frame.width, height: frame.height });
  const mean = new Float32Array(64);
  const sd = new Float32Array(64);
  for (let c = 0; c < 64; c++) {
    const i = c & 7;
    const j = c >> 3;
    const v: number[] = [];
    for (let a = 0; a < 5; a++)
      for (let b = 0; b < 5; b++) {
        const p = applyH(hb, [i + 0.1 + 0.2 * a, j + 0.1 + 0.2 * b]);
        v.push(sampleGray(g, frame.width, frame.height, p[0], p[1]));
      }
    const m = v.reduce((s, x) => s + x, 0) / v.length;
    mean[c] = m;
    sd[c] = Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length);
  }
  const avg = (cells: number[], f: Float32Array) => cells.reduce((s, c) => s + f[c]!, 0) / Math.max(1, cells.length);
  const best = { k: 0, margin: -Infinity, tex: -Infinity };
  for (const axis of [0, 1]) {
    const line = (c: number) => (axis === 0 ? c >> 3 : c & 7);
    const cells = (lo: number, hi: number) => Array.from({ length: 64 }, (_, c) => c).filter((c) => line(c) >= lo && line(c) <= hi);
    const mid = cells(2, 5);
    const lo = cells(0, 1);
    const hi = cells(6, 7);
    const tex = avg(lo, sd) + avg(hi, sd) - 2 * avg(mid, sd);
    if (tex <= best.tex) continue;
    const midPar = [0, 1].map((p) => avg(mid.filter((c) => ((c & 7) + (c >> 3)) % 2 === p), mean));
    const rel = (cs: number[]) => cs.reduce((s, c) => s + mean[c]! - midPar[((c & 7) + (c >> 3)) % 2]!, 0) / cs.length;
    const dLo = rel(lo);
    const dHi = rel(hi);
    const spread = Math.max(1, Math.abs(midPar[0]! - midPar[1]!));
    // Edge indices of BOARD order: y = 0 is edge 0 (corner 0 -> 1), y = 8 edge 2, x = 8 edge 1, x = 0 edge 3.
    const k = axis === 0 ? (dLo > dHi ? 0 : 2) : dLo > dHi ? 3 : 1;
    best.k = k;
    best.tex = tex;
    best.margin = Math.abs(dLo - dHi) / spread;
  }
  return { k: best.k, margin: best.margin };
}

export interface ProposeOptions {
  sizes?: number[];
  params?: Params;
  force?: boolean;
  filter?: string;
  refine?: boolean;
}

export function proposeOne(cv: CV, det: Detector, img: Rgba, opt: ProposeOptions = {}): Omit<DraftEntry, 'reviewed' | 'reviewer' | 'crossChecked' | 'crossChecker' | 'notes'> {
  const sizes = opt.sizes ?? [640, 960];
  const params = opt.params ?? {};
  const flags: string[] = [];
  const dets: { size: number; conf: number; corners: Point[] }[] = [];
  for (const size of sizes) {
    const frame = toFrame(cv, img, size);
    const s = img.width / frame.width;
    const res = det.detect(frame as unknown as ImageData, params);
    if (res.corners) dets.push({ size, conf: res.confidence, corners: res.corners.map(([x, y]) => [x * s, y * s] as Point) });
  }
  const base = {
    width: img.width, height: img.height, confidence: null, detectSize: null, sizeSpread: null, refine: null,
    whiteEdgeMethod: null, whiteEdgeTracker: null, whiteEdgeHeuristic: null, heuristicMargin: null,
  };
  if (!dets.length) return { ...base, corners: null, whiteEdge: null, source: 'none', flags: ['no detection: annotate by hand (ANNOT=at)'] };
  dets.sort((a, b) => b.conf - a.conf);
  const bestDet = dets[0]!;
  let spread: number | null = null;
  if (dets.length > 1) {
    spread = Math.max(...dets.slice(1).map((d) => cornerErrors(d.corners, bestDet.corners).max));
    if (spread > 0.02) flags.push(`detection sizes disagree by ${(spread * 100).toFixed(1)}% of the diagonal (possible off-by-one cell)`);
  }
  if (dets.length < sizes.length) flags.push(`detected at ${dets.map((d) => d.size).join(',')} only (of ${sizes.join(',')})`);
  let quad = normaliseQuad(bestDet.corners)!.corners;
  let refine: DraftEntry['refine'] = null;
  if (opt.refine !== false) {
    const g = grayOf(img);
    const r = refineLattice(g, img.width, img.height, homographyFrom4(BOARD, quad)!);
    if (r) {
      const rq = normaliseQuad(BOARD.map((p) => applyH(r.hb, p)))?.corners ?? null;
      const shift = rq ? cornerErrors(rq, quad).max : Infinity;
      const used = !!rq && r.inliers >= 12 && r.rms < 3 && shift < 0.03;
      refine = { used, inliers: r.inliers, rms: Math.round(r.rms * 100) / 100, shift: Math.round(shift * 10000) / 100 };
      if (used) quad = rq!;
      else flags.push(`lattice refinement rejected (inliers ${r.inliers}, rms ${r.rms.toFixed(2)} px, shift ${(shift * 100).toFixed(2)}%)`);
    } else flags.push('lattice refinement failed (too few clear inner grid corners)');
  }
  quad = quad.map(([x, y]) => [r1(x), r1(y)] as Point);
  // whiteEdge: the app's tracker at the worker size on the proposed corners; band heuristic as fallback.
  const frame = toFrame(cv, img, 640);
  const s = frame.width / img.width;
  const q640 = quad.map(([x, y]) => [x * s, y * s] as Point);
  const hb640 = homographyFrom4(BOARD, q640)!;
  const run = runTracker(frame, hb640, params, null, true);
  let weTracker: number | null = null;
  const o = run.out.orientation;
  if (o) for (let k = 0; k < 4; k++) if (expectedOrientation(q640, k, hb640)?.every((v, sq) => v === o[sq])) weTracker = k;
  const heur = bandHeuristic(frame, hb640);
  const whiteEdge = weTracker ?? heur.k;
  if (weTracker === null) flags.push(`tracker did not orient (boot ${run.r.boot}${run.r.orientReason ? `, ${run.r.orientReason}` : ''}): whiteEdge from the band heuristic, CHECK IT`);
  else if (weTracker !== heur.k) flags.push(`whiteEdge: tracker ${weTracker} vs heuristic ${heur.k}: check`);
  return {
    ...base, corners: quad, whiteEdge, source: 'detector', confidence: Math.round(bestDet.conf * 1000) / 1000, detectSize: bestDet.size,
    sizeSpread: spread === null ? null : Math.round(spread * 10000) / 100, refine,
    whiteEdgeMethod: weTracker !== null ? 'tracker' : 'heuristic', whiteEdgeTracker: weTracker, whiteEdgeHeuristic: heur.k,
    heuristicMargin: Math.round(heur.margin * 100) / 100, flags,
  };
}

export function listJpegs(dir: string, filter?: string): string[] {
  return fs.readdirSync(dir).filter((f) => JPG_RE.test(f) && (!filter || f.includes(filter))).sort();
}

export function loadImage(dir: string, file: string): Rgba {
  const b = fs.readFileSync(path.join(dir, file));
  const { img, orientation } = decodeImage(b);
  if (orientation !== 1) console.warn(`[warn] ${file}: EXIF orientation ${orientation} (run ingest to bake it in; the bench ignores EXIF)`);
  return img;
}

export function propose(cv: CV, det: Detector, dir: string, opt: ProposeOptions = {}): string[] {
  const log: string[] = [];
  const f = path.join(dir, 'corners.draft.json');
  const draft = readJson<Record<string, unknown>>(f, {});
  draft._comment ??= 'Draft corners for annotation (tests/fixtures/web/ANNOTATION.md). Same corner order and whiteEdge meaning as corners.json; promote copies entries with "reviewed": true into corners.json.';
  for (const file of listJpegs(dir, opt.filter)) {
    const prev = draft[file] as DraftEntry | undefined;
    if (prev?.reviewed) {
      log.push(`[keep] ${file}: reviewed, not re-proposed`);
      continue;
    }
    if (prev && !opt.force) {
      log.push(`[keep] ${file}: draft entry exists (ANNOT_FORCE=1 to re-propose unreviewed entries)`);
      continue;
    }
    const img = loadImage(dir, file);
    if (Math.max(img.width, img.height) > 2000) log.push(`[warn] ${file}: ${img.width}x${img.height}, run ingest first`);
    const p = proposeOne(cv, det, img, opt);
    const negative = file.startsWith('neg-');
    // Annotation fields first (readable one-line entries), then provenance.
    const e: DraftEntry = Object.assign({ width: 0, height: 0, corners: null, whiteEdge: null, source: 'none' as const, reviewed: false }, p, { reviewer: null, crossChecked: false, crossChecker: null, notes: prev?.notes ?? null });
    if (negative) {
      e.flags = [p.corners ? 'NEGATIVE image but the detector found a board (false positive); corners must stay null' : 'negative image, no detection (ok)'];
      e.corners = null;
      e.whiteEdge = null;
    }
    draft[file] = e;
    log.push(`[propose] ${file}: ${e.corners ? `conf ${e.confidence} @${e.detectSize}, refine ${e.refine?.used ? `${e.refine.inliers} pts rms ${e.refine.rms}px shift ${e.refine.shift}%` : 'no'}, whiteEdge ${e.whiteEdge} (${e.whiteEdgeMethod})` : negative ? 'negative' : 'NO DETECTION'}${e.flags.length ? `  FLAGS: ${e.flags.join('; ')}` : ''}`);
  }
  writeCornersJson(f, draft);
  log.push(`wrote ${path.relative(REPO, f)}`);
  return log;
}

// ---------------------------------------------------------------------------------------------------------------
// Crops and previews

/**
 * Zoomed crop around (cx, cy): nearest-neighbour upscale x zoom, a pixel tick grid (every 10 px, labelled every 50 px
 * with absolute image coordinates), the projected board grid lines of hb (yellow, boundary lines extended by one cell,
 * cyan on the white edge) and a red crosshair (with a gap, so the pixel under it stays visible) at (cx, cy). Extra
 * marks (e.g. the corners.json corner) as green crosses.
 */
export function renderCrop(img: Rgba, cx: number, cy: number, title: string, hb: Mat3 | null, whiteEdge: number | null, opt: { size?: number; zoom?: number; marks?: Point[]; ticks?: boolean } = {}): { data: Uint8ClampedArray; width: number; height: number } {
  const size = opt.size ?? 200;
  const zoom = opt.zoom ?? 3;
  const W = Math.round(size * zoom);
  const x0 = Math.floor(cx - size / 2);
  const y0 = Math.floor(cy - size / 2);
  const out = new Uint8ClampedArray(W * W * 4);
  for (let Y = 0; Y < W; Y++)
    for (let X = 0; X < W; X++) {
      const sx = x0 + Math.floor(X / zoom);
      const sy = y0 + Math.floor(Y / zoom);
      const d = (Y * W + X) * 4;
      out[d + 3] = 255;
      if (sx < 0 || sy < 0 || sx >= img.width || sy >= img.height) {
        out[d] = out[d + 1] = out[d + 2] = ((X >> 3) + (Y >> 3)) % 2 ? 40 : 60; // outside the image: dark checker
        continue;
      }
      const s = (sy * img.width + sx) * 4;
      out[d] = img.data[s]!;
      out[d + 1] = img.data[s + 1]!;
      out[d + 2] = img.data[s + 2]!;
    }
  const c = new Canvas(out, W, W);
  // Pixel centres at integer coordinates (OpenCV convention): pixel x covers [x - 0.5, x + 0.5).
  const P = (p: Point): Point => [(p[0] - x0 + 0.5) * zoom, (p[1] - y0 + 0.5) * zoom];
  if (opt.ticks !== false) {
    const step = size <= 300 ? 10 : 50;
    const major = step * 5;
    for (let v = Math.ceil(x0 / step) * step; v <= x0 + size; v += step) {
      const X = (v - x0 + 0.5) * zoom;
      const maj = v % major === 0;
      c.line([X, 0], [X, W - 1], maj ? WHITE : BLACK, 1, maj ? 0.35 : 0.18);
      if (maj) c.text(String(v), Math.round(X) + 2, 2, 1, WHITE, BLACK, 0.6);
    }
    for (let v = Math.ceil(y0 / step) * step; v <= y0 + size; v += step) {
      const Y = (v - y0 + 0.5) * zoom;
      const maj = v % major === 0;
      c.line([0, Y], [W - 1, Y], maj ? WHITE : BLACK, 1, maj ? 0.35 : 0.18);
      if (maj) c.text(String(v), 2, Math.round(Y) + 2, 1, WHITE, BLACK, 0.6);
    }
  }
  if (hb) {
    for (let k = 0; k <= 8; k++) {
      const ext = k === 0 || k === 8 ? 1 : 0;
      for (const axis of [0, 1]) {
        const a: Point = axis === 0 ? [k, -ext] : [-ext, k];
        const b: Point = axis === 0 ? [k, 8 + ext] : [8 + ext, k];
        const pa = P(applyH(hb, a));
        const pb = P(applyH(hb, b));
        if (Math.max(Math.abs(pa[0]), Math.abs(pa[1]), Math.abs(pb[0]), Math.abs(pb[1])) > 1e5) continue;
        c.line(pa, pb, YELLOW, 1, ext ? 0.8 : 0.45);
      }
    }
    if (whiteEdge !== null) {
      const e = [BOARD[whiteEdge]!, BOARD[(whiteEdge + 1) % 4]!];
      c.line(P(applyH(hb, e[0]!)), P(applyH(hb, e[1]!)), CYAN, 2, 0.8);
    }
  }
  const cross = (p: Point, col: RGB, gap: number, len: number) => {
    const [X, Y] = P(p);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) c.line([X + dx * gap, Y + dy * gap], [X + dx * len, Y + dy * len], col, 1);
  };
  for (const m of opt.marks ?? []) cross(m, GREEN, 4, 14);
  cross([cx, cy], RED, Math.max(4, zoom * 2), Math.max(20, zoom * 12));
  c.text(title, 4, W - 14, 1, WHITE, BLACK, 0.7);
  return { data: out, width: W, height: W };
}

export interface CropJob {
  dir: string;
  outDir: string;
  /** draft (corners.draft.json, default) or gt (corners.json). */
  from?: 'draft' | 'gt';
  filter?: string;
  size?: number;
  zoom?: number;
  previewSize?: number;
}

function entryOf(dir: string, file: string, from: 'draft' | 'gt'): CornerEntry | null {
  const f = path.join(dir, from === 'gt' ? 'corners.json' : 'corners.draft.json');
  return (readJson<Record<string, CornerEntry>>(f, {})[file] as CornerEntry | undefined) ?? null;
}

/** Corner entry scaled to the image's pixel size (entries record the size they were annotated at). */
function scaledCorners(e: CornerEntry | null, img: Rgba): Point[] | null {
  if (!e?.corners) return null;
  return e.corners.map(([x, y]) => [(x * img.width) / (e.width || img.width), (y * img.height) / (e.height || img.height)] as Point);
}

export function preview(cv: CV, img: Rgba, quad: Point[] | null, whiteEdge: number | null, title: string, size = 1000): { data: Uint8ClampedArray; width: number; height: number } {
  const vis = resizeRgba(cv, img, size);
  const s = vis.width / img.width;
  const c = new Canvas(vis.data, vis.width, vis.height);
  if (quad) drawGrid(c, quad.map(([x, y]) => [x * s, y * s] as Point), whiteEdge ?? null, title);
  else c.text(`${title} NO CORNERS`, 6, 6, 2, WHITE);
  return vis;
}

export function crops(cv: CV, job: CropJob): string[] {
  const log: string[] = [];
  const from = job.from ?? 'draft';
  fs.mkdirSync(job.outDir, { recursive: true });
  for (const file of listJpegs(job.dir, job.filter)) {
    const e = entryOf(job.dir, file, from);
    const other = entryOf(job.dir, file, from === 'gt' ? 'draft' : 'gt');
    const img = loadImage(job.dir, file);
    const q = scaledCorners(e, img);
    const oq = scaledCorners(other, img);
    const stem = file.replace(JPG_RE, '');
    const we = e?.whiteEdge ?? null;
    const pv = preview(cv, img, q, we, `${file} ${from.toUpperCase()} WHITEEDGE ${we ?? '?'}`, job.previewSize ?? 1000);
    fs.writeFileSync(path.join(job.outDir, `${stem}.preview.png`), encodePng(pv.data, pv.width, pv.height));
    if (!q) {
      log.push(`[crops] ${file}: no ${from} corners (preview only); use ANNOT=at with ANNOT_AT=x,y;... for manual guesses`);
      continue;
    }
    const hb = homographyFrom4(BOARD, q);
    q.forEach((p, k) => {
      const marks = oq ? [oq.reduce((b, m) => (Math.hypot(m[0] - p[0], m[1] - p[1]) < Math.hypot(b[0] - p[0], b[1] - p[1]) ? m : b))] : [];
      const t = `${stem} C${k} X ${p[0].toFixed(1)} Y ${p[1].toFixed(1)}`;
      const z = renderCrop(img, p[0], p[1], t, hb, we, { size: job.size, zoom: job.zoom, marks });
      fs.writeFileSync(path.join(job.outDir, `${stem}.c${k}.png`), encodePng(z.data, z.width, z.height));
      const wide = renderCrop(img, p[0], p[1], `${stem} C${k} WIDE`, hb, we, { size: 600, zoom: 1, marks, ticks: true });
      fs.writeFileSync(path.join(job.outDir, `${stem}.c${k}.wide.png`), encodePng(wide.data, wide.width, wide.height));
    });
    log.push(`[crops] ${file}: preview + 4 corner crops (+ wide) from ${from}${oq ? ` (green cross: ${from === 'gt' ? 'draft' : 'corners.json'} corner)` : ''}`);
  }
  log.push(`PNGs in ${path.relative(REPO, job.outDir)}`);
  return log;
}

/** Crops at arbitrary points "x,y;x,y;..." of one image (manual first guesses), named <stem>.at-<x>-<y>.png. */
export function cropsAt(cv: CV, dir: string, file: string, points: Point[], outDir: string, opt: { size?: number; zoom?: number } = {}): string[] {
  fs.mkdirSync(outDir, { recursive: true });
  const img = loadImage(dir, file);
  const stem = file.replace(JPG_RE, '');
  const q = scaledCorners(entryOf(dir, file, 'draft'), img);
  const hb = q ? homographyFrom4(BOARD, q) : null;
  const log: string[] = [];
  for (const [x, y] of points) {
    const z = renderCrop(img, x, y, `${stem} X ${x} Y ${y}`, hb, null, opt);
    const name = `${stem}.at-${Math.round(x)}-${Math.round(y)}.png`;
    fs.writeFileSync(path.join(outDir, name), encodePng(z.data, z.width, z.height));
    log.push(`[at] ${name}`);
  }
  if (points.length === 4) {
    const n = normaliseQuad(points);
    const pv = preview(cv, img, n?.corners ?? null, null, `${file} MANUAL GUESS`);
    fs.writeFileSync(path.join(outDir, `${stem}.at-preview.png`), encodePng(pv.data, pv.width, pv.height));
    log.push(`[at] ${stem}.at-preview.png (the 4 points as a quad${n ? `, normalised order ${JSON.stringify(n.corners)}` : ', NOT a convex quad'})`);
  }
  log.push(`image ${img.width}x${img.height}; PNGs in ${path.relative(REPO, outDir)}`);
  return log;
}

// ---------------------------------------------------------------------------------------------------------------
// Promote / compare

export function promote(dir: string, opt: { force?: boolean; filter?: string } = {}): string[] {
  const log: string[] = [];
  const draft = readJson<Record<string, DraftEntry>>(path.join(dir, 'corners.draft.json'), {});
  const cf = path.join(dir, 'corners.json');
  const gt = readJson<Record<string, unknown>>(cf, {});
  gt._comment ??= 'whiteEdge k: the white pieces start along edge corners[k] -> corners[k+1]. Outer corners of the 8x8 playing area (the 64 squares) in the pixel coordinates of the stored file, clockwise starting from the corner with the smallest x+y. Proposed by the detector + lattice refinement (tests/tools/annotate.ts), reviewed and adjusted on zoomed crops, cross-checked by a second annotator.';
  let n = 0;
  for (const [file, e] of Object.entries(draft)) {
    if (file.startsWith('_') || (opt.filter && !file.includes(opt.filter))) continue;
    if (!e.reviewed) {
      log.push(`[skip] ${file}: not reviewed`);
      continue;
    }
    if (!fs.existsSync(path.join(dir, file))) {
      log.push(`[error] ${file}: image not found`);
      continue;
    }
    let entry: CornerEntry;
    if (file.startsWith('neg-') || e.corners === null) {
      if (!file.startsWith('neg-')) {
        log.push(`[error] ${file}: reviewed but corners null (only neg-* images may have null corners)`);
        continue;
      }
      entry = { width: e.width, height: e.height, corners: null };
    } else {
      const nq = normaliseQuad(e.corners, e.whiteEdge ?? null);
      if (!nq) {
        log.push(`[error] ${file}: corners are not a convex quad`);
        continue;
      }
      if (nq.whiteEdge === null) {
        log.push(`[error] ${file}: whiteEdge missing`);
        continue;
      }
      if (nq.corners.some((p, i) => p[0] !== e.corners![i]![0] || p[1] !== e.corners![i]![1])) log.push(`[note] ${file}: corners reordered to clockwise-from-smallest-x+y, whiteEdge ${e.whiteEdge} -> ${nq.whiteEdge}`);
      entry = { width: e.width, height: e.height, corners: nq.corners, whiteEdge: nq.whiteEdge };
    }
    if (!e.crossChecked) log.push(`[warn] ${file}: promoted without a cross-check (crossChecked false)`);
    const prev = gt[file];
    if (prev !== undefined && JSON.stringify(prev) !== JSON.stringify(entry) && !opt.force) {
      log.push(`[skip] ${file}: corners.json already has a different entry (ANNOT_FORCE=1 to overwrite): ${JSON.stringify(prev)}`);
      continue;
    }
    gt[file] = entry;
    n++;
    log.push(`[promote] ${file}: ${JSON.stringify(entry)}`);
  }
  writeCornersJson(cf, gt);
  log.push(`${n} entr${n === 1 ? 'y' : 'ies'} promoted to ${path.relative(REPO, cf)}. Review overlays: REALOCC_GRID_ONLY=1 REALOCC_DIRS=${path.relative(REPO, dir)} BENCH_ONLY=realocc npm run bench`);
  return log;
}

export interface CompareRow {
  file: string;
  errMean: number | null;
  errMax: number | null;
  perCornerPx: number[] | null;
  weA: number | null;
  weB: number | null;
  /** a1 / h1 / h8 / a8 of A and B coincide (whiteEdge agrees up to the corner relabelling). */
  weMatch: boolean | null;
  note: string;
}

/** Compares two corner files (e.g. a draft against corners.json, or two annotators' drafts), in A's pixel frame. */
export function compare(fa: string, fb: string, filter?: string): { rows: CompareRow[]; text: string } {
  const A = readJson<Record<string, CornerEntry>>(fa, {});
  const B = readJson<Record<string, CornerEntry>>(fb, {});
  const rows: CompareRow[] = [];
  for (const file of Object.keys(A).filter((k) => !k.startsWith('_') && (!filter || k.includes(filter))).sort()) {
    const a = A[file]!;
    const b = B[file];
    const row: CompareRow = { file, errMean: null, errMax: null, perCornerPx: null, weA: a.whiteEdge ?? null, weB: b?.whiteEdge ?? null, weMatch: null, note: '' };
    if (!b) row.note = 'not in B';
    else if (!a.corners || !b.corners) row.note = a.corners || b.corners ? `corners null in ${a.corners ? 'B' : 'A'}` : 'both null (negative)';
    else {
      const bq = b.corners.map(([x, y]) => [(x * a.width) / (b.width || a.width), (y * a.height) / (b.height || a.height)] as Point);
      const e = cornerErrors(a.corners, bq);
      row.errMean = e.mean;
      row.errMax = e.max;
      row.perCornerPx = a.corners.map((p) => Math.min(...bq.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1]))));
      if (a.whiteEdge !== null && a.whiteEdge !== undefined && b.whiteEdge !== null && b.whiteEdge !== undefined) {
        const ca = chessCorners(a.corners, a.whiteEdge);
        const cb = chessCorners(bq, b.whiteEdge);
        row.weMatch = ca.every((p, i) => Math.hypot(p[0] - cb[i]![0], p[1] - cb[i]![1]) < 0.1 * diagOf(bq));
      }
    }
    rows.push(row);
  }
  const f2 = (x: number | null) => (x === null ? '-' : (x * 100).toFixed(2));
  const L = [`A = ${path.relative(REPO, fa)}  B = ${path.relative(REPO, fb)}  (errors in % of B's diagonal; px in A's pixels)`, 'image                                  mean%   max%   per-corner px (A order)      whiteEdge A/B  match'];
  for (const r of rows)
    L.push(`${r.file.padEnd(38)} ${f2(r.errMean).padStart(6)} ${f2(r.errMax).padStart(6)}   ${(r.perCornerPx?.map((v) => v.toFixed(1).padStart(5)).join(' ') ?? '-').padEnd(27)}  ${`${r.weA ?? '-'}/${r.weB ?? '-'}`.padStart(9)}  ${r.weMatch === null ? '-' : r.weMatch ? 'yes' : 'NO'}  ${r.note}`);
  const errs = rows.filter((r) => r.errMean !== null);
  if (errs.length) L.push(`mean over ${errs.length}: ${f2(errs.reduce((s, r) => s + r.errMean!, 0) / errs.length)}% mean, worst max ${f2(Math.max(...errs.map((r) => r.errMax!)))}%; whiteEdge match ${rows.filter((r) => r.weMatch).length}/${rows.filter((r) => r.weMatch !== null).length}`);
  return { rows, text: L.join('\n') };
}

