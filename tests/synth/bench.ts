import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import type { Point } from '../../src/geom/homography';
import { inferProfile, type BoardProfile } from '../../src/vision/profile';
import { Detector, type DetectResult } from '../../src/vision/detector';
import type { CV } from '../../src/vision/preprocess';
import type { Params } from '../../src/worker/protocol';
import { makeBoardSample, makeNegative, NEGATIVE_KINDS, type Sample, type SampleSpec } from './generate';
import { encodePng } from './png';

export const ERR_THRESHOLD = 0.02;

export type Quad = readonly Point[];

const diag = (q: Quad) => Math.max(Math.hypot(q[0]![0] - q[2]![0], q[0]![1] - q[2]![1]), Math.hypot(q[1]![0] - q[3]![0], q[1]![1] - q[3]![1]));

/** Mean corner error over the best of 8 corner assignments (4 rotations x 2 windings), relative to the GT diagonal. */
export function cornerError(det: Quad, gt: Quad): number {
  let best = Infinity;
  for (const rev of [false, true]) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let i = 0; i < 4; i++) {
        const j = rev ? (r - i + 8) % 4 : (r + i) % 4;
        s += Math.hypot(det[j]![0] - gt[i]![0], det[j]![1] - gt[i]![1]);
      }
      best = Math.min(best, s / 4);
    }
  }
  return best / diag(gt);
}

export interface Case {
  label: string;
  name: string;
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  gt: Quad | null;
}

export interface Outcome {
  c: Case;
  res: DetectResult;
  err: number | null;
  ok: boolean;
  falsePositive: boolean;
  ms: number;
}

export const SPECS: SampleSpec[] = [];
for (const elev of ['overhead', 'oblique'] as const)
  for (const palette of ['wood', 'vinyl', 'printed'] as const)
    for (const pieces of ['none', 'some', 'outer'] as const) SPECS.push({ elev, palette, pieces });

export function synthCases(cv: CV, n: number, seedBase = 1000, filter?: (s: SampleSpec) => boolean): Case[] {
  const cases: Case[] = [];
  let k = 0;
  for (const spec of SPECS) {
    if (filter && !filter(spec)) continue;
    k++;
    for (let i = 0; i < n; i++) {
      const seed = seedBase + k * 1000 + i;
      const s: Sample = makeBoardSample(cv, seed, spec);
      cases.push({ label: s.label, name: `${s.label}-${seed}`, rgba: s.rgba, width: s.width, height: s.height, gt: s.corners });
    }
  }
  return cases;
}

export function negativeCases(cv: CV, seedBase = 900000): Case[] {
  return NEGATIVE_KINDS.map((kind, i) => {
    const s = makeNegative(cv, seedBase + i, kind);
    return { label: 'negative', name: `negative-${kind}-${seedBase + i}`, rgba: s.rgba, width: s.width, height: s.height, gt: null };
  });
}

const REAL_DIR = path.resolve(import.meta.dirname, '../fixtures/real');

interface RealEntry {
  width: number;
  height: number;
  corners: [number, number][] | null;
}

/** Real photos downscaled to a long side of `longSide` px; corners scaled accordingly. */
export function realCases(cv: CV, longSide: number): Case[] {
  const require = createRequire(import.meta.url);
  const jpeg = require('jpeg-js') as { decode(b: Buffer, o: { useTArray: boolean }): { width: number; height: number; data: Uint8Array } };
  const meta = JSON.parse(fs.readFileSync(path.join(REAL_DIR, 'corners.json'), 'utf8')) as Record<string, RealEntry>;
  const cases: Case[] = [];
  for (const name of fs.readdirSync(REAL_DIR).filter((f) => f.endsWith('.jpg')).sort()) {
    const img = jpeg.decode(fs.readFileSync(path.join(REAL_DIR, name)), { useTArray: true });
    const scale = longSide / Math.max(img.width, img.height);
    const w = Math.round(img.width * scale);
    const h = Math.round(img.height * scale);
    const src = new cv.Mat(img.height, img.width, cv.CV_8UC4);
    const dst = new cv.Mat();
    try {
      src.data.set(img.data);
      cv.resize(src, dst, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
      const e = meta[name];
      const gt = e?.corners ? (e.corners.map(([x, y]) => [(x * w) / img.width, (y * h) / img.height]) as Point[]) : null;
      cases.push({ label: `real-${longSide}`, name: `${name.replace('.jpg', '')}@${longSide}`, rgba: new Uint8ClampedArray(dst.data), width: w, height: h, gt });
    } finally {
      src.delete();
      dst.delete();
    }
  }
  return cases;
}

/** Minimal ImageData look-alike (Node has no ImageData). */
const imageOf = (c: Case) => ({ data: c.rgba, width: c.width, height: c.height }) as unknown as ImageData;

export function runCases(det: Detector, cases: Case[], params: Params = {}, debug = false): Outcome[] {
  // Warm-up so JIT/wasm compilation does not distort the first timings.
  const first = cases[0];
  if (first) det.detect(imageOf(first), params);
  return cases.map((c) => {
    const t = performance.now();
    const res = det.detect(imageOf(c), params, { debug });
    const ms = performance.now() - t;
    let err: number | null = null;
    if (res.corners && c.gt) err = cornerError(res.corners, c.gt);
    const ok = err !== null && err < ERR_THRESHOLD;
    return { c, res, err, ok, falsePositive: !c.gt && res.corners !== null, ms };
  });
}

/**
 * Locked run: each case is first detected unlocked (that detection defines the profile, as the first
 * confident detection would in the app), then detected again with the profile locked. Returns the locked
 * outcomes (times are those of the locked run) and the profile per case (null when nothing was inferred, in
 * which case the locked run is the unlocked one).
 */
export function runCasesLocked(det: Detector, cases: Case[], params: Params = {}, debug = false): { outs: Outcome[]; profiles: (BoardProfile | null)[] } {
  const first = cases[0];
  if (first) {
    det.detect(imageOf(first), params);
    const p = inferProfile(det.detect(imageOf(first), params));
    if (p) det.detect(imageOf(first), params, { profile: p });
  }
  const profiles: (BoardProfile | null)[] = [];
  const outs = cases.map((c) => {
    const unlocked = det.detect(imageOf(c), params);
    const profile = inferProfile(unlocked);
    profiles.push(profile);
    const t = performance.now();
    const res = profile ? det.detect(imageOf(c), params, { debug, profile }) : det.detect(imageOf(c), params, { debug });
    const ms = performance.now() - t;
    let err: number | null = null;
    if (res.corners && c.gt) err = cornerError(res.corners, c.gt);
    const ok = err !== null && err < ERR_THRESHOLD;
    return { c, res, err, ok, falsePositive: !c.gt && res.corners !== null, ms };
  });
  return { outs, profiles };
}

export interface Row {
  label: string;
  n: number;
  recall: number;
  wrong: number;
  meanErr: number;
  fp: number;
  ms: number;
}

export function summarize(outs: Outcome[]): Row[] {
  const groups = new Map<string, Outcome[]>();
  for (const o of outs) {
    const g = groups.get(o.c.label) ?? [];
    g.push(o);
    groups.set(o.c.label, g);
  }
  const rows: Row[] = [];
  for (const [label, g] of groups) {
    const pos = g.filter((o) => o.c.gt);
    const okN = pos.filter((o) => o.ok).length;
    const errs = pos.filter((o) => o.ok).map((o) => o.err!);
    rows.push({
      label,
      n: g.length,
      recall: pos.length ? okN / pos.length : NaN,
      wrong: pos.filter((o) => o.err !== null && !o.ok).length,
      meanErr: errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : NaN,
      fp: g.filter((o) => o.falsePositive).length,
      ms: g.reduce((a, o) => a + o.ms, 0) / g.length,
    });
  }
  return rows;
}

export function formatTable(rows: Row[]): string {
  const pct = (x: number) => (Number.isNaN(x) ? '   -  ' : `${(x * 100).toFixed(0).padStart(4)}% `);
  const lines = [
    'category                          n  recall  wrong  err%(ok)   FP    ms',
    '-------------------------------------------------------------------------',
  ];
  for (const r of rows) {
    lines.push(
      `${r.label.padEnd(30)} ${String(r.n).padStart(4)}  ${pct(r.recall)}  ${String(r.wrong).padStart(4)}   ${Number.isNaN(r.meanErr) ? '  -  ' : (r.meanErr * 100).toFixed(2).padStart(5)}   ${String(r.fp).padStart(3)}  ${r.ms.toFixed(1).padStart(5)}`,
    );
  }
  return lines.join('\n');
}

/** Aggregates the pieces / no-pieces recall over all positive synthetic rows. */
export function recallWhere(outs: Outcome[], pred: (label: string) => boolean): { n: number; recall: number } {
  const pos = outs.filter((o) => o.c.gt && pred(o.c.label));
  return { n: pos.length, recall: pos.length ? pos.filter((o) => o.ok).length / pos.length : NaN };
}

// ---- failure dumps ----

function line(img: Uint8ClampedArray, w: number, h: number, x1: number, y1: number, x2: number, y2: number, c: [number, number, number], th = 1): void {
  const n = Math.ceil(Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1))) + 1;
  for (let i = 0; i <= n; i++) {
    const x = Math.round(x1 + ((x2 - x1) * i) / n);
    const y = Math.round(y1 + ((y2 - y1) * i) / n);
    for (let dy = 0; dy < th; dy++)
      for (let dx = 0; dx < th; dx++) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const o = (yy * w + xx) * 4;
        img[o] = c[0];
        img[o + 1] = c[1];
        img[o + 2] = c[2];
      }
  }
}

/** Writes an annotated PNG: segments by family (blue/yellow/grey), GT quad (green), detection (red), candidate (orange). */
export function dumpFailure(o: Outcome, dir: string): string {
  fs.mkdirSync(dir, { recursive: true });
  const { width: w, height: h } = o.c;
  const img = new Uint8ClampedArray(o.c.rgba);
  const d = o.res.debug;
  if (d) {
    d.segments.forEach((s, i) => {
      const f = d.family[i];
      line(img, w, h, s.x1, s.y1, s.x2, s.y2, f === 1 ? [40, 120, 255] : f === 2 ? [255, 220, 0] : [150, 150, 150]);
    });
    if (d.best) {
      const q = [d.best.hb].map((hb) => {
        const p = (x: number, y: number): Point => {
          const ww = hb[6] * x + hb[7] * y + hb[8];
          return [(hb[0] * x + hb[1] * y + hb[2]) / ww, (hb[3] * x + hb[4] * y + hb[5]) / ww];
        };
        return [p(0, 0), p(8, 0), p(8, 8), p(0, 8)];
      })[0]!;
      for (let i = 0; i < 4; i++) line(img, w, h, q[i]![0], q[i]![1], q[(i + 1) % 4]![0], q[(i + 1) % 4]![1], [255, 140, 0], 2);
    }
  }
  if (o.c.gt) for (let i = 0; i < 4; i++) line(img, w, h, o.c.gt[i]![0], o.c.gt[i]![1], o.c.gt[(i + 1) % 4]![0], o.c.gt[(i + 1) % 4]![1], [0, 255, 0], 2);
  if (o.res.corners) for (let i = 0; i < 4; i++) line(img, w, h, o.res.corners[i]![0], o.res.corners[i]![1], o.res.corners[(i + 1) % 4]![0], o.res.corners[(i + 1) % 4]![1], [255, 0, 0], 2);
  const file = path.join(dir, `${o.c.name}.png`);
  fs.writeFileSync(file, encodePng(img, w, h));
  return file;
}
