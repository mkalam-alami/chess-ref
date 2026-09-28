import type { Params } from '../worker/protocol';
import { applyH, type Mat3 } from '../geom/homography';
import type { Segment } from '../geom/segments';
import { param, type ParamSpec } from './preprocess';

export const GRID_PARAMS: readonly ParamSpec[] = [
  { name: 'combTol', min: 0.03, max: 0.25, step: 0.01, default: 0.1 },
  { name: 'combK', min: 1, max: 6, step: 1, default: 4 },
  { name: 'maxCandidates', min: 4, max: 40, step: 1, default: 20 },
];

export interface LinePos {
  pos: number;
  /** Support weight in [0, 1]. */
  w: number;
}

/** 9 teeth at start + k * period, k = 0..8. */
export interface CombHypothesis {
  start: number;
  period: number;
  score: number;
  /** Per tooth support in [0, 1]. */
  support: number[];
  shifted?: boolean;
}

export interface CombOptions {
  /** Tooth match tolerance as a fraction of the period. */
  tol: number;
  topK: number;
  /** Also emit +-1 cell shifted variants of this many best hypotheses. */
  shiftBest: number;
}

const TEETH = 9;

/** Support of tooth positions for a given comb and the best 9-tooth window. */
function evaluate(
  pos: readonly LinePos[],
  origin: number,
  p: number,
  tol: number,
): { start: number; score: number; support: number[] } | null {
  const KMIN = -40;
  const sup = new Float64Array(120);
  const ks = new Int32Array(pos.length);
  const rs = new Float64Array(pos.length);
  let kLo = 1e9;
  let kHi = -1e9;
  for (let i = 0; i < pos.length; i++) {
    const t = (pos[i]!.pos - origin) / p;
    const k = Math.round(t);
    ks[i] = k;
    rs[i] = t - k;
    const a = Math.abs(rs[i]!);
    if (a <= tol && k - KMIN >= 0 && k - KMIN < 120) {
      const q = a / tol;
      sup[k - KMIN]! += pos[i]!.w * (1 - q * q);
      if (k < kLo) kLo = k;
      if (k > kHi) kHi = k;
    }
  }
  if (kHi < kLo) return null;
  let best = -1e9;
  let bestM = 0;
  for (let m = kLo - (TEETH - 1); m <= kHi; m++) {
    let s = 0;
    for (let k = m; k < m + TEETH; k++) {
      const j = k - KMIN;
      if (j >= 0 && j < 120) s += Math.min(1, sup[j]!);
    }
    if (s < best - 1e-9) continue;
    // Clutter: strong lines strictly inside the window but off the comb.
    let pen = 0;
    for (let i = 0; i < pos.length; i++) {
      const t = (pos[i]!.pos - origin) / p - m;
      if (t > 0.4 && t < TEETH - 1 - 0.4 && Math.abs(rs[i]!) > 0.3) pen += 0.5 * pos[i]!.w;
    }
    s -= pen;
    if (s > best) {
      best = s;
      bestM = m;
    }
  }
  const support: number[] = [];
  for (let k = bestM; k < bestM + TEETH; k++) {
    const j = k - KMIN;
    support.push(j >= 0 && j < 120 ? Math.min(1, sup[j]!) : 0);
  }
  return { start: origin + bestM * p, score: best, support };
}

/** Weighted least-squares refit of start/period on lines matched to teeth. */
export function refit(pos: readonly LinePos[], start: number, p: number, tol: number): { start: number; period: number } {
  let sw = 0;
  let sk = 0;
  let sp = 0;
  let skk = 0;
  let skp = 0;
  for (const l of pos) {
    const t = (l.pos - start) / p;
    const k = Math.round(t);
    if (k < 0 || k >= TEETH || Math.abs(t - k) > tol) continue;
    sw += l.w;
    sk += l.w * k;
    sp += l.w * l.pos;
    skk += l.w * k * k;
    skp += l.w * k * l.pos;
  }
  const den = sw * skk - sk * sk;
  if (sw < 3 || Math.abs(den) < 1e-9) return { start, period: p };
  const period = (sw * skp - sk * sp) / den;
  const s0 = (sp - period * sk) / sw;
  if (Math.abs(period - p) > 0.08 * p) return { start, period: p };
  return { start: s0, period };
}

/**
 * Comb fit: finds start/period so that 9 evenly spaced teeth explain the observed line positions, tolerating
 * missing teeth. The frame border is just an extra off-comb line outside the window.
 */
export function fitComb(pos: readonly LinePos[], opt: CombOptions): CombHypothesis[] {
  if (pos.length < 4) return [];
  let lo = Infinity;
  let hi = -Infinity;
  for (const l of pos) {
    if (l.pos < lo) lo = l.pos;
    if (l.pos > hi) hi = l.pos;
  }
  const extent = hi - lo;
  if (extent <= 0) return [];
  const pMin = extent / 30;
  const pMax = extent / 5.5;
  const ratio = 1 + Math.min(0.03, opt.tol / 4);
  // Phase anchors: the strongest lines.
  const anchors = [...pos].sort((a, b) => b.w - a.w).slice(0, 14);
  const found: CombHypothesis[] = [];
  for (let p = pMin; p <= pMax; p *= ratio) {
    let bestForP: CombHypothesis | null = null;
    for (const a of anchors) {
      const ev = evaluate(pos, a.pos, p, opt.tol);
      if (!ev) continue;
      if (!bestForP || ev.score > bestForP.score) bestForP = { start: ev.start, period: p, score: ev.score, support: ev.support };
    }
    if (bestForP) found.push(bestForP);
  }
  found.sort((a, b) => b.score - a.score);
  // Refine and de-duplicate.
  const out: CombHypothesis[] = [];
  for (const h of found) {
    if (out.length >= opt.topK * 3) break;
    const r = refit(pos, h.start, h.period, opt.tol * 1.5);
    const ev = evaluate(pos, r.start, r.period, opt.tol);
    if (!ev) continue;
    // The refit start may sit on another window; re-anchor to the evaluated one.
    const cand: CombHypothesis = { start: ev.start, period: r.period, score: ev.score, support: ev.support };
    const dup = out.some((o) => Math.abs(o.period - cand.period) < 0.06 * cand.period && Math.abs(o.start - cand.start) < 0.35 * cand.period);
    if (!dup) out.push(cand);
  }
  out.sort((a, b) => b.score - a.score);
  const top = out.slice(0, opt.topK);
  const extra: CombHypothesis[] = [];
  for (const h of top.slice(0, opt.shiftBest)) {
    for (const d of [-1, 1]) {
      const s = h.start + d * h.period;
      if (top.some((o) => Math.abs(o.period - h.period) < 0.06 * h.period && Math.abs(o.start - s) < 0.35 * h.period)) continue;
      const support = h.support.map((_, k) => {
        let v = 0;
        for (const l of pos) {
          const t = Math.abs((l.pos - (s + k * h.period)) / h.period);
          if (t <= opt.tol) v += l.w * (1 - (t / opt.tol) ** 2);
        }
        return Math.min(1, v);
      });
      extra.push({ start: s, period: h.period, score: support.reduce((a, b) => a + b, 0) - 0.5, support, shifted: true });
    }
  }
  return [...top, ...extra];
}

export interface FamilyLines {
  /** Line positions along the perpendicular axis of the rectified space (family 1: x, family 2: y). */
  pos: LinePos[];
}

/** Position of segments along an axis (0 = x, 1 = y) in rectified space. Segments crossing the horizon are dropped. */
export function segmentPositions(segs: readonly Segment[], idx: readonly number[], H: Mat3, axis: 0 | 1, refLen: number): LinePos[] {
  const res: LinePos[] = [];
  for (const i of idx) {
    const s = segs[i]!;
    const w1 = H[6] * s.x1 + H[7] * s.y1 + H[8];
    const w2 = H[6] * s.x2 + H[7] * s.y2 + H[8];
    if (w1 < 0.03 || w2 < 0.03) continue;
    const a = applyH(H, [s.x1, s.y1]);
    const b = applyH(H, [s.x2, s.y2]);
    res.push({ pos: (a[axis] + b[axis]) / 2, w: Math.min(1, s.weight / refLen) });
  }
  return res;
}

export function combOptions(params: Params): CombOptions {
  return { tol: param(params, GRID_PARAMS, 'combTol'), topK: param(params, GRID_PARAMS, 'combK'), shiftBest: 2 };
}

/** Indices of segments whose rectified position lies on a tooth of the hypothesis (teeth 0..8). */
export function toothSegments(
  segs: readonly Segment[],
  idx: readonly number[],
  H: Mat3,
  axis: 0 | 1,
  hyp: Pick<CombHypothesis, 'start' | 'period'>,
  tolFrac: number,
): number[] {
  const out: number[] = [];
  for (const i of idx) {
    const s = segs[i]!;
    const w1 = H[6] * s.x1 + H[7] * s.y1 + H[8];
    const w2 = H[6] * s.x2 + H[7] * s.y2 + H[8];
    if (w1 < 0.03 || w2 < 0.03) continue;
    const a = applyH(H, [s.x1, s.y1]);
    const b = applyH(H, [s.x2, s.y2]);
    const t = ((a[axis] + b[axis]) / 2 - hyp.start) / hyp.period;
    const k = Math.round(t);
    if (k >= 0 && k < TEETH && Math.abs(t - k) <= tolFrac) out.push(i);
  }
  return out;
}
