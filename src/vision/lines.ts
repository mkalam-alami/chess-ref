import type { Params } from '../worker/protocol';
import { makeSegment, type Segment } from '../geom/segments';
import type { CV, ParamSpec } from './preprocess';
import { param } from './preprocess';

type Mat = InstanceType<CV['Mat']>;

/** Tunables of the line stage. Length-like params are fractions of the image diagonal (x1000). */
export const LINES_PARAMS: readonly ParamSpec[] = [
  { name: 'houghVotes', min: 10, max: 80, step: 1, default: 24 },
  { name: 'minSegLen', min: 10, max: 120, step: 1, default: 40 },
  { name: 'houghGap', min: 2, max: 40, step: 1, default: 12 },
  { name: 'mergeAngle', min: 0.5, max: 8, step: 0.5, default: 2.5 },
  { name: 'mergeOffset', min: 1, max: 12, step: 0.5, default: 4 },
  { name: 'maxSegments', min: 40, max: 400, step: 10, default: 140 },
];

export interface LineOptions {
  votes: number;
  minLen: number;
  gap: number;
  mergeAngleRad: number;
  mergeOffset: number;
  maxSegments: number;
}

/** Scales the thresholds to the image size (they are tuned for an 800 px diagonal ~ 640x480). */
export function lineOptions(params: Params, width: number, height: number): LineOptions {
  const diag = Math.hypot(width, height);
  const k = diag / 800;
  return {
    votes: Math.round(param(params, LINES_PARAMS, 'houghVotes') * k),
    minLen: param(params, LINES_PARAMS, 'minSegLen') * k,
    gap: param(params, LINES_PARAMS, 'houghGap') * k,
    mergeAngleRad: (param(params, LINES_PARAMS, 'mergeAngle') * Math.PI) / 180,
    mergeOffset: param(params, LINES_PARAMS, 'mergeOffset') * k,
    maxSegments: param(params, LINES_PARAMS, 'maxSegments'),
  };
}

/** Raw HoughLinesP segments as [x1,y1,x2,y2] quads. */
export function houghSegments(cv: CV, edges: Mat, o: LineOptions): Float64Array {
  const out = new cv.Mat();
  try {
    cv.HoughLinesP(edges, out, 1, Math.PI / 180, Math.max(5, o.votes), o.minLen, o.gap);
    const n = Math.floor(out.data32S.length / 4);
    const res = new Float64Array(n * 4);
    const d = out.data32S;
    for (let i = 0; i < n * 4; i++) res[i] = d[i]!;
    return res;
  } finally {
    out.delete();
  }
}

interface Cluster {
  /** Unit direction and point on the line (weighted centroid). */
  dx: number;
  dy: number;
  px: number;
  py: number;
  members: number[];
  wsum: number;
}

/**
 * Merges near-collinear segments (similar angle, small perpendicular offset), also across gaps, drops short
 * results and returns at most `maxSegments` sorted by weight. Input: [x1,y1,x2,y2]* quads.
 */
export function mergeSegments(raw: ArrayLike<number>, o: Pick<LineOptions, 'mergeAngleRad' | 'mergeOffset' | 'minLen' | 'maxSegments'>, maxGap = Infinity): Segment[] {
  const n = Math.floor(raw.length / 4);
  const segs: Segment[] = [];
  for (let i = 0; i < n; i++) segs.push(makeSegment(raw[i * 4]!, raw[i * 4 + 1]!, raw[i * 4 + 2]!, raw[i * 4 + 3]!));
  segs.sort((a, b) => b.len - a.len);
  const cosTol = Math.cos(o.mergeAngleRad);
  const clusters: Cluster[] = [];
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i]!;
    let bestC: Cluster | null = null;
    let bestOff = Infinity;
    for (const c of clusters) {
      if (Math.abs(c.dx * s.dx + c.dy * s.dy) < cosTol) continue;
      // Perpendicular offsets of both endpoints from the cluster line.
      const o1 = Math.abs(-c.dy * (s.x1 - c.px) + c.dx * (s.y1 - c.py));
      const o2 = Math.abs(-c.dy * (s.x2 - c.px) + c.dx * (s.y2 - c.py));
      const off = Math.max(o1, o2);
      if (off > o.mergeOffset || off >= bestOff) continue;
      if (maxGap < Infinity) {
        // Gap along the line between the segment and the cluster's current extent.
        let lo = Infinity;
        let hi = -Infinity;
        for (const m of c.members) {
          const g = segs[m]!;
          for (const t of [(g.x1 - c.px) * c.dx + (g.y1 - c.py) * c.dy, (g.x2 - c.px) * c.dx + (g.y2 - c.py) * c.dy]) {
            if (t < lo) lo = t;
            if (t > hi) hi = t;
          }
        }
        const a = (s.x1 - c.px) * c.dx + (s.y1 - c.py) * c.dy;
        const b = (s.x2 - c.px) * c.dx + (s.y2 - c.py) * c.dy;
        const gap = Math.max(Math.min(a, b) - hi, lo - Math.max(a, b), 0);
        if (gap > maxGap) continue;
      }
      bestC = c;
      bestOff = off;
    }
    if (bestC) {
      const c = bestC;
      c.members.push(i);
      // Update the line as the length-weighted mean (orientation-consistent direction).
      const sign = c.dx * s.dx + c.dy * s.dy < 0 ? -1 : 1;
      const w = s.len;
      const nx = c.dx * c.wsum + sign * s.dx * w;
      const ny = c.dy * c.wsum + sign * s.dy * w;
      const nn = Math.hypot(nx, ny) || 1;
      c.px = (c.px * c.wsum + s.mx * w) / (c.wsum + w);
      c.py = (c.py * c.wsum + s.my * w) / (c.wsum + w);
      c.dx = nx / nn;
      c.dy = ny / nn;
      c.wsum += w;
    } else {
      clusters.push({ dx: s.dx, dy: s.dy, px: s.mx, py: s.my, members: [i], wsum: s.len });
    }
  }
  const out: Segment[] = [];
  for (const c of clusters) {
    let lo = Infinity;
    let hi = -Infinity;
    let sum = 0;
    for (const m of c.members) {
      const g = segs[m]!;
      sum += g.len;
      for (const t of [(g.x1 - c.px) * c.dx + (g.y1 - c.py) * c.dy, (g.x2 - c.px) * c.dx + (g.y2 - c.py) * c.dy]) {
        if (t < lo) lo = t;
        if (t > hi) hi = t;
      }
    }
    if (hi - lo < o.minLen) continue;
    out.push(makeSegment(c.px + c.dx * lo, c.py + c.dy * lo, c.px + c.dx * hi, c.py + c.dy * hi, Math.min(sum, hi - lo)));
  }
  out.sort((a, b) => b.weight - a.weight);
  return out.slice(0, o.maxSegments);
}

/** Hough + merge on an edge map. */
export function extractLines(cv: CV, edges: Mat, params: Params, width: number, height: number): Segment[] {
  const o = lineOptions(params, width, height);
  const raw = houghSegments(cv, edges, o);
  return mergeSegments(raw, o, 0.35 * Math.hypot(width, height));
}
