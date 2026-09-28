import { describe, expect, it } from 'vitest';
import { applyH, homographyFrom4, type Mat3, type Point } from '../src/geom/homography';
import { makeSegment, type Segment } from '../src/geom/segments';
import { affineRectification, findVanishingPoints, mulberry32, vpDirection } from '../src/geom/vanishing';
import { fitComb, type LinePos } from '../src/vision/grid';
import { mergeSegments } from '../src/vision/lines';

const opts = () => ({ iterations: 300, angleDeg: 2, minSeparationDeg: 20, rng: mulberry32(1) });

/** Segments of a 9+9 line grid seen through a homography, in roughly [-1, 1] coordinates. */
function gridSegments(h: Mat3, skip: (family: number, k: number) => boolean = () => false): Segment[] {
  const segs: Segment[] = [];
  for (let k = 0; k <= 8; k++) {
    for (let f = 0; f < 2; f++) {
      if (skip(f, k)) continue;
      // Split each line in two pieces with a gap, like a line hidden behind a piece.
      const pts: Array<[number, number]> = [[0.2, 3.5], [4.5, 7.8]];
      for (const [a, b] of pts) {
        const p: Point = f === 0 ? [k, a] : [a, k];
        const q: Point = f === 0 ? [k, b] : [b, k];
        const P = applyH(h, p);
        const Q = applyH(h, q);
        segs.push(makeSegment(P[0], P[1], Q[0], Q[1]));
      }
    }
  }
  return segs;
}

const board = (a: Point[]): Mat3 => homographyFrom4([[0, 0], [8, 0], [8, 8], [0, 8]], a)!;

describe('vanishing points and rectification', () => {
  it('recovers finite VPs of a perspective grid', () => {
    const Q: Point[] = [[-0.4, -0.5], [0.3, -0.4], [0.6, 0.5], [-0.7, 0.55]];
    const h = board(Q);
    const pair = findVanishingPoints(gridSegments(h), opts())!;
    expect(pair).not.toBeNull();
    // Truth: intersections of opposite edges.
    const truth = (a: Point, b: Point, c: Point, d: Point) => {
      const l = (p: Point, q: Point) => [p[1] - q[1], q[0] - p[0], p[0] * q[1] - q[0] * p[1]];
      const l1 = l(a, b);
      const l2 = l(c, d);
      const x = l1[1]! * l2[2]! - l1[2]! * l2[1]!;
      const y = l1[2]! * l2[0]! - l1[0]! * l2[2]!;
      const w = l1[0]! * l2[1]! - l1[1]! * l2[0]!;
      return [x / w, y / w];
    };
    const tv = [truth(Q[0]!, Q[3]!, Q[1]!, Q[2]!), truth(Q[0]!, Q[1]!, Q[3]!, Q[2]!)];
    const est = [pair.vp1.vp, pair.vp2.vp].map((v) => [v[0] / v[2], v[1] / v[2]]);
    for (const t of tv) {
      const dmin = Math.min(...est.map((e) => Math.hypot(e[0]! - t![0]!, e[1]! - t![1]!)));
      expect(dmin).toBeLessThan(0.05 * Math.hypot(t![0]!, t![1]!));
    }
  });

  it('handles an overhead view (both VPs at infinity)', () => {
    const c = Math.cos(0.4);
    const s = Math.sin(0.4);
    const h: Mat3 = [0.1 * c, -0.1 * s, 0.1, 0.1 * s, 0.1 * c, -0.2, 0, 0, 1];
    const pair = findVanishingPoints(gridSegments(h), opts())!;
    expect(pair).not.toBeNull();
    expect(Math.abs(pair.vp1.vp[2])).toBeLessThan(1e-2);
    expect(Math.abs(pair.vp2.vp[2])).toBeLessThan(1e-2);
    const d1 = vpDirection(pair.vp1.vp, pair.ref);
    const d2 = vpDirection(pair.vp2.vp, pair.ref);
    expect(Math.abs(d1[0] * d2[0] + d1[1] * d2[1])).toBeLessThan(0.02);
  });

  it('rectifies so the families become axis aligned and evenly spaced', () => {
    const h = board([[-0.4, -0.5], [0.3, -0.45], [0.6, 0.5], [-0.7, 0.55]]);
    const segs = gridSegments(h);
    const pair = findVanishingPoints(segs, opts())!;
    const rect = affineRectification(pair.vp1.vp, pair.vp2.vp, pair.ref)!;
    expect(rect).not.toBeNull();
    const pos1: number[] = [];
    for (const i of pair.vp1.inliers) {
      const a = applyH(rect.H, [segs[i]!.x1, segs[i]!.y1]);
      const b = applyH(rect.H, [segs[i]!.x2, segs[i]!.y2]);
      expect(Math.abs(a[0] - b[0])).toBeLessThan(0.02 * Math.abs(a[1] - b[1]) + 1e-3); // vertical
      pos1.push((a[0] + b[0]) / 2);
    }
    const uniq = [...new Set(pos1.map((p) => Math.round(p * 1e4) / 1e4))].sort((x, y) => x - y);
    const gaps = uniq.slice(1).map((v, i) => v - uniq[i]!).filter((g) => g > 1e-3);
    const mean = gaps.reduce((x, y) => x + y, 0) / gaps.length;
    for (const g of gaps) expect(Math.abs(g / mean - 1)).toBeLessThan(0.1);
  });
});

describe('comb fit', () => {
  const lines = (start: number, p: number, skip: number[], extra: number[] = []): LinePos[] => [
    ...Array.from({ length: 9 }, (_, k) => k).filter((k) => !skip.includes(k)).map((k) => ({ pos: start + k * p, w: 1 })),
    ...extra.map((pos) => ({ pos, w: 0.6 })),
  ];

  it('finds start and period with missing teeth and frame lines', () => {
    // Teeth 0, 1 and 7 hidden; frame lines just outside; a clutter line far away.
    const pos = lines(2, 0.5, [0, 1, 7], [2 - 0.2, 2 + 8 * 0.5 + 0.22, 9]);
    const h = fitComb(pos, { tol: 0.1, topK: 4, shiftBest: 2 })[0]!;
    expect(h.period).toBeCloseTo(0.5, 1);
    expect(h.period).toBeGreaterThan(0.48);
    expect(h.period).toBeLessThan(0.52);
    expect(h.start).toBeGreaterThan(1.95);
    expect(h.start).toBeLessThan(2.05);
  });

  it('offers the +-1 cell shifted hypotheses', () => {
    const hyps = fitComb(lines(0, 1, []), { tol: 0.1, topK: 3, shiftBest: 1 });
    expect(hyps.some((h) => h.shifted && Math.abs(h.start - -1) < 0.1)).toBe(true);
    expect(hyps.some((h) => h.shifted && Math.abs(h.start - 1) < 0.1)).toBe(true);
  });
});

describe('segment merging', () => {
  it('merges collinear pieces across a gap and keeps distinct lines apart', () => {
    const raw = [0, 0, 60, 1, 100, 2, 160, 3, 0, 40, 100, 40];
    const out = mergeSegments(raw, { mergeAngleRad: 0.05, mergeOffset: 4, minLen: 30, maxSegments: 50 });
    expect(out).toHaveLength(2);
    expect(Math.max(...out.map((s) => s.len))).toBeGreaterThan(155);
  });

  it('drops short segments', () => {
    expect(mergeSegments([0, 0, 10, 0], { mergeAngleRad: 0.05, mergeOffset: 4, minLen: 30, maxSegments: 50 })).toHaveLength(0);
  });
});
