import { cross, invert3, mul3, type Mat3, type Vec3 } from './homography';
import type { Segment } from './segments';

/** Deterministic PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface VPOptions {
  iterations: number;
  /** Inlier angle threshold in degrees. */
  angleDeg: number;
  /** Minimum sin(angle) between directions to the two VPs, seen from the reference point. */
  minSeparationDeg: number;
  rng: () => number;
}

export interface VPResult {
  /** Homogeneous VP in the coordinates of the input segments, unit length. w ~ 0 means at infinity. */
  vp: Vec3;
  inliers: number[];
  score: number;
}

/** sin of the angle between the segment direction and the direction from its midpoint to the VP; NaN if degenerate. */
export function vpResidual(s: Segment, vp: Vec3): number {
  const ex = vp[0] - s.mx * vp[2];
  const ey = vp[1] - s.my * vp[2];
  const n = Math.hypot(ex, ey);
  if (n < 1e-9) return NaN;
  return Math.abs(s.dx * ey - s.dy * ex) / n;
}

function unit(v: Vec3): Vec3 | null {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n < 1e-9 ? null : [v[0] / n, v[1] / n, v[2] / n];
}

/** Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (Jacobi). */
export function smallestEigenvector(m: number[][]): Vec3 {
  const a = m.map((r) => r.slice());
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 20; sweep++) {
    const off = Math.abs(a[0]![1]!) + Math.abs(a[0]![2]!) + Math.abs(a[1]![2]!);
    if (off < 1e-14) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[p]![q]!;
      if (Math.abs(apq) < 1e-18) continue;
      const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k]![p]!;
        const akq = a[k]![q]!;
        a[k]![p] = c * akp - s * akq;
        a[k]![q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p]![k]!;
        const aqk = a[q]![k]!;
        a[p]![k] = c * apk - s * aqk;
        a[q]![k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k]![p]!;
        const vkq = v[k]![q]!;
        v[k]![p] = c * vkp - s * vkq;
        v[k]![q] = s * vkp + c * vkq;
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 3; i++) if (a[i]![i]! < a[best]![best]!) best = i;
  return [v[0]![best]!, v[1]![best]!, v[2]![best]!];
}

/** Least-squares VP for a set of segments: minimises the weighted sum of squared line residuals. */
export function refineVP(segs: readonly Segment[], idx: readonly number[]): Vec3 | null {
  const m = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const i of idx) {
    const s = segs[i]!;
    const l = s.line;
    const w = s.weight;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[r]![c]! += w * l[r]! * l[c]!;
  }
  return unit(smallestEigenvector(m));
}

function collectInliers(segs: readonly Segment[], pool: readonly number[], vp: Vec3, sinTol: number): { idx: number[]; score: number } {
  const idx: number[] = [];
  let score = 0;
  for (const i of pool) {
    const r = vpResidual(segs[i]!, vp);
    if (r <= sinTol) {
      idx.push(i);
      score += segs[i]!.weight;
    }
  }
  return { idx, score };
}

/** Direction (unit 2-vector) from `ref` towards the VP, or along it when at infinity. */
export function vpDirection(vp: Vec3, ref: [number, number]): [number, number] {
  const ex = vp[0] - ref[0] * vp[2];
  const ey = vp[1] - ref[1] * vp[2];
  const n = Math.hypot(ex, ey) || 1;
  return [ex / n, ey / n];
}

/**
 * RANSAC over pairs of segments. `pool` are indices into `segs`. `accept` may veto hypotheses (used to keep
 * VP2 away from VP1's family). All geometry is homogeneous, so VPs at infinity are ordinary.
 */
export function ransacVP(
  segs: readonly Segment[],
  pool: readonly number[],
  opts: VPOptions,
  accept?: (vp: Vec3) => boolean,
): VPResult | null {
  if (pool.length < 2) return null;
  const sinTol = Math.sin((opts.angleDeg * Math.PI) / 180);
  const cum: number[] = [];
  let tot = 0;
  for (const i of pool) {
    tot += segs[i]!.weight;
    cum.push(tot);
  }
  const pick = (): number => {
    const r = opts.rng() * tot;
    let lo = 0;
    let hi = cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid]! < r) lo = mid + 1;
      else hi = mid;
    }
    return pool[lo]!;
  };
  let best: VPResult | null = null;
  for (let it = 0; it < opts.iterations; it++) {
    const i = pick();
    const j = pick();
    if (i === j) continue;
    const vp = unit(cross(segs[i]!.line, segs[j]!.line));
    if (!vp) continue;
    if (accept && !accept(vp)) continue;
    const { idx, score } = collectInliers(segs, pool, vp, sinTol);
    if (!best || score > best.score) best = { vp, inliers: idx, score };
  }
  if (!best) return null;
  // Refine on the consensus set with a shrinking inlier tolerance: the loose RANSAC threshold lets noisy
  // segments bias the VP, which matters most for the far, foreshortened rows of oblique views.
  const tols = [1, 1, 0.5, 0.5];
  let vp = best.vp;
  for (const f of tols) {
    const r = refineVP(segs, best.inliers);
    if (!r || (accept && !accept(r))) break;
    const { idx } = collectInliers(segs, pool, r, sinTol * f);
    if (idx.length < Math.max(3, 0.5 * best.inliers.length)) break;
    vp = r;
    best = { vp: r, inliers: idx, score: idx.reduce((a, i) => a + segs[i]!.weight, 0) };
  }
  // Report the loose inlier set for the final VP (segments slightly off still belong to the family).
  const { idx, score } = collectInliers(segs, pool, vp, sinTol);
  return { vp, inliers: idx, score };
}

export interface VPPair {
  vp1: VPResult;
  vp2: VPResult;
  /** Reference point (weighted centroid of segment midpoints). */
  ref: [number, number];
}

/** Finds two dominant, non-parallel vanishing points. Segments must be in normalised coordinates. */
export function findVanishingPoints(segs: readonly Segment[], opts: VPOptions): VPPair | null {
  if (segs.length < 4) return null;
  let sx = 0;
  let sy = 0;
  let sw = 0;
  for (const s of segs) {
    sx += s.mx * s.weight;
    sy += s.my * s.weight;
    sw += s.weight;
  }
  const ref: [number, number] = [sx / sw, sy / sw];
  const all = segs.map((_, i) => i);
  const vp1 = ransacVP(segs, all, opts);
  if (!vp1 || vp1.inliers.length < 3) return null;
  const d1 = vpDirection(vp1.vp, ref);
  const minSin = Math.sin((opts.minSeparationDeg * Math.PI) / 180);
  const taken = new Set(vp1.inliers);
  const rest = all.filter((i) => !taken.has(i));
  const vp2 = ransacVP(segs, rest, opts, (vp) => {
    const d2 = vpDirection(vp, ref);
    return Math.abs(d1[0] * d2[1] - d1[1] * d2[0]) >= minSin;
  });
  if (!vp2 || vp2.inliers.length < 3) return null;
  return { vp1, vp2, ref };
}

export interface Rectification {
  /** Normalised image coordinates -> rectified coordinates. Family 1 becomes vertical, family 2 horizontal. */
  H: Mat3;
  Hinv: Mat3;
  /** Normalised coordinates of the image of the line at infinity. */
  horizon: Vec3;
}

/**
 * Affine rectification from two VPs (normalised coordinates). The origin is moved to `ref` (a point on the board
 * side of the horizon) first, so the rectifying homography stays well conditioned even when the horizon passes
 * near the image centre. Returns null for degenerate configurations.
 */
export function affineRectification(vp1: Vec3, vp2: Vec3, ref: [number, number]): Rectification | null {
  const t0: Mat3 = [1, 0, -ref[0], 0, 1, -ref[1], 0, 0, 1];
  const shift = (v: Vec3): Vec3 => [v[0] - ref[0] * v[2], v[1] - ref[1] * v[2], v[2]];
  const v1 = shift(vp1);
  const v2 = shift(vp2);
  let l = cross(v1, v2);
  const n12 = Math.hypot(l[0], l[1]);
  if (n12 < 1e-12) {
    // Both VPs at infinity (and so is the line joining them): an affine view already.
    l = [0, 0, 1];
  } else if (Math.abs(l[2]) < 0.02 * n12) {
    return null; // horizon passes through the board reference point
  }
  const a1 = l[0] / l[2];
  const a2 = l[1] / l[2];
  const ha: Mat3 = [1, 0, 0, 0, 1, 0, a1, a2, 1];
  const dir = (v: Vec3): [number, number] => {
    // Ha v; its w component is ~0 because v lies on the horizon.
    const x = v[0];
    const y = v[1];
    const n = Math.hypot(x, y) || 1;
    return [x / n, y / n];
  };
  const d1 = dir(v1);
  const d2 = dir(v2);
  const det = d2[0] * d1[1] - d1[0] * d2[1];
  if (Math.abs(det) < 0.05) return null;
  // Columns d2, d1 -> inverse maps d2 to (1,0) and d1 to (0,1).
  const m: Mat3 = [d1[1] / det, -d1[0] / det, 0, -d2[1] / det, d2[0] / det, 0, 0, 0, 1];
  const H = mul3(m, mul3(ha, t0));
  const Hinv = invert3(H);
  if (!Hinv) return null;
  return { H, Hinv, horizon: l };
}
