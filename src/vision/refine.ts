import { applyH, homographyFrom4, invert3, mul3, solveLinear, type Mat3, type Point } from '../geom/homography';
import { mulberry32 } from '../geom/vanishing';
import type { Params } from '../worker/protocol';
import { param, type ParamSpec } from './preprocess';
import type { Lab3 } from './verify';

export const REFINE_PARAMS: readonly ParamSpec[] = [
  { name: 'refine', min: 0, max: 1, step: 1, default: 0 },
  { name: 'refineIters', min: 1, max: 5, step: 1, default: 3 },
  /** Search window radius around each predicted corner, as a fraction of a cell (first iteration). */
  { name: 'refineWindow', min: 0.1, max: 0.45, step: 0.01, default: 0.35 },
  /** RANSAC inlier threshold in cell units. */
  { name: 'refineInlier', min: 0.02, max: 0.2, step: 0.01, default: 0.06 },
  { name: 'refineMinInliers', min: 8, max: 40, step: 1, default: 14 },
  /** Corner responses below this fraction of the median response are rejected. */
  { name: 'refineMinResp', min: 0.1, max: 0.9, step: 0.05, default: 0.35 },
  /** A refined homography is kept unless its checker score drops by more than this. */
  /** Weight of the detected inner corners in the joint edge + corner fit of the outer quad (0 = edges only). */
  { name: 'refineJoint', min: 0, max: 4, step: 0.25, default: 1 },
  /** 1 = use the inner-corner homography alone (extrapolated to the outer quad) instead of the edge fit. */
  { name: 'refineReplace', min: 0, max: 1, step: 1, default: 0 },
  { name: 'refineKeepTol', min: 0, max: 0.3, step: 0.01, default: 0.05 },
];

export interface RefineOptions {
  iters: number;
  window: number;
  inlier: number;
  minInliers: number;
  minResp: number;
  /** Channel of the Lab3 image to use (0 = CLAHE-L, 1 = a*, 2 = b*). */
  channel: number;
  /** Absolute response floor in channel units. */
  absFloor: number;
}

export function refineOptions(params: Params, channel: number): RefineOptions {
  return {
    iters: param(params, REFINE_PARAMS, 'refineIters'),
    window: param(params, REFINE_PARAMS, 'refineWindow'),
    inlier: param(params, REFINE_PARAMS, 'refineInlier'),
    minInliers: param(params, REFINE_PARAMS, 'refineMinInliers'),
    minResp: param(params, REFINE_PARAMS, 'refineMinResp'),
    channel,
    absFloor: 1.2,
  };
}

export interface RefineResult {
  hb: Mat3;
  /** Number of RANSAC inliers of the final fit, and of accepted corner detections in the last iteration. */
  inliers: number;
  accepted: number;
  /** RMS reprojection error of the inliers, px. */
  rms: number;
  ok: boolean;
  /** Inlier corner detections of the final fit (grid coordinates -> image), for joint fitting. */
  points: CornerPoint[];
}

export interface CornerPoint {
  gi: number;
  gj: number;
  x: number;
  y: number;
}

// Quadrant sampling offsets, in cells from the corner (a point per quadrant is at sign * offset).
const QO: Array<[number, number]> = (() => {
  const o: Array<[number, number]> = [];
  for (const u of [0.1, 0.21, 0.32]) for (const v of [0.1, 0.21, 0.32]) o.push([u, v]);
  return o;
})();

function makeSampler(img: Lab3, channel: number) {
  const { data, width, height } = img;
  const stride = img.stride ?? 3;
  return (x: number, y: number): number => {
    if (!(x >= 0 && y >= 0 && x < width - 1 && y < height - 1)) return NaN;
    const x0 = x | 0;
    const y0 = y | 0;
    const fx = x - x0;
    const fy = y - y0;
    const o = (y0 * width + x0) * stride + channel;
    const o2 = o + width * stride;
    return (
      data[o]! * (1 - fx) * (1 - fy) + data[o + stride]! * fx * (1 - fy) +
      data[o2]! * (1 - fx) * fy + data[o2 + stride]! * fx * fy
    );
  };
}

interface Detection {
  /** Grid coordinates and detected image position. */
  gi: number;
  gj: number;
  x: number;
  y: number;
  /** Signed strength of the best response matching the expected parity polarity. */
  strength: number;
}

/**
 * Board -> image homography refinement from checkerboard X-corners.
 *
 * For each of the 49 inner corners the current homography predicts a position and a local affine frame. A quadrant
 * template (mean of TL + BR minus TR + BL over a few samples per quadrant, sampled in that local frame so perspective
 * is respected) is evaluated over a small window in cell space; its peak is refined to sub-pixel with a parabola.
 * Pieces and clutter give weak or inconsistent responses, which the RANSAC homography fit treats as outliers.
 */
export class CornerRefiner {
  private readonly rng = mulberry32(777);

  refine(img: Lab3, hb0: Mat3, opt: RefineOptions): RefineResult {
    const sample = makeSampler(img, opt.channel);
    let hb = hb0;
    let last: RefineResult = { hb: hb0, inliers: 0, accepted: 0, rms: Infinity, ok: false, points: [] };
    for (let it = 0; it < opt.iters; it++) {
      const win = opt.window * (it === 0 ? 1 : it === 1 ? 0.75 : 0.55);
      const dets = this.detect(sample, hb, win, opt);
      if (dets.length < opt.minInliers) return { ...last, accepted: dets.length, ok: false };
      const fit = fitHomographyRansac(dets, opt.inlier, this.rng);
      if (!fit || fit.inliers < opt.minInliers) return { ...last, accepted: dets.length, ok: false };
      hb = fit.h;
      last = { hb, inliers: fit.inliers, accepted: dets.length, rms: fit.rms, ok: true, points: fit.points };
    }
    return last;
  }

  /** Corner detections around the predictions of `hb`. */
  detect(sample: (x: number, y: number) => number, hb: Mat3, win: number, opt: RefineOptions): Detection[] {
    const cands: Array<{ gi: number; gj: number; par: number; pos: Point | null; neg: Point | null; sp: number; sn: number }> = [];
    const e = 0.25;
    for (let gj = 1; gj <= 7; gj++) {
      for (let gi = 1; gi <= 7; gi++) {
        const p = applyH(hb, [gi, gj]);
        const px1 = applyH(hb, [gi + e, gj]);
        const px0 = applyH(hb, [gi - e, gj]);
        const py1 = applyH(hb, [gi, gj + e]);
        const py0 = applyH(hb, [gi, gj - e]);
        if (![p, px1, px0, py1, py0].every((q) => Number.isFinite(q[0]) && Number.isFinite(q[1]))) continue;
        const ex: Point = [(px1[0] - px0[0]) / (2 * e), (px1[1] - px0[1]) / (2 * e)];
        const ey: Point = [(py1[0] - py0[0]) / (2 * e), (py1[1] - py0[1]) / (2 * e)];
        const lx = Math.hypot(ex[0], ex[1]);
        const ly = Math.hypot(ey[0], ey[1]);
        if (Math.min(lx, ly) < 3) continue; // cells too small to resolve
        const na = Math.min(13, Math.max(3, Math.ceil(2 * win * lx) + 1)) | 1;
        const nb = Math.min(13, Math.max(3, Math.ceil(2 * win * ly) + 1)) | 1;
        const R = new Float32Array(na * nb).fill(NaN);
        // Precompute sample offsets for the four quadrants (image px relative to the candidate).
        const qs: number[][] = [[], [], [], []];
        const signs: Array<[number, number]> = [[-1, -1], [1, -1], [1, 1], [-1, 1]]; // TL, TR, BR, BL
        for (let q = 0; q < 4; q++) {
          for (const [u, v] of QO) {
            const sx = signs[q]![0] * u;
            const sy = signs[q]![1] * v;
            qs[q]!.push(sx * ex[0] + sy * ey[0], sx * ex[1] + sy * ey[1]);
          }
        }
        for (let jb = 0; jb < nb; jb++) {
          const b = -win + (2 * win * jb) / (nb - 1);
          for (let ia = 0; ia < na; ia++) {
            const a = -win + (2 * win * ia) / (na - 1);
            const cx = p[0] + a * ex[0] + b * ey[0];
            const cy = p[1] + a * ex[1] + b * ey[1];
            const m = [0, 0, 0, 0];
            let bad = false;
            for (let q = 0; q < 4 && !bad; q++) {
              const off = qs[q]!;
              let s = 0;
              for (let k = 0; k < off.length; k += 2) {
                const v = sample(cx + off[k]!, cy + off[k + 1]!);
                if (Number.isNaN(v)) {
                  bad = true;
                  break;
                }
                s += v;
              }
              m[q] = s / QO.length;
            }
            if (!bad) R[jb * na + ia] = (m[0]! + m[2]! - m[1]! - m[3]!) / 4;
          }
        }
        const pk = (sign: number): { pos: Point; s: number } | null => {
          let bi = -1;
          let bv = 0;
          for (let k = 0; k < R.length; k++) {
            const v = R[k]! * sign;
            if (v > bv) {
              bv = v;
              bi = k;
            }
          }
          if (bi < 0) return null;
          const ia = bi % na;
          const ja = (bi / na) | 0;
          // Peaks on the window border are ambiguous (the true corner is probably outside).
          if (ia === 0 || ia === na - 1 || ja === 0 || ja === nb - 1) return null;
          const at = (i: number, j: number) => R[j * na + i]! * sign;
          const sub = (l: number, c: number, r: number) => {
            const d = l - 2 * c + r;
            return Number.isFinite(d) && d < -1e-9 ? Math.max(-0.5, Math.min(0.5, (0.5 * (l - r)) / d)) : 0;
          };
          const da = sub(at(ia - 1, ja), bv, at(ia + 1, ja));
          const db = sub(at(ia, ja - 1), bv, at(ia, ja + 1));
          const stepA = (2 * win) / (na - 1);
          const stepB = (2 * win) / (nb - 1);
          const a = -win + (ia + da) * stepA;
          const b = -win + (ja + db) * stepB;
          return { pos: [p[0] + a * ex[0] + b * ey[0], p[1] + a * ex[1] + b * ey[1]], s: bv };
        };
        const pp = pk(1);
        const pn = pk(-1);
        cands.push({ gi, gj, par: (gi + gj) & 1 ? -1 : 1, pos: pp?.pos ?? null, neg: pn?.pos ?? null, sp: pp?.s ?? 0, sn: pn?.s ?? 0 });
      }
    }
    // The global polarity: expected sign of R at a corner is parity * g.
    let vote = 0;
    for (const c of cands) vote += c.par * (c.sp - c.sn);
    const g = vote >= 0 ? 1 : -1;
    const picked: Detection[] = [];
    for (const c of cands) {
      const want = c.par * g;
      const pos = want > 0 ? c.pos : c.neg;
      const s = want > 0 ? c.sp : c.sn;
      if (pos && s >= opt.absFloor) picked.push({ gi: c.gi, gj: c.gj, x: pos[0], y: pos[1], strength: s });
    }
    if (picked.length === 0) return picked;
    const strengths = picked.map((d) => d.strength).sort((a, b) => a - b);
    const med = strengths[strengths.length >> 1]!;
    return picked.filter((d) => d.strength >= opt.minResp * med);
  }
}

interface Fit {
  points: CornerPoint[];
  h: Mat3;
  inliers: number;
  rms: number;
}

/** Least-squares homography (h33 = 1) in normalised coordinates over the given correspondences. */
function lsHomography(src: Point[], dst: Point[]): Mat3 | null {
  const n = src.length;
  if (n < 4) return null;
  // Normalise: centre and scale each side.
  const norm = (pts: Point[]): { t: Mat3; p: Point[] } => {
    let cx = 0;
    let cy = 0;
    for (const q of pts) {
      cx += q[0];
      cy += q[1];
    }
    cx /= pts.length;
    cy /= pts.length;
    let d = 0;
    for (const q of pts) d += Math.hypot(q[0] - cx, q[1] - cy);
    d /= pts.length;
    const s = d > 1e-9 ? Math.SQRT2 / d : 1;
    return { t: [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1], p: pts.map((q) => [(q[0] - cx) * s, (q[1] - cy) * s] as Point) };
  };
  const S = norm(src);
  const D = norm(dst);
  let h: Mat3 | null;
  if (n === 4) {
    h = homographyFrom4(S.p, D.p);
  } else {
    const ata: number[][] = Array.from({ length: 8 }, () => new Array<number>(8).fill(0));
    const atb = new Array<number>(8).fill(0);
    for (let i = 0; i < n; i++) {
      const [x, y] = S.p[i]!;
      const [u, v] = D.p[i]!;
      const r1 = [x, y, 1, 0, 0, 0, -u * x, -u * y];
      const r2 = [0, 0, 0, x, y, 1, -v * x, -v * y];
      for (let a = 0; a < 8; a++) {
        for (let b = 0; b < 8; b++) ata[a]![b]! += r1[a]! * r1[b]! + r2[a]! * r2[b]!;
        atb[a]! += r1[a]! * u + r2[a]! * v;
      }
    }
    const s = solveLinear(ata, atb);
    h = s ? [s[0]!, s[1]!, s[2]!, s[3]!, s[4]!, s[5]!, s[6]!, s[7]!, 1] : null;
  }
  if (!h) return null;
  const dInv = invert3(D.t);
  if (!dInv) return null;
  const out = mul3(mul3(dInv, h), S.t);
  const k = out[8];
  if (Math.abs(k) < 1e-12) return null;
  return out.map((v) => v / k) as Mat3;
}

/** RANSAC + local least-squares polish of the board -> image homography from corner detections. */
export function fitHomographyRansac(dets: readonly Detection[], inlierCells: number, rng: () => number): Fit | null {
  const n = dets.length;
  if (n < 6) return null;
  const src: Point[] = dets.map((d) => [d.gi, d.gj]);
  const dst: Point[] = dets.map((d) => [d.x, d.y]);
  // Inlier threshold: in px, from the local scale of the detections (cell size at the centre of the board).
  // Estimated from the spread of the detected points: use the mean nearest-neighbour spacing along grid axes.
  let cell = 0;
  let cn = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(src[i]![0] - src[j]![0]) + Math.abs(src[i]![1] - src[j]![1]) === 1) {
        cell += Math.hypot(dst[i]![0] - dst[j]![0], dst[i]![1] - dst[j]![1]);
        cn++;
      }
    }
  }
  cell = cn > 0 ? cell / cn : 20;
  const thr = Math.max(0.6, inlierCells * cell);
  const thr2 = thr * thr;

  const countInliers = (h: Mat3, mask?: Uint8Array): number => {
    let c = 0;
    for (let i = 0; i < n; i++) {
      const w = h[6] * src[i]![0] + h[7] * src[i]![1] + h[8];
      let ok = false;
      if (w > 1e-9) {
        const x = (h[0] * src[i]![0] + h[1] * src[i]![1] + h[2]) / w;
        const y = (h[3] * src[i]![0] + h[4] * src[i]![1] + h[5]) / w;
        const dx = x - dst[i]![0];
        const dy = y - dst[i]![1];
        ok = dx * dx + dy * dy <= thr2;
      }
      if (mask) mask[i] = ok ? 1 : 0;
      if (ok) c++;
    }
    return c;
  };

  let bestH: Mat3 | null = null;
  let bestC = 0;
  const idx = new Array<number>(4);
  const maxIter = 150;
  for (let it = 0; it < maxIter; it++) {
    for (let k = 0; k < 4; k++) {
      let r: number;
      do r = (rng() * n) | 0;
      while (idx.slice(0, k).includes(r));
      idx[k] = r;
    }
    const s4 = idx.map((i) => src[i]!);
    // Reject nearly collinear samples.
    if (collinear(s4)) continue;
    const h = lsHomography(s4, idx.map((i) => dst[i]!));
    if (!h) continue;
    const c = countInliers(h);
    if (c > bestC) {
      bestC = c;
      bestH = h;
      if (c >= 0.95 * n) break;
    }
  }
  if (!bestH || bestC < 6) return null;
  // Local optimisation: refit on inliers a few times.
  const mask = new Uint8Array(n);
  for (let k = 0; k < 4; k++) {
    countInliers(bestH, mask);
    const s: Point[] = [];
    const d: Point[] = [];
    for (let i = 0; i < n; i++) if (mask[i]) (s.push(src[i]!), d.push(dst[i]!));
    const h = lsHomography(s, d);
    if (!h) break;
    const c = countInliers(h);
    if (c >= bestC) {
      bestH = h;
      bestC = c;
    } else break;
  }
  countInliers(bestH, mask);
  let se = 0;
  let m = 0;
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    const p = applyH(bestH, src[i]!);
    se += (p[0] - dst[i]![0]) ** 2 + (p[1] - dst[i]![1]) ** 2;
    m++;
  }
  // Coverage: inliers must span at least 3 distinct rows and columns of the grid.
  const rows = new Set<number>();
  const cols = new Set<number>();
  for (let i = 0; i < n; i++)
    if (mask[i]) {
      cols.add(src[i]![0]);
      rows.add(src[i]![1]);
    }
  if (rows.size < 3 || cols.size < 3) return null;
  const points: CornerPoint[] = dets.filter((_, i) => mask[i]).map((d) => ({ gi: d.gi, gj: d.gj, x: d.x, y: d.y }));
  return { h: bestH, points, inliers: m, rms: Math.sqrt(se / Math.max(1, m)) };
}

function collinear(p: Point[]): boolean {
  // Any three of the four collinear (grid points: exact test on integers).
  for (let a = 0; a < 4; a++)
    for (let b = a + 1; b < 4; b++)
      for (let c = b + 1; c < 4; c++) {
        const cr = (p[b]![0] - p[a]![0]) * (p[c]![1] - p[a]![1]) - (p[b]![1] - p[a]![1]) * (p[c]![0] - p[a]![0]);
        if (Math.abs(cr) < 1e-9) return true;
      }
  return false;
}
