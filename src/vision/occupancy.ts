import { OCC_BLACK, OCC_EMPTY, OCC_WHITE, type Observation, type OccupancyStats, type Params } from '../worker/protocol';
import { solveLinear, type Mat3, type Point, type Vec3 } from '../geom/homography';
import { param, type ParamSpec } from './preprocess';

/** Tunables of the square-occupancy stage (milestone 8); registered as debug sliders via ALL_PARAMS. */
export const OCCUPANCY_PARAMS: readonly ParamSpec[] = [
  /** 0 disables occupancy in the worker. */
  { name: 'occupancy', min: 0, max: 1, step: 1, default: 1 },
  /** Nominal horizontal FOV (long side) used when the focal length cannot be recovered from hb. */
  { name: 'occFovDeg', min: 40, max: 90, step: 1, default: 65 },
  /** Occluder model: a cylinder of this radius and height (cells) on every occupied square. */
  { name: 'occPieceRadius', min: 0.15, max: 0.5, step: 0.01, default: 0.33 },
  { name: 'occPieceHeight', min: 0.5, max: 2.5, step: 0.1, default: 1.5 },
  /** Footprint: base disc radius and stem height (cells). */
  { name: 'occBaseRadius', min: 0.1, max: 0.4, step: 0.01, default: 0.27 },
  { name: 'occStemHeight', min: 0, max: 0.8, step: 0.05, default: 0.4 },
  /** Summed capped squared z-scores above which a cell deviates from a model. */
  { name: 'occDeviation', min: 4, max: 60, step: 1, default: 16 },
  /** A cell further than this from every class model is an outlier (hand, glare): confidence 0. */
  { name: 'occOutlier', min: 20, max: 125, step: 5, default: 90 },
  /** Classifier margin (NLL units) that gives full confidence. */
  { name: 'occMarginScale', min: 1, max: 20, step: 0.5, default: 4 },
  /** Cells below this confidence are not updated on this frame. */
  { name: 'occCellMin', min: 0, max: 1, step: 0.05, default: 0.2 },
  /** The frame is dropped when more cells than this are low-confidence ... */
  { name: 'occMaxLowCells', min: 0, max: 64, step: 1, default: 28 },
  /** ... or when the mean confidence is below this. */
  { name: 'occFrameMin', min: 0, max: 1, step: 0.05, default: 0.25 },
  /** Consecutive consistent frames before a cell changes its committed state. */
  { name: 'occHoldFrames', min: 1, max: 10, step: 1, default: 3 },
  /** This many cells flipping between consecutive frames freezes commits (hand over the board). */
  { name: 'occFreezeCells', min: 2, max: 32, step: 1, default: 6 },
  /** After a freeze (or a frame without board), the board must be stable this long before commits resume. */
  { name: 'occSettleMs', min: 0, max: 2000, step: 50, default: 400 },
  /** Consecutive frames the starting position must be recognised before calibrating on it. */
  { name: 'occBootFrames', min: 1, max: 10, step: 1, default: 2 },
  /** Without a starting position for this long, calibrate unsupervised. */
  { name: 'occFallbackMs', min: 1000, max: 30000, step: 500, default: 8000 },
  /** Temperature of the per-cell class log-likelihoods (the five features are correlated, so their summed NLL is
   *  over-confident by about this factor). */
  { name: 'occLikTemp', min: 1, max: 10, step: 0.25, default: 3 },
  /** Exponent on the visible fraction in the log-likelihood evidence weight: partly hidden cells are still mostly
   *  right, so a linear weight left them badly under-confident (vis 0 stays uniform). */
  { name: 'occLikVisGamma', min: 0.1, max: 1, step: 0.05, default: 0.25 },
  /** EMA rate of the model updates from confident cells. */
  { name: 'occLearnRate', min: 0, max: 0.2, step: 0.005, default: 0.02 },
  /** Without a board for longer than this (ms), the orientation (which cell is a1) must be re-verified. */
  { name: 'occLossMs', min: 200, max: 10000, step: 100, default: 1000 },
  /** Orientation: summed log-likelihood (nats) by which the best dihedral placement of the target grid (position hint,
   *  else the starting position) must beat the best placement predicting a different cell grid. */
  { name: 'occOrientMargin', min: 1, max: 60, step: 1, default: 12 },
  /** Orientation: most cells allowed to confidently contradict the best placement (p of its class < 0.1). A hint one
   *  move behind the board (a move made while it was hidden) differs by 2-4 cells, and the frame itself may have a few
   *  confident errors (light squares read as white pieces on low-contrast boards); the margin does the real work. */
  { name: 'occOrientMaxMiss', min: 0, max: 16, step: 1, default: 8 },
  /** Orientation: consecutive stable frames that must agree on the same placement before it is adopted. */
  { name: 'occOrientFrames', min: 1, max: 10, step: 1, default: 2 },
  /** Dark-a1 check: least lightness difference (L*) between the two parities' empty models for the check to count;
   *  below it the parity is inconclusive and the handedness alone decides. */
  { name: 'occParityMinDL', min: 0, max: 30, step: 1, default: 4 },
  /** Framing: height (cells) above the board corners that must be in view too, so the tops of tall pieces on the far
   *  rank count (a cropped board reads wrong near the edge; see boardFramed). */
  { name: 'occFrameTopZ', min: 0, max: 3, step: 0.1, default: 1.2 },
  /** Framing: the visible rect is shrunk by this fraction of the frame's short side before the check. */
  { name: 'occFrameMargin', min: 0, max: 0.1, step: 0.005, default: 0.02 },
];

// ---------------------------------------------------------------------------------------------------------------
// Camera from the board homography

export interface BoardCamera {
  f: number;
  cx: number;
  cy: number;
  /** True when f is the nominal-FOV fallback (near-overhead or implausible estimate). */
  nominal: boolean;
  /** Camera centre in board coordinates (cell units, z up towards the camera). */
  centre: Vec3;
  /** Image point of board point (x, y) at height z (cells); null behind the camera. */
  project(x: number, y: number, z: number): Point | null;
}

/**
 * Pose from a board (cell units) -> image homography, assuming square pixels and the principal point at the image
 * centre. The focal length comes from the square-board constraint (r1 . r2 = 0, |r1| = |r2|); it falls back to a
 * nominal FOV when the view is nearly fronto-parallel (the constraint degenerates) or the estimate is implausible.
 * The plane terms keep hb exactly (z = 0 reproduces hb); r3 is signed so that +z points to the camera side.
 */
export function cameraFromHomography(hb: Mat3, width: number, height: number, fovDeg = 65): BoardCamera | null {
  const cx = width / 2;
  const cy = height / 2;
  const x1 = hb[0] - cx * hb[6], y1 = hb[3] - cy * hb[6], z1 = hb[6];
  const x2 = hb[1] - cx * hb[7], y2 = hb[4] - cy * hb[7], z2 = hb[7];
  let x3 = hb[2] - cx * hb[8], y3 = hb[5] - cy * hb[8], z3 = hb[8];
  const long = Math.max(width, height);
  const fNominal = long / 2 / Math.tan((fovDeg * Math.PI) / 360);
  // Relative change of the projective depth across the board: ~0 for a fronto-parallel view.
  const wc = 4 * z1 + 4 * z2 + z3;
  if (Math.abs(wc) < 1e-12) return null;
  const persp = (8 * Math.hypot(z1, z2)) / Math.abs(wc);
  const a1 = x1 * x2 + y1 * y2;
  const b1 = z1 * z2;
  const a2 = x1 * x1 + y1 * y1 - x2 * x2 - y2 * y2;
  const b2 = z1 * z1 - z2 * z2;
  const den = a1 * a1 + a2 * a2;
  const u = den > 0 ? -(a1 * b1 + a2 * b2) / den : 0;
  let f = u > 0 ? 1 / Math.sqrt(u) : NaN;
  let nominal = false;
  if (!(persp >= 0.06 && f > 0.3 * long && f < 3 * long)) {
    f = fNominal;
    nominal = true;
  }
  let k1: Vec3 = [x1 / f, y1 / f, z1];
  let k2: Vec3 = [x2 / f, y2 / f, z2];
  let k3: Vec3 = [x3 / f, y3 / f, z3];
  const lam = (Math.hypot(...k1) + Math.hypot(...k2)) / 2;
  if (!(lam > 0)) return null;
  // Board centre in front of the camera.
  const s = (wc > 0 ? 1 : -1) / lam;
  k1 = [k1[0] * s, k1[1] * s, k1[2] * s];
  k2 = [k2[0] * s, k2[1] * s, k2[2] * s];
  k3 = [k3[0] * s, k3[1] * s, k3[2] * s];
  x3 = k3[0];
  y3 = k3[1];
  z3 = k3[2];
  const n1 = Math.hypot(...k1);
  const n2 = Math.hypot(...k2);
  let r3: Vec3 = [
    (k1[1] * k2[2] - k1[2] * k2[1]) / (n1 * n2),
    (k1[2] * k2[0] - k1[0] * k2[2]) / (n1 * n2),
    (k1[0] * k2[1] - k1[1] * k2[0]) / (n1 * n2),
  ];
  const rn = Math.hypot(...r3);
  if (!(rn > 1e-9)) return null;
  r3 = [r3[0] / rn, r3[1] / rn, r3[2] / rn];
  const sol = solveLinear(
    [[k1[0], k2[0], r3[0]], [k1[1], k2[1], r3[1]], [k1[2], k2[2], r3[2]]],
    [-x3, -y3, -z3],
  );
  if (!sol) return null;
  let centre: Vec3 = [sol[0]!, sol[1]!, sol[2]!];
  if (centre[2] < 0) {
    r3 = [-r3[0], -r3[1], -r3[2]];
    centre = [centre[0], centre[1], -centre[2]];
  }
  const R3 = r3;
  const project = (x: number, y: number, z: number): Point | null => {
    const X = x * k1[0] + y * k2[0] + z * R3[0] + x3;
    const Y = x * k1[1] + y * k2[1] + z * R3[1] + y3;
    const Z = x * k1[2] + y * k2[2] + z * R3[2] + z3;
    if (Z <= 1e-9) return null;
    return [(f * X) / Z + cx, (f * Y) / Z + cy];
  };
  return { f, cx, cy, nominal, centre, project };
}

/** A rect of the frame, [x0, y0, x1, y1] in pixels (FrameMessage.visibleRect). */
export type FrameRect = readonly [number, number, number, number];

/**
 * Whether the whole board is in view: its 4 corners at z = 0 and again at z = `occFrameTopZ` (the tops of tall
 * pieces on the corner squares, so the far rank's pieces count too) must all lie inside `rect` shrunk by
 * `occFrameMargin` of the frame's short side (the frame is 2 cx x 2 cy). A point behind the camera is outside.
 * `worst` is the smallest distance (px) of the 8 points inside the unshrunk rect: negative when one is outside,
 * -Infinity when one is behind the camera.
 */
export function boardFramed(cam: BoardCamera, rect: FrameRect, params: Params = {}): { framed: boolean; worst: number } {
  const zTop = param(params, OCCUPANCY_PARAMS, 'occFrameTopZ');
  const margin = param(params, OCCUPANCY_PARAMS, 'occFrameMargin') * 2 * Math.min(cam.cx, cam.cy);
  let worst = Infinity;
  for (const z of [0, zTop])
    for (const [x, y] of [[0, 0], [8, 0], [8, 8], [0, 8]] as const) {
      const p = cam.project(x, y, z);
      worst = p ? Math.min(worst, p[0] - rect[0], rect[2] - p[0], p[1] - rect[1], rect[3] - p[1]) : -Infinity;
      if (worst === -Infinity) return { framed: false, worst };
    }
  return { framed: worst >= margin, worst };
}

// ---------------------------------------------------------------------------------------------------------------
// Footprints

export interface Footprints {
  /** Samples per cell; sample s of cell k is at index k * n + s. */
  n: number;
  /** Board coordinates (cells) of each sample. */
  bx: Float32Array;
  by: Float32Array;
  bz: Float32Array;
  /** Image position; NaN when not projectable / outside the frame. */
  px: Float32Array;
  py: Float32Array;
  /** Base weight (0 when outside the frame). */
  w0: Float32Array;
  /** Template weight per sample (independent of the frame), for the visible fraction. */
  wt: Float32Array;
  /** Cells whose piece could cover sample k: occCells[occStart[k] .. occStart[k + 1]). */
  occStart: Int32Array;
  occCells: Uint8Array;
  /** Pixel offset of the local gradient per cell (~6% of the cell size). */
  step: Uint8Array;
}

interface Tmpl {
  dx: number;
  dy: number;
  z: number;
  /** Stem samples are offset sideways relative to the viewing direction. */
  lateral: boolean;
  w: number;
}

function template(baseR: number, stemH: number): Tmpl[] {
  const t: Tmpl[] = [{ dx: 0, dy: 0, z: 0, lateral: false, w: 1 }];
  for (const [r, n, ph] of [[baseR * 0.5, 6, 0], [baseR, 8, Math.PI / 8]] as const)
    for (let k = 0; k < n; k++) t.push({ dx: r * Math.cos(ph + (2 * Math.PI * k) / n), dy: r * Math.sin(ph + (2 * Math.PI * k) / n), z: 0, lateral: false, w: 1 });
  if (stemH > 0)
    for (const z of [stemH / 2, stemH]) for (const l of [-0.13, 0, 0.13]) t.push({ dx: l, dy: 0, z, lateral: true, w: 0.7 });
  return t;
}

/** Sample points per cell with the cells whose piece (cylinder radius x height) could hide each of them. */
export function cellFootprints(cam: BoardCamera, width: number, height: number, params: Params = {}): Footprints {
  const tm = template(param(params, OCCUPANCY_PARAMS, 'occBaseRadius'), param(params, OCCUPANCY_PARAMS, 'occStemHeight'));
  const R = param(params, OCCUPANCY_PARAMS, 'occPieceRadius');
  const Hp = param(params, OCCUPANCY_PARAMS, 'occPieceHeight');
  const n = tm.length;
  const N = 64 * n;
  const fp: Footprints = {
    n, bx: new Float32Array(N), by: new Float32Array(N), bz: new Float32Array(N), px: new Float32Array(N), py: new Float32Array(N),
    w0: new Float32Array(N), wt: new Float32Array(N), occStart: new Int32Array(N + 1), occCells: new Uint8Array(0), step: new Uint8Array(64),
  };
  const occ: number[] = [];
  const [Cx, Cy, Cz] = cam.centre;
  for (let c = 0; c < 64; c++) {
    const i = c & 7;
    const j = c >> 3;
    const ox = i + 0.5;
    const oy = j + 0.5;
    // Local cell size in pixels for the gradient step.
    const p0 = cam.project(i, j, 0);
    const p1 = cam.project(i + 1, j + 1, 0);
    const q0 = cam.project(i + 1, j, 0);
    const q1 = cam.project(i, j + 1, 0);
    const size = p0 && p1 && q0 && q1 ? Math.sqrt(Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * Math.hypot(q1[0] - q0[0], q1[1] - q0[1]) / 2) : 10;
    fp.step[c] = Math.max(1, Math.min(4, Math.round(size * 0.06)));
    // Viewing direction on the board (towards the camera) for the lateral stem offsets.
    let vx = Cx - ox;
    let vy = Cy - oy;
    const vn = Math.hypot(vx, vy);
    if (vn > 1e-6) {
      vx /= vn;
      vy /= vn;
    } else {
      vx = 1;
      vy = 0;
    }
    for (let s = 0; s < n; s++) {
      const t = tm[s]!;
      const k = c * n + s;
      const x = t.lateral ? ox - vy * t.dx : ox + t.dx;
      const y = t.lateral ? oy + vx * t.dx : oy + t.dy;
      fp.bx[k] = x;
      fp.by[k] = y;
      fp.bz[k] = t.z;
      fp.wt[k] = t.w;
      const p = cam.project(x, y, t.z);
      if (p && p[0] >= 0 && p[1] >= 0 && p[0] < width && p[1] < height) {
        fp.px[k] = p[0];
        fp.py[k] = p[1];
        fp.w0[k] = t.w;
      } else {
        fp.px[k] = NaN;
        fp.py[k] = NaN;
        fp.w0[k] = 0;
      }
      fp.occStart[k] = occ.length;
      // Segment from the sample towards the camera, up to the occluder height; a piece on cell c2 hides the
      // sample when its axis passes within R of the segment's ground projection.
      let ex: number;
      let ey: number;
      if (Cz <= Hp) {
        ex = Cx;
        ey = Cy;
      } else {
        const u = Math.max(0, (Hp - t.z) / (Cz - t.z));
        ex = x + u * (Cx - x);
        ey = y + u * (Cy - y);
      }
      const minx = Math.min(x, ex) - R;
      const maxx = Math.max(x, ex) + R;
      const miny = Math.min(y, ey) - R;
      const maxy = Math.max(y, ey) + R;
      const dx = ex - x;
      const dy = ey - y;
      const len2 = dx * dx + dy * dy;
      for (let c2 = 0; c2 < 64; c2++) {
        if (c2 === c) continue;
        const qx = (c2 & 7) + 0.5;
        const qy = (c2 >> 3) + 0.5;
        if (qx < minx || qx > maxx || qy < miny || qy > maxy) continue;
        const u = len2 > 1e-12 ? Math.max(0, Math.min(1, ((qx - x) * dx + (qy - y) * dy) / len2)) : 0;
        if (Math.hypot(x + u * dx - qx, y + u * dy - qy) < R) occ.push(c2);
      }
    }
  }
  fp.occStart[N] = occ.length;
  fp.occCells = Uint8Array.from(occ);
  return fp;
}

// ---------------------------------------------------------------------------------------------------------------
// Sampling and features

export interface RawFrame {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/** Per-sample linear RGB (5-tap mean) and local luminance gradient from the raw RGBA frame. Lab is computed per cell
 *  in `cellFeatures`, after the illumination gain of the cell is applied. */
export interface Samples {
  r: Float32Array;
  gr: Float32Array;
  bl: Float32Array;
  g: Float32Array;
}

const SRGB_LIN = Float32Array.from({ length: 256 }, (_, v) => {
  const c = v / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

const labF = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);

export function sampleFrame(frame: RawFrame, fp: Footprints, out?: Samples): Samples {
  const N = fp.w0.length;
  const s = out ?? { r: new Float32Array(N), gr: new Float32Array(N), bl: new Float32Array(N), g: new Float32Array(N) };
  const { data, width: w, height: h } = frame;
  const off = [0, 0, -1, 0, 1, 0, 0, -1, 0, 1];
  const lum = new Float32Array(5);
  for (let k = 0; k < N; k++) {
    if (fp.w0[k] === 0) continue;
    const x = Math.round(fp.px[k]!);
    const y = Math.round(fp.py[k]!);
    const d = fp.step[(k / fp.n) | 0]!;
    let r = 0;
    let gr = 0;
    let bl = 0;
    for (let q = 0; q < 5; q++) {
      const xx = Math.max(0, Math.min(w - 1, x + off[2 * q]! * d));
      const yy = Math.max(0, Math.min(h - 1, y + off[2 * q + 1]! * d));
      const o = (yy * w + xx) * 4;
      const R = data[o]!;
      const G = data[o + 1]!;
      const B = data[o + 2]!;
      r += SRGB_LIN[R]!;
      gr += SRGB_LIN[G]!;
      bl += SRGB_LIN[B]!;
      lum[q] = 0.299 * R + 0.587 * G + 0.114 * B;
    }
    s.r[k] = r / 5;
    s.gr[k] = gr / 5;
    s.bl[k] = bl / 5;
    s.g[k] = ((Math.abs(lum[2]! - lum[1]!) + Math.abs(lum[4]! - lum[3]!)) * 100) / 255;
  }
  return s;
}

/** Feature vector per cell: L, a*, b*, spread (p90 - p10 of L), mean gradient. */
export const NF = 5;

export interface CellFeatures {
  x: Float32Array;
  /** Visible fraction of each cell's footprint under the occupancy prior. */
  vis: Float32Array;
  /** Log of the weighted mean linear luminance of each cell (before its gain), for the illumination field. */
  logY: Float32Array;
}

const MIN_OCC_WEIGHT = 0.08;

/** Weighted per-cell features; sample weights are reduced by the probability that a nearer piece covers them.
 *  `gain` (per cell) multiplies the linear colour first, to undo the local illumination. */
export function cellFeatures(fp: Footprints, s: Samples, pOcc: ArrayLike<number>, gain?: ArrayLike<number>): CellFeatures {
  const x = new Float32Array(64 * NF);
  const vis = new Float32Array(64);
  const logY = new Float32Array(64);
  const n = fp.n;
  const w = new Float32Array(n);
  const Ls = new Float32Array(n);
  const idx: number[] = [];
  for (let c = 0; c < 64; c++) {
    const k0 = gain ? gain[c]! : 1;
    const kg = k0 === 1 ? 1 : k0 ** (1 / 2.2);
    let sw = 0;
    let st = 0;
    let sv = 0;
    let L = 0;
    let A = 0;
    let B = 0;
    let G = 0;
    let Y = 0;
    idx.length = 0;
    for (let q = 0; q < n; q++) {
      const k = c * n + q;
      st += fp.wt[k]!;
      w[q] = 0;
      if (fp.w0[k] === 0) continue;
      let keep = 1;
      for (let o = fp.occStart[k]!; o < fp.occStart[k + 1]!; o++) keep *= 1 - pOcc[fp.occCells[o]!]!;
      sv += fp.w0[k]! * keep;
      const ww = fp.w0[k]! * Math.max(MIN_OCC_WEIGHT, keep);
      w[q] = ww;
      sw += ww;
      const r = s.r[k]! * k0;
      const gr = s.gr[k]! * k0;
      const bl = s.bl[k]! * k0;
      const yl = 0.2126 * r + 0.7152 * gr + 0.0722 * bl;
      const fx = labF((0.4124 * r + 0.3576 * gr + 0.1805 * bl) / 0.9505);
      const fy = labF(yl);
      const fz = labF((0.0193 * r + 0.1192 * gr + 0.9505 * bl) / 1.089);
      const l = 116 * fy - 16;
      Ls[q] = l;
      Y += (ww * yl) / k0;
      L += ww * l;
      A += ww * 500 * (fx - fy);
      B += ww * 200 * (fy - fz);
      G += ww * s.g[k]! * kg;
      idx.push(q);
    }
    vis[c] = st > 0 ? sv / st : 0;
    if (sw <= 0) continue;
    logY[c] = Math.log(Math.max(1e-4, Y / sw));
    idx.sort((p, q) => Ls[p]! - Ls[q]!);
    let acc = 0;
    let p10 = NaN;
    let p90 = NaN;
    for (const q of idx) {
      acc += w[q]!;
      const v = Ls[q]!;
      if (Number.isNaN(p10) && acc >= 0.1 * sw) p10 = v;
      if (Number.isNaN(p90) && acc >= 0.9 * sw) p90 = v;
    }
    const o = c * NF;
    x[o] = L / sw;
    x[o + 1] = A / sw;
    x[o + 2] = B / sw;
    x[o + 3] = p90 - p10;
    x[o + 4] = G / sw;
  }
  return { x, vis, logY };
}

// ---------------------------------------------------------------------------------------------------------------
// Illumination

/**
 * Lighting varies across the board (gradients, shadow bands), so absolute colour models fitted on one part of the
 * board misread the rest. Each cell's log luminance is modelled as reflectance(class, parity) + illumination(cell),
 * the illumination being estimated from the cell's neighbours (the median of their residuals, excluding the cell
 * itself so a piece that just arrived does not correct itself away). Returns per-cell linear gains.
 */
/** Fewest neighbour residuals for the local illumination (the neighbourhood grows from 3x3 up to 7x7). */
const ILLUM_MIN_NB = 4;

function illumGains(feats: CellFeatures, grid: Uint8Array, refl: Float64Array): Float32Array {
  const res = new Float32Array(64).fill(NaN);
  const all: number[] = [];
  for (let c = 0; c < 64; c++)
    if (feats.vis[c]! >= VIS_FIT) {
      res[c] = feats.logY[c]! - refl[grid[c]! * 2 + parityOf(c)]!;
      all.push(res[c]!);
    }
  const gains = new Float32Array(64).fill(1);
  if (all.length < 8) return gains;
  const ref = medianOf(all);
  const nb: number[] = [];
  for (let c = 0; c < 64; c++) {
    const i = c & 7;
    const j = c >> 3;
    for (let rad = 1; rad <= 3; rad++) {
      nb.length = 0;
      for (let dj = -rad; dj <= rad; dj++)
        for (let di = -rad; di <= rad; di++) {
          const ii = i + di;
          const jj = j + dj;
          if ((di === 0 && dj === 0) || ii < 0 || jj < 0 || ii > 7 || jj > 7) continue;
          const v = res[jj * 8 + ii]!;
          if (!Number.isNaN(v)) nb.push(v);
        }
      if (nb.length >= ILLUM_MIN_NB) break;
    }
    const I = nb.length ? medianOf(nb) : ref;
    gains[c] = Math.exp(Math.max(-1.5, Math.min(1.5, ref - I)));
  }
  return gains;
}

/** Reflectance per (class, parity) and the illumination field, jointly from a labelled grid. */
function fitIllum(feats: CellFeatures, grid: Uint8Array): { refl: Float64Array; gains: Float32Array } {
  const refl = new Float64Array(6);
  let gains: Float32Array = new Float32Array(64).fill(1);
  for (let it = 0; it < 4; it++) {
    const byK: number[][] = [[], [], [], [], [], []];
    for (let c = 0; c < 64; c++) if (feats.vis[c]! >= VIS_FIT) byK[grid[c]! * 2 + parityOf(c)]!.push(feats.logY[c]! + Math.log(gains[c]!));
    const allK = byK.flat();
    for (let k = 0; k < 6; k++) refl[k] = byK[k]!.length ? medianOf(byK[k]!) : allK.length ? medianOf(allK) : 0;
    gains = illumGains(feats, grid, refl);
  }
  return { refl, gains };
}

// ---------------------------------------------------------------------------------------------------------------
// Models

/** Minimum standard deviation per feature (L, a*, b*, spread, gradient). */
const FLOOR = [2.5, 1.5, 1.5, 3, 2];
const ZCAP = 25;

interface Gauss {
  mu: Float64Array;
  v: Float64Array;
}

const parityOf = (c: number) => ((c & 7) + (c >> 3)) & 1;

/** Capped sum of squared z-scores. */
function zdist(x: Float32Array, c: number, g: Gauss): number {
  let d = 0;
  for (let f = 0; f < NF; f++) {
    const z = x[c * NF + f]! - g.mu[f]!;
    d += Math.min(ZCAP, (z * z) / g.v[f]!);
  }
  return d;
}

function nll(x: Float32Array, c: number, g: Gauss): number {
  let d = 0;
  for (let f = 0; f < NF; f++) {
    const z = x[c * NF + f]! - g.mu[f]!;
    d += 0.5 * (Math.min(ZCAP, (z * z) / g.v[f]!) + Math.log(g.v[f]!));
  }
  return d;
}

function medianOf(a: number[]): number {
  const s = [...a].sort((p, q) => p - q);
  const n = s.length;
  return n === 0 ? 0 : n & 1 ? s[n >> 1]! : (s[n / 2 - 1]! + s[n / 2]!) / 2;
}

/** Median / MAD model of the given cells; null with fewer than `minN` cells. */
function robustModel(x: Float32Array, cells: number[], floorMul = 1, minN = 3): Gauss | null {
  if (cells.length < minN) return null;
  const mu = new Float64Array(NF);
  const v = new Float64Array(NF);
  for (let f = 0; f < NF; f++) {
    const vals = cells.map((c) => x[c * NF + f]!);
    const m = medianOf(vals);
    const mad = medianOf(vals.map((q) => Math.abs(q - m)));
    const sd = Math.max(1.4826 * mad, FLOOR[f]! * floorMul);
    mu[f] = m;
    v[f] = sd * sd;
  }
  return { mu, v };
}

const cloneGauss = (g: Gauss): Gauss => ({ mu: new Float64Array(g.mu), v: new Float64Array(g.v) });

/** models[class][parity], class = OCC_EMPTY / OCC_WHITE / OCC_BLACK. */
type Models = [Gauss, Gauss][];

/** Starting-position grid for a rank axis (0: ranks along j, 1: along i) and side (0: white on lines 0-1). */
export function startGrid(axis: number, side: number): Uint8Array {
  const g = new Uint8Array(64);
  for (let c = 0; c < 64; c++) {
    const line = axis === 0 ? c >> 3 : c & 7;
    if (line <= 1) g[c] = side === 0 ? OCC_WHITE : OCC_BLACK;
    else if (line >= 6) g[c] = side === 0 ? OCC_BLACK : OCC_WHITE;
  }
  return g;
}

const VIS_FIT = 0.4;
/** Evidence weight kept by a cell that matches no class model. */
const OUTLIER_EVIDENCE = 0.25;

/** Occupancy prior of the occluder model from a grid (empty squares keep a small chance of a piece). */
const priorOf = (grid: Uint8Array): number[] => Array.from(grid, (v) => (v === OCC_EMPTY ? 0.1 : 1));
/** Occluder prior while the committed grid is stale (after an unresolved relabelling): no cell is trusted. */
const STALE_PRIOR: readonly number[] = new Array<number>(64).fill(0.3);

interface StartTest {
  score: number;
  side: number;
  dL: number;
  feats: CellFeatures;
}

/** Scores the starting-position hypothesis along one axis: middle lines fit one empty model per parity, the four
 *  outer lines deviate from it, and the two edge groups differ in lightness (the lighter one is white). */
function testStart(fp: Footprints, s: Samples, axis: number, dev: number): StartTest {
  const g = startGrid(axis, 0);
  const prior = Array.from(g, (v) => (v === OCC_EMPTY ? 0 : 1));
  const feats = cellFeatures(fp, s, prior);
  const { x, vis } = feats;
  const fail: StartTest = { score: 0, side: 0, dL: 0, feats };
  const empty: [number[], number[]] = [[], []];
  const groups: [number[], number[]] = [[], []];
  for (let c = 0; c < 64; c++) {
    // Edge-group cells count even when hidden: whatever covers them is a piece of the same group.
    if (g[c] !== OCC_EMPTY) groups[g[c] === OCC_WHITE ? 0 : 1]!.push(c);
    else if (vis[c]! >= VIS_FIT) empty[parityOf(c)]!.push(c);
  }
  if (empty[0].length < 6 || empty[1].length < 6 || groups[0].length < 6 || groups[1].length < 6) return fail;
  const em = [robustModel(x, empty[0])!, robustModel(x, empty[1])!];
  let eOk = 0;
  for (const p of [0, 1]) for (const c of empty[p]!) if (zdist(x, c, em[p]!) < dev) eOk++;
  let oOk = 0;
  const lsum = [[0, 0], [0, 0]];
  const lcnt = [[0, 0], [0, 0]];
  for (const gi of [0, 1])
    for (const c of groups[gi]!) {
      if (zdist(x, c, em[parityOf(c)]!) >= dev) oOk++;
      lsum[gi]![parityOf(c)]! += x[c * NF]!;
      lcnt[gi]![parityOf(c)]!++;
    }
  const score = (eOk / (empty[0].length + empty[1].length)) * (oOk / (groups[0].length + groups[1].length));
  let dL = 0;
  let np = 0;
  for (const p of [0, 1])
    if (lcnt[0]![p]! > 0 && lcnt[1]![p]! > 0) {
      dL += lsum[0]![p]! / lcnt[0]![p]! - lsum[1]![p]! / lcnt[1]![p]!;
      np++;
    }
  dL = np ? dL / np : 0;
  return { score, side: dL >= 0 ? 0 : 1, dL, feats };
}

/** Per-class, per-parity models from a labelled grid; null when a class has too few visible cells. */
function fitModels(feats: CellFeatures, grid: Uint8Array, dev: number): Models | null {
  const { x, vis } = feats;
  const out: (Gauss | null)[][] = [];
  for (const cls of [OCC_EMPTY, OCC_WHITE, OCC_BLACK]) {
    const byP: [number[], number[]] = [[], []];
    for (let c = 0; c < 64; c++) if (grid[c] === cls && vis[c]! >= VIS_FIT) byP[parityOf(c)]!.push(c);
    const mul = cls === OCC_EMPTY ? 1 : 1.5;
    let m = [robustModel(x, byP[0], mul), robustModel(x, byP[1], mul)];
    if (cls === OCC_EMPTY && m[0] && m[1]) {
      // Second pass without the cells that deviate (a missed piece, a shadow).
      m = [0, 1].map((p) => robustModel(x, byP[p]!.filter((c) => zdist(x, c, m[p]!) < dev), mul) ?? m[p]!);
    }
    if (!m[0] && !m[1]) {
      const pooled = robustModel(x, [...byP[0], ...byP[1]], mul, 2);
      if (!pooled) return null;
      m = [pooled, cloneGauss(pooled)];
    }
    out.push([m[0] ?? cloneGauss(m[1]!), m[1] ?? cloneGauss(m[0]!)]);
  }
  return out as Models;
}

/** Unsupervised fit: per-parity empty model (iterated median), occupied = deviating cells, 2-means on L. */
function fitUnsupervised(feats: CellFeatures, dev: number): { grid: Uint8Array } | null {
  const { x, vis } = feats;
  const grid = new Uint8Array(64);
  const occupied: number[] = [];
  for (const p of [0, 1]) {
    let cells: number[] = [];
    for (let c = 0; c < 64; c++) if (parityOf(c) === p && vis[c]! >= VIS_FIT) cells.push(c);
    let m = robustModel(x, cells);
    if (!m) return null;
    for (let it = 0; it < 2; it++) {
      const inl = cells.filter((c) => zdist(x, c, m!) < dev);
      m = robustModel(x, inl) ?? m;
    }
    cells = cells.filter((c) => zdist(x, c, m!) >= dev);
    occupied.push(...cells);
  }
  if (occupied.length < 4) return null;
  const Ls = occupied.map((c) => x[c * NF]!);
  let lo = Math.min(...Ls);
  let hi = Math.max(...Ls);
  if (hi - lo < 8) return null;
  for (let it = 0; it < 10; it++) {
    const mid = (lo + hi) / 2;
    const a = Ls.filter((v) => v < mid);
    const b = Ls.filter((v) => v >= mid);
    if (!a.length || !b.length) return null;
    lo = a.reduce((s, v) => s + v, 0) / a.length;
    hi = b.reduce((s, v) => s + v, 0) / b.length;
  }
  const mid = (lo + hi) / 2;
  for (const c of occupied) grid[c] = x[c * NF]! >= mid ? OCC_WHITE : OCC_BLACK;
  return { grid };
}

// ---------------------------------------------------------------------------------------------------------------
// Temporal filter

/**
 * Per-cell hysteresis: a cell changes its committed state after `occHoldFrames` consistent frames. Commits are
 * frozen while many cells flip between consecutive frames (a hand over the board) or the board was lost, until the
 * board has been stable for `occSettleMs`.
 */
export class OccupancyFilter {
  committed: Uint8Array | null = null;
  private readonly cand = new Uint8Array(64);
  private readonly count = new Uint8Array(64);
  /** Last confident raw state per cell (255 = none). */
  private readonly prev = new Uint8Array(64).fill(255);
  private unstableAt = -Infinity;

  reset(grid: Uint8Array | null = null): void {
    this.committed = grid ? new Uint8Array(grid) : null;
    this.count.fill(0);
    this.prev.fill(255);
    this.unstableAt = -Infinity;
  }

  /** Board not seen (or tracking lost): freeze. */
  noBoard(nowMs: number): void {
    this.unstableAt = nowMs;
    this.prev.fill(255);
  }

  frozen(nowMs: number, params: Params): boolean {
    return nowMs - this.unstableAt < param(params, OCCUPANCY_PARAMS, 'occSettleMs');
  }

  /** Overrides the committed grid (position hint) without touching the freeze state. */
  setCommitted(grid: Uint8Array): void {
    if (!this.committed) this.committed = new Uint8Array(grid);
    else this.committed.set(grid);
  }

  /**
   * Cells with conf below `occCellMin` are ignored. Returns whether commits are frozen. With `commit` false (a
   * position hint owns the committed grid) only the freeze detection runs: the committed grid and the per-cell
   * hysteresis are left alone.
   */
  update(raw: Uint8Array, conf: Float32Array, params: Params, nowMs: number, commit = true): boolean {
    if (!this.committed) this.committed = new Uint8Array(raw);
    const cellMin = param(params, OCCUPANCY_PARAMS, 'occCellMin');
    const hold = param(params, OCCUPANCY_PARAMS, 'occHoldFrames');
    let flips = 0;
    for (let c = 0; c < 64; c++) if (conf[c]! >= cellMin && this.prev[c] !== 255 && this.prev[c] !== raw[c]) flips++;
    if (flips >= param(params, OCCUPANCY_PARAMS, 'occFreezeCells')) this.unstableAt = nowMs;
    const frozen = this.frozen(nowMs, params);
    for (let c = 0; c < 64; c++) {
      if (conf[c]! < cellMin) continue;
      const r = raw[c]!;
      this.prev[c] = r;
      if (!commit) {
        this.count[c] = 0;
        continue;
      }
      if (r === this.committed[c]) {
        this.count[c] = 0;
        continue;
      }
      if (r === this.cand[c] && this.count[c]! > 0) this.count[c] = Math.min(255, this.count[c]! + 1);
      else {
        this.cand[c] = r;
        this.count[c] = 1;
      }
      if (!frozen && this.count[c]! >= hold) {
        this.committed[c] = r;
        this.count[c] = 0;
      }
    }
    return frozen;
  }

  /** Re-indexes the state after a relabelling of the board cells: new cell c was old cell map[c]. */
  permute(map: ArrayLike<number>): void {
    const re = (a: Uint8Array) => {
      const b = Uint8Array.from(a);
      for (let c = 0; c < 64; c++) a[c] = b[map[c]!]!;
    };
    if (this.committed) re(this.committed);
    re(this.cand);
    re(this.count);
    re(this.prev);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Tracker

export interface OccupancyResult {
  /** Committed grid (cell (i, j) at j * 8 + i), or null when this frame was dropped / not calibrated yet. While a
   *  position hint is in effect (see OccupancyTracker.setPosition) this is the hint in board cell order. */
  grid: Uint8Array | null;
  /** This frame's raw classification (all empty before calibration). */
  raw: Uint8Array;
  /** Per-cell confidence: classifier margin x visible fraction. */
  conf: Float32Array;
  /**
   * Per-cell class log-probabilities, cell c's (empty, white, black) at c * 3 + (0, 1, 2), normalised per cell
   * (logsumexp = 0). Hidden / low-visibility cells and outliers are flattened towards uniform (log 1/3) by the evidence
   * lost (weight vis^occLikVisGamma, exactly uniform at vis 0). All uniform before calibration and on frames without a camera. Intended for scoring
   * candidate (legal) positions: sum logLik[c * 3 + class(c)] over the cells. Always in board cell order (the
   * observation carries the chess-square-ordered copy).
   */
  logLik: Float32Array;
  /**
   * Per cell, this frame's probability (exp of logLik) of the cell's committed class, even when the frame is dropped
   * (shows why). Null when not calibrated / no committed grid / no camera.
   */
  committedProb: Float32Array | null;
  stats: OccupancyStats;
  /**
   * The frame's evidence for the game layer: logLik and the visible fractions in chess square order when oriented
   * (board cell order otherwise), the calibration state, `framed` (see boardFramed) and `stable` = framed, calibrated,
   * not dropped and not frozen (a freeze covers mass changes such as a hand, and the `occSettleMs` after any frame
   * without a board or with the board not fully in view).
   */
  observation: Observation;
  /** orientation[sq] = board cell of chess square sq (a1 = 0 ... h8 = 63); null while not oriented. */
  orientation: Uint8Array | null;
}

export interface OccupancyDebug {
  fp: Footprints;
  weights: Float32Array;
  raw: Uint8Array;
  conf: Float32Array;
  /** Illumination-corrected features of the frame. */
  feats: CellFeatures;
}

/** The 8 relabellings of the board square: new board point (x, y) is old point g(x, y). The first 4 are rotations
 *  (they keep the handedness of the board frame), the last 4 reflections. */
const DIHEDRAL: ReadonlyArray<(x: number, y: number) => Point> = [
  (x, y) => [x, y], (x, y) => [8 - y, x], (x, y) => [8 - x, 8 - y], (x, y) => [y, 8 - x],
  (x, y) => [y, x], (x, y) => [8 - x, y], (x, y) => [x, 8 - y], (x, y) => [8 - y, 8 - x],
];

const applyHb = (h: Mat3, x: number, y: number): Point => {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
};

export interface DihedralMatch {
  /** Cell map (new cell -> old cell) when `hb` is a clear non-identity relabelling of `prev`; null otherwise. */
  map: Uint8Array | null;
  /**
   * Whether `hb` is clearly one of the 8 labellings of `prev` (the identity included). False when the board moved too
   * much between the two (e.g. the camera moved during a long loss) to tell which labelling it is: the cell state
   * (filter, per-parity models, orientation) may then be misaligned.
   */
  resolved: boolean;
}

/** Matches `hb` against the previous frame's homography within the board's dihedral ambiguity (corner distances). */
export function dihedralMatch(prev: Mat3, hb: Mat3): DihedralMatch {
  const corners: Point[] = [[0, 0], [8, 0], [8, 8], [0, 8]];
  const now = corners.map(([x, y]) => applyHb(hb, x, y));
  const diag = Math.hypot(now[0]![0] - now[2]![0], now[0]![1] - now[2]![1]);
  const errs = DIHEDRAL.map((g) => {
    let e = 0;
    corners.forEach(([x, y], k) => {
      const [u, v] = g(x, y);
      const p = applyHb(prev, u, v);
      e += Math.hypot(p[0] - now[k]![0], p[1] - now[k]![1]);
    });
    return e / 4;
  });
  let best = 0;
  for (let g = 1; g < 8; g++) if (errs[g]! < errs[best]!) best = g;
  let second = Infinity;
  for (let g = 0; g < 8; g++) if (g !== best) second = Math.min(second, errs[g]!);
  const close = errs[best]! < 0.1 * diag;
  const resolved = close && second > 2 * errs[best]!;
  if (best === 0 || !close || !(errs[0]! > 2 * errs[best]!)) return { map: null, resolved };
  const map = new Uint8Array(64);
  for (let c = 0; c < 64; c++) {
    const [u, v] = DIHEDRAL[best]!((c & 7) + 0.5, (c >> 3) + 0.5);
    map[c] = Math.floor(v) * 8 + Math.floor(u);
  }
  return { map, resolved };
}

/** Cell map (new cell -> old cell) when `hb` is a clear dihedral relabelling of `prev`; null when it is not. */
export function dihedralRemap(prev: Mat3, hb: Mat3): Uint8Array | null {
  return dihedralMatch(prev, hb).map;
}

// ---------------------------------------------------------------------------------------------------------------
// Orientation (board cell -> chess square)

/**
 * The 8 candidate orientations: ORIENT_MAPS[g][sq] = board cell of chess square sq (file sq & 7, rank sq >> 3) when
 * the chess frame (files a->h, ranks 1->8) sits on the board as DIHEDRAL[g]: square (f, r) covers board point
 * g(f + 0.5, r + 0.5). g = 0 is the identity (a1 = cell (0, 0), files along i, ranks along j). g < 4 keep the
 * handedness of the board cell frame, g >= 4 reverse it.
 */
export const ORIENT_MAPS: readonly Uint8Array[] = DIHEDRAL.map((g) => {
  const m = new Uint8Array(64);
  for (let sq = 0; sq < 64; sq++) {
    const [u, v] = g((sq & 7) + 0.5, (sq >> 3) + 0.5);
    m[sq] = Math.floor(v) * 8 + Math.floor(u);
  }
  return m;
});

/** The starting position's occupancy in chess square order (white on ranks 1-2). */
export const START_SQUARES: Uint8Array = Uint8Array.from({ length: 64 }, (_, sq) =>
  sq >> 3 <= 1 ? OCC_WHITE : sq >> 3 >= 6 ? OCC_BLACK : OCC_EMPTY,
);

/** Re-orders a per-cell array (`stride` values per cell) into chess square order: out[sq] = a[orientation[sq]]. */
export function toSquareOrder<T extends Float32Array | Uint8Array>(a: T, orientation: ArrayLike<number>, stride = 1): T {
  const out = new (a.constructor as new (n: number) => T)(a.length);
  for (let sq = 0; sq < 64; sq++) for (let k = 0; k < stride; k++) out[sq * stride + k] = a[orientation[sq]! * stride + k]!;
  return out;
}

/** Inverse of toSquareOrder for a 64-entry grid: out[orientation[sq]] = g[sq] (a chess-ordered grid to board cells). */
export function toCellOrder(g: Uint8Array, orientation: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(64);
  for (let sq = 0; sq < 64; sq++) out[orientation[sq]!] = g[sq]!;
  return out;
}

/**
 * Handedness of the board cell frame seen by the camera: +1 when (i, j, up) is right-handed, with "up" the side of
 * the board the camera looks at. The camera is always above the board, so this is the sign of the pose's
 * (r1 x r2) . (towards the camera) in cameraFromHomography terms, which reduces to the orientation of the image
 * of the board: a right-handed frame seen from above appears counter-clockwise (x then y), i.e. with a negative
 * Jacobian determinant in image coordinates (y down). The Jacobian of a homography at a point is det(H) / w^3, so
 * its sign is that of det(H) * w, taken at the board centre. 0 for a degenerate hb.
 */
export function boardHandedness(hb: Mat3): number {
  const det =
    hb[0] * (hb[4] * hb[8] - hb[5] * hb[7]) - hb[1] * (hb[3] * hb[8] - hb[5] * hb[6]) + hb[2] * (hb[3] * hb[7] - hb[4] * hb[6]);
  const j = det * (4 * hb[6] + 4 * hb[7] + hb[8]);
  return j < 0 ? 1 : j > 0 ? -1 : 0;
}

export type OrientRefusal = 'margin' | 'miss' | 'mirror' | 'ambiguous' | 'parity';

export type OrientChoice = { g: number; score: number; gap: number } | { g: -1; reason: OrientRefusal };

/**
 * Chooses the orientation from one frame's evidence. The target grid (chess square order: the game's position hint,
 * else the starting position) is placed on the cells under each of the 8 maps and scored by the summed logLik; the
 * best placement must beat every placement predicting a DIFFERENT cell grid by `margin` nats, and at most `maxMiss`
 * cells may confidently contradict it (p < 0.1). Placements predicting the same cell grid (a symmetric target: the
 * starting position is symmetric under the a<->h mirror) are told apart by the handedness: the chess frame
 * (files, ranks, up) must be right-handed, i.e. white's left is the a-file. The handedness also cross-checks an
 * asymmetric winner ('mirror' when it contradicts: a mirrored image). Finally a1 must be a dark square:
 * `darkParity` (parity of the darker empty squares, -1 when inconclusive) must match a1's cell parity.
 */
export function chooseOrientation(
  logLik: Float32Array, target: Uint8Array, handed: number, darkParity: number, margin: number, maxMiss: number,
): OrientChoice {
  const grids = ORIENT_MAPS.map((m) => toCellOrder(target, m));
  const scores = grids.map((g) => {
    let s = 0;
    for (let c = 0; c < 64; c++) s += logLik[c * 3 + g[c]!]!;
    return s;
  });
  let best = 0;
  for (let g = 1; g < 8; g++) if (scores[g]! > scores[best]!) best = g;
  const same = (a: Uint8Array, b: Uint8Array) => a.every((v, c) => v === b[c]);
  const group: number[] = [];
  let second = -Infinity;
  for (let g = 0; g < 8; g++) {
    if (same(grids[g]!, grids[best]!)) group.push(g);
    else second = Math.max(second, scores[g]!);
  }
  const gap = scores[best]! - second;
  if (!(gap >= margin)) return { g: -1, reason: 'margin' };
  let miss = 0;
  const lo = Math.log(0.1);
  for (let c = 0; c < 64; c++) if (logLik[c * 3 + grids[best]![c]!]! < lo) miss++;
  if (miss > maxMiss) return { g: -1, reason: 'miss' };
  const cands = group.filter((g) => (g < 4 ? 1 : -1) === handed);
  if (cands.length === 0) return { g: -1, reason: 'mirror' };
  if (cands.length > 1) return { g: -1, reason: 'ambiguous' };
  const g = cands[0]!;
  if (darkParity >= 0 && parityOf(ORIENT_MAPS[g]![0]!) !== darkParity) return { g: -1, reason: 'parity' };
  return { g, score: scores[g]!, gap };
}

/**
 * Square occupancy (empty / white / black) from the raw frame and the board homography: camera-aware footprints,
 * per-parity class models calibrated on the starting position (or unsupervised after `occFallbackMs`), per-cell
 * confidence, frame dropping and the temporal filter.
 *
 * Orientation (milestone 9): which cell is which chess square. Found by chooseOrientation on stable frames, against
 * the position hint when one is set, else against the starting position (so the first orientation comes right after
 * the start-position calibration), and adopted after `occOrientFrames` consecutive agreeing frames. It is permuted
 * with the cells on dihedral relabellings, and marked unverified (not dropped: the class models survive too) after
 * more than `occLossMs` without a board or a relabelling that cannot be resolved; it is then re-found the same way.
 * Only reset() (the reset / resetProfile messages) drops the models and the orientation.
 *
 * Position hint (setPosition): while set and oriented it replaces the filter's committed grid (occluder prior,
 * illumination and learning labels, the returned grid); the filter then only detects freezes.
 */
export class OccupancyTracker {
  private models: Models | null = null;
  /** Log reflectance per (class, parity) for the illumination field. */
  private refl: Float64Array = new Float64Array(6);
  private state: OccupancyStats['state'] = 'start';
  private readonly filter = new OccupancyFilter();
  private bootKey = -1;
  private bootCount = 0;
  private firstSeenAt = NaN;
  private lastHb: Mat3 | null = null;
  private lastBoardAt = NaN;
  private lastStats: OccupancyStats = { state: 'start', empty: 0, white: 0, black: 0, lowCells: 0, dropped: false, frozen: false };
  /** Position hint from the game, chess square order; kept across reset() (the main thread owns it). */
  private hint: Uint8Array | null = null;
  /** Last orientation found (square -> cell), permuted with the cells; trusted only while `orientOk`. */
  private orient: Uint8Array | null = null;
  private orientOk = false;
  /** Placement (ORIENT_MAPS index in the current cell frame) chosen on the last evaluated frames, and their count. */
  private orientCand = -1;
  private orientCount = 0;
  /** After an unresolved relabelling: the per-parity models must be checked against the frame, and the filter's
   *  committed grid is stale until the next orientation. */
  private parityUnchecked = false;
  private filterStale = false;
  /** Why the last orientation attempt failed (debug); '' when oriented or not tried. */
  orientReason: OrientRefusal | 'uncalibrated' | '' = '';
  /** Footprints and weights of the last frame, for the debug view. */
  debug: OccupancyDebug | null = null;

  /** Drops the models, the filter and the orientation (keeps the position hint). */
  reset(): void {
    this.models = null;
    this.state = 'start';
    this.filter.reset();
    this.bootKey = -1;
    this.bootCount = 0;
    this.firstSeenAt = NaN;
    this.lastHb = null;
    this.lastBoardAt = NaN;
    this.orient = null;
    this.unverify();
    this.parityUnchecked = false;
    this.filterStale = false;
    this.orientReason = '';
    this.debug = null;
  }

  get calibrated(): boolean {
    return this.models !== null;
  }

  get oriented(): boolean {
    return this.orientOk && this.orient !== null;
  }

  /** orientation[sq] = board cell of chess square sq, or null while not (re-)verified. */
  get orientation(): Uint8Array | null {
    return this.oriented ? new Uint8Array(this.orient!) : null;
  }

  /** The game's position (64 entries, chess square order) or null (no game): see the class doc. */
  setPosition(grid: Uint8Array | null): void {
    this.hint = grid && grid.length === 64 ? new Uint8Array(grid) : null;
  }

  noBoard(nowMs: number): OccupancyStats {
    this.filter.noBoard(nowMs);
    this.bootCount = 0;
    this.lastStats = { ...this.lastStats, dropped: true, frozen: true, lowCells: 0 };
    return this.lastStats;
  }

  /**
   * One frame with a board. `visibleRect` (default: the whole frame) is the part of the frame the user sees; when the
   * board is not wholly inside it (boardFramed) the frame is treated like a hand over the board: the filter freezes
   * (commits resume `occSettleMs` after the board is fully back), nothing is calibrated, learnt or oriented, and the
   * result is `stable` = `framed` = false with no grid. Occupancy is still computed (debug view, committedProb).
   */
  update(frame: RawFrame, hb: Mat3, params: Params, nowMs: number, visibleRect?: FrameRect): OccupancyResult {
    const raw = new Uint8Array(64);
    const conf = new Float32Array(64);
    const logLik = new Float32Array(64 * 3).fill(-Math.log(3));
    if (Number.isNaN(this.firstSeenAt)) this.firstSeenAt = nowMs;
    // A long loss: the board may have been turned, or the camera moved; the orientation must be re-verified.
    if (nowMs - this.lastBoardAt > param(params, OCCUPANCY_PARAMS, 'occLossMs')) this.unverify();
    this.lastBoardAt = nowMs;
    // Keep the grid in hb's frame if the detector relabelled the board corners.
    if (this.lastHb) {
      const m = dihedralMatch(this.lastHb, hb);
      if (m.map) this.permute(m.map);
      else if (!m.resolved) {
        this.unverify();
        this.parityUnchecked = this.models !== null;
        this.filterStale = true;
      }
    }
    this.lastHb = hb;
    const cam = cameraFromHomography(hb, frame.width, frame.height, param(params, OCCUPANCY_PARAMS, 'occFovDeg'));
    // Without a camera the framing is unknown (the frame is unstable anyway): not reported as cropped.
    const framed = !cam || boardFramed(cam, visibleRect ?? [0, 0, frame.width, frame.height], params).framed;
    const fail = (): OccupancyResult => {
      this.lastStats = this.stats(0, true, false);
      return this.result(nowMs, null, raw, conf, logLik, new Float32Array(64), null, false, framed);
    };
    if (!cam) return fail();
    if (!framed) {
      this.filter.noBoard(nowMs);
      this.bootCount = 0;
    }
    const fp = cellFootprints(cam, frame.width, frame.height, params);
    const s = sampleFrame(frame, fp);
    const dev = param(params, OCCUPANCY_PARAMS, 'occDeviation');

    if (framed && this.state !== 'calibrated') this.tryBootstrap(fp, s, params, dev);
    if (framed && !this.models && nowMs - this.firstSeenAt > param(params, OCCUPANCY_PARAMS, 'occFallbackMs')) {
      const u = fitUnsupervised(cellFeatures(fp, s, new Float32Array(64).fill(0.3)), dev);
      if (u && this.calibrate(fp, s, u.grid, dev)) this.state = 'fallback';
    }
    if (!this.models || !this.filter.committed) {
      this.orientReason = 'uncalibrated';
      return fail();
    }

    // The position hint (when oriented) is the committed grid; the filter's own hysteresis is bypassed.
    const hintCells = this.hintCells();
    if (hintCells) this.filter.setCommitted(hintCells);
    const committed = this.filter.committed;
    // After an unresolved relabelling the committed grid belongs to another cell frame: used as the occluder prior and
    // the illumination labels it would make the frame read as that stale grid (and never re-orient), so until the next
    // orientation the features use a flat prior and no illumination correction, and nothing is learnt.
    const stale = this.filterStale;
    const prior = stale ? STALE_PRIOR : priorOf(committed);
    const featsNow = () => cellFeatures(fp, s, prior, stale ? undefined : illumGains(cellFeatures(fp, s, prior), committed, this.refl));
    let feats = featsNow();
    if (framed && this.parityUnchecked) {
      const r = this.recheckParity(feats);
      if (r !== 'inconclusive') this.parityUnchecked = false;
      if (r === 'swapped') feats = featsNow();
    }
    const models = this.models;
    const outlier = param(params, OCCUPANCY_PARAMS, 'occOutlier');
    const scale = param(params, OCCUPANCY_PARAMS, 'occMarginScale');
    const temp = param(params, OCCUPANCY_PARAMS, 'occLikTemp');
    const visGamma = param(params, OCCUPANCY_PARAMS, 'occLikVisGamma');
    const d3 = [0, 0, 0];
    let sum = 0;
    for (let c = 0; c < 64; c++) {
      const p = parityOf(c);
      let best = Infinity;
      let second = Infinity;
      let arg = 0;
      let near = Infinity;
      for (let cls = 0; cls < 3; cls++) {
        const g = models[cls]![p]!;
        const d = nll(feats.x, c, g);
        d3[cls] = d;
        near = Math.min(near, zdist(feats.x, c, g));
        if (d < best) {
          second = best;
          best = d;
          arg = cls;
        } else if (d < second) second = d;
      }
      raw[c] = arg;
      // Evidence weight: visible fraction (to the power occLikVisGamma), reduced for outliers (hand, glare) that match no model.
      const ew = Math.max(0, Math.min(1, feats.vis[c]!)) ** visGamma * (near > outlier ? OUTLIER_EVIDENCE : 1);
      let lse = -Infinity;
      for (let cls = 0; cls < 3; cls++) {
        const l = (-(d3[cls]! - best) / temp) * ew;
        logLik[c * 3 + cls] = l;
        lse = Math.max(lse, l) + Math.log1p(Math.exp(-Math.abs(lse - l)));
      }
      for (let cls = 0; cls < 3; cls++) logLik[c * 3 + cls]! -= lse;
      conf[c] = near > outlier ? 0 : Math.min(1, (second - best) / scale) * Math.min(1, feats.vis[c]!);
      sum += conf[c]!;
    }
    const cellMin = param(params, OCCUPANCY_PARAMS, 'occCellMin');
    let low = 0;
    for (let c = 0; c < 64; c++) if (conf[c]! < cellMin) low++;
    const dropped = low > param(params, OCCUPANCY_PARAMS, 'occMaxLowCells') || sum / 64 < param(params, OCCUPANCY_PARAMS, 'occFrameMin');
    let frozen = this.filter.frozen(nowMs, params);
    if (!dropped && framed) {
      frozen = this.filter.update(raw, conf, params, nowMs, !hintCells);
      if (!stale) this.learn(feats, raw, conf, this.filter.committed!, params);
    }
    const stable = framed && !dropped && !frozen;
    if (!this.orientOk && stable) this.tryOrient(logLik, hb, params);
    this.debug = { fp, weights: this.weights(fp, prior), raw, conf, feats };
    this.lastStats = this.stats(low, dropped, frozen);
    const committedNow = this.filter.committed ?? committed;
    const committedProb = new Float32Array(64);
    for (let c = 0; c < 64; c++) committedProb[c] = Math.exp(logLik[c * 3 + committedNow[c]!]!);
    const grid = dropped || !framed ? null : new Uint8Array(committedNow);
    return this.result(nowMs, grid, raw, conf, logLik, feats.vis, committedProb, stable, framed);
  }

  private result(
    t: number, grid: Uint8Array | null, raw: Uint8Array, conf: Float32Array, logLik: Float32Array, vis: Float32Array,
    committedProb: Float32Array | null, stable: boolean, framed: boolean,
  ): OccupancyResult {
    const orientation = this.orientation;
    const observation: Observation = {
      t,
      logLik: orientation ? toSquareOrder(logLik, orientation, 3) : new Float32Array(logLik),
      vis: orientation ? toSquareOrder(vis, orientation) : new Float32Array(vis),
      oriented: orientation !== null,
      calibration: this.lastStats.state,
      stable,
      framed,
    };
    return { grid, raw, conf, logLik, committedProb, stats: this.lastStats, observation, orientation };
  }

  /** The position hint in board cell order, when there is one and the orientation is verified. */
  private hintCells(): Uint8Array | null {
    return this.hint && this.oriented ? toCellOrder(this.hint, this.orient!) : null;
  }

  private unverify(): void {
    this.orientOk = false;
    this.orientCand = -1;
    this.orientCount = 0;
  }

  /** Parity of the dark squares from the calibrated empty models (the darker one), -1 when their lightness is too
   *  close to tell (`occParityMinDL`). */
  private darkParity(params: Params): number {
    const e = this.models![OCC_EMPTY]!;
    const d = e[0].mu[0]! - e[1].mu[0]!;
    if (!(Math.abs(d) >= param(params, OCCUPANCY_PARAMS, 'occParityMinDL'))) return -1;
    return d < 0 ? 0 : 1;
  }

  /** One orientation attempt on a stable frame (see chooseOrientation); adopted after `occOrientFrames` agreeing ones. */
  private tryOrient(logLik: Float32Array, hb: Mat3, params: Params): void {
    const target = this.hint ?? START_SQUARES;
    const ch = chooseOrientation(
      logLik, target, boardHandedness(hb), this.darkParity(params),
      param(params, OCCUPANCY_PARAMS, 'occOrientMargin'), param(params, OCCUPANCY_PARAMS, 'occOrientMaxMiss'),
    );
    if (ch.g < 0) {
      this.orientReason = (ch as { reason: OrientRefusal }).reason;
      this.orientCand = -1;
      this.orientCount = 0;
      return;
    }
    this.orientCount = ch.g === this.orientCand ? this.orientCount + 1 : 1;
    this.orientCand = ch.g;
    if (this.orientCount < param(params, OCCUPANCY_PARAMS, 'occOrientFrames')) return;
    this.orient = new Uint8Array(ORIENT_MAPS[ch.g]!);
    this.orientOk = true;
    this.orientReason = '';
    // After an unresolved relabelling the filter's grid belongs to another cell frame: restart it from the target.
    if (this.filterStale) this.filter.setCommitted(toCellOrder(target, this.orient));
    this.filterStale = false;
  }

  /**
   * After a relabelling that could not be resolved, the per-parity models may be swapped (a quarter turn moves cell
   * (0, 0) to the other parity). Keeps whichever assignment explains the frame's visible cells better (each cell's
   * best class NLL, summed), swapping the models and the reflectances when that is clearly the swapped one.
   */
  private recheckParity(feats: CellFeatures): 'kept' | 'swapped' | 'inconclusive' {
    const models = this.models;
    if (!models) return 'inconclusive';
    let asIs = 0;
    let swapped = 0;
    let n = 0;
    for (let c = 0; c < 64; c++) {
      if (feats.vis[c]! < VIS_FIT) continue;
      const p = parityOf(c);
      let a = Infinity;
      let b = Infinity;
      for (let cls = 0; cls < 3; cls++) {
        a = Math.min(a, nll(feats.x, c, models[cls]![p]!));
        b = Math.min(b, nll(feats.x, c, models[cls]![1 - p]!));
      }
      asIs += a;
      swapped += b;
      n++;
    }
    if (n < 16) return 'inconclusive';
    if (!(swapped < asIs - 0.5 * n)) return 'kept';
    this.swapParity();
    return 'swapped';
  }

  /** Swaps the per-parity models and reflectances (cell (0, 0) changed parity). */
  private swapParity(): void {
    if (this.models) for (const m of this.models) m.reverse();
    for (let cls = 0; cls < 3; cls++) {
      const t = this.refl[cls * 2]!;
      this.refl[cls * 2] = this.refl[cls * 2 + 1]!;
      this.refl[cls * 2 + 1] = t;
    }
  }

  private tryBootstrap(fp: Footprints, s: Samples, params: Params, dev: number): void {
    const tests = [testStart(fp, s, 0, dev), testStart(fp, s, 1, dev)];
    const axis = tests[0]!.score >= tests[1]!.score ? 0 : 1;
    const t = tests[axis]!;
    const other = tests[1 - axis]!;
    if (!(t.score >= 0.75 && t.score - other.score >= 0.25 && Math.abs(t.dL) >= 6)) {
      this.bootCount = 0;
      return;
    }
    const key = axis * 2 + t.side;
    this.bootCount = key === this.bootKey ? this.bootCount + 1 : 1;
    this.bootKey = key;
    if (this.bootCount < param(params, OCCUPANCY_PARAMS, 'occBootFrames')) return;
    if (this.calibrate(fp, s, startGrid(axis, t.side), dev)) this.state = 'calibrated';
  }

  /** Fits the illumination (reflectance per class and parity) and the class models on a labelled frame, using the
   *  same occupancy prior as `update` so the models describe exactly the features `update` computes. */
  private calibrate(fp: Footprints, s: Samples, grid: Uint8Array, dev: number): boolean {
    const prior = priorOf(grid);
    const il = fitIllum(cellFeatures(fp, s, prior), grid);
    const models = fitModels(cellFeatures(fp, s, prior, il.gains), grid, dev);
    if (!models) return false;
    this.models = models;
    this.refl = il.refl;
    this.filter.reset(grid);
    this.parityUnchecked = false;
    this.filterStale = false;
    return true;
  }

  /**
   * Slow EMA update of the class models. Learning guard: only cells whose label (the committed grid, i.e. the
   * position hint in game mode) AND this frame's own confident classification (conf >= 0.6) agree are used, so a
   * wrong game commit (or a wrong filter commit) never teaches a model the wrong label.
   */
  private learn(feats: CellFeatures, raw: Uint8Array, conf: Float32Array, labels: Uint8Array, params: Params): void {
    const a = param(params, OCCUPANCY_PARAMS, 'occLearnRate');
    if (!(a > 0) || !this.models) return;
    for (let c = 0; c < 64; c++) {
      if (conf[c]! < 0.6 || raw[c] !== labels[c]) continue;
      const g = this.models[raw[c]!]![parityOf(c)]!;
      const floorMul = raw[c] === OCC_EMPTY ? 1 : 1.5;
      for (let f = 0; f < NF; f++) {
        const d = feats.x[c * NF + f]! - g.mu[f]!;
        g.mu[f] = g.mu[f]! + a * d;
        g.v[f] = Math.max((FLOOR[f]! * floorMul) ** 2, g.v[f]! + a * (d * d - g.v[f]!));
      }
    }
  }

  /** Re-indexes the cell state after a relabelling: new cell c was old cell map[c]. */
  private permute(map: Uint8Array): void {
    this.filter.permute(map);
    // A relabelling that moves cell (0,0) to a cell of the other parity swaps the per-parity models (and reflectances).
    if (parityOf(map[0]!) !== 0) this.swapParity();
    if (this.orient) {
      const inv = new Uint8Array(64);
      for (let c = 0; c < 64; c++) inv[map[c]!] = c;
      for (let sq = 0; sq < 64; sq++) this.orient[sq] = inv[this.orient[sq]!]!;
    }
    // A pending candidate was expressed in the old cell frame.
    this.orientCand = -1;
    this.orientCount = 0;
    this.bootCount = 0;
  }

  private weights(fp: Footprints, prior: readonly number[]): Float32Array {
    const w = new Float32Array(fp.w0.length);
    for (let k = 0; k < w.length; k++) {
      let keep = 1;
      for (let o = fp.occStart[k]!; o < fp.occStart[k + 1]!; o++) keep *= 1 - prior[fp.occCells[o]!]!;
      w[k] = fp.w0[k]! * keep;
    }
    return w;
  }

  private stats(lowCells: number, dropped: boolean, frozen: boolean): OccupancyStats {
    const g = this.filter.committed;
    let white = 0;
    let black = 0;
    if (g && this.models) for (const v of g) v === OCC_WHITE ? white++ : v === OCC_BLACK ? black++ : 0;
    const empty = g && this.models ? 64 - white - black : 0;
    return { state: this.state, empty, white, black, lowCells, dropped, frozen };
  }
}

type Ctx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/** Debug view: footprint samples (green = visible, red = possibly hidden by a nearer piece) and one dot per cell
 *  with the raw class (black = empty, white, green = black piece), its radius scaled by the confidence. */
export function drawOccupancy(ctx: Ctx, d: OccupancyDebug): void {
  const { fp } = d;
  for (let k = 0; k < fp.w0.length; k++) {
    if (fp.w0[k] === 0) continue;
    const v = d.weights[k]! / fp.w0[k]!;
    ctx.fillStyle = `rgb(${Math.round(255 * (1 - v))},${Math.round(255 * v)},0)`;
    ctx.fillRect(fp.px[k]! - 1, fp.py[k]! - 1, 2, 2);
  }
  for (let c = 0; c < 64; c++) {
    const k = c * fp.n;
    if (fp.w0[k] === 0) continue;
    ctx.fillStyle = d.raw[c] === OCC_WHITE ? '#fff' : d.raw[c] === OCC_BLACK ? '#0c0' : '#000';
    ctx.strokeStyle = '#444';
    ctx.beginPath();
    ctx.arc(fp.px[k]!, fp.py[k]!, 2 + 5 * d.conf[c]!, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}
