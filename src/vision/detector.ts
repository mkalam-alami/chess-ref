import type { Corners, Params } from '../worker/protocol';
import { mul3, type Mat3, type Point, type Vec3 } from '../geom/homography';
import { normFrame, segmentsToNormalised, type NormFrame, type Segment } from '../geom/segments';
import { affineRectification, findVanishingPoints, mulberry32, refineVP, type Rectification } from '../geom/vanishing';
import { combOptions, fitComb, GRID_PARAMS, segmentPositions, toothSegments, type CombHypothesis } from './grid';
import { extractLines, LINES_PARAMS } from './lines';
import { param, Preprocessor, PREPROCESS_PARAMS, type CV, type ParamSpec, type PreprocessResult } from './preprocess';
import { EdgePolisher } from './polish';
import { CornerRefiner, refineOptions, REFINE_PARAMS, type RefineResult } from './refine';
import { boardCorners, verifyBoard, verifyOptions, VERIFY_PARAMS, type Lab3, type VerifyResult } from './verify';

type Mat = InstanceType<CV['Mat']>;

export const VANISHING_PARAMS: readonly ParamSpec[] = [
  { name: 'vpAngle', min: 1, max: 8, step: 0.5, default: 3 },
  { name: 'vpIterations', min: 50, max: 1500, step: 50, default: 400 },
  { name: 'vpMinSeparation', min: 5, max: 60, step: 1, default: 20 },
];

export const POLISH_PARAMS: readonly ParamSpec[] = [
  { name: 'polish', min: 0, max: 1, step: 1, default: 1 },
  { name: 'refit', min: 0, max: 1, step: 1, default: 1 },
];

/** All tunables of the detection pipeline; the main thread registers them as debug sliders. */
export const ALL_PARAMS: readonly ParamSpec[] = [
  ...PREPROCESS_PARAMS,
  ...LINES_PARAMS,
  ...VANISHING_PARAMS,
  ...GRID_PARAMS,
  ...VERIFY_PARAMS,
  ...POLISH_PARAMS,
  ...REFINE_PARAMS,
];

export interface CandidateInfo {
  hx: CombHypothesis;
  hy: CombHypothesis;
  verify: VerifyResult;
  /** Board cell units -> image pixels. */
  hb: Mat3;
}

export interface DetectDebug {
  width: number;
  height: number;
  segments: Segment[];
  /** Per segment: 0 = unassigned, 1 = family 1, 2 = family 2. */
  family: Uint8Array;
  /** Vanishing points in pixel-homogeneous coordinates. */
  vps: [Vec3, Vec3] | null;
  frame: NormFrame;
  rect: Rectification | null;
  hypsX: CombHypothesis[];
  hypsY: CombHypothesis[];
  candidates: CandidateInfo[];
  best: CandidateInfo | null;
  /** Board->image homography after the edge polish. */
  polished?: Mat3;
  refined?: { hb: Mat3; inliers: number; accepted: number; rms: number };
  /** Why no board was reported, when applicable. */
  reason?: string;
}

export interface DetectResult {
  corners: Corners | null;
  confidence: number;
  timings: Record<string, number>;
  debug?: DetectDebug;
}

export interface DetectOptions {
  debug?: boolean;
}

function isConvex(q: readonly Point[]): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i]!;
    const b = q[(i + 1) % 4]!;
    const c = q[(i + 2) % 4]!;
    const cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (cr === 0) return false;
    const s = Math.sign(cr);
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}

function quadArea(q: readonly Point[]): number {
  let a = 0;
  for (let i = 0; i < 4; i++) {
    const p = q[i]!;
    const r = q[(i + 1) % 4]!;
    a += p[0] * r[1] - r[0] * p[1];
  }
  return Math.abs(a) / 2;
}

export class Detector {
  private readonly pre: Preprocessor;
  private readonly polisher: EdgePolisher;
  private readonly refiner = new CornerRefiner();
  private rgba: Mat | null = null;
  /** Preprocessing outputs of the last detect() (valid until the next call). */
  lastPre: PreprocessResult | null = null;

  constructor(private readonly cv: CV) {
    this.pre = new Preprocessor(cv);
    this.polisher = new EdgePolisher(cv);
  }

  dispose(): void {
    this.pre.dispose();
    this.polisher.dispose();
    this.rgba?.delete();
    this.rgba = null;
  }

  /** Accepts an RGBA Mat (not owned) or ImageData. */
  detect(input: Mat | ImageData, params: Params, opts: DetectOptions = {}): DetectResult {
    const cv = this.cv;
    const timings: Record<string, number> = {};
    const t0 = performance.now();
    let t = t0;
    const lap = (name: string) => {
      const now = performance.now();
      timings[name] = now - t;
      t = now;
    };

    let rgba: Mat;
    if ('cols' in input) {
      rgba = input;
    } else {
      if (!this.rgba || this.rgba.cols !== input.width || this.rgba.rows !== input.height) {
        this.rgba?.delete();
        this.rgba = new cv.Mat(input.height, input.width, cv.CV_8UC4);
      }
      this.rgba.data.set(input.data);
      rgba = this.rgba;
    }
    const width = rgba.cols;
    const height = rgba.rows;

    const pre = this.pre.run(rgba, params);
    this.lastPre = pre;
    Object.assign(timings, pre.timings);
    t = performance.now();

    const debug: DetectDebug | undefined = opts.debug
      ? {
          width, height, segments: [], family: new Uint8Array(0), vps: null, frame: normFrame(width, height),
          rect: null, hypsX: [], hypsY: [], candidates: [], best: null,
        }
      : undefined;
    const finish = (corners: Corners | null, confidence: number, reason?: string): DetectResult => {
      timings.total = performance.now() - t0;
      if (debug && reason) debug.reason = reason;
      return { corners, confidence, timings, debug };
    };

    // 3. Lines.
    const segments = extractLines(cv, pre.edges, params, width, height);
    lap('lines');
    if (debug) debug.segments = segments;
    if (segments.length < 6) return finish(null, 0, 'few segments');

    // 4. Vanishing points and affine rectification.
    const frame = normFrame(width, height);
    if (debug) debug.frame = frame;
    const nsegs = segmentsToNormalised(segments, frame);
    const rng = mulberry32(12345);
    const pair = findVanishingPoints(nsegs, {
      iterations: param(params, VANISHING_PARAMS, 'vpIterations'),
      angleDeg: param(params, VANISHING_PARAMS, 'vpAngle'),
      minSeparationDeg: param(params, VANISHING_PARAMS, 'vpMinSeparation'),
      rng,
    });
    if (debug) debug.family = new Uint8Array(segments.length);
    if (!pair) {
      lap('vanishing');
      return finish(null, 0, 'no vanishing points');
    }
    const ctx: SolveContext = {
      nsegs, frame, width, height, params,
      lab: { data: pre.labEq.data, width, height },
    };
    // 5-6. Rectify, comb fit, verify.
    const pass1 = solveGrid(ctx, pair.vp1.vp, pair.vp2.vp, pair.vp1.inliers, pair.vp2.inliers, param(params, GRID_PARAMS, 'maxCandidates'));
    lap('grid');
    let solved = pass1;
    // Second pass: re-estimate both VPs from the segments that sit on the fitted grid lines only (no clutter,
    // no frame lines), then repeat. This sharpens the VPs, which matters for the far rows of oblique views.
    if (param(params, POLISH_PARAMS, 'refit') > 0 && pass1.best && pass1.best.verify.score > 0.15 && pass1.rect) {
      const m1 = toothSegments(nsegs, pair.vp1.inliers, pass1.rect.H, 0, pass1.best.hx, 0.2);
      const m2 = toothSegments(nsegs, pair.vp2.inliers, pass1.rect.H, 1, pass1.best.hy, 0.2);
      if (m1.length >= 4 && m2.length >= 4) {
        const v1 = refineVP(nsegs, m1);
        const v2 = refineVP(nsegs, m2);
        if (v1 && v2) {
          const pass2 = solveGrid(ctx, v1, v2, m1, m2, 8);
          if (pass2.best && pass2.best.verify.score >= pass1.best.verify.score - 0.05) solved = pass2;
        }
      }
    }
    lap('refit');
    if (debug) {
      debug.rect = solved.rect;
      debug.hypsX = solved.hypsX;
      debug.hypsY = solved.hypsY;
      debug.candidates = solved.candidates;
      debug.best = solved.best;
      pair.vp1.inliers.forEach((i) => (debug.family[i] = 1));
      pair.vp2.inliers.forEach((i) => (debug.family[i] = 2));
      const toPix = (v: Vec3): Vec3 => [v[0] * frame.s + frame.cx * v[2], v[1] * frame.s + frame.cy * v[2], v[2]];
      debug.vps = [toPix(pair.vp1.vp), toPix(pair.vp2.vp)];
    }
    const best = solved.best;
    if (!solved.rect) return finish(null, 0, 'degenerate rectification');
    if (solved.hypsX.length === 0 || solved.hypsY.length === 0) return finish(null, 0, 'no comb fit');
    if (!best) return finish(null, 0, 'no valid candidate');
    if (best.verify.score < param(params, VERIFY_PARAMS, 'verifyAccept')) return finish(null, best.verify.score, 'below threshold');

    // 7. Edge-alignment polish of the accepted candidate. (The corner-level refinement of the next milestone
    // would be inserted here, after verification.)
    let hb = best.hb;
    let score = best.verify.score;
    const doRefine = param(params, REFINE_PARAMS, 'refine') > 0;
    let rr: RefineResult | null = null;
    if (doRefine) {
      rr = this.refiner.refine(ctx.lab, hb, refineOptions(params, best.verify.channel));
      lap('refine');
    }
    if (param(params, POLISH_PARAMS, 'polish') > 0) {
      const joint = rr?.ok && param(params, REFINE_PARAMS, 'refineJoint') > 0;
      hb = this.polisher.polish(pre.edges, hb, {
        maxShiftFrac: 0.05,
        points: joint ? rr!.points : undefined,
        pointWeight: param(params, REFINE_PARAMS, 'refineJoint'),
      }).hb;
      if (debug) debug.polished = hb;
      lap('polish');
    }
    if (rr?.ok && param(params, REFINE_PARAMS, 'refineReplace') > 0 && isValidQuad(boardCorners(rr.hb), width, height)) {
      const v = verifyBoard(ctx.lab, rr.hb, verifyOptions(params));
      if (v.score >= score - param(params, REFINE_PARAMS, 'refineKeepTol')) {
        hb = rr.hb;
        score = Math.max(v.score, score * 0.999);
      }
      lap('refineVerify');
    }
    if (debug && rr?.ok) debug.refined = { hb: rr.hb, inliers: rr.inliers, accepted: rr.accepted, rms: rr.rms };
    const corners = orientedCorners(boardCorners(hb));
    return finish(corners, score);
  }
}

interface SolveContext {
  nsegs: Segment[];
  frame: NormFrame;
  width: number;
  height: number;
  params: Params;
  lab: Lab3;
}

interface Solved {
  rect: Rectification | null;
  hypsX: CombHypothesis[];
  hypsY: CombHypothesis[];
  candidates: CandidateInfo[];
  best: CandidateInfo | null;
}

/** Rectification from two VPs, per-axis comb fits and checker verification of the best combinations. */
function solveGrid(ctx: SolveContext, vp1: Vec3, vp2: Vec3, idx1: readonly number[], idx2: readonly number[], maxCand: number): Solved {
  const { nsegs, frame, width, height, params } = ctx;
  const out: Solved = { rect: null, hypsX: [], hypsY: [], candidates: [], best: null };
  // Reference point: centroid of the family segments (on the board side of the horizon).
  let rx = 0;
  let ry = 0;
  let rw = 0;
  for (const i of [...idx1, ...idx2]) {
    const s = nsegs[i]!;
    rx += s.mx * s.weight;
    ry += s.my * s.weight;
    rw += s.weight;
  }
  const rect = affineRectification(vp1, vp2, [rx / rw, ry / rw]);
  out.rect = rect;
  if (!rect) return out;
  const refLen = (0.15 * Math.hypot(width, height)) / frame.s;
  const copt = combOptions(params);
  out.hypsX = fitComb(segmentPositions(nsegs, idx1, rect.H, 0, refLen), copt);
  out.hypsY = fitComb(segmentPositions(nsegs, idx2, rect.H, 1, refLen), copt);
  if (out.hypsX.length === 0 || out.hypsY.length === 0) return out;

  const combos: Array<[CombHypothesis, CombHypothesis]> = [];
  for (const hx of out.hypsX) for (const hy of out.hypsY) combos.push([hx, hy]);
  combos.sort((a, b) => b[0].score + b[1].score - (a[0].score + a[1].score));
  const vopt = verifyOptions(params);
  const toPix: Mat3 = [frame.s, 0, frame.cx, 0, frame.s, frame.cy, 0, 0, 1];
  const rectInvPix = mul3(toPix, rect.Hinv);
  for (const [hx, hy] of combos.slice(0, maxCand)) {
    const a: Mat3 = [hx.period, 0, hx.start, 0, hy.period, hy.start, 0, 0, 1];
    const hb = mul3(rectInvPix, a);
    if (!isValidQuad(boardCorners(hb), width, height)) continue;
    const info: CandidateInfo = { hx, hy, verify: verifyBoard(ctx.lab, hb, vopt), hb };
    out.candidates.push(info);
    if (!out.best || info.verify.score > out.best.verify.score) out.best = info;
  }
  return out;
}

/** Corner order with a positive shoelace area (consistent winding). */
function orientedCorners(q: [Point, Point, Point, Point]): Corners {
  let a = 0;
  for (let i = 0; i < 4; i++) a += q[i]![0] * q[(i + 1) % 4]![1] - q[(i + 1) % 4]![0] * q[i]![1];
  return a < 0 ? [q[0], q[3], q[2], q[1]] : q;
}

function isValidQuad(q: readonly Point[], width: number, height: number): boolean {
  const mx = 0.03 * width;
  const my = 0.03 * height;
  for (const p of q) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return false;
    if (p[0] < -mx || p[0] > width + mx || p[1] < -my || p[1] > height + my) return false;
  }
  if (!isConvex(q)) return false;
  return quadArea(q) >= 0.03 * width * height;
}

const detectors = new WeakMap<object, Detector>();

/** Convenience wrapper around a per-`cv` cached Detector. */
export function detect(cv: CV, input: Mat | ImageData, params: Params = {}, opts: DetectOptions = {}): DetectResult {
  let d = detectors.get(cv);
  if (!d) {
    d = new Detector(cv);
    detectors.set(cv, d);
  }
  return d.detect(input, params, opts);
}

