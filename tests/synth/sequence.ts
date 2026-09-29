import { applyH, homographyFrom4, type Point } from '../../src/geom/homography';
import { mulberry32 } from '../../src/geom/vanishing';
import type { CV } from '../../src/vision/preprocess';
import type { Params } from '../../src/worker/protocol';
import { TrackingSession, type SessionResult } from '../../src/worker/tracker';
import type { DetectOptions, Detector } from '../../src/vision/detector';
import { cornerError, type Quad } from './bench';

export interface SeqSpec {
  frames: number;
  /** Per-corner random jitter amplitude (px, uniform +-). */
  jitter: number;
  /** Drift velocity (px per frame) along a fixed random direction. */
  drift: number;
  /** Rotation about the image centre (degrees per frame). */
  rotate: number;
  /** Extra Gaussian-ish pixel noise sigma added per frame. */
  noise: number;
  seed: number;
}

export interface SeqFrame {
  rgba: Uint8ClampedArray;
  gt: Quad;
}

/** Warps one rendered frame by a series of small homographies (the board moves, the scene moves with it). */
export function makeSequence(cv: CV, base: Uint8ClampedArray, W: number, H: number, gt0: Quad, spec: SeqSpec): SeqFrame[] {
  const rng = mulberry32(spec.seed);
  const dirA = rng() * Math.PI * 2;
  const src = new cv.Mat(H, W, cv.CV_8UC4);
  const dst = new cv.Mat();
  src.data.set(base);
  const frames: SeqFrame[] = [];
  const rect: Point[] = [[0, 0], [W, 0], [W, H], [0, H]];
  try {
    for (let f = 0; f < spec.frames; f++) {
      const th = (spec.rotate * f * Math.PI) / 180;
      const c = Math.cos(th);
      const s = Math.sin(th);
      const dx = Math.cos(dirA) * spec.drift * f;
      const dy = Math.sin(dirA) * spec.drift * f;
      const moved = rect.map(([x, y]): Point => {
        const ux = x - W / 2;
        const uy = y - H / 2;
        return [
          W / 2 + c * ux - s * uy + dx + (f === 0 ? 0 : (rng() * 2 - 1) * spec.jitter),
          H / 2 + s * ux + c * uy + dy + (f === 0 ? 0 : (rng() * 2 - 1) * spec.jitter),
        ];
      });
      const M = homographyFrom4(rect, moved)!;
      const m = cv.matFromArray(3, 3, cv.CV_64F, M);
      try {
        cv.warpPerspective(src, dst, m, new cv.Size(W, H), cv.INTER_LINEAR, cv.BORDER_REPLICATE);
      } finally {
        m.delete();
      }
      const out = new Uint8ClampedArray(dst.data);
      if (spec.noise > 0) {
        for (let i = 0; i < out.length; i += 4) {
          const n = (rng() + rng() + rng() - 1.5) * 2 * spec.noise;
          out[i] = out[i]! + n;
          out[i + 1] = out[i + 1]! + n;
          out[i + 2] = out[i + 2]! + n;
        }
      }
      frames.push({ rgba: out, gt: gt0.map((p) => applyH(M, p as Point)) });
    }
  } finally {
    src.delete();
    dst.delete();
  }
  return frames;
}

export interface SeqOutcome {
  results: SessionResult[];
  errs: (number | null)[];
  trackMs: number[];
  fullMs: number[];
}

const asImage = (rgba: Uint8ClampedArray, W: number, H: number) => ({ data: rgba, width: W, height: H }) as unknown as ImageData;

/** Runs frames through the tracking state machine (frame i is "time" i * dtMs). */
export function runSequence(det: Detector, frames: SeqFrame[], W: number, H: number, params: Params = {}, dtMs = 66, opts: DetectOptions = {}): SeqOutcome {
  const session = new TrackingSession(det);
  const results: SessionResult[] = [];
  const errs: (number | null)[] = [];
  const trackMs: number[] = [];
  const fullMs: number[] = [];
  frames.forEach((f, i) => {
    const t = performance.now();
    const r = session.process(asImage(f.rgba, W, H), W, H, params, opts, i * dtMs);
    const ms = performance.now() - t;
    results.push(r);
    errs.push(r.corners ? cornerError(r.corners, f.gt) : null);
    (r.mode === 'tracking' ? trackMs : fullMs).push(ms);
  });
  return { results, errs, trackMs, fullMs };
}
