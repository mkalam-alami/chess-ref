// Tracking-mode part of `npm run bench` (BENCH_ONLY=track runs it alone): the worker's per-frame pipeline
// (frame copy into the Mat, TrackingSession, occupancy) on perturbed sequences, with the mean time per stage over the
// tracked frames. Also reports the corner error and a checksum of the occupancy outputs, so a speed-up can be checked
// for identical results (same BENCH_SEED).
import type { Detector } from '../../src/vision/detector';
import { OccupancyTracker } from '../../src/vision/occupancy';
import type { CV } from '../../src/vision/preprocess';
import type { Params } from '../../src/worker/protocol';
import { TrackingSession } from '../../src/worker/tracker';
import { cornerError, realCases, type Quad } from './bench';
import { makeBoardSample } from './generate';
import { makeSequence, type SeqSpec } from './sequence';

interface Clip {
  name: string;
  width: number;
  height: number;
  frames: { rgba: Uint8ClampedArray; gt: Quad }[];
}

function clips(cv: CV, nFrames: number, seed: number): Clip[] {
  const spec = (s: number): SeqSpec => ({ frames: nFrames, jitter: 1, drift: 0.5, rotate: 0.05, noise: 1.5, seed: s });
  const out: Clip[] = [];
  for (const c of realCases(cv, 640).filter((x) => x.gt))
    out.push({ name: c.name, width: c.width, height: c.height, frames: makeSequence(cv, c.rgba, c.width, c.height, c.gt!, spec(seed)) });
  let k = 0;
  for (const elev of ['overhead', 'oblique'] as const)
    for (const palette of ['wood', 'printed'] as const) {
      const s = makeBoardSample(cv, seed + 7000 + k++, { elev, palette, pieces: 'start' });
      out.push({ name: s.label, width: s.width, height: s.height, frames: makeSequence(cv, s.rgba, s.width, s.height, s.corners!, spec(seed + k)) });
    }
  return out;
}

export function trackingBench(cv: CV, det: Detector, nFrames: number, seed: number, params: Params = {}): string {
  const all = clips(cv, nFrames, seed);
  const sum: Record<string, number> = {};
  let tracked = 0;
  let full = 0;
  let errSum = 0;
  let errN = 0;
  let lost = 0;
  let occHash = 0;
  let occFrames = 0;
  let stableFrames = 0;
  const run = (clip: Clip, record: boolean) => {
    const session = new TrackingSession(det);
    const occ = new OccupancyTracker();
    const mat = new cv.Mat(clip.height, clip.width, cv.CV_8UC4);
    try {
      clip.frames.forEach((f, i) => {
        const now = i * 66;
        const t0 = performance.now();
        mat.data.set(f.rgba);
        const tCopy = performance.now() - t0;
        const r = session.process(mat, clip.width, clip.height, params, {}, now);
        const t1 = performance.now();
        const frame = { data: f.rgba, width: clip.width, height: clip.height };
        const o = r.hb ? occ.update(frame, r.hb, params, now) : null;
        if (!r.hb) occ.noBoard(now);
        const tOcc = performance.now() - t1;
        if (!record) return;
        if (r.mode !== 'tracking') {
          full++;
          if (!r.corners) lost++;
          return;
        }
        tracked++;
        const add = (k: string, v: number) => (sum[k] = (sum[k] ?? 0) + v);
        add('copy', tCopy);
        for (const [k, v] of Object.entries(r.timings)) add(k === 'total' ? 'track' : k, v);
        add('occupancy', tOcc);
        for (const [k, v] of Object.entries(occ.timings)) add(k, v);
        add('frame', tCopy + (r.timings.total ?? 0) + tOcc);
        if (r.corners) {
          errSum += cornerError(r.corners, f.gt);
          errN++;
        }
        if (o) {
          occFrames++;
          if (o.observation.stable) stableFrames++;
          for (let c = 0; c < 64; c++) occHash = (occHash * 31 + o.raw[c]! * 3 + (o.grid ? o.grid[c]! : 7) + Math.round(o.conf[c]! * 1000)) % 1000000007;
        }
      });
    } finally {
      mat.delete();
    }
  };
  // Warm-up (JIT, wasm) on the first clip, then the measured pass.
  if (all[0]) run(all[0], false);
  for (const c of all) run(c, true);
  const order = ['copy', 'clahe', 'trackGradient', 'trackCanny', 'polish', 'verify', 'track', 'occFootprints', 'occSample', 'occCalib', 'occFeatures', 'occClassify', 'occupancy', 'frame'];
  const keys = [...order.filter((k) => k in sum), ...Object.keys(sum).filter((k) => !order.includes(k))];
  const n = Math.max(1, tracked);
  return (
    `tracking: ${all.length} clips x ${nFrames} frames, tracked ${tracked}, full ${full} (lost ${lost})\n` +
    `  mean ms per tracked frame: ${keys.map((k) => `${k} ${(sum[k]! / n).toFixed(2)}`).join(', ')}\n` +
    `  tracked corner error ${((errSum / Math.max(1, errN)) * 100).toFixed(3)}% | occupancy frames ${occFrames}, stable ${stableFrames}, checksum ${occHash}`
  );
}
