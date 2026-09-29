import { beforeAll, describe, expect, it } from 'vitest';
import { Detector } from '../src/vision/detector';
import type { CV } from '../src/vision/preprocess';
import { TrackingSession } from '../src/worker/tracker';
import { negativeCases, realCases } from './synth/bench';
import { loadCv } from './synth/cvNode';
import { makeSequence, runSequence, type SeqSpec } from './synth/sequence';
import { isSimpleQuad, repairQuad } from '../src/geom/cornerOrder';

let cv: CV;
let det: Detector;
beforeAll(async () => {
  ({ cv } = await loadCv());
  det = new Detector(cv);
}, 60_000);

const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const img = (c: { rgba: Uint8ClampedArray; width: number; height: number }) => ({ data: c.rgba, width: c.width, height: c.height }) as unknown as ImageData;

describe('tracking on perturbed sequences', () => {
  it('stays locked (<2% error), is mostly tracked and at least 3x faster than full detection', () => {
    const spec: SeqSpec = { frames: 16, jitter: 2, drift: 1.5, rotate: 0.15, noise: 2, seed: 5 };
    const cases = realCases(cv, 640).filter((c) => c.gt).slice(0, 3);
    const trackMs: number[] = [];
    const fullMs: number[] = [];
    for (const c of cases) {
      const frames = makeSequence(cv, c.rgba, c.width, c.height, c.gt!, spec);
      const o = runSequence(det, frames, c.width, c.height, { trackFullEvery: 0 });
      o.errs.forEach((e) => {
        expect(e).not.toBeNull();
        expect(e!).toBeLessThan(0.02);
      });
      expect(o.trackMs.length).toBeGreaterThanOrEqual(spec.frames - 3);
      trackMs.push(...o.trackMs);
      fullMs.push(...o.fullMs);
    }
    console.log(`tracking ${mean(trackMs).toFixed(1)} ms vs full ${mean(fullMs).toFixed(1)} ms`);
    expect(mean(fullMs) / mean(trackMs)).toBeGreaterThanOrEqual(3);
  }, 120_000);

  it('falls back to full detection periodically', () => {
    const c = realCases(cv, 640).filter((x) => x.gt)[0]!;
    const frames = makeSequence(cv, c.rgba, c.width, c.height, c.gt!, { frames: 8, jitter: 1, drift: 1, rotate: 0.1, noise: 1, seed: 3 });
    const o = runSequence(det, frames, c.width, c.height, { trackFullEvery: 4 });
    expect(o.results.map((r) => r.mode).join(',')).toBe('full,tracking,tracking,tracking,tracking,full,tracking,tracking');
  }, 60_000);
});

describe('tracking rejection', () => {
  it('rejects a frame without a board and falls back to full detection', () => {
    const real = realCases(cv, 640).filter((c) => c.gt)[0]!;
    const first = det.detect(img(real), {});
    expect(first.hb).toBeDefined();
    const neg = negativeCases(cv)[3]!;
    const tr = det.track(img(neg), {}, first.hb!);
    expect(tr.corners).toBeNull();
    const session = new TrackingSession(det);
    session.process(img(real), real.width, real.height, {});
    const r = session.process(img(neg), neg.width, neg.height, {}, {}, 10);
    expect(r.mode).toBe('full');
    expect(r.fullReason).toBe('track-failed');
    expect(r.corners).toBeNull();
    expect(session.locked).toBe(false);
  }, 60_000);
});

describe('quad repair', () => {
  it('reorders a bow-tie', () => {
    const bow: [[number, number], [number, number], [number, number], [number, number]] = [[0, 0], [10, 10], [10, 0], [0, 10]];
    expect(isSimpleQuad(bow)).toBe(false);
    const fixed = repairQuad(bow);
    expect(fixed && isSimpleQuad(fixed)).toBe(true);
  });
});
