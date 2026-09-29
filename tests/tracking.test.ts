import { beforeAll, describe, expect, it } from 'vitest';
import { inferProfile } from '../src/vision/profile';
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

  it('L-only profile tracking stays locked (<2% error) and is cheaper', () => {
    const cases = realCases(cv, 640).filter((c) => c.gt);
    const spec: SeqSpec = { frames: 12, jitter: 1, drift: 1, rotate: 0.1, noise: 1, seed: 5 };
    const plain: number[] = [];
    const lonly: number[] = [];
    for (const c of cases) {
      const first = det.detect(img(c), {});
      if (!first.corners) continue;
      const profile = inferProfile(first);
      if (profile?.channels.length !== 1 || profile.channels[0] !== 0) continue;
      const frames = makeSequence(cv, c.rgba, c.width, c.height, c.gt!, spec);
      plain.push(...runSequence(det, frames, c.width, c.height, { trackFullEvery: 0 }).trackMs);
      const o = runSequence(det, frames, c.width, c.height, { trackFullEvery: 0 }, 66, { profile });
      o.errs.forEach((e) => {
        expect(e).not.toBeNull();
        expect(e!).toBeLessThan(0.02);
      });
      lonly.push(...o.trackMs);
    }
    expect(lonly.length).toBeGreaterThan(0);
    console.log(`tracking no-profile ${mean(plain).toFixed(1)} ms vs L-only profile ${mean(lonly).toFixed(1)} ms`);
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
    const n0 = negativeCases(cv)[3]!;
    // Same frame size as the real photo, so the session does not reset for a size change.
    const src = new cv.Mat(n0.height, n0.width, cv.CV_8UC4);
    const dst = new cv.Mat();
    src.data.set(n0.rgba);
    cv.resize(src, dst, new cv.Size(real.width, real.height));
    const neg = { rgba: new Uint8ClampedArray(dst.data), width: real.width, height: real.height };
    src.delete();
    dst.delete();
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
