import { beforeAll, describe, expect, it } from 'vitest';
import { Detector } from '../src/vision/detector';
import type { CV } from '../src/vision/preprocess';
import { inferProfile } from '../src/vision/profile';
import { cornerError, negativeCases, realCases, recallWhere, runCases, synthCases, type Outcome } from './synth/bench';
import { loadCv } from './synth/cvNode';

let cv: CV;
let det: Detector;
beforeAll(async () => {
  ({ cv } = await loadCv());
  det = new Detector(cv);
}, 60_000);

describe('detector on synthetic boards', () => {
  it('meets a floor recall on a small subset and rejects negatives', () => {
    const cases = [
      ...synthCases(cv, 2, 4242, (s) => s.pieces !== 'outer'),
      ...negativeCases(cv).slice(0, 4),
    ];
    const outs: Outcome[] = runCases(det, cases);
    const clear = recallWhere(outs, (l) => l.endsWith('none'));
    const all = recallWhere(outs, (l) => l.startsWith('overhead') || l.startsWith('oblique'));
    expect(clear.recall).toBeGreaterThanOrEqual(0.7);
    expect(all.recall).toBeGreaterThanOrEqual(0.6);
    expect(outs.filter((o) => o.falsePositive)).toHaveLength(0);
  }, 60_000);
});

describe('detector on real photos', () => {
  it('finds the board in at least 3 of 5 photos and nothing in the negative', () => {
    const outs = runCases(det, realCases(cv, 640));
    const pos = outs.filter((o) => o.c.gt);
    expect(pos.filter((o) => o.ok).length).toBeGreaterThanOrEqual(3);
    expect(outs.filter((o) => !o.c.gt && o.res.corners)).toHaveLength(0);
  }, 60_000);
});

describe('board profile lock', () => {
  it('infers L-only from every real photo, stays accurate when locked and still rejects the negative', () => {
    const cases = realCases(cv, 640);
    const unlocked = runCases(det, cases);
    const img = (c: (typeof cases)[number]) => ({ data: c.rgba, width: c.width, height: c.height }) as unknown as ImageData;
    const profiles = unlocked.map((o) => (o.res.corners ? inferProfile(o.res) : null));
    for (const [i, o] of unlocked.entries()) {
      if (!o.c.gt) continue;
      expect(profiles[i]?.channels).toEqual([0]);
      const res = det.detect(img(o.c), {}, { profile: profiles[i] });
      expect(res.corners).not.toBeNull();
      expect(cornerError(res.corners!, o.c.gt)).toBeLessThan(0.02);
    }
    const profile = profiles.find((p) => p)!;
    const neg = cases.find((c) => !c.gt)!;
    expect(det.detect(img(neg), {}, { profile }).corners).toBeNull();
  }, 60_000);
});
