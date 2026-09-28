import { beforeAll, describe, expect, it } from 'vitest';
import { Detector } from '../src/vision/detector';
import type { CV } from '../src/vision/preprocess';
import { negativeCases, realCases, recallWhere, runCases, synthCases, type Outcome } from './synth/bench';
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
