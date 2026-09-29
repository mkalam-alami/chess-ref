import { describe, expect, it } from 'vitest';
import type { DetectResult } from '../src/vision/detector';
import { inferProfile, ProfileLock } from '../src/vision/profile';

const det = (channel = 0, contrast = 100, ok = true): DetectResult => ({
  corners: ok ? [[0, 0], [1, 0], [1, 1], [0, 1]] : null,
  confidence: ok ? 1 : 0,
  timings: {},
  verify: ok ? { channel, contrast, ringFactor: 1, ringLevel: 0.3 } : undefined,
});

describe('ProfileLock', () => {
  it('locks after 3 consecutive agreeing detections', () => {
    const l = new ProfileLock();
    expect(l.update(det(), 0)).toBeNull();
    expect(l.update(det(), 100)).toBeNull();
    const p = l.update(det(), 200);
    expect(p).toEqual({ channels: [0], verifyChannel: 0, contrast: 100, surround: 'dark' });
    expect(l.profile).toBe(p);
  });
  it('restarts the streak on a miss or a channel disagreement', () => {
    const l = new ProfileLock();
    l.update(det(), 0);
    l.update(det(2, 10), 100);
    l.update(det(), 200);
    l.update(det(0, 100, false), 300);
    l.update(det(), 400);
    expect(l.update(det(), 500)).toBeNull();
    expect(l.update(det(), 600)).not.toBeNull();
  });
  it('unlocks after profileUnlockMs without detections and on reset', () => {
    const l = new ProfileLock();
    for (let i = 0; i < 3; i++) l.update(det(), i * 100);
    expect(l.update(null, 3000)).not.toBeNull();
    expect(l.update(null, 8000)).toBeNull();
    for (let i = 0; i < 3; i++) l.update(det(), 9000 + i);
    l.reset();
    expect(l.profile).toBeNull();
  });
  it('infers chroma channels alongside L', () => {
    expect(inferProfile(det(2, -8))!.channels).toEqual([0, 2]);
    expect(inferProfile(det(0, 0))).toBeNull();
  });
});
