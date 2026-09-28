import { describe, expect, it } from 'vitest';
import { coverMap, frameToScreen, quadAlpha } from '../src/overlay';

describe('coverMap', () => {
  it('maps identically when aspect ratios match', () => {
    const m = coverMap(640, 360, 1280, 720);
    expect(m).toEqual({ scale: 2, offsetX: 0, offsetY: 0 });
    expect(frameToScreen([320, 180], m)).toEqual([640, 360]);
  });

  it('crops vertically when the viewport is wider than the frame', () => {
    const m = coverMap(400, 400, 800, 400);
    expect(m.scale).toBe(2);
    expect(m.offsetX).toBe(0);
    expect(m.offsetY).toBe(-200);
    expect(frameToScreen([0, 100], m)).toEqual([0, 0]);
    expect(frameToScreen([400, 300], m)).toEqual([800, 400]);
  });

  it('crops horizontally when the viewport is taller than the frame', () => {
    const m = coverMap(400, 200, 200, 200);
    expect(m.scale).toBe(1);
    expect(m.offsetX).toBe(-100);
    expect(frameToScreen([200, 100], m)).toEqual([100, 100]);
  });
});

describe('quadAlpha', () => {
  it('holds, fades, then vanishes', () => {
    expect(quadAlpha(0)).toBe(1);
    expect(quadAlpha(500)).toBe(1);
    expect(quadAlpha(650)).toBeCloseTo(0.5);
    expect(quadAlpha(800)).toBe(0);
    expect(quadAlpha(5000)).toBe(0);
  });
});
