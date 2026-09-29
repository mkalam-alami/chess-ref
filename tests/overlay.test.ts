import { describe, expect, it } from 'vitest';
import { coverMap, DOT_BLACK, DOT_EMPTY, DOT_WHITE, frameToScreen, occupancyDots, quadAlpha } from '../src/overlay';
import { homographyFrom4, applyH, type Point } from '../src/geom/homography';
import { OCC_BLACK, OCC_EMPTY, OCC_WHITE } from '../src/worker/protocol';

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

describe('occupancyDots', () => {
  const grid = new Uint8Array(64).fill(OCC_EMPTY);
  grid[0] = OCC_WHITE; // cell (0, 0)
  grid[7 * 8 + 2] = OCC_BLACK; // cell (2, 7)

  it('places one dot per cell centre of an axis-aligned board, with the colour of its state', () => {
    const dots = occupancyDots([[100, 100], [260, 100], [260, 260], [100, 260]], grid);
    expect(dots).toHaveLength(64);
    expect(dots[0]!.x).toBeCloseTo(110);
    expect(dots[0]!.y).toBeCloseTo(110);
    expect(dots[0]!.fill).toBe(DOT_WHITE);
    expect(dots[0]!.outline).toBe(true);
    const b = dots[7 * 8 + 2]!;
    expect(b.x).toBeCloseTo(150);
    expect(b.y).toBeCloseTo(250);
    expect(b.fill).toBe(DOT_BLACK);
    const e = dots[1]!;
    expect(e.x).toBeCloseTo(130);
    expect(e.fill).toBe(DOT_EMPTY);
    expect(e.outline).toBe(false);
    // Piece dots are larger than empty ones and scale with the 20 px cells.
    expect(dots[0]!.r).toBeGreaterThan(e.r);
    expect(dots[0]!.r).toBeCloseTo(4.8);
  });

  it('follows the corner order: cell (0,0) sits next to corners[0]', () => {
    // Same square, labelled from another corner: board (0,0) is now the bottom-right image corner.
    const dots = occupancyDots([[180, 180], [100, 180], [100, 100], [180, 100]], grid);
    expect(dots[0]!.x).toBeCloseTo(175);
    expect(dots[0]!.y).toBeCloseTo(175);
    expect(dots[0]!.fill).toBe(DOT_WHITE);
  });

  it('projects cell centres through perspective and shrinks far dots', () => {
    const corners: Point[] = [[200, 100], [440, 100], [600, 400], [40, 400]];
    const h = homographyFrom4([[0, 0], [8, 0], [8, 8], [0, 8]], corners)!;
    const dots = occupancyDots(corners, grid);
    const c = applyH(h, [3.5, 4.5]);
    expect(dots[4 * 8 + 3]!.x).toBeCloseTo(c[0]);
    expect(dots[4 * 8 + 3]!.y).toBeCloseTo(c[1]);
    // Row 0 is the far (short) edge: smaller cells, smaller dots than row 7.
    expect(dots[3]!.r).toBeLessThan(dots[7 * 8 + 3]!.r);
  });

  it('returns nothing for a wrong-sized grid', () => {
    expect(occupancyDots([[0, 0], [8, 0], [8, 8], [0, 8]], new Uint8Array(10))).toEqual([]);
  });
});
