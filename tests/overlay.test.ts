import { describe, expect, it } from 'vitest';
import { coverMap, DOT_BLACK, DOT_EMPTY, DOT_LOW_CONF, DOT_WHITE, dotAlpha, frameToScreen, occupancyDots, quadAlpha } from '../src/overlay';
import { formatOccupancy } from '../src/debug/panel';
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

  it('encodes committed-class probability as opacity and flags low-confidence cells', () => {
    const corners: Point[] = [[100, 100], [260, 100], [260, 260], [100, 260]];
    const prob = new Float32Array(64).fill(0.99);
    prob[0] = 0.5; // white piece, low
    prob[1] = 0; // empty, very low
    prob[2] = DOT_LOW_CONF; // exactly at the threshold: not low
    const dots = occupancyDots(corners, grid, prob);
    expect(dots[0]!.fill).toBe(DOT_WHITE); // class colour kept
    expect(dots[0]!.conf).toBeCloseTo(0.5);
    expect(dots[0]!.alpha).toBeCloseTo(0.35 + 0.65 * 0.5);
    expect(dots[0]!.low).toBe(true);
    expect(dots[1]!.alpha).toBeCloseTo(0.35);
    expect(dots[1]!.low).toBe(true);
    expect(dots[2]!.low).toBe(false);
    expect(dots[3]!.alpha).toBeCloseTo(0.35 + 0.65 * 0.99);
    expect(dots[3]!.low).toBe(false);
    // Size stays class-based.
    const plain = occupancyDots(corners, grid);
    expect(dots.map((d) => d.r)).toEqual(plain.map((d) => d.r));
  });

  it('draws missing / malformed probabilities at full confidence', () => {
    const corners: Point[] = [[100, 100], [260, 100], [260, 260], [100, 260]];
    for (const prob of [null, new Float32Array(10)]) {
      for (const d of occupancyDots(corners, grid, prob)) {
        expect(d.alpha).toBe(1);
        expect(d.conf).toBe(1);
        expect(d.low).toBe(false);
      }
    }
    const nan = new Float32Array(64).fill(NaN);
    expect(occupancyDots(corners, grid, nan)[0]!.alpha).toBe(1);
    expect(dotAlpha(undefined)).toBe(1);
    expect(dotAlpha(2)).toBe(1);
    expect(dotAlpha(-1)).toBeCloseTo(0.35);
  });

  it('returns nothing for a wrong-sized grid', () => {
    expect(occupancyDots([[0, 0], [8, 0], [8, 8], [0, 8]], new Uint8Array(10))).toEqual([]);
  });
});

describe('formatOccupancy confidence line', () => {
  const stats = { state: 'calibrated', empty: 32, white: 16, black: 16, lowCells: 0, dropped: false, frozen: false } as const;

  it('summarises the minimum probability and the low-confidence count', () => {
    const prob = new Float32Array(64).fill(0.95);
    prob[5] = 0.42;
    prob[9] = 0.55;
    expect(formatOccupancy(stats, 0, prob)).toContain('occ conf min 0.42, <0.6: 2 cells');
  });

  it('omits the line without probabilities', () => {
    expect(formatOccupancy(stats, 0)).not.toContain('conf');
  });
});
