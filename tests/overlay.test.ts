import { describe, expect, it } from 'vitest';
import {
  coverMap,
  DOT_BLACK,
  DOT_EMPTY,
  DOT_LOW_CONF,
  DOT_WHITE,
  dotAlpha,
  frameToScreen,
  gameMarks,
  occupancyDots,
  PIECE_BASE,
  PIECE_SIZE,
  quadAlpha,
} from '../src/overlay';
import { formatGame, formatOccupancy } from '../src/debug/panel';
import type { PieceCode } from '../src/game/types';
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

describe('gameMarks', () => {
  const square: Point[] = [[100, 100], [260, 100], [260, 260], [100, 260]]; // 20 px cells
  const identity = Uint8Array.from({ length: 64 }, (_, k) => k); // a1 on cell (0,0), h1 on (7,0), a8 on (0,7)
  const pieces: Array<PieceCode | null> = new Array(64).fill(null);
  pieces[0] = 'wR'; // a1
  pieces[4] = 'wK'; // e1
  pieces[60] = 'bK'; // e8

  it('places one screen-upright icon per piece, sized by the cell and anchored toward its base', () => {
    const m = gameMarks(square, identity, pieces);
    expect(m.pieces.map((p) => p.code).sort()).toEqual(['bK', 'wK', 'wR']);
    const a1 = m.pieces.find((p) => p.sq === 0)!;
    expect(a1.size).toBeCloseTo(20 * PIECE_SIZE);
    expect(a1.x + a1.size / 2).toBeCloseTo(110); // centred horizontally on the cell
    expect(a1.y + a1.size).toBeCloseTo(110 + 20 * PIECE_BASE); // bottom edge just below the cell centre
    expect(m.tint).toEqual([]);
  });

  it('maps squares through the orientation (square -> board cell)', () => {
    // Board seen from the other side: a1 is on cell (7,7), e1 on (3,7), e8 on (3,0).
    const flipped = Uint8Array.from({ length: 64 }, (_, sq) => 63 - sq);
    const m = gameMarks(square, flipped, pieces);
    const e8 = m.pieces.find((p) => p.sq === 60)!;
    expect(e8.x + e8.size / 2).toBeCloseTo(100 + 3.5 * 20);
    const a1 = m.pieces.find((p) => p.sq === 0)!;
    expect(a1.x + a1.size / 2).toBeCloseTo(250);
    expect(a1.y + a1.size).toBeCloseTo(250 + 20 * PIECE_BASE);
  });

  it('draws far pieces first and shrinks them with perspective', () => {
    const corners: Point[] = [[200, 100], [440, 100], [600, 400], [40, 400]]; // row 0 is the far edge
    const m = gameMarks(corners, identity, pieces);
    const ys = m.pieces.map((p) => p.y);
    expect(ys).toEqual([...ys].sort((a, b) => a - b));
    const e1 = m.pieces.find((p) => p.sq === 4)!;
    const e8 = m.pieces.find((p) => p.sq === 60)!;
    expect(e1.size).toBeLessThan(e8.size);
    expect(m.pieces[m.pieces.length - 1]!.sq).toBe(60);
  });

  it('tints the last move cells', () => {
    const m = gameMarks(square, identity, pieces, { from: 12, to: 28 }); // e2 -> e4
    expect(m.tint).toHaveLength(2);
    const [x0, y0] = m.tint[0]![0]!;
    expect(x0).toBeCloseTo(180);
    expect(y0).toBeCloseTo(120);
    expect(m.tint[1]![2]).toEqual([expect.closeTo(200), expect.closeTo(180)]); // e4 = cell (4,3), far corner (5,4)
  });

  it('returns nothing for malformed input or a degenerate quad', () => {
    expect(gameMarks(square, new Uint8Array(10), pieces)).toEqual({ pieces: [], tint: [] });
    expect(gameMarks(square, identity, pieces.slice(0, 10))).toEqual({ pieces: [], tint: [] });
    expect(gameMarks([[0, 0], [0, 0], [0, 0], [0, 0]], identity, pieces, { from: 0, to: 1 })).toEqual({ pieces: [], tint: [] });
    const bad = identity.slice();
    bad[0] = 200; // out-of-range cell: that square is skipped
    expect(gameMarks(square, bad, pieces).pieces.map((p) => p.sq).sort()).toEqual([4, 60]);
  });
});

describe('formatGame', () => {
  it('shows the state and the top-3 hypotheses', () => {
    const text = formatGame({
      state: 'playing',
      turn: 'b',
      plies: [{ san: 'e4', from: 12, to: 28, tentative: true }],
      top: [{ line: 'e4', score: -1.25 }, { line: '', score: -9 }, { line: 'd4', score: -12 }, { line: 'c4', score: -30 }],
    });
    expect(text).toContain('game    playing b ply 1');
    expect(text).toContain('top e4');
    expect(text).toContain('-1.3');
    expect(text).toContain('(anchor)');
    expect(text).toContain('d4');
    expect(text).not.toContain('c4');
  });

  it('is short while waiting', () => {
    expect(formatGame({ state: 'waiting', turn: 'w', plies: [], top: [] })).toBe('game    waiting\n');
    expect(formatGame(undefined)).toBe('game    -\n');
  });
});
