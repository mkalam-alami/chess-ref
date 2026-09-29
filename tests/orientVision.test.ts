// Milestone 9, vision side: board orientation (cell -> chess square), dihedral relabellings, the position hint,
// re-orientation after a loss, model survival across losses, the learning guard and the Observation square order.
// Pure-math tests first (synthetic homographies, hand-built logLik); then a few rendered boards (tests/synth).
import { beforeAll, describe, expect, it } from 'vitest';
import { BOARD_DIHEDRAL } from '../src/geom/cornerOrder';
import { applyH, mul3, type Mat3, type Point } from '../src/geom/homography';
import { Detector } from '../src/vision/detector';
import {
  boardHandedness, chooseOrientation, OccupancyFilter, OccupancyTracker, ORIENT_MAPS, START_SQUARES, toCellOrder,
  toSquareOrder, type OccupancyResult,
} from '../src/vision/occupancy';
import type { CV } from '../src/vision/preprocess';
import { OCC_BLACK, OCC_EMPTY, OCC_WHITE, type Params } from '../src/worker/protocol';
import { loadCv } from './synth/cvNode';
import { H, makeBoardSample, makeCamera, W, type SampleSpec } from './synth/generate';
import { alignGt, cellCorrespondence, expectedOrientation, FRAME_CORNERS_ONLY, GAME } from './synth/occBench';

const sq = (s: string) => (s.charCodeAt(1) - 49) * 8 + (s.charCodeAt(0) - 97);
const parity = (c: number) => ((c & 7) + (c >> 3)) & 1;
const same = (a: ArrayLike<number> | null, b: ArrayLike<number> | null) =>
  !!a && !!b && a.length === b.length && Array.from(a).every((v, i) => v === b[i]);
const argmax3 = (l: Float32Array, k: number) => {
  let a = 0;
  for (let q = 1; q < 3; q++) if (l[k * 3 + q]! > l[k * 3 + a]!) a = q;
  return a;
};

/** Generator camera (principal point at the image centre) and its board -> image homography in cell units (0..8). */
function genHb(azDeg: number, elDeg: number, rollDeg = 0, f = 560, D = 14): Mat3 {
  const c = makeCamera((azDeg * Math.PI) / 180, (elDeg * Math.PI) / 180, (rollDeg * Math.PI) / 180, f, D, W / 2, H / 2);
  return mul3(c.hBoard, [1, 0, -4, 0, 1, -4, 0, 0, 1]);
}
const genCorners = (hb: Mat3): Point[] => ([[0, 0], [8, 0], [8, 8], [0, 8]] as Point[]).map((p) => applyH(hb, p));
/** Image-plane rotation by `deg` about the image centre, and a left-right image mirror. */
const imgRot = (deg: number): Mat3 => {
  const a = (deg * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [c, -s, W / 2 - c * (W / 2) + s * (H / 2), s, c, H / 2 - s * (W / 2) - c * (H / 2), 0, 0, 1];
};
const imgMirror: Mat3 = [-1, 0, W, 0, 1, 0, 0, 0, 1];

/** Hand-built logLik (cell order): the class of `cells` gets probability p, the two others (1 - p) / 2. */
function likFor(cells: Uint8Array, p = 0.9): Float32Array {
  const l = new Float32Array(192);
  for (let c = 0; c < 64; c++) for (let k = 0; k < 3; k++) l[c * 3 + k] = Math.log(k === cells[c] ? p : (1 - p) / 2);
  return l;
}

/** A mid-game position (chess square order) with no dihedral symmetry: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6. */
const MID: Uint8Array = (() => {
  const g = new Uint8Array(START_SQUARES);
  for (const [a, b] of GAME) {
    g[b] = g[a]!;
    g[a] = OCC_EMPTY;
  }
  return g;
})();

/** Checks that `o` is a chess orientation: a bijection with files and ranks along perpendicular unit cell steps. */
function expectChessMap(o: Uint8Array): void {
  expect(new Set(o).size).toBe(64);
  const step = (a: number, b: number): Point => [(o[b]! & 7) - (o[a]! & 7), (o[b]! >> 3) - (o[a]! >> 3)];
  const df = step(0, 1);
  const dr = step(0, 8);
  expect(Math.abs(df[0]) + Math.abs(df[1])).toBe(1);
  expect(Math.abs(dr[0]) + Math.abs(dr[1])).toBe(1);
  expect(df[0] * dr[0] + df[1] * dr[1] === 0).toBe(true);
  for (let s = 0; s < 64; s++) {
    if ((s & 7) < 7) expect(step(s, s + 1)).toEqual(df);
    if (s >> 3 < 7) expect(step(s, s + 8)).toEqual(dr);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Pure math

describe('boardHandedness', () => {
  it('is +1 for the generator (right-handed) board frame, overhead or oblique, from any azimuth and roll', () => {
    for (const el of [89.5, 75, 50, 30])
      for (const az of [0, 37, 90, 160, 215, 300])
        for (const roll of [-8, 0, 8]) expect(boardHandedness(genHb(az, el, roll))).toBe(1);
  });

  it('keeps the sign under image rotations (0/90/180/270) and board rotations; mirrors flip it', () => {
    for (const hb of [genHb(20, 85), genHb(130, 45), genHb(250, 32)]) {
      for (const deg of [0, 90, 180, 270]) {
        const rot = mul3(imgRot(deg), hb);
        expect(boardHandedness(rot)).toBe(1);
        expect(boardHandedness(mul3(imgMirror, rot))).toBe(-1);
      }
      BOARD_DIHEDRAL.forEach((d, g) => expect(boardHandedness(mul3(hb, d))).toBe(g < 4 ? 1 : -1));
    }
  });

  it('reads the orientation of simple affine views (image y down) and is 0 when degenerate', () => {
    // Board x to the right, board y up the screen: seen from above, (x, y, up) is right-handed.
    expect(boardHandedness([60, 0, 80, 0, -60, 560, 0, 0, 1])).toBe(1);
    // Board y down the screen: the image of a left-handed frame.
    expect(boardHandedness([60, 0, 80, 0, 60, 40, 0, 0, 1])).toBe(-1);
    // Scaled by a negative factor (same homography): same answer.
    expect(boardHandedness([-60, 0, -80, 0, 60, -560, 0, 0, -1])).toBe(1);
    expect(boardHandedness([60, 0, 80, 60, 0, 40, 0, 0, 1])).toBe(0);
  });

  it('puts the a-file at white\'s left: chooseOrientation on a start position under every labelling', () => {
    // World truth: generator cell (i, j) is chess square j * 8 + i (white on j = 0-1, a1 at (0, 0), dark), and the
    // generator frame (i, j, up) is right-handed, so a->h runs to white's right.
    for (const base of [genHb(0, 85), genHb(75, 50), genHb(200, 35, 6)]) {
      const corners = genCorners(base);
      BOARD_DIHEDRAL.forEach((d) => {
        const hb = mul3(base, d);
        const corr = cellCorrespondence(corners, hb)!; // hb cell -> generator cell = chess square
        expect(corr).not.toBeNull();
        const cells = Uint8Array.from(corr, (s) => START_SQUARES[s]!);
        const a1Cell = corr.indexOf(0);
        const ch = chooseOrientation(likFor(cells), START_SQUARES, boardHandedness(hb), parity(a1Cell), 12, 8);
        expect(ch.g).toBeGreaterThanOrEqual(0);
        const o = ORIENT_MAPS[ch.g]!;
        for (let s = 0; s < 64; s++) expect(corr[o[s]!]).toBe(s);
        // The mirror placement (same cell grid) with the wrong dark square: refused, never mirrored.
        expect(chooseOrientation(likFor(cells), START_SQUARES, boardHandedness(hb), 1 - parity(a1Cell), 12, 8)).toEqual({ g: -1, reason: 'parity' });
      });
    }
  });
});

describe('ORIENT_MAPS, toSquareOrder, toCellOrder', () => {
  it('are 8 distinct chess orientations, rotations first; the orders are inverse', () => {
    const keys = new Set(ORIENT_MAPS.map((m) => m.join(',')));
    expect(keys.size).toBe(8);
    ORIENT_MAPS.forEach((m, g) => {
      expectChessMap(m);
      // Rotations keep (file, rank) as a right-handed pair of cell steps, reflections reverse it.
      const df = [(m[1]! & 7) - (m[0]! & 7), (m[1]! >> 3) - (m[0]! >> 3)];
      const dr = [(m[8]! & 7) - (m[0]! & 7), (m[8]! >> 3) - (m[0]! >> 3)];
      expect(Math.sign(df[0]! * dr[1]! - df[1]! * dr[0]!)).toBe(g < 4 ? 1 : -1);
      const cells = toCellOrder(MID, m);
      expect(same(toSquareOrder(cells, m), MID)).toBe(true);
      const lik = likFor(cells);
      const sqLik = toSquareOrder(lik, m, 3);
      for (let s = 0; s < 64; s++) for (let k = 0; k < 3; k++) expect(sqLik[s * 3 + k]).toBe(lik[m[s]! * 3 + k]);
    });
  });
});

describe('chooseOrientation', () => {
  it('finds each of the 8 placements of an asymmetric mid-game grid', () => {
    ORIENT_MAPS.forEach((m, g) => {
      const lik = likFor(toCellOrder(MID, m));
      const ch = chooseOrientation(lik, MID, g < 4 ? 1 : -1, parity(m[0]!), 12, 8);
      expect(ch.g).toBe(g);
      // Without the parity check (inconclusive) too; a contradicting handedness is a mirrored image: refused.
      expect(chooseOrientation(lik, MID, g < 4 ? 1 : -1, -1, 12, 8).g).toBe(g);
      expect(chooseOrientation(lik, MID, g < 4 ? -1 : 1, -1, 12, 8)).toEqual({ g: -1, reason: 'mirror' });
    });
  });

  it('refuses a symmetric target without handedness support, and a rotation-symmetric one', () => {
    const lik = likFor(toCellOrder(START_SQUARES, ORIENT_MAPS[0]!));
    expect(chooseOrientation(lik, START_SQUARES, 0, -1, 12, 8)).toEqual({ g: -1, reason: 'mirror' });
    // White on a1 and h8 only: the 180 degree rotation predicts the same cells, and handedness cannot tell.
    const sym = new Uint8Array(64);
    sym[sq('a1')] = OCC_WHITE;
    sym[sq('h8')] = OCC_WHITE;
    sym[sq('c3')] = OCC_BLACK;
    sym[sq('f6')] = OCC_BLACK;
    expect(chooseOrientation(likFor(sym), sym, 1, -1, 1, 8)).toEqual({ g: -1, reason: 'ambiguous' });
  });

  it('refuses weak evidence (margin) and too many confidently contradicting cells (occOrientMaxMiss)', () => {
    const cells = toCellOrder(MID, ORIENT_MAPS[3]!);
    // Nearly uniform evidence: every placement scores within a few nats.
    expect(chooseOrientation(likFor(cells, 0.36), MID, 1, -1, 12, 8)).toEqual({ g: -1, reason: 'margin' });
    expect(chooseOrientation(likFor(cells, 0.36), MID, 1, -1, 0.5, 8).g).toBe(3);
    // k cells confidently read as another class (p of the placed class 0.025 < 0.1), the rest right.
    const wrong = (k: number) => {
      const l = likFor(cells, 0.95);
      const empties = [...cells.keys()].filter((c) => cells[c] === OCC_EMPTY).slice(0, k);
      for (const c of empties) for (let q = 0; q < 3; q++) l[c * 3 + q] = Math.log(q === OCC_BLACK ? 0.95 : 0.025);
      return l;
    };
    expect(chooseOrientation(wrong(8), MID, 1, -1, 12, 8).g).toBe(3);
    expect(chooseOrientation(wrong(9), MID, 1, -1, 12, 8)).toEqual({ g: -1, reason: 'miss' });
    // A target with nothing like the board (all empty) is fully symmetric but misses every piece.
    expect(chooseOrientation(likFor(cells, 0.95), new Uint8Array(64), 1, -1, 12, 8)).toEqual({ g: -1, reason: 'miss' });
  });
});

describe('OccupancyFilter', () => {
  const params: Params = { occHoldFrames: 3, occFreezeCells: 6, occSettleMs: 400, occCellMin: 0.2 };
  const conf = new Float32Array(64).fill(1);

  it('permute() carries the committed grid and the pending hysteresis to the new cell labels', () => {
    for (let g = 1; g < 8; g++) {
      const m = ORIENT_MAPS[g]!; // as a cell map: new cell c was old cell m[c] (any permutation works)
      const inv = new Uint8Array(64);
      for (let c = 0; c < 64; c++) inv[m[c]!] = c;
      const f = new OccupancyFilter();
      const start = toCellOrder(START_SQUARES, ORIENT_MAPS[0]!);
      f.reset(start);
      const raw = new Uint8Array(start);
      raw[sq('e2')] = OCC_EMPTY;
      raw[sq('e4')] = OCC_WHITE;
      f.update(raw, conf, params, 1000);
      f.update(raw, conf, params, 1100); // two of the three hold frames
      f.permute(m);
      const want = Uint8Array.from({ length: 64 }, (_, c) => start[m[c]!]!);
      expect(same(f.committed, want)).toBe(true);
      const raw2 = Uint8Array.from({ length: 64 }, (_, c) => raw[m[c]!]!);
      f.update(raw2, conf, params, 1200); // the third frame, in the new labels, commits
      expect(f.committed![inv[sq('e4')]!]).toBe(OCC_WHITE);
      expect(f.committed![inv[sq('e2')]!]).toBe(OCC_EMPTY);
    }
  });

  it('update(..., commit = false) leaves the committed grid and the hysteresis alone but still freezes', () => {
    const f = new OccupancyFilter();
    const start = toCellOrder(START_SQUARES, ORIENT_MAPS[0]!);
    f.reset(start);
    const moved = new Uint8Array(start);
    moved[sq('e2')] = OCC_EMPTY;
    moved[sq('e4')] = OCC_WHITE;
    for (let t = 0; t < 5; t++) f.update(moved, conf, params, 1000 + t * 100, false);
    expect(same(f.committed, start)).toBe(true);
    // Back to committing: the count restarts, so two more frames do not commit, the third does.
    f.update(moved, conf, params, 1500);
    f.update(moved, conf, params, 1600);
    expect(f.committed![sq('e4')]).toBe(OCC_EMPTY);
    f.update(moved, conf, params, 1700);
    expect(f.committed![sq('e4')]).toBe(OCC_WHITE);
    const hand = new Uint8Array(moved);
    for (let c = 16; c < 26; c++) hand[c] = OCC_BLACK;
    expect(f.update(hand, conf, params, 1800, false)).toBe(true);
    expect(f.frozen(2100, params)).toBe(true);
    expect(f.frozen(2200, params)).toBe(false);
  });
});

describe('OccupancyTracker learning guard', () => {
  it('learn() only updates the models from confident cells whose label agrees with the frame class', () => {
    const tr = new OccupancyTracker();
    const g = (mu: number) => ({ mu: new Float64Array(5).fill(mu), v: new Float64Array(5).fill(100) });
    const models = [[g(50), g(50)], [g(80), g(80)], [g(20), g(20)]];
    const internals = tr as unknown as {
      models: typeof models;
      learn(f: { x: Float32Array }, raw: Uint8Array, conf: Float32Array, labels: Uint8Array, p: Params): void;
    };
    internals.models = models;
    const x = new Float32Array(64 * 5);
    const raw = new Uint8Array(64);
    const labels = new Uint8Array(64);
    const conf = new Float32Array(64);
    // Cell 0 (parity 0): label and frame agree (empty), confident -> learnt.
    x.fill(60, 0, 5);
    conf[0] = 1;
    // Cell 1 (parity 1): the frame confidently sees a white piece, the label (hint) says empty -> ignored.
    x.fill(0, 5, 10);
    raw[1] = OCC_WHITE;
    conf[1] = 1;
    // Cell 3 (parity 1): agree on black but not confident enough -> ignored.
    x.fill(0, 15, 20);
    raw[3] = labels[3] = OCC_BLACK;
    conf[3] = 0.5;
    // Cell 2 (parity 0): the label says black, the frame (confidently) white -> ignored.
    x.fill(0, 10, 15);
    raw[2] = OCC_WHITE;
    labels[2] = OCC_BLACK;
    conf[2] = 1;
    internals.learn({ x }, raw, conf, labels, { occLearnRate: 0.1 });
    expect(models[OCC_EMPTY]![0]!.mu[0]).toBeCloseTo(51, 6);
    expect(models[OCC_EMPTY]![1]!.mu[0]).toBe(50);
    for (const cls of [OCC_WHITE, OCC_BLACK]) for (const p of [0, 1]) expect(models[cls]![p]!.mu.every((v) => v === (cls === OCC_WHITE ? 80 : 20))).toBe(true);
    // Once the label agrees with the confident white reading of cell 1, it is learnt.
    labels[1] = OCC_WHITE;
    internals.learn({ x }, raw, conf, labels, { occLearnRate: 0.1 });
    expect(models[OCC_WHITE]![1]!.mu[0]).toBeCloseTo(72, 6);
    expect(models[OCC_BLACK]![0]!.mu[0]).toBe(20);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Rendered boards

/** Seeds found to detect and calibrate: dark-a1 boards (flip 0) and light-a1 boards (meta.flip = 1). */
const DARK: ReadonlyArray<readonly [number, SampleSpec['elev'], SampleSpec['palette']]> = [
  [9000, 'overhead', 'wood'], [9005, 'overhead', 'vinyl'], [9020, 'oblique', 'vinyl'], [9026, 'oblique', 'printed'],
];
const LIGHT: ReadonlyArray<readonly [number, SampleSpec['elev'], SampleSpec['palette']]> = [
  [9003, 'overhead', 'wood'], [9019, 'oblique', 'wood'],
];

/** The rendered boards keep only the board itself (z = 0) inside the image: tall pieces on the far rank may reach
 *  the frame edge, which the whole-board-in-view check (boardFramed) rejects. These tests are about orientation on
 *  boards in view, so they check the corners only (their own framing tests are below). */
const P: Params = FRAME_CORNERS_ONLY;

interface Board {
  frame: { data: Uint8ClampedArray; width: number; height: number };
  /** Detected homography of the start frame (reused for later positions: fixed camera). */
  hb: Mat3;
  corners: Point[];
  flip: boolean;
  /** Ground-truth occupancy, generator cells (= chess squares on a dark-a1 board). */
  occ: Uint8Array;
}

describe('orientation on rendered boards', () => {
  let cv: CV;
  let det: Detector;
  const cache = new Map<string, Board>();
  beforeAll(async () => {
    ({ cv } = await loadCv());
    det = new Detector(cv);
  }, 60_000);

  function board(seed: number, elev: SampleSpec['elev'], palette: SampleSpec['palette'], moves = 0): Board {
    const key = `${seed}/${moves}`;
    const hit = cache.get(key);
    if (hit) return hit;
    const s = makeBoardSample(cv, seed, { elev, palette, pieces: 'start' }, GAME.slice(0, moves));
    const frame = { data: s.rgba, width: s.width, height: s.height };
    let hb: Mat3;
    if (moves === 0) {
      const r = det.detect(frame as unknown as ImageData, {});
      expect(r.hb).toBeTruthy();
      hb = r.hb!;
    } else hb = board(seed, elev, palette).hb;
    const b = { frame, hb, corners: s.corners!, flip: s.meta.flip === 1, occ: s.occupancy! };
    cache.set(key, b);
    return b;
  }

  /** A tracker calibrated on the start frame; returns every result (frames 100 ms apart from t = 0). */
  function calibrate(b: Board, n = 5, params: Params = P): { tr: OccupancyTracker; outs: OccupancyResult[]; t: number } {
    const tr = new OccupancyTracker();
    const outs: OccupancyResult[] = [];
    let t = 0;
    for (let i = 0; i < n; i++, t += 100) outs.push(tr.update(b.frame, b.hb, params, t));
    return { tr, outs, t: t - 100 };
  }

  it('orients dark-a1 start boards after calibration (a1 on the right cell, ranks and files right)', () => {
    for (const [seed, elev, palette] of DARK) {
      const b = board(seed, elev, palette);
      expect(b.flip).toBe(false);
      const { tr, outs } = calibrate(b);
      const want = expectedOrientation(b.corners, b.hb, false)!;
      expect(outs[0]!.orientation).toBeNull();
      expect(outs[0]!.observation.oriented).toBe(false);
      const last = outs[outs.length - 1]!;
      expect(last.stats.state).toBe('calibrated');
      expect(tr.oriented).toBe(true);
      expect(same(last.orientation, want)).toBe(true);
      expect(last.observation.oriented).toBe(true);
      expectChessMap(last.orientation!);
      // Never oriented to anything else on the way.
      for (const o of outs) if (o.orientation) expect(same(o.orientation, want)).toBe(true);
      // Every square's cell holds that square's start-position content (white on ranks 1-2 etc.).
      const gt = alignGt(b.occ, b.corners, b.hb)!;
      for (let s = 0; s < 64; s++) expect(gt[last.orientation![s]!]).toBe(START_SQUARES[s]);
    }
  });

  it('leaves light-a1 boards (meta.flip) unoriented, even with a position hint', () => {
    for (const [seed, elev, palette] of LIGHT) {
      const b = board(seed, elev, palette);
      expect(b.flip).toBe(true);
      expect(expectedOrientation(b.corners, b.hb, true)).toBeNull();
      const { tr, outs } = calibrate(b, 8);
      expect(outs[outs.length - 1]!.stats.state).toBe('calibrated');
      for (const o of outs) {
        expect(o.orientation).toBeNull();
        expect(o.observation.oriented).toBe(false);
        // Unoriented: the observation stays in board cell order.
        expect(same(o.observation.logLik, o.logLik)).toBe(true);
      }
      expect(tr.orientReason).toBe('parity');
      // A hint is ignored while unoriented: the grid is the filter's own.
      const twin = calibrate(b, 8).tr;
      tr.setPosition(MID);
      for (let t = 800; t < 1200; t += 100) {
        const o = tr.update(b.frame, b.hb, P, t);
        const w = twin.update(b.frame, b.hb, P, t);
        expect(o.orientation).toBeNull();
        expect(same(o.grid, w.grid)).toBe(true);
      }
    }
  });

  it('observation arrays are in chess square order when oriented', () => {
    const b = board(...DARK[0]!);
    const { tr, outs } = calibrate(b);
    const o = outs[outs.length - 1]!;
    const orient = o.orientation!;
    const cellVis = tr.debug!.feats.vis;
    const a1Cell = expectedOrientation(b.corners, b.hb, false)![0]!;
    expect(orient[0]).toBe(a1Cell);
    for (let k = 0; k < 3; k++) expect(o.observation.logLik[k]).toBe(o.logLik[a1Cell * 3 + k]);
    for (let s = 0; s < 64; s++) {
      expect(o.observation.vis[s]).toBe(cellVis[orient[s]!]);
      for (let k = 0; k < 3; k++) expect(o.observation.logLik[s * 3 + k]).toBe(o.logLik[orient[s]! * 3 + k]);
    }
    // In square order the evidence reads as the starting position.
    let ok = 0;
    for (let s = 0; s < 64; s++) if (argmax3(o.observation.logLik, s) === START_SQUARES[s]) ok++;
    expect(ok).toBeGreaterThanOrEqual(56);
    expect(o.observation.calibration).toBe('calibrated');
    expect(o.observation.stable).toBe(true);
  });

  it('permutes the orientation and the filter state with each of the 8 relabellings of hb', () => {
    const b = board(...DARK[2]!);
    for (let g = 0; g < 8; g++) {
      const { tr, outs, t } = calibrate(b);
      const o0 = outs[outs.length - 1]!;
      expect(o0.orientation).not.toBeNull();
      const hb2 = mul3(b.hb, BOARD_DIHEDRAL[g]!);
      const o = tr.update(b.frame, hb2, P, t + 100);
      expect(same(o.orientation, expectedOrientation(b.corners, hb2, false))).toBe(true);
      // Same physical squares: the committed grid in square order is unchanged.
      expect(same(toSquareOrder(o.grid!, o.orientation!), toSquareOrder(o0.grid!, o0.orientation!))).toBe(true);
      // The per-parity models followed (a quarter turn swaps the parity of cell (0, 0)): same readings per square.
      let agree = 0;
      for (let s = 0; s < 64; s++) if (argmax3(o.observation.logLik, s) === argmax3(o0.observation.logLik, s)) agree++;
      expect(agree).toBeGreaterThanOrEqual(60);
      expect(o.stats.state).toBe('calibrated');
    }
  });

  it('marks the orientation unverified on an unresolvable relabelling, then re-finds it (all 8 labellings)', () => {
    // Lost, and the camera moved meanwhile: the last homography seen is not a dihedral relabelling of the next one,
    // which comes back in any of the 8 labellings. Regression: the stale committed grid (another cell frame) used to
    // be kept as the occluder prior and illumination labels, making the frame read as that grid; most labellings
    // then never re-oriented.
    const cases: Array<{ b: Board; start: Board; hint: Uint8Array | null }> = [];
    for (const spec of DARK) cases.push({ b: board(...spec), start: board(...spec), hint: null });
    cases.push({ b: board(DARK[2]![0], DARK[2]![1], DARK[2]![2], GAME.length), start: board(...DARK[2]!), hint: MID });
    for (const { b, start, hint } of cases) {
      const target = hint ?? START_SQUARES;
      for (let g = 0; g < 8; g++) {
        const { tr, t: t0 } = calibrate(start);
        expect(tr.oriented).toBe(true);
        let t = t0;
        tr.setPosition(hint);
        tr.noBoard((t += 100));
        (tr as unknown as { lastHb: Mat3 }).lastHb = mul3(b.hb, [1, 0, 2.5, 0, 1, 1.5, 0, 0, 1]);
        const hb2 = mul3(b.hb, BOARD_DIHEDRAL[g]!);
        const want = expectedOrientation(b.corners, hb2, false)!;
        let found = -1;
        for (let i = 1; i <= 8; i++) {
          const o = tr.update(b.frame, hb2, P, (t += 100));
          if (i === 1) {
            expect(o.orientation).toBeNull();
            expect(o.observation.oriented).toBe(false);
          }
          if (!o.orientation) continue;
          expect(same(o.orientation, want)).toBe(true);
          if (found > 0) continue;
          found = i;
          // The stale filter grid restarts from the target in the new cell frame.
          expect(same(toSquareOrder(o.grid!, o.orientation), target)).toBe(true);
        }
        expect(found).toBeGreaterThan(0);
        expect(tr.calibrated).toBe(true);
      }
    }
  });

  it('position hint: grid follows the hint at once while oriented; setPosition(null) restores the filter', () => {
    const b = board(...DARK[2]!);
    const { tr, t: t0 } = calibrate(b);
    const orient = tr.orientation!;
    let t = t0;
    // The game says 1. e4, the frame still shows the starting position.
    const hint = new Uint8Array(START_SQUARES);
    hint[sq('e2')] = OCC_EMPTY;
    hint[sq('e4')] = OCC_WHITE;
    tr.setPosition(hint);
    const internals = tr as unknown as { learn: (...a: unknown[]) => void };
    const learn = internals.learn;
    let labels: Uint8Array | null = null;
    internals.learn = function (this: unknown, ...a: unknown[]) {
      labels = new Uint8Array(a[3] as Uint8Array);
      return learn.apply(this, a);
    };
    for (let i = 0; i < 4; i++) {
      const o = tr.update(b.frame, b.hb, P, (t += 100));
      // Hysteresis bypassed from the first frame: the hint in board cells.
      expect(same(o.grid, toCellOrder(hint, orient))).toBe(true);
      expect(same(o.orientation, orient)).toBe(true);
      // The hint is the learning label (the guard then drops e2 / e4, where the frame disagrees).
      expect(same(labels, toCellOrder(hint, orient))).toBe(true);
      expect(o.raw[orient[sq('e2')]!]).toBe(OCC_WHITE);
    }
    internals.learn = learn;
    // A hint for a position the board now shows (after 1. e4) keeps the grid in place too.
    const e4 = board(DARK[2]![0], DARK[2]![1], DARK[2]![2], 1);
    for (let i = 0; i < 3; i++) expect(same(tr.update(e4.frame, e4.hb, P, (t += 100)).grid, toCellOrder(hint, orient))).toBe(true);
    // No game: back to the per-cell filter, whose hysteresis starts from the hint's grid.
    tr.setPosition(null);
    const e2 = orient[sq('e2')]!;
    const e4c = orient[sq('e4')]!;
    const hold = 3; // occHoldFrames default
    for (let i = 1; i <= hold + 2; i++) {
      const o = tr.update(b.frame, b.hb, P, (t += 100));
      if (i < hold) {
        expect(o.grid![e2]).toBe(OCC_EMPTY);
        expect(o.grid![e4c]).toBe(OCC_WHITE);
      } else if (i >= hold + 1) {
        expect(o.grid![e2]).toBe(OCC_WHITE);
        expect(o.grid![e4c]).toBe(OCC_EMPTY);
      }
    }
  });

  it('re-orients from a mid-game hint after a loss > occLossMs, under each of the 8 relabellings', () => {
    const b = board(...DARK[2]!);
    const mid = board(DARK[2]![0], DARK[2]![1], DARK[2]![2], GAME.length);
    expect(same(mid.occ, MID)).toBe(true);
    const { tr, t: t0 } = calibrate(b);
    expect(tr.oriented).toBe(true);
    let t = t0;
    // Follow the game with the hint so the models see the mid-game board.
    tr.setPosition(MID);
    for (let i = 0; i < 3; i++) tr.update(mid.frame, mid.hb, P, (t += 100));
    for (let g = 0; g < 8; g++) {
      tr.noBoard((t += 100));
      t += 1500; // > occLossMs
      const hb2 = mul3(b.hb, BOARD_DIHEDRAL[g]!);
      const want = expectedOrientation(b.corners, hb2, false)!;
      const o1 = tr.update(mid.frame, hb2, P, t);
      // Unverified after the loss; the first agreeing frame is not enough (occOrientFrames = 2).
      expect(o1.orientation).toBeNull();
      expect(o1.observation.oriented).toBe(false);
      expect(o1.stats.state).toBe('calibrated');
      const o2 = tr.update(mid.frame, hb2, P, (t += 100));
      expect(same(o2.orientation, want)).toBe(true);
      // From the next frame on the hint owns the grid again (on the orienting frame itself the grid is still the
      // filter's own, which ran before the orientation was adopted).
      const o3 = tr.update(mid.frame, hb2, P, (t += 100));
      expect(same(o3.orientation, want)).toBe(true);
      expect(same(o3.grid, toCellOrder(MID, want))).toBe(true);
    }
    // More agreeing frames required: still unoriented after two.
    tr.noBoard((t += 100));
    t += 1500;
    const p3: Params = { ...P, occOrientFrames: 3 };
    expect(tr.update(mid.frame, b.hb, p3, t).orientation).toBeNull();
    expect(tr.update(mid.frame, b.hb, p3, (t += 100)).orientation).toBeNull();
    expect(same(tr.update(mid.frame, b.hb, p3, (t += 100)).orientation, expectedOrientation(b.corners, b.hb, false))).toBe(true);
  });

  it('refuses to re-orient against a hint that does not explain the board', () => {
    const b = board(...DARK[2]!);
    const mid = board(DARK[2]![0], DARK[2]![1], DARK[2]![2], GAME.length);
    const { tr, t: t0 } = calibrate(b);
    let t = t0;
    tr.noBoard((t += 100));
    t += 1500;
    // An empty board: fully symmetric, but it contradicts every visible piece.
    tr.setPosition(new Uint8Array(64));
    for (let i = 0; i < 5; i++) expect(tr.update(mid.frame, b.hb, P, (t += 100)).orientation).toBeNull();
    expect(tr.orientReason).toBe('miss');
    // A demanding margin refuses even the right hint.
    tr.setPosition(MID);
    const strict: Params = { ...P, occOrientMargin: 1e6 };
    for (let i = 0; i < 3; i++) expect(tr.update(mid.frame, b.hb, strict, (t += 100)).orientation).toBeNull();
    expect(tr.orientReason).toBe('margin');
    // With the default margin it is found.
    tr.update(mid.frame, b.hb, P, (t += 100));
    expect(same(tr.update(mid.frame, b.hb, P, (t += 100)).orientation, expectedOrientation(b.corners, b.hb, false))).toBe(true);
  });

  it('does not count frames with the board partly out of view: no calibration, learning or orientation; settles after', () => {
    // Default framing rule (not P): this overhead board is wholly in view, far-rank piece tops included.
    const b = board(...DARK[0]!);
    /** The same scene moved dx px to the right (image and homography). */
    const shifted = (dx: number) => {
      const data = new Uint8ClampedArray(b.frame.data.length);
      for (let y = 0; y < H; y++) data.set(b.frame.data.subarray(y * W * 4, (y * W + W - dx) * 4), (y * W + dx) * 4);
      return { frame: { data, width: W, height: H }, hb: mul3([1, 0, dx, 0, 1, 0, 0, 0, 1], b.hb) };
    };
    const out = shifted(150);
    expect(Math.max(...genCorners(out.hb).map((p) => p[0]))).toBeGreaterThan(W);
    const internal = ['learn', 'tryBootstrap', 'calibrate', 'tryOrient', 'recheckParity'];
    const spy = (tr: OccupancyTracker) => {
      const calls: Record<string, number> = {};
      const internals = tr as unknown as Record<string, (...a: unknown[]) => unknown>;
      for (const k of internal) {
        const f = internals[k]!;
        calls[k] = 0;
        internals[k] = function (this: unknown, ...a: unknown[]) {
          calls[k]!++;
          return f.apply(this, a);
        };
      }
      return calls;
    };
    const models = (tr: OccupancyTracker) =>
      JSON.stringify((tr as unknown as { models: unknown }).models, (_, v) => (v instanceof Float64Array ? Array.from(v) : v));
    /** One frame; on a frame not in view, checks that it counted for nothing. */
    const step = (tr: OccupancyTracker, calls: Record<string, number>, sc: { frame: Board['frame']; hb: Mat3 }, t: number, rect?: [number, number, number, number]) => {
      const m0 = models(tr);
      const c0 = { ...calls };
      const o = tr.update(sc.frame, sc.hb, {}, t, rect);
      if (!o.observation.framed) {
        expect(o.observation.stable).toBe(false);
        expect(o.grid).toBeNull();
        expect(calls).toEqual(c0);
        expect(models(tr)).toBe(m0);
      }
      return o;
    };

    // Not calibrated yet: cropped frames never calibrate, not even unsupervised past occFallbackMs.
    const tr0 = new OccupancyTracker();
    const calls0 = spy(tr0);
    for (let t = 0; t <= 9000; t += 250) expect(step(tr0, calls0, out, t).observation.framed).toBe(false);
    expect(tr0.calibrated).toBe(false);

    // Calibrated and oriented, then the board slides out to the right (25 px per frame, a moving camera) and back.
    const { tr, t: t0 } = calibrate(b, 5, {});
    const orient = tr.orientation!;
    expect(orient).not.toBeNull();
    const calls = spy(tr);
    let t = t0;
    let o = step(tr, calls, b, (t += 100));
    expect(o.observation.framed).toBe(true);
    expect(o.observation.stable).toBe(true);
    expect(calls.learn).toBe(1);
    const g0 = o.grid;
    let lastCrop = -Infinity;
    const path = [25, 50, 75, 100, 125, 150, 150, 150, 150, 125, 100, 75, 50, 25, 0];
    for (const dx of path) {
      o = step(tr, calls, shifted(dx), (t += 100));
      // Frozen for occSettleMs after the last frame not in view.
      if (!o.observation.framed) lastCrop = t;
      else expect(o.observation.stable).toBe(t - lastCrop >= 400);
      // The orientation is kept throughout.
      expect(same(o.orientation, orient)).toBe(true);
    }
    expect(lastCrop).toBeGreaterThan(t0);
    // The visible rect counts, not the frame: the unshifted frame with its left part off screen.
    const left = Math.min(...genCorners(b.hb).map((p) => p[0]));
    o = step(tr, calls, b, (t += 100), [left + 20, 0, W, H]);
    expect(o.observation.framed).toBe(false);
    expect(o.stats.frozen).toBe(true);
    lastCrop = t;
    for (let i = 0; i < 6; i++) {
      o = step(tr, calls, b, (t += 100), [0, 0, W, H]);
      expect(o.observation.framed).toBe(true);
      expect(o.observation.stable).toBe(t - lastCrop >= 400);
    }
    expect(o.observation.stable).toBe(true);
    // Nothing was committed from the cropped frames.
    expect(same(o.grid, g0)).toBe(true);

    // After a long loss the orientation is re-verified: never on frames not in view, then again once back in view.
    tr.noBoard((t += 100));
    t += 1500;
    for (let i = 0; i < 4; i++) expect(step(tr, calls, out, (t += 100)).orientation).toBeNull();
    for (const dx of [125, 100, 75, 50, 25, 0, 0, 0, 0, 0, 0, 0, 0]) if (step(tr, calls, shifted(dx), (t += 100)).orientation) break;
    expect(same(tr.orientation, orient)).toBe(true);
  });

  it('keeps the models and orientation across noBoard; stable is false while settling; only reset() drops them', () => {
    const b = board(...DARK[0]!);
    const fresh = new OccupancyTracker().update(b.frame, b.hb, P, 0);
    // Before calibration: not stable, not oriented.
    expect(fresh.observation.stable).toBe(false);
    expect(fresh.observation.calibration).toBe('start');
    const { tr, t: t0 } = calibrate(b);
    const orient = tr.orientation!;
    let t = t0;
    // Short loss (< occLossMs): orientation kept; frozen for occSettleMs after the board-less frame.
    tr.noBoard((t += 100));
    const lostAt = t;
    let o = tr.update(b.frame, b.hb, P, (t += 100));
    expect(same(o.orientation, orient)).toBe(true);
    expect(o.observation.oriented).toBe(true);
    expect(o.observation.calibration).toBe('calibrated');
    expect(o.observation.stable).toBe(false);
    expect(o.stats.frozen).toBe(true);
    o = tr.update(b.frame, b.hb, P, lostAt + 350);
    expect(o.observation.stable).toBe(false);
    o = tr.update(b.frame, b.hb, P, (t = lostAt + 450));
    expect(o.observation.stable).toBe(true);
    // Long loss: the models survive, the orientation is re-verified (not dropped) and comes back.
    tr.noBoard((t += 100));
    t += 2000;
    o = tr.update(b.frame, b.hb, P, t);
    expect(tr.calibrated).toBe(true);
    expect(o.stats.state).toBe('calibrated');
    expect(o.orientation).toBeNull();
    o = tr.update(b.frame, b.hb, P, (t += 100));
    expect(same(o.orientation, orient)).toBe(true);
    // reset(): models and orientation gone; the next frame is uncalibrated and unstable.
    tr.reset();
    expect(tr.calibrated).toBe(false);
    expect(tr.orientation).toBeNull();
    o = tr.update(b.frame, b.hb, P, (t += 100));
    expect(o.observation.calibration).toBe('start');
    expect(o.observation.stable).toBe(false);
    expect(o.orientation).toBeNull();
  });
});
