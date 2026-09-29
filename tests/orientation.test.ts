import { describe, expect, it } from 'vitest';
import { BOARD_DIHEDRAL, hbCorners, orientHomography, signedArea } from '../src/geom/cornerOrder';
import type { Detector } from '../src/vision/detector';
import { TrackingSession } from '../src/worker/tracker';
import type { Params } from '../src/worker/protocol';
import { applyH, homographyFrom4, mul3, type Mat3, type Point, type Quad } from '../src/geom/homography';

const BOARD: Point[] = [[0, 0], [8, 0], [8, 8], [0, 8]];
/** A perspective board -> image homography (positive winding: clockwise on screen). */
const img: Quad = [[100, 80], [520, 110], [560, 420], [60, 380]];
const H = homographyFrom4(BOARD, img)!;

const close = (a: readonly Point[], b: readonly Point[], tol = 1e-6) =>
  a.every((p, i) => Math.hypot(p[0] - b[i]![0], p[1] - b[i]![1]) < tol);

/** Maps board point p through a D-relabelled homography to the same image point as through the original. */
function sameBoardMapping(a: Mat3, b: Mat3): boolean {
  const pts: Point[] = [[0.5, 0.5], [3.5, 6.5], [7.5, 1.5]];
  return pts.every((p) => {
    const q = applyH(a, p);
    const r = applyH(b, p);
    return Math.hypot(q[0] - r[0], q[1] - r[1]) < 1e-6;
  });
}

describe('BOARD_DIHEDRAL', () => {
  it('maps the board square onto itself, 4 rotations then 4 reflections', () => {
    BOARD_DIHEDRAL.forEach((d, k) => {
      const q = BOARD.map((p) => applyH(d, p));
      for (const p of q) expect(BOARD.some((b) => Math.hypot(b[0] - p[0], b[1] - p[1]) < 1e-9)).toBe(true);
      expect(Math.sign(signedArea(q))).toBe(k < 4 ? 1 : -1);
    });
  });
});

describe('orientHomography', () => {
  it('keeps the labelling of a homography already aligned with the previous one', () => {
    const out = orientHomography(H, H);
    expect(sameBoardMapping(out, H)).toBe(true);
    expect(close(hbCorners(out), img)).toBe(true);
  });

  it('undoes any of the 8 dihedral relabellings of a re-detection', () => {
    for (const d of BOARD_DIHEDRAL) {
      const redetected = mul3(H, d);
      const out = orientHomography(redetected, H);
      expect(sameBoardMapping(out, H)).toBe(true);
      expect(close(hbCorners(out), img)).toBe(true);
    }
  });

  it('follows small motion between frames', () => {
    const moved: Quad = img.map(([x, y]) => [x + 7, y - 4]) as Quad;
    const Hm = homographyFrom4(BOARD, moved)!;
    for (const d of BOARD_DIHEDRAL) {
      const out = orientHomography(mul3(Hm, d), H);
      expect(close(hbCorners(out), moved)).toBe(true);
    }
  });

  it('tracks a board rotating in the image by steps under 45 degrees', () => {
    const c: Point = [310, 250];
    const rot = (q: Quad, a: number) =>
      q.map(([x, y]) => [c[0] + Math.cos(a) * (x - c[0]) - Math.sin(a) * (y - c[1]), c[1] + Math.sin(a) * (x - c[0]) + Math.cos(a) * (y - c[1])]) as Quad;
    let prev = H;
    for (let step = 1; step <= 12; step++) {
      const q = rot(img, (step * 30 * Math.PI) / 180);
      // Each re-detection comes back in an arbitrary labelling.
      const raw = mul3(homographyFrom4(BOARD, q)!, BOARD_DIHEDRAL[(step * 5) % 8]!);
      prev = orientHomography(raw, prev);
      expect(close(hbCorners(prev), q, 1e-4)).toBe(true);
    }
  });

  it('without a previous frame, only normalises to a positive winding', () => {
    const flipped = mul3(H, BOARD_DIHEDRAL[4]!); // transpose: negative winding
    expect(signedArea(hbCorners(flipped))).toBeLessThan(0);
    const out = orientHomography(flipped, null);
    expect(signedArea(hbCorners(out))).toBeGreaterThan(0);
    expect(close(hbCorners(out), img)).toBe(true);
    // An already positive homography is left as is.
    expect(sameBoardMapping(orientHomography(H, null), H)).toBe(true);
  });

  it('never flips the winding relative to the previous frame, even when a reflection is closer', () => {
    // The new quad is the previous one mirrored left-right: the reflection would match corners better.
    const mirrored: Quad = [img[1], img[0], img[3], img[2]].map(([x, y]) => [x, y]) as Quad;
    const out = orientHomography(homographyFrom4(BOARD, mirrored)!, H);
    expect(signedArea(hbCorners(out))).toBeGreaterThan(0);
  });
});

describe('TrackingSession orientation', () => {
  /** Fake detector: every call returns the same board in the next dihedral labelling (detector corner order). */
  function fakeDetector(board: () => Mat3 | null) {
    let n = 0;
    const res = () => {
      const b = board();
      if (!b) return { corners: null, confidence: 0, timings: {} };
      const hb = mul3(b, BOARD_DIHEDRAL[n++ % 8]!);
      const q = hbCorners(hb);
      return { corners: signedArea(q) < 0 ? [q[0], q[3], q[2], q[1]] : q, confidence: 1, timings: {}, hb };
    };
    return { detect: res, track: res } as unknown as Detector;
  }
  const input = {} as ImageData;

  it('returns corners in a stable order, equal to hb applied to the board corners', () => {
    let board: Mat3 | null = H;
    const s = new TrackingSession(fakeDetector(() => board));
    // Force a full detection every frame (tracking disabled) and also exercise the tracking path.
    for (const params of [{ tracking: 0 }, { tracking: 1, trackFullEvery: 0 }] as Params[]) {
      s.reset();
      let first: Mat3 | null = null;
      for (let f = 0; f < 10; f++) {
        const r = s.process(input, 640, 480, params, {}, f);
        expect(close(hbCorners(r.hb!), r.corners!)).toBe(true);
        expect(signedArea(r.corners!)).toBeGreaterThan(0);
        first ??= r.hb!;
        // Same labelling as the first frame of the lock, whatever labelling the detector returned.
        expect(sameBoardMapping(r.hb!, first)).toBe(true);
        expect(close(r.corners!, hbCorners(first))).toBe(true);
      }
    }
    // Losing the board drops the orientation memory.
    board = null;
    expect(s.process(input, 640, 480, { tracking: 0 }, {}, 20).corners).toBeNull();
    expect(s.locked).toBe(false);
  });
});
