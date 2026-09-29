import { applyH, mul3, type Mat3, type Point, type Quad } from './homography';

function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/**
 * The board has 4-fold symmetry, so a new detection may list its corners starting
 * anywhere (and in either winding). Pick the cyclic rotation, and reversal if needed,
 * of `next` that minimises total distance to `prev`.
 */
export function stabiliseCorners(prev: Quad, next: Quad): Quad {
  let best: Quad = next;
  let bestCost = Infinity;
  for (const reversed of [false, true]) {
    const seq = reversed ? [next[0], next[3], next[2], next[1]] : [...next];
    for (let r = 0; r < 4; r++) {
      const cand = [0, 1, 2, 3].map((i) => seq[(i + r) % 4]!) as Quad;
      const cost = cand.reduce((s, p, i) => s + dist(p, prev[i]!), 0);
      if (cost < bestCost) {
        bestCost = cost;
        best = cand;
      }
    }
  }
  return best;
}

/** True when no two opposite edges of the quad cross (i.e. it is not a bow-tie). */
export function isSimpleQuad(q: Quad): boolean {
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const hit = (p1: Point, p2: Point, p3: Point, p4: Point) =>
    cross(p1, p2, p3) * cross(p1, p2, p4) < 0 && cross(p3, p4, p1) * cross(p3, p4, p2) < 0;
  return !hit(q[0], q[1], q[2], q[3]) && !hit(q[1], q[2], q[3], q[0]);
}

/** Reorders a self-intersecting quad into a simple one (convex-hull order around the centroid); null if degenerate. */
export function repairQuad(q: Quad): Quad | null {
  if (isSimpleQuad(q)) return q;
  const cx = q.reduce((s, p) => s + p[0], 0) / 4;
  const cy = q.reduce((s, p) => s + p[1], 0) / 4;
  const r = [...q].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx)) as Quad;
  return isSimpleQuad(r) ? r : null;
}

/** Board points of the 4 corners, in the order `corners[k]` reports them: (0,0), (8,0), (8,8), (0,8). */
export const BOARD_CORNERS: Quad = [[0, 0], [8, 0], [8, 8], [0, 8]];

/**
 * The 8 symmetries of the board square [0,8]^2 as board -> board homographies: the 4 rotations
 * (orientation-preserving) followed by the 4 reflections.
 */
export const BOARD_DIHEDRAL: readonly Mat3[] = [
  [1, 0, 0, 0, 1, 0, 0, 0, 1], // identity
  [0, -1, 8, 1, 0, 0, 0, 0, 1], // (x, y) -> (8 - y, x)
  [-1, 0, 8, 0, -1, 8, 0, 0, 1], // (x, y) -> (8 - x, 8 - y)
  [0, 1, 0, -1, 0, 8, 0, 0, 1], // (x, y) -> (y, 8 - x)
  [0, 1, 0, 1, 0, 0, 0, 0, 1], // (x, y) -> (y, x)
  [-1, 0, 8, 0, 1, 0, 0, 0, 1], // (x, y) -> (8 - x, y)
  [1, 0, 0, 0, -1, 8, 0, 0, 1], // (x, y) -> (x, 8 - y)
  [0, -1, 8, -1, 0, 8, 0, 0, 1], // (x, y) -> (8 - y, 8 - x)
];

/** Image points of board corners (0,0), (8,0), (8,8), (0,8) under `hb`. */
export function hbCorners(hb: Mat3): Quad {
  return BOARD_CORNERS.map((p) => applyH(hb, p)) as Quad;
}

/** Shoelace signed area (positive = clockwise on screen, where y points down). */
export function signedArea(q: readonly Point[]): number {
  let a = 0;
  for (let i = 0; i < q.length; i++) {
    const p = q[i]!;
    const n = q[(i + 1) % q.length]!;
    a += p[0] * n[1] - n[0] * p[1];
  }
  return a / 2;
}

function normaliseH(h: Mat3): Mat3 {
  const s = h[8];
  return Math.abs(s) > 1e-12 ? (h.map((v) => v / s) as Mat3) : h;
}

/**
 * Relabels a board homography (board cell units -> image) within the board's 8-fold dihedral ambiguity so its
 * orientation is stable across frames: returns `hb * D` for the symmetry D minimising the summed distance between
 * its corners and `prev`'s. Only candidates with the same image winding as `prev` are considered (a real board
 * cannot flip over). Without `prev`, `hb` is only normalised to a positive winding (like the detector's corner
 * order). Either way `hbCorners(result)` is a relabelling of `hbCorners(hb)`.
 */
export function orientHomography(hb: Mat3, prev: Mat3 | null): Mat3 {
  const prevQ = prev ? hbCorners(prev) : null;
  const wantPositive = prevQ ? signedArea(prevQ) >= 0 : true;
  let best: Mat3 = hb;
  let bestCost = Infinity;
  for (const d of BOARD_DIHEDRAL) {
    const cand = mul3(hb, d);
    const q = hbCorners(cand);
    if (signedArea(q) >= 0 !== wantPositive) continue;
    // Without a previous frame, the first candidate with the right winding (identity or the transpose) wins.
    const cost = prevQ ? q.reduce((s, p, i) => s + dist(p, prevQ[i]!), 0) : 0;
    if (cost < bestCost) {
      bestCost = cost;
      best = cand;
    }
  }
  return normaliseH(best);
}
