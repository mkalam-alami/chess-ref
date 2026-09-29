import type { Point, Quad } from './homography';

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
