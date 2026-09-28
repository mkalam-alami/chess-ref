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
