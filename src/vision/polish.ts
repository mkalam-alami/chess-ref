import { applyH, homographyFrom4, type Mat3, type Point } from '../geom/homography';
import type { CV } from './preprocess';

type Mat = InstanceType<CV['Mat']>;

export interface PolishResult {
  hb: Mat3;
  /** Mean capped edge distance (px) before and after. */
  before: number;
  after: number;
}

const SAMPLES_PER_LINE = 20;

/** Grid line sample points in board cell units: the 9 vertical and 9 horizontal lines, away from the corners. */
const GRID_POINTS: Array<[number, number]> = (() => {
  const pts: Array<[number, number]> = [];
  for (let i = 0; i <= 8; i++) {
    for (let k = 0; k < SAMPLES_PER_LINE; k++) {
      const t = 0.2 + (7.6 * (k + 0.5)) / SAMPLES_PER_LINE;
      pts.push([i, t]);
      pts.push([t, i]);
    }
  }
  return pts;
})();

/**
 * Coarse edge-alignment polish of the board homography: a small pattern search over the 4 corners minimising the
 * mean capped distance from the 18 grid lines to the nearest edge pixel. Pulls a VP-derived fit onto the real
 * lines (lens distortion and VP noise leave a few pixels of error). Not a substitute for the corner-level
 * refinement of the next milestone.
 */
export class EdgePolisher {
  private readonly inv: Mat;
  private readonly dt: Mat;

  constructor(private readonly cv: CV) {
    this.inv = new cv.Mat();
    this.dt = new cv.Mat();
  }

  dispose(): void {
    this.inv.delete();
    this.dt.delete();
  }

  polish(edges: Mat, hb: Mat3, opts: { maxShiftFrac: number } = { maxShiftFrac: 0.05 }): PolishResult {
    const cv = this.cv;
    cv.bitwise_not(edges, this.inv);
    cv.distanceTransform(this.inv, this.dt, cv.DIST_L2, 3);
    const w = this.dt.cols;
    const h = this.dt.rows;
    const dt = this.dt.data32F;

    const sample = (x: number, y: number, cap: number): number => {
      if (!(x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1)) return cap;
      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const fx = x - x0;
      const fy = y - y0;
      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);
      const v =
        dt[y0 * w + x0]! * (1 - fx) * (1 - fy) + dt[y0 * w + x1]! * fx * (1 - fy) +
        dt[y1 * w + x0]! * (1 - fx) * fy + dt[y1 * w + x1]! * fx * fy;
      return v < cap ? v : cap;
    };

    const board: Point[] = [[0, 0], [8, 0], [8, 8], [0, 8]];
    const c0: Point[] = [applyH(hb, board[0]!), applyH(hb, board[1]!), applyH(hb, board[2]!), applyH(hb, board[3]!)];
    const diag = Math.hypot(c0[2]![0] - c0[0]![0], c0[2]![1] - c0[0]![1]);
    // Cap the distance below half a cell so a line never locks onto its neighbour.
    const cell = Math.min(
      Math.hypot(c0[1]![0] - c0[0]![0], c0[1]![1] - c0[0]![1]),
      Math.hypot(c0[2]![0] - c0[3]![0], c0[2]![1] - c0[3]![1]),
      Math.hypot(c0[3]![0] - c0[0]![0], c0[3]![1] - c0[0]![1]),
      Math.hypot(c0[2]![0] - c0[1]![0], c0[2]![1] - c0[1]![1]),
    ) / 8;
    const cap = Math.max(2.5, Math.min(8, 0.3 * cell));
    const maxShift = opts.maxShiftFrac * diag;

    const cost = (c: Point[]): number => {
      const H = homographyFrom4(board, c);
      if (!H) return Infinity;
      let s = 0;
      for (const [u, v] of GRID_POINTS) {
        const wv = H[6] * u + H[7] * v + 1;
        if (wv <= 1e-6) return Infinity;
        s += sample((H[0] * u + H[1] * v + H[2]) / wv, (H[3] * u + H[4] * v + H[5]) / wv, cap);
      }
      return s / GRID_POINTS.length;
    };

    let cur = c0.map((p) => [p[0], p[1]] as Point);
    let curCost = cost(cur);
    const before = curCost;
    const dirs: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    for (let step = Math.max(1, 0.02 * diag); step >= 0.3; step *= 0.5) {
      for (let iter = 0; iter < 6; iter++) {
        let improved = false;
        for (let k = 0; k < 4; k++) {
          for (const [dx, dy] of dirs) {
            const nx = cur[k]![0] + dx * step;
            const ny = cur[k]![1] + dy * step;
            if (Math.hypot(nx - c0[k]![0], ny - c0[k]![1]) > maxShift) continue;
            const cand = cur.map((p, i) => (i === k ? ([nx, ny] as Point) : p));
            const cc = cost(cand);
            if (cc < curCost - 1e-4) {
              cur = cand;
              curCost = cc;
              improved = true;
            }
          }
        }
        if (!improved) break;
      }
    }
    const H = homographyFrom4(board, cur);
    return { hb: H ?? hb, before, after: curCost };
  }
}
