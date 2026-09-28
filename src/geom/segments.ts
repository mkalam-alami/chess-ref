import type { Vec3 } from './homography';

/** A line segment with its homogeneous line (unit normal: a^2 + b^2 = 1). */
export interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Extent length. */
  len: number;
  /** Evidence weight (sum of merged member lengths, at most `len`). */
  weight: number;
  mx: number;
  my: number;
  /** Unit direction. */
  dx: number;
  dy: number;
  line: Vec3;
}

export function makeSegment(x1: number, y1: number, x2: number, y2: number, weight?: number): Segment {
  const len = Math.hypot(x2 - x1, y2 - y1);
  const dx = len > 0 ? (x2 - x1) / len : 1;
  const dy = len > 0 ? (y2 - y1) / len : 0;
  // Normal (a, b) = (dy, -dx); c makes the line pass through the midpoint.
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const a = dy;
  const b = -dx;
  return { x1, y1, x2, y2, len, weight: weight ?? len, mx, my, dx, dy, line: [a, b, -(a * mx + b * my)] };
}

/** Maps pixel coordinates to roughly [-1, 1]: x_n = (x - cx) / s. */
export interface NormFrame {
  cx: number;
  cy: number;
  s: number;
}

export function normFrame(width: number, height: number): NormFrame {
  return { cx: width / 2, cy: height / 2, s: Math.max(width, height) / 2 };
}

export function segmentsToNormalised(segs: readonly Segment[], f: NormFrame): Segment[] {
  return segs.map((g) =>
    makeSegment((g.x1 - f.cx) / f.s, (g.y1 - f.cy) / f.s, (g.x2 - f.cx) / f.s, (g.y2 - f.cy) / f.s, g.weight / f.s),
  );
}
