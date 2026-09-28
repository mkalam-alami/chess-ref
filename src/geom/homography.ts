export type Point = [number, number];
export type Vec3 = [number, number, number];
/** Row-major 3x3 matrix. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];
export type Quad = [Point, Point, Point, Point];

export const IDENTITY: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function applyH(h: Mat3, [x, y]: Point): Point {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

/** Returns null when the matrix is singular. */
export function invert3(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  const k = 1 / det;
  return [
    A * k, -(b * i - c * h) * k, (b * f - c * e) * k,
    B * k, (a * i - c * g) * k, -(a * f - c * d) * k,
    C * k, -(a * h - b * g) * k, (a * e - b * d) * k,
  ];
}

export function mul3(p: Mat3, q: Mat3): Mat3 {
  const r = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) r[i * 3 + j]! += p[i * 3 + k]! * q[k * 3 + j]!;
  return r as Mat3;
}

/** Solves A x = b by Gaussian elimination with partial pivoting; null if singular. */
export function solveLinear(a: number[][], b: number[]): number[] | null {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r]![col]!) > Math.abs(m[piv]![col]!)) piv = r;
    if (Math.abs(m[piv]![col]!) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv]!, m[col]!];
    for (let r = col + 1; r < n; r++) {
      const f = m[r]![col]! / m[col]![col]!;
      for (let c = col; c <= n; c++) m[r]![c]! -= f * m[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r]![n]!;
    for (let c = r + 1; c < n; c++) s -= m[r]![c]! * x[c]!;
    x[r] = s / m[r]![r]!;
  }
  return x;
}

/** Homography mapping src[i] -> dst[i] from 4 correspondences (h33 fixed to 1). */
export function homographyFrom4(src: readonly Point[], dst: readonly Point[]): Mat3 | null {
  if (src.length !== 4 || dst.length !== 4) return null;
  const a: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i]!;
    const [u, v] = dst[i]!;
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const s = solveLinear(a, b);
  return s ? [s[0]!, s[1]!, s[2]!, s[3]!, s[4]!, s[5]!, s[6]!, s[7]!, 1] : null;
}

export function toHomogeneous([x, y]: Point): Vec3 {
  return [x, y, 1];
}

/** Returns null for points at infinity. */
export function fromHomogeneous([x, y, w]: Vec3, eps = 1e-12): Point | null {
  return Math.abs(w) < eps ? null : [x / w, y / w];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function lineThrough(p: Point, q: Point): Vec3 {
  return cross(toHomogeneous(p), toHomogeneous(q));
}

/** Intersection of two homogeneous lines; the result may be at infinity (w ~ 0). */
export function intersect(l1: Vec3, l2: Vec3): Vec3 {
  return cross(l1, l2);
}

export function normalize3(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n === 0 ? v : [v[0] / n, v[1] / n, v[2] / n];
}

/** Signed distance from a point to a homogeneous line. */
export function pointLineDistance(l: Vec3, [x, y]: Point): number {
  return (l[0] * x + l[1] * y + l[2]) / Math.hypot(l[0], l[1]);
}
