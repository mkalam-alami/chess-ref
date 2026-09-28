import { describe, expect, it } from 'vitest';
import {
  applyH, cross, fromHomogeneous, homographyFrom4, IDENTITY, intersect, invert3, lineThrough,
  mul3, pointLineDistance, solveLinear, type Mat3, type Point, type Quad,
} from '../src/geom/homography';
import { OneEuroFilter, PointsFilter } from '../src/geom/oneEuro';
import { stabiliseCorners } from '../src/geom/cornerOrder';

const close = (a: Point | null, b: Point, eps = 1e-6) => {
  expect(a).not.toBeNull();
  expect(a![0]).toBeCloseTo(b[0], -Math.log10(eps));
  expect(a![1]).toBeCloseTo(b[1], -Math.log10(eps));
};

describe('homography', () => {
  const H: Mat3 = [1.2, 0.1, 30, -0.2, 0.9, 12, 0.0005, 0.0002, 1];

  it('applies identity and a translation', () => {
    close(applyH(IDENTITY, [3, 4]), [3, 4]);
    close(applyH([1, 0, 5, 0, 1, -2, 0, 0, 1], [3, 4]), [8, 2]);
  });

  it('inverts', () => {
    const inv = invert3(H)!;
    const p: Point = [123, 45];
    close(applyH(inv, applyH(H, p)), p);
    const prod = mul3(H, inv);
    const scale = prod[8];
    prod.forEach((v, i) => expect(v / scale).toBeCloseTo(IDENTITY[i]!, 8));
  });

  it('returns null for singular matrices', () => {
    expect(invert3([1, 2, 3, 2, 4, 6, 0, 0, 1])).toBeNull();
  });

  it('solves linear systems', () => {
    const x = solveLinear([[2, 1], [1, 3]], [5, 10])!;
    expect(x[0]).toBeCloseTo(1);
    expect(x[1]).toBeCloseTo(3);
    expect(solveLinear([[1, 2], [2, 4]], [1, 2])).toBeNull();
  });

  it('computes H from 4 correspondences', () => {
    const src: Point[] = [[0, 0], [8, 0], [8, 8], [0, 8]];
    const dst = src.map((p) => applyH(H, p));
    const est = homographyFrom4(src, dst)!;
    for (const p of [[1, 2], [4, 4], [7.5, 0.3]] as Point[]) close(applyH(est, p), applyH(H, p), 1e-5);
  });

  it('returns null for degenerate correspondences', () => {
    const src: Point[] = [[0, 0], [1, 1], [2, 2], [3, 3]];
    expect(homographyFrom4(src, src)).toBeNull();
  });
});

describe('homogeneous helpers', () => {
  it('cross product', () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
  });

  it('line through two points and intersection', () => {
    const l1 = lineThrough([0, 0], [10, 10]);
    const l2 = lineThrough([0, 10], [10, 0]);
    close(fromHomogeneous(intersect(l1, l2)), [5, 5]);
    expect(Math.abs(pointLineDistance(l1, [10, 0]))).toBeCloseTo(10 * Math.SQRT1_2, 6);
  });

  it('parallel lines meet at infinity', () => {
    const p = intersect(lineThrough([0, 0], [1, 0]), lineThrough([0, 1], [1, 1]));
    expect(fromHomogeneous(p)).toBeNull();
  });
});

describe('One-Euro filter', () => {
  it('passes the first sample through', () => {
    expect(new OneEuroFilter().filter(5, 0)).toBe(5);
  });

  it('attenuates jitter around a constant', () => {
    const f = new OneEuroFilter(1, 0, 1);
    let maxDev = 0;
    for (let i = 0; i < 100; i++) {
      const y = f.filter(10 + (i % 2 ? 1 : -1), i / 30);
      if (i > 10) maxDev = Math.max(maxDev, Math.abs(y - 10));
    }
    expect(maxDev).toBeLessThan(0.5);
  });

  it('follows fast motion with small lag when beta is high', () => {
    const f = new OneEuroFilter(1, 5, 1);
    let y = 0;
    for (let i = 0; i <= 30; i++) y = f.filter(i * 10, i / 30);
    expect(300 - y).toBeLessThan(30);
  });

  it('filters points independently', () => {
    const pf = new PointsFilter(4);
    const pts: Point[] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    expect(pf.filter(pts, 0)).toEqual(pts);
    expect(pf.filter(pts, 0.03)).toEqual(pts);
  });
});

describe('stabiliseCorners', () => {
  const prev: Quad = [[0, 0], [100, 0], [100, 100], [0, 100]];

  it('keeps identical order', () => {
    expect(stabiliseCorners(prev, prev)).toEqual(prev);
  });

  it('undoes a cyclic rotation', () => {
    const rotated: Quad = [prev[2], prev[3], prev[0], prev[1]];
    const noisy = rotated.map(([x, y]) => [x + 2, y - 1] as Point) as Quad;
    const out = stabiliseCorners(prev, noisy);
    out.forEach((p, i) => close(p, [prev[i]![0] + 2, prev[i]![1] - 1]));
  });

  it('undoes a reversal', () => {
    const rev: Quad = [prev[0], prev[3], prev[2], prev[1]];
    expect(stabiliseCorners(prev, rev)).toEqual(prev);
  });

  it('undoes reversal plus rotation', () => {
    const rev: Quad = [prev[1], prev[0], prev[3], prev[2]];
    expect(stabiliseCorners(prev, rev)).toEqual(prev);
  });
});
