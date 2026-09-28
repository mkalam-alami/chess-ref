import type { Point } from './homography';

/** One-Euro filter (Casiez et al. 2012) for a scalar signal. Times in seconds. */
export class OneEuroFilter {
  private xPrev: number | null = null;
  private dxPrev = 0;
  private tPrev = 0;

  constructor(
    private minCutoff = 1.0,
    private beta = 0.02,
    private dCutoff = 1.0,
  ) {}

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x: number, t: number): number {
    if (this.xPrev === null) {
      this.xPrev = x;
      this.tPrev = t;
      return x;
    }
    const dt = Math.max(t - this.tPrev, 1e-6);
    const dx = (x - this.xPrev) / dt;
    const dxHat = this.dxPrev + OneEuroFilter.alpha(this.dCutoff, dt) * (dx - this.dxPrev);
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const xHat = this.xPrev + OneEuroFilter.alpha(cutoff, dt) * (x - this.xPrev);
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    this.tPrev = t;
    return xHat;
  }

  reset(): void {
    this.xPrev = null;
    this.dxPrev = 0;
  }
}

/** Smooths N 2D points independently (x and y each get a filter). */
export class PointsFilter {
  private filters: OneEuroFilter[][];

  constructor(count = 4, minCutoff = 1.0, beta = 0.02, dCutoff = 1.0) {
    this.filters = Array.from({ length: count }, () => [
      new OneEuroFilter(minCutoff, beta, dCutoff),
      new OneEuroFilter(minCutoff, beta, dCutoff),
    ]);
  }

  filter(points: readonly Point[], t: number): Point[] {
    return points.map((p, i) => {
      const [fx, fy] = this.filters[i]!;
      return [fx!.filter(p[0], t), fy!.filter(p[1], t)] as Point;
    });
  }

  reset(): void {
    for (const [fx, fy] of this.filters) {
      fx!.reset();
      fy!.reset();
    }
  }
}
