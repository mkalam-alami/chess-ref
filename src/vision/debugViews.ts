import { applyH, mul3, type Mat3, type Point } from '../geom/homography';
import type { DetectDebug } from './detector';

type Ctx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

const FAMILY_COLORS = ['rgba(160,160,160,0.7)', '#2f80ff', '#ffd400'];

/** Merged segments coloured by family (blue / yellow, grey = unassigned) and the VPs that fall inside the frame. */
export function drawLines(ctx: Ctx, d: DetectDebug): void {
  ctx.lineWidth = 2;
  d.segments.forEach((s, i) => {
    ctx.strokeStyle = FAMILY_COLORS[d.family[i] ?? 0]!;
    ctx.beginPath();
    ctx.moveTo(s.x1, s.y1);
    ctx.lineTo(s.x2, s.y2);
    ctx.stroke();
  });
  if (d.vps) {
    d.vps.forEach((v, i) => {
      if (Math.abs(v[2]) < 1e-9) return;
      const x = v[0] / v[2];
      const y = v[1] / v[2];
      if (x < 0 || y < 0 || x > d.width || y > d.height) return;
      ctx.fillStyle = FAMILY_COLORS[i + 1]!;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

const cellQuad = (hb: Mat3, i: number, j: number): Point[] => [
  applyH(hb, [i, j]), applyH(hb, [i + 1, j]), applyH(hb, [i + 1, j + 1]), applyH(hb, [i, j + 1]),
];

/** Per-cell alternation agreement heat map (red = disagrees, green = agrees) over the frame, with the quad. */
export function drawVerify(ctx: Ctx, d: DetectDebug): void {
  const c = d.best;
  if (!c) return;
  const hb = d.polished ?? c.hb;
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 8; i++) {
      const a = c.verify.cells[j * 8 + i]!;
      ctx.fillStyle = `rgba(${Math.round(255 * (1 - a))},${Math.round(200 * a)},40,0.5)`;
      const q = cellQuad(hb, i, j);
      ctx.beginPath();
      ctx.moveTo(q[0]![0], q[0]![1]);
      for (let k = 1; k < 4; k++) ctx.lineTo(q[k]![0], q[k]![1]);
      ctx.closePath();
      ctx.fill();
    }
  }
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const q = [applyH(hb, [0, 0]), applyH(hb, [8, 0]), applyH(hb, [8, 8]), applyH(hb, [0, 8])];
  ctx.moveTo(q[0]![0], q[0]![1]);
  for (let k = 1; k < 4; k++) ctx.lineTo(q[k]![0], q[k]![1]);
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = '16px monospace';
  ctx.fillText(`score ${c.verify.score.toFixed(2)} alt ${c.verify.alternation.toFixed(2)} ring x${c.verify.ringFactor}`, 8, 20);
}

/**
 * Homography (frame pixels -> view pixels) that shows the rectified board with a one-cell margin, plus the
 * fitted grid lines in view pixels. Null without a candidate.
 */
export function rectifiedView(d: DetectDebug, w: number, h: number): { h: Mat3; lines: Array<[Point, Point]> } | null {
  const c = d.best;
  if (!c || !d.rect) return null;
  const f = d.frame;
  const toNorm: Mat3 = [1 / f.s, 0, -f.cx / f.s, 0, 1 / f.s, -f.cy / f.s, 0, 0, 1];
  const sx = w / (10 * c.hx.period);
  const sy = h / (10 * c.hy.period);
  const toView: Mat3 = [sx, 0, -(c.hx.start - c.hx.period) * sx, 0, sy, -(c.hy.start - c.hy.period) * sy, 0, 0, 1];
  const H = mul3(toView, mul3(d.rect.H, toNorm));
  const lines: Array<[Point, Point]> = [];
  for (let k = 0; k <= 8; k++) {
    lines.push([[(1 + k) * (w / 10), h / 10], [(1 + k) * (w / 10), (9 * h) / 10]]);
    lines.push([[w / 10, (1 + k) * (h / 10)], [(9 * w) / 10, (1 + k) * (h / 10)]]);
  }
  return { h: H, lines };
}
