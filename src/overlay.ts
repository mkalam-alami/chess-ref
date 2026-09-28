import type { Point } from './geom/homography';

export interface CoverMap {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/** Mapping for an element showing a frame with `object-fit: cover` in a viewport. */
export function coverMap(frameW: number, frameH: number, viewW: number, viewH: number): CoverMap {
  const scale = Math.max(viewW / frameW, viewH / frameH);
  return {
    scale,
    offsetX: (viewW - frameW * scale) / 2,
    offsetY: (viewH - frameH * scale) / 2,
  };
}

/** Frame coordinates -> viewport (CSS pixel) coordinates. */
export function frameToScreen(p: Point, m: CoverMap): Point {
  return [p[0] * m.scale + m.offsetX, p[1] * m.scale + m.offsetY];
}

export const HOLD_MS = 500;
export const FADE_MS = 300;

/** Full alpha for `hold` ms after the last detection, then a linear fade over `fade` ms. */
export function quadAlpha(elapsedMs: number, hold = HOLD_MS, fade = FADE_MS): number {
  if (elapsedMs <= hold) return 1;
  if (elapsedMs >= hold + fade) return 0;
  return 1 - (elapsedMs - hold) / fade;
}

export function drawQuad(
  ctx: CanvasRenderingContext2D,
  pts: readonly Point[],
  alpha: number,
  color = '#00e676',
): void {
  if (alpha <= 0 || pts.length < 3) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineJoin = 'round';
  ctx.lineWidth = 4;
  ctx.strokeStyle = color;
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = 4;
  ctx.beginPath();
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = color;
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

interface HeldQuad {
  points: Point[];
  frameW: number;
  frameH: number;
  time: number;
}

/** Fullscreen canvas above the video: draws the held quad and an optional debug image. */
export class Overlay {
  private ctx: CanvasRenderingContext2D;
  private quad: HeldQuad | null = null;
  private debugImage: { image: ImageBitmap; alpha: number } | null = null;
  private viewW = 0;
  private viewH = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    canvas.dataset.debugImage = 'off';
    this.resize();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('orientationchange', () => this.resize());
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    this.viewW = window.innerWidth;
    this.viewH = window.innerHeight;
    this.canvas.width = Math.round(this.viewW * dpr);
    this.canvas.height = Math.round(this.viewH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Points are in the coordinates of a frame of size frameW x frameH. */
  setQuad(points: Point[], frameW: number, frameH: number, timeMs: number): void {
    this.quad = { points, frameW, frameH, time: timeMs };
  }

  /** Time since the last quad was set, or Infinity. */
  msSinceQuad(nowMs: number): number {
    return this.quad ? nowMs - this.quad.time : Infinity;
  }

  setDebugImage(image: ImageBitmap | null, alpha = 0.6): void {
    this.debugImage?.image.close();
    this.debugImage = image ? { image, alpha } : null;
    this.canvas.dataset.debugImage = image ? 'on' : 'off';
  }

  draw(nowMs: number, frameW: number, frameH: number): void {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.viewW, this.viewH);
    if (frameW <= 0 || frameH <= 0) return;
    if (this.debugImage) {
      const m = coverMap(frameW, frameH, this.viewW, this.viewH);
      ctx.save();
      ctx.globalAlpha = this.debugImage.alpha;
      ctx.drawImage(this.debugImage.image, m.offsetX, m.offsetY, frameW * m.scale, frameH * m.scale);
      ctx.restore();
    }
    const q = this.quad;
    if (!q) return;
    const alpha = quadAlpha(nowMs - q.time);
    if (alpha <= 0) return;
    const m = coverMap(q.frameW, q.frameH, this.viewW, this.viewH);
    drawQuad(ctx, q.points.map((p) => frameToScreen(p, m)), alpha);
  }
}
