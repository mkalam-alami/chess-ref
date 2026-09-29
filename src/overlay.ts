import { applyH, homographyFrom4, type Point } from './geom/homography';
import { BOARD_CORNERS, signedArea } from './geom/cornerOrder';
import { OCC_BLACK, OCC_WHITE } from './worker/protocol';
import type { PieceCode } from './game/types';

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

export const DOT_EMPTY = '#000000';
export const DOT_WHITE = '#ffffff';
export const DOT_BLACK = '#00c853';
export const DOT_OUTLINE = 'rgba(0,0,0,0.85)';
/** Probability of the committed class below which a dot gets the low-confidence ring. */
export const DOT_LOW_CONF = 0.6;
/** Low-confidence ring: red dashes over a dark halo, legible on light and dark squares. */
export const DOT_RING = '#ff1744';
export const DOT_RING_HALO = 'rgba(0,0,0,0.75)';
/** Fill opacity at probability 0; it rises linearly to 1 at probability 1. */
export const DOT_MIN_ALPHA = 0.35;

/** Fill opacity for a committed-class probability (missing / non-finite -> full confidence). */
export function dotAlpha(p: number | undefined): number {
  if (p === undefined || !Number.isFinite(p)) return 1;
  return DOT_MIN_ALPHA + (1 - DOT_MIN_ALPHA) * Math.max(0, Math.min(1, p));
}

export interface OccDot {
  x: number;
  y: number;
  r: number;
  fill: string;
  /** Whether the dot gets the thin dark outline (piece dots, so white shows on light squares). */
  outline: boolean;
  /** Probability of the committed class (1 when unknown). */
  conf: number;
  /** Fill / outline opacity encoding `conf` (see dotAlpha). */
  alpha: number;
  /** Whether `conf` is below DOT_LOW_CONF: drawn with a dashed red ring. */
  low: boolean;
}

/**
 * One dot per board cell for an occupancy grid (index j * 8 + i, see ResultMessage.occupancy), placed at the
 * projected centre of cell (i, j) under the homography mapping board (0,0), (8,0), (8,8), (0,8) to `corners`.
 * The radius scales with the projected cell size (sqrt of its area): small black dots for empty cells, larger
 * white / green dots for white / black pieces. `prob` (ResultMessage.occupancyProb, same indexing) encodes
 * confidence: opacity 0.35 + 0.65 p, and a low-confidence ring below DOT_LOW_CONF; without it every dot is
 * drawn at full confidence. Returns [] for a degenerate quad or a grid of the wrong size.
 */
export function occupancyDots(
  corners: readonly Point[],
  grid: ArrayLike<number>,
  prob: ArrayLike<number> | null = null,
): OccDot[] {
  if (corners.length !== 4 || grid.length !== 64) return [];
  const h = homographyFrom4(BOARD_CORNERS, corners);
  if (!h) return [];
  const probs = prob && prob.length === 64 ? prob : null;
  const dots: OccDot[] = [];
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 8; i++) {
      const v = grid[j * 8 + i]!;
      const cell = [applyH(h, [i, j]), applyH(h, [i + 1, j]), applyH(h, [i + 1, j + 1]), applyH(h, [i, j + 1])];
      const size = Math.sqrt(Math.abs(signedArea(cell)));
      if (!Number.isFinite(size)) continue;
      const [x, y] = applyH(h, [i + 0.5, j + 0.5]);
      const piece = v === OCC_WHITE || v === OCC_BLACK;
      const p = probs ? probs[j * 8 + i]! : NaN;
      const known = Number.isFinite(p);
      const conf = known ? Math.max(0, Math.min(1, p)) : 1;
      dots.push({
        x,
        y,
        r: Math.max(piece ? 2.5 : 1.5, size * (piece ? 0.24 : 0.09)),
        fill: v === OCC_WHITE ? DOT_WHITE : v === OCC_BLACK ? DOT_BLACK : DOT_EMPTY,
        outline: piece,
        conf,
        alpha: known ? dotAlpha(conf) : 1,
        low: known && conf < DOT_LOW_CONF,
      });
    }
  }
  return dots;
}

export function drawDots(ctx: CanvasRenderingContext2D, dots: readonly OccDot[], alpha: number): void {
  if (alpha <= 0 || dots.length === 0) return;
  ctx.save();
  for (const d of dots) {
    ctx.globalAlpha = alpha * d.alpha;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = DOT_OUTLINE;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
    ctx.fillStyle = d.fill;
    ctx.fill();
    if (d.outline) ctx.stroke();
    if (d.low) {
      // Ring at full (quad) opacity so it stays visible even though the dot itself is faint.
      ctx.globalAlpha = alpha;
      const rr = d.r + Math.max(2.5, d.r * 0.35);
      ctx.beginPath();
      ctx.arc(d.x, d.y, rr, 0, Math.PI * 2);
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = DOT_RING_HALO;
      ctx.stroke();
      ctx.lineWidth = 2;
      ctx.strokeStyle = DOT_RING;
      ctx.setLineDash([3, 2.5]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  ctx.restore();
}

/** Icon size relative to the projected cell size. */
export const PIECE_SIZE = 0.9;
/** Where the icon's bottom edge sits below the cell centre, in projected cell sizes (anchors it toward the base). */
export const PIECE_BASE = 0.32;
/** Icon opacity at full quad alpha (slightly translucent so the real board shows through). */
export const PIECE_ALPHA = 0.85;
export const LAST_MOVE_TINT = 'rgba(255, 214, 0, 0.38)';

export interface PieceMark {
  /** Chess square (a1 = 0). */
  sq: number;
  code: PieceCode;
  /** Top-left corner and side of the screen-upright icon box. */
  x: number;
  y: number;
  size: number;
}

export interface GameMarks {
  pieces: PieceMark[];
  /** Screen polygons of the last move's from / to cells. */
  tint: Point[][];
}

/** Board cell (i, j) of a chess square under the orientation map (ResultMessage.orientation), or null if invalid. */
function squareCell(orientation: ArrayLike<number>, sq: number): [number, number] | null {
  const cell = orientation[sq];
  if (cell === undefined || !(cell >= 0 && cell < 64)) return null;
  return [cell % 8, Math.floor(cell / 8)];
}

/**
 * Screen placement of the game's pieces on the quad `corners` (board (0,0), (8,0), (8,8), (0,8)): one screen-upright
 * icon box per occupied square, sized by the projected cell size (as in occupancyDots) and anchored with its bottom
 * edge slightly below the cell centre, plus the last move's cells as polygons. `pieces` is in chess square order and
 * square sq lies on board cell orientation[sq]. Pieces are sorted far-to-near (by screen y) so nearer icons overlap
 * farther ones. Returns empty marks for a degenerate quad or malformed arrays.
 */
export function gameMarks(
  corners: readonly Point[],
  orientation: ArrayLike<number>,
  pieces: ReadonlyArray<PieceCode | null>,
  lastMove: { from: number; to: number } | null = null,
): GameMarks {
  const out: GameMarks = { pieces: [], tint: [] };
  if (corners.length !== 4 || orientation.length !== 64 || pieces.length !== 64) return out;
  const h = homographyFrom4(BOARD_CORNERS, corners);
  if (!h) return out;
  const cellPoly = ([i, j]: [number, number]): Point[] => [applyH(h, [i, j]), applyH(h, [i + 1, j]), applyH(h, [i + 1, j + 1]), applyH(h, [i, j + 1])];
  for (let sq = 0; sq < 64; sq++) {
    const code = pieces[sq];
    if (!code) continue;
    const ij = squareCell(orientation, sq);
    if (!ij) continue;
    const cell = Math.sqrt(Math.abs(signedArea(cellPoly(ij))));
    const [cx, cy] = applyH(h, [ij[0] + 0.5, ij[1] + 0.5]);
    if (!Number.isFinite(cell) || !Number.isFinite(cx) || !Number.isFinite(cy) || cell <= 0) continue;
    const size = cell * PIECE_SIZE;
    out.pieces.push({ sq, code, x: cx - size / 2, y: cy + cell * PIECE_BASE - size, size });
  }
  out.pieces.sort((a, b) => a.y - b.y);
  if (lastMove) {
    for (const sq of [lastMove.from, lastMove.to]) {
      const ij = squareCell(orientation, sq);
      if (!ij) continue;
      const poly = cellPoly(ij);
      if (poly.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))) out.tint.push(poly);
    }
  }
  return out;
}

/** Icon source for a piece, or null to draw its glyph. */
export type PieceImages = (code: PieceCode) => CanvasImageSource | null;

const GLYPHS: Record<string, string> = { P: '\u265F', N: '\u265E', B: '\u265D', R: '\u265C', Q: '\u265B', K: '\u265A' };

export function drawGameMarks(ctx: CanvasRenderingContext2D, marks: GameMarks, alpha: number, images: PieceImages): void {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = LAST_MOVE_TINT;
  for (const poly of marks.tint) {
    ctx.beginPath();
    poly.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
    ctx.closePath();
    ctx.fill();
  }
  ctx.globalAlpha = alpha * PIECE_ALPHA;
  ctx.shadowColor = 'rgba(0,0,0,0.5)';
  ctx.shadowBlur = 3;
  for (const m of marks.pieces) {
    const img = images(m.code);
    if (img) {
      ctx.drawImage(img, m.x, m.y, m.size, m.size);
      continue;
    }
    // Fallback until the icon is loaded (or when it is not vendored): a filled glyph with a contrasting outline.
    ctx.font = `${Math.round(m.size * 0.9)}px serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const g = GLYPHS[m.code[1]!] ?? '?';
    const x = m.x + m.size / 2;
    const y = m.y + m.size;
    ctx.lineWidth = Math.max(1, m.size * 0.04);
    ctx.strokeStyle = m.code[0] === 'w' ? '#000' : '#fff';
    ctx.strokeText(g, x, y);
    ctx.fillStyle = m.code[0] === 'w' ? '#fff' : '#000';
    ctx.fillText(g, x, y);
  }
  ctx.restore();
}

/** What the overlay shows of the game in play mode: the engine position, or null before lock-in. */
export interface OverlayGame {
  pieces: ReadonlyArray<PieceCode | null>;
  lastMove: { from: number; to: number } | null;
}

interface HeldQuad {
  points: Point[];
  frameW: number;
  frameH: number;
  time: number;
  /** Occupancy grid shown with the quad (corners[k] <-> board corner k), or null. */
  grid: Uint8Array | null;
  /** Committed-class probability per cell (ResultMessage.occupancyProb), or null for full-confidence styling. */
  prob: Float32Array | null;
  /** Chess square -> board cell (ResultMessage.orientation), or null while unknown. */
  orientation: Uint8Array | null;
}

/**
 * Fullscreen canvas above the video: draws the held quad and an optional debug image. In debug mode the quad carries
 * the occupancy dots; in play mode it carries the game's pieces (once a game is playing and the orientation is known).
 */
export class Overlay {
  private ctx: CanvasRenderingContext2D;
  private quad: HeldQuad | null = null;
  private playMode = false;
  private game: OverlayGame | null = null;
  private images: PieceImages = () => null;
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

  /**
   * Points are in the coordinates of a frame of size frameW x frameH; points[k] must be board corner k
   * ((0,0), (8,0), (8,8), (0,8)) for the optional occupancy grid to line up.
   */
  setQuad(
    points: Point[],
    frameW: number,
    frameH: number,
    timeMs: number,
    grid: Uint8Array | null = null,
    prob: Float32Array | null = null,
    orientation: Uint8Array | null = null,
  ): void {
    this.quad = { points, frameW, frameH, time: timeMs, grid, prob, orientation };
  }

  /** Play mode (debug panel hidden): pieces instead of dots. */
  setPlayMode(on: boolean): void {
    this.playMode = on;
  }

  /** The game position to draw in play mode, or null (then only the quad is drawn). */
  setGame(game: OverlayGame | null): void {
    this.game = game;
  }

  setPieceImages(images: PieceImages): void {
    this.images = images;
  }

  /** Drops the held quad and occupancy grid (e.g. when the source changes). */
  clearQuad(): void {
    this.quad = null;
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
    const screen = q.points.map((p) => frameToScreen(p, m));
    drawQuad(ctx, screen, alpha);
    if (!this.playMode) {
      if (q.grid) drawDots(ctx, occupancyDots(screen, q.grid, q.prob), alpha);
    } else if (this.game && q.orientation) {
      drawGameMarks(ctx, gameMarks(screen, q.orientation, this.game.pieces, this.game.lastMove), alpha, this.images);
    }
  }
}
