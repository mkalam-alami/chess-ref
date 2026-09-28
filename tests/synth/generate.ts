import { applyH, homographyFrom4, invert3, type Mat3, type Point } from '../../src/geom/homography';
import { mulberry32 } from '../../src/geom/vanishing';
import type { CV } from '../../src/vision/preprocess';

export const W = 640;
export const H = 480;

export interface SampleSpec {
  elev: 'overhead' | 'oblique';
  /** 'printed' = black/light-grey print inside a thin black coordinate ring and a red-brown wooden frame. */
  palette: 'wood' | 'vinyl' | 'printed';
  /** 'outer' = occluders concentrated in ranks 1-2 and 7-8, like a starting position. */
  pieces: 'none' | 'some' | 'outer';
}

export interface Sample {
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  /** Ground-truth corners of the 8x8 area (null for negatives). */
  corners: [Point, Point, Point, Point] | null;
  label: string;
  seed: number;
  meta: Record<string, number | string>;
}

type RGB = [number, number, number];
const hex = (s: string): RGB => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const luma = (c: RGB) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
const desat = (c: RGB, s: number): RGB => mix([luma(c), luma(c), luma(c)], c, s);

type Rng = () => number;

function hash2(x: number, y: number, seed: number): number {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(seed, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function vnoise(x: number, y: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const fx = x - xi;
  const fy = y - yi;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const a = hash2(xi, yi, seed);
  const b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed);
  const d = hash2(xi + 1, yi + 1, seed);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

const TEX = 256;
/** Mirror-repeat index (no visible seams). */
const mir = (i: number): number => {
  const m = ((i % (2 * TEX)) + 2 * TEX) % (2 * TEX);
  return m < TEX ? m : 2 * TEX - 1 - m;
};
/** Tileable-ish grain texture in [-1, 1]; anisotropic for wood. */
function makeGrain(rng: Rng, wood: boolean): Float32Array {
  const seed = Math.floor(rng() * 1e6);
  const t = new Float32Array(TEX * TEX);
  const stretch = wood ? 6 : 1;
  for (let y = 0; y < TEX; y++) {
    for (let x = 0; x < TEX; x++) {
      const u = x / 12;
      const v = y / (12 * stretch);
      let n = 0;
      let amp = 1;
      let tot = 0;
      for (let o = 0; o < 3; o++) {
        n += amp * vnoise(u * (1 << o), v * (1 << o), seed + o);
        tot += amp;
        amp *= 0.5;
      }
      n /= tot;
      let val = (n - 0.5) * 2;
      if (wood) val = 0.65 * Math.sin(v * 9 + 6 * n) + 0.35 * val + 0.15 * (hash2(x, y, seed) - 0.5);
      else val = val * 0.5 + 0.3 * (hash2(x, y, seed) - 0.5);
      t[y * TEX + x] = val;
    }
  }
  return t;
}

function gauss(rng: Rng): number {
  const u = Math.max(1e-9, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/** Fills a convex polygon (pixel-centre rule) with a flat colour or a per-pixel shader. */
function fillConvex(img: Float32Array, pts: readonly Point[], color: RGB, shade?: (x: number, y: number) => number, alpha = 1): void {
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const p of pts) {
    ymin = Math.min(ymin, p[1]);
    ymax = Math.max(ymax, p[1]);
  }
  const y0 = Math.max(0, Math.ceil(ymin - 0.5));
  const y1 = Math.min(H - 1, Math.floor(ymax - 0.5));
  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5;
    let xl = Infinity;
    let xr = -Infinity;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i]!;
      const b = pts[(i + 1) % pts.length]!;
      if ((a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc)) {
        const x = a[0] + ((yc - a[1]) * (b[0] - a[0])) / (b[1] - a[1]);
        xl = Math.min(xl, x);
        xr = Math.max(xr, x);
      }
    }
    if (xl > xr) continue;
    const x0 = Math.max(0, Math.ceil(xl - 0.5));
    const x1 = Math.min(W - 1, Math.floor(xr - 0.5));
    for (let x = x0; x <= x1; x++) {
      const k = shade ? shade(x, y) : 1;
      const o = (y * W + x) * 3;
      img[o] = img[o]! * (1 - alpha) + color[0] * k * alpha;
      img[o + 1] = img[o + 1]! * (1 - alpha) + color[1] * k * alpha;
      img[o + 2] = img[o + 2]! * (1 - alpha) + color[2] * k * alpha;
    }
  }
}

function convexHull(pts: Point[]): Point[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo: Point[] = [];
  for (const q of p) {
    while (lo.length >= 2 && cross(lo[lo.length - 2]!, lo[lo.length - 1]!, q) <= 0) lo.pop();
    lo.push(q);
  }
  const up: Point[] = [];
  for (const q of [...p].reverse()) {
    while (up.length >= 2 && cross(up[up.length - 2]!, up[up.length - 1]!, q) <= 0) up.pop();
    up.push(q);
  }
  lo.pop();
  up.pop();
  return lo.concat(up);
}

interface Camera {
  project(x: number, y: number, z: number): [number, number, number];
  hBoard: Mat3;
  dist: number;
}

function makeCamera(az: number, el: number, roll: number, fpx: number, D: number, cx: number, cy: number): Camera {
  const C: [number, number, number] = [D * Math.cos(el) * Math.cos(az), D * Math.cos(el) * Math.sin(az), D * Math.sin(el)];
  const f: [number, number, number] = [-Math.cos(el) * Math.cos(az), -Math.cos(el) * Math.sin(az), -Math.sin(el)];
  const r0: [number, number, number] = [-Math.sin(az), Math.cos(az), 0];
  const d0: [number, number, number] = [
    f[1] * r0[2] - f[2] * r0[1],
    f[2] * r0[0] - f[0] * r0[2],
    f[0] * r0[1] - f[1] * r0[0],
  ];
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const r = [0, 1, 2].map((i) => r0[i]! * cr + d0[i]! * sr);
  const d = [0, 1, 2].map((i) => -r0[i]! * sr + d0[i]! * cr);
  const project = (x: number, y: number, z: number): [number, number, number] => {
    const rx = x - C[0];
    const ry = y - C[1];
    const rz = z - C[2];
    const zc = rx * f[0] + ry * f[1] + rz * f[2];
    const xc = rx * r[0]! + ry * r[1]! + rz * r[2]!;
    const yc = rx * d[0]! + ry * d[1]! + rz * d[2]!;
    return [(fpx * xc) / zc + cx, (fpx * yc) / zc + cy, zc];
  };
  const src: Point[] = [[-4, -4], [4, -4], [4, 4], [-4, 4]];
  const dst = src.map(([x, y]) => project(x, y, 0).slice(0, 2) as Point);
  return { project, hBoard: homographyFrom4(src, dst)!, dist: D };
}

function jitterColor(rng: Rng, c: RGB, amt: number): RGB {
  return [c[0] * (1 + (rng() - 0.5) * amt), c[1] * (1 + (rng() - 0.5) * amt), c[2] * (1 + (rng() - 0.5) * amt)];
}

function drawBackground(rng: Rng, img: Float32Array, mode: 'clutter' | 'wood' | 'plain' | 'greywood' = 'clutter'): void {
  const tables: RGB[] = [hex('#a07040'), hex('#8a8a86'), hex('#d8d4c8'), hex('#3a3835'), hex('#5a6070'), hex('#b09a80'), hex('#6b4a2e')];
  const base = jitterColor(rng, mode === 'greywood' ? hex('#8c8880') : tables[Math.floor(rng() * tables.length)]!, 0.2);
  const gx = (rng() - 0.5) * 0.4;
  const gy = (rng() - 0.5) * 0.4;
  const blobs = Array.from({ length: 4 }, () => ({ x: rng() * W, y: rng() * H, r: 100 + rng() * 250, a: (rng() - 0.5) * 0.4 }));
  const grain = makeGrain(rng, mode === 'wood' || mode === 'greywood' || rng() < 0.4);
  const ang = rng() * Math.PI;
  const ca = Math.cos(ang);
  const sa = Math.sin(ang);
  const grainAmp = mode === 'wood' || mode === 'greywood' ? 0.13 : 0.05 + rng() * 0.05;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let k = 1 + gx * (x / W - 0.5) + gy * (y / H - 0.5);
      for (const b of blobs) k += b.a * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.r * b.r));
      const u = (x * ca + y * sa) * 0.9;
      const v = (-x * sa + y * ca) * 0.9;
      k *= 1 + grainAmp * grain[mir(Math.floor(v)) * TEX + mir(Math.floor(u))]!;
      const o = (y * W + x) * 3;
      img[o] = base[0] * k;
      img[o + 1] = base[1] * k;
      img[o + 2] = base[2] * k;
    }
  }
  if (mode === 'plain') return;
  // Table edges: half-planes with another tone.
  const edges = rng() < 0.55 ? 1 + Math.floor(rng() * 2) : 0;
  for (let e = 0; e < edges; e++) {
    const a = rng() * Math.PI * 2;
    const nx = Math.cos(a);
    const ny = Math.sin(a);
    const off = (rng() - 0.5) * 300;
    const tone = mix(base, rng() < 0.5 ? [30, 30, 30] : [230, 225, 215], 0.2 + rng() * 0.4);
    const big = 4000;
    const dx = -ny;
    const dy = nx;
    const cx = W / 2 + nx * off;
    const cy = H / 2 + ny * off;
    fillConvex(img, [
      [cx - dx * big, cy - dy * big],
      [cx + dx * big, cy + dy * big],
      [cx + dx * big + nx * big, cy + dy * big + ny * big],
      [cx - dx * big + nx * big, cy - dy * big + ny * big],
    ], tone);
  }
  // Objects: rotated rectangles.
  const nRect = Math.floor(rng() * 4);
  for (let i = 0; i < nRect; i++) {
    const cx = rng() * W;
    const cy = rng() * H;
    const w = 40 + rng() * 160;
    const h = 30 + rng() * 120;
    const a = rng() * Math.PI;
    const c: RGB = [rng() * 255, rng() * 255, rng() * 255];
    const pts: Point[] = ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as Point[]).map(([sx, sy]) => [
      cx + (sx * w * Math.cos(a) - sy * h * Math.sin(a)) / 2,
      cy + (sx * w * Math.sin(a) + sy * h * Math.cos(a)) / 2,
    ]);
    fillConvex(img, pts, c);
  }
}

interface BoardStyle {
  /** Width of the black ring right outside the squares (0 = none), with its light inner rule. */
  ringW: number;
  ring: RGB;
  light: RGB;
  dark: RGB;
  frame: RGB;
  wood: boolean;
  fw: number;
  grainAmp: number;
  frameRule: boolean;
  flip: boolean;
}

function makeStyle(rng: Rng, palette: 'wood' | 'vinyl' | 'printed'): BoardStyle {
  let light: RGB;
  let dark: RGB;
  let frame: RGB;
  const wood = palette !== 'vinyl';
  if (palette === 'printed') {
    light = jitterColor(rng, hex('#b4b4b0'), 0.18);
    dark = jitterColor(rng, hex('#1c1c1c'), 0.3);
    frame = jitterColor(rng, hex('#8e3f1e'), 0.25);
    return {
      ringW: 0.16 + rng() * 0.12, ring: mix(dark, [0, 0, 0], 0.2), light, dark, frame, wood, fw: 0.45 + rng() * 0.5,
      grainAmp: 0.1 + rng() * 0.12, frameRule: false, flip: rng() < 0.5,
    };
  }
  if (wood) {
    light = jitterColor(rng, hex('#d9b58a'), 0.14);
    dark = jitterColor(rng, hex('#8b5a2b'), 0.18);
    const r = rng();
    frame = r < 0.5 ? mix(dark, [20, 12, 6], 0.15 + rng() * 0.4) : r < 0.75 ? mix(light, dark, 0.3) : jitterColor(rng, hex('#4a2c14'), 0.3);
  } else {
    light = jitterColor(rng, hex('#eeeed2'), 0.08);
    dark = jitterColor(rng, hex('#769656'), 0.16);
    const r = rng();
    frame = r < 0.35 ? mix(dark, [10, 20, 10], 0.4) : r < 0.6 ? light : r < 0.85 ? jitterColor(rng, hex('#302820'), 0.4) : dark;
  }
  if (rng() < 0.5) {
    const s = 0.3 + rng() * 0.7;
    light = desat(light, s);
    dark = desat(dark, s);
    frame = desat(frame, s);
  }
  if (rng() < 0.4) {
    const c = 0.55 + rng() * 0.45;
    const m = mix(light, dark, 0.5);
    light = mix(m, light, c);
    dark = mix(m, dark, c);
  }
  return { ringW: 0, ring: dark, light, dark, frame, wood, fw: 0.12 + rng() * 0.5, grainAmp: wood ? 0.1 + rng() * 0.16 : 0.02 + rng() * 0.05, frameRule: rng() < 0.5, flip: rng() < 0.5 };
}

function renderBoard(rng: Rng, img: Float32Array, hb: Mat3, style: BoardStyle): void {
  const inv = invert3(hb)!;
  const grain = makeGrain(rng, style.wood);
  const tint = Float32Array.from({ length: 64 }, () => 1 + (rng() - 0.5) * (style.wood ? 0.1 : 0.05));
  const half = 4 + style.ringW + style.fw;
  const corners = ([[-half, -half], [half, -half], [half, half], [-half, half]] as Point[]).map((p) => applyH(hb, p));
  const xs = corners.map((c) => c[0]);
  const ys = corners.map((c) => c[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const x1 = Math.min(W - 1, Math.ceil(Math.max(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const y1 = Math.min(H - 1, Math.ceil(Math.max(...ys)));
  const ruleW = 0.04 + rng() * 0.05;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const px = x + 0.5;
      const py = y + 0.5;
      const w = inv[6] * px + inv[7] * py + inv[8];
      const bx = (inv[0] * px + inv[1] * py + inv[2]) / w;
      const by = (inv[3] * px + inv[4] * py + inv[5]) / w;
      if (Math.abs(bx) > half || Math.abs(by) > half) continue;
      const sx = bx + 4;
      const sy = by + 4;
      const inside = sx >= 0 && sx < 8 && sy >= 0 && sy < 8;
      let c: RGB;
      let gAmp: number;
      let k = 1;
      if (inside) {
        const i = Math.floor(sx);
        const j = Math.floor(sy);
        const dark = ((i + j) & 1) === (style.flip ? 1 : 0);
        c = dark ? style.dark : style.light;
        k = tint[j * 8 + i]!;
        gAmp = style.grainAmp;
        const swap = style.wood && (hash2(i, j, 7) < 0.5);
        const u = Math.floor((swap ? by : bx) * 26 + 1000);
        const v = Math.floor((swap ? bx : by) * 26 + 1000);
        k *= 1 + gAmp * grain[mir(v) * TEX + mir(u)]!;
      } else if (style.ringW > 0 && Math.max(-sx, sx - 8, -sy, sy - 8) < style.ringW) {
        // Printed black ring with a thin light rule on its inner boundary and small light glyphs.
        const d = Math.max(-sx, sx - 8, -sy, sy - 8);
        c = style.ring;
        if (d < 0.035) c = style.light;
        const along = sx >= 0 && sx < 8 ? sx : sy;
        const fa = along - Math.floor(along);
        if (d > 0.05 && d < style.ringW - 0.04 && fa > 0.42 && fa < 0.58 && (sx >= 0 && sx < 8) !== (sy >= 0 && sy < 8)) c = mix(style.ring, style.light, 0.7);
      } else {
        c = style.frame;
        const u = Math.floor(bx * 26 + 1000);
        const v = Math.floor(by * 26 + 1000);
        k = 1 + style.grainAmp * 0.8 * grain[mir(v) * TEX + mir(u)]!;
        if (style.frameRule) {
          const d = Math.max(-sx, sx - 8, -sy, sy - 8);
          if (d < ruleW) k *= 0.55;
        }
      }
      const o = (y * W + x) * 3;
      img[o] = c[0] * k;
      img[o + 1] = c[1] * k;
      img[o + 2] = c[2] * k;
    }
  }
}

function drawPieces(rng: Rng, img: Float32Array, cam: Camera, count: number, outer = false): void {
  const squares = Array.from({ length: 64 }, (_, i) => i).filter((i) => !outer || [0, 1, 6, 7].includes(Math.floor(i / 8)));
  for (let i = squares.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [squares[i], squares[j]] = [squares[j]!, squares[i]!];
  }
  const chosen = squares.slice(0, count).map((s) => {
    const x = (s % 8) - 3.5 + (rng() - 0.5) * 0.16;
    const y = Math.floor(s / 8) - 3.5 + (rng() - 0.5) * 0.16;
    return { x, y, depth: cam.project(x, y, 0)[2] };
  });
  chosen.sort((a, b) => b.depth - a.depth);
  const lightPieces = rng() < 0.5;
  for (const p of chosen) {
    const isLight = outer ? p.y > 0 : rng() < (lightPieces ? 0.7 : 0.3);
    const body = isLight ? jitterColor(rng, hex('#e8e0c8'), 0.15) : jitterColor(rng, hex('#2a2622'), 0.4);
    const r = 0.27 + rng() * 0.13;
    const h = outer ? 0.8 + rng() * 1.3 : 0.8 + rng() * 1.1;
    const rt = r * (0.4 + rng() * 0.6);
    const n = 20;
    const bottom: Point[] = [];
    const top: Point[] = [];
    const rim: Point[] = [];
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2;
      const b = cam.project(p.x + r * Math.cos(a), p.y + r * Math.sin(a), 0);
      const t = cam.project(p.x + rt * Math.cos(a), p.y + rt * Math.sin(a), h);
      bottom.push([b[0], b[1]]);
      top.push([t[0], t[1]]);
      const m = cam.project(p.x + r * 0.8 * Math.cos(a), p.y + r * 0.8 * Math.sin(a), h * 0.3);
      rim.push([m[0], m[1]]);
    }
    const hull = convexHull([...bottom, ...top, ...rim]);
    const xs = hull.map((q) => q[0]);
    const xmin = Math.min(...xs);
    const span = Math.max(1, Math.max(...xs) - xmin);
    const dir = rng() < 0.5 ? 1 : -1;
    fillConvex(img, hull, body, (x) => {
      const t = (x - xmin) / span;
      return 0.8 + 0.4 * (dir > 0 ? t : 1 - t);
    });
    fillConvex(img, top, mix(body, isLight ? [255, 255, 255] : [90, 85, 80], 0.25));
  }
}

/** Lighting gradient, shadow band, blur, chroma softening (JPEG-like) and sensor noise. */
function degrade(cv: CV, rng: Rng, rgb: Float32Array, meta: Record<string, number | string>): Uint8ClampedArray {
  const gA = rng() * 0.5;
  const gT = rng() * Math.PI * 2;
  const gC = Math.cos(gT);
  const gS = Math.sin(gT);
  const bright = 0.7 + rng() * 0.45;
  const cast: RGB = [0.93 + rng() * 0.14, 0.95 + rng() * 0.1, 0.9 + rng() * 0.16];
  const shadow = rng() < 0.55;
  const sA = rng() * Math.PI;
  const sC = Math.cos(sA);
  const sS = Math.sin(sA);
  const sOff = (rng() - 0.5) * 350;
  const sW = 25 + rng() * 90;
  const sD = 0.25 + rng() * 0.4;
  const sSoft = 8 + rng() * 30;
  meta.shadow = shadow ? 1 : 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let k = bright * (1 + (gA * ((x - W / 2) * gC + (y - H / 2) * gS)) / 320);
      if (shadow) {
        const d = Math.abs((x - W / 2) * sC + (y - H / 2) * sS - sOff);
        const t = Math.max(0, Math.min(1, (sW - d) / sSoft + 0.5));
        k *= 1 - sD * t;
      }
      const o = (y * W + x) * 3;
      rgb[o] = rgb[o]! * k * cast[0];
      rgb[o + 1] = rgb[o + 1]! * k * cast[1];
      rgb[o + 2] = rgb[o + 2]! * k * cast[2];
    }
  }
  const sigma = 0.7 + rng() * 1.1;
  const chromaSigma = 0.8 + rng() * 2.2;
  const noise = rng() < 0.85 ? rng() * 6 : 0;
  meta.blur = +sigma.toFixed(2);
  meta.noise = +noise.toFixed(2);
  const src = new cv.Mat(H, W, cv.CV_8UC3);
  const ycc = new cv.Mat();
  const mv = new cv.MatVector();
  const merged = new cv.Mat();
  const out = new cv.Mat();
  try {
    const d = src.data;
    for (let i = 0; i < W * H * 3; i++) d[i] = Math.max(0, Math.min(255, rgb[i]!));
    cv.GaussianBlur(src, src, new cv.Size(0, 0), sigma);
    cv.cvtColor(src, ycc, cv.COLOR_RGB2YCrCb);
    cv.split(ycc, mv);
    for (const c of [1, 2]) {
      const ch = mv.get(c);
      try {
        cv.GaussianBlur(ch, ch, new cv.Size(0, 0), chromaSigma);
        // The blurred header shares data with the vector element; nothing else to do.
      } finally {
        ch.delete();
      }
    }
    cv.merge(mv, merged);
    cv.cvtColor(merged, out, cv.COLOR_YCrCb2RGB);
    const o = out.data;
    const res = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      const n = noise > 0 ? gauss(rng) * noise : 0;
      res[i * 4] = o[i * 3]! + n;
      res[i * 4 + 1] = o[i * 3 + 1]! + n * 0.8 + (noise > 0 ? gauss(rng) * noise * 0.3 : 0);
      res[i * 4 + 2] = o[i * 3 + 2]! + n * 0.9 + (noise > 0 ? gauss(rng) * noise * 0.3 : 0);
      res[i * 4 + 3] = 255;
    }
    return res;
  } finally {
    src.delete();
    ycc.delete();
    mv.delete();
    merged.delete();
    out.delete();
  }
}

/** Renders a board sample; deterministic for a given seed. */
export function makeBoardSample(cv: CV, seed: number, spec: SampleSpec): Sample {
  const rng = mulberry32(seed * 7919 + 13);
  const U = (a: number, b: number) => a + (b - a) * rng();
  const style = makeStyle(rng, spec.palette);
  const meta: Record<string, number | string> = { palette: spec.palette };
  let cam: Camera | null = null;
  let corners: Point[] = [];
  for (let attempt = 0; attempt < 40 && !cam; attempt++) {
    const el = ((spec.elev === 'overhead' ? U(72, 90) : U(30, 55)) * Math.PI) / 180;
    const az = U(0, Math.PI * 2);
    const roll = (U(-8, 8) * Math.PI) / 180;
    const fpx = U(430, 700);
    let fill = U(0.3, 0.9) * (attempt > 10 ? 0.85 : 1);
    const half = 4 + style.ringW + style.fw;
    const bbox = (D: number) => {
      const c = makeCamera(az, el, roll, fpx, D, W / 2, H / 2);
      const pts = ([[-half, -half], [half, -half], [half, half], [-half, half]] as Point[]).map(([x, y]) => c.project(x, y, 0));
      if (pts.some((p) => p[2] <= 0.1)) return null;
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      return { c, minx: Math.min(...xs), maxx: Math.max(...xs), miny: Math.min(...ys), maxy: Math.max(...ys) };
    };
    let lo = 6;
    let hi = 200;
    for (let it = 0; it < 40; it++) {
      const mid = (lo + hi) / 2;
      const b = bbox(mid);
      if (!b || (b.maxx - b.minx) / W > fill) {
        if (!b) lo = mid;
        else lo = mid;
      } else hi = mid;
    }
    const b = bbox(hi);
    if (!b) continue;
    const bw = b.maxx - b.minx;
    const bh = b.maxy - b.miny;
    fill = bw / W;
    if (bh > 0.96 * H || bw > 0.98 * W || fill < 0.28) continue;
    // Random placement inside the image.
    const tx = U(0.01 * W - b.minx, 0.99 * W - b.maxx);
    const ty = U(0.02 * H - b.miny, 0.98 * H - b.maxy);
    cam = makeCamera(az, el, roll, fpx, hi, W / 2 + tx, H / 2 + ty);
    corners = ([[-4, -4], [4, -4], [4, 4], [-4, 4]] as Point[]).map(([x, y]) => cam!.project(x, y, 0).slice(0, 2) as Point);
    Object.assign(meta, { elev: +((el * 180) / Math.PI).toFixed(1), fill: +fill.toFixed(2), az: +((az * 180) / Math.PI).toFixed(0) });
  }
  if (!cam) throw new Error(`could not place a board for seed ${seed}`);
  const img = new Float32Array(W * H * 3);
  drawBackground(rng, img, spec.palette === 'printed' ? 'greywood' : 'clutter');
  renderBoard(rng, img, cam.hBoard, style);
  const nPieces = spec.pieces === 'none' ? 0 : spec.pieces === 'outer' ? Math.floor(U(14, 33)) : Math.floor(U(8, 33));
  meta.pieces = nPieces;
  if (nPieces > 0) drawPieces(rng, img, cam, nPieces, spec.pieces === 'outer');
  const rgba = degrade(cv, rng, img, meta);
  return {
    rgba, width: W, height: H, corners: corners as Sample['corners'],
    label: `${spec.elev}-${spec.palette}-${spec.pieces}`, seed, meta,
  };
}

export type NegativeKind = 'plain' | 'wood' | 'clutter' | 'tiles' | 'stripes' | 'pieces';

export const NEGATIVE_KINDS: readonly NegativeKind[] = ['plain', 'wood', 'clutter', 'tiles', 'stripes', 'pieces', 'tiles', 'clutter'];

/** An image without a chessboard: textures, clutter, and a uniform tile grid (strong parallel lines, no checker). */
export function makeNegative(cv: CV, seed: number, kind: NegativeKind): Sample {
  const rng = mulberry32(seed * 104729 + 7);
  const U = (a: number, b: number) => a + (b - a) * rng();
  const img = new Float32Array(W * H * 3);
  const meta: Record<string, number | string> = { kind };
  drawBackground(rng, img, kind === 'wood' ? 'wood' : kind === 'plain' ? 'plain' : 'clutter');
  if (kind === 'tiles' || kind === 'stripes') {
    const az = U(0, Math.PI * 2);
    const el = (U(35, 90) * Math.PI) / 180;
    const cam = makeCamera(az, el, 0, 560, U(9, 16), W / 2, H / 2);
    const inv = invert3(cam.hBoard)!;
    const base = jitterColor(rng, hex(rng() < 0.5 ? '#c9c2b0' : '#8a9a92'), 0.2);
    const grout = mix(base, [40, 40, 40], 0.5);
    const cell = kind === 'tiles' ? U(0.6, 1.4) : 1;
    const jit = Float32Array.from({ length: 4096 }, () => 1 + (rng() - 0.5) * 0.06);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const w = inv[6] * (x + 0.5) + inv[7] * (y + 0.5) + inv[8];
        const bx = (inv[0] * (x + 0.5) + inv[1] * (y + 0.5) + inv[2]) / w;
        const by = (inv[3] * (x + 0.5) + inv[4] * (y + 0.5) + inv[5]) / w;
        if (!(w > 0)) continue;
        const u = bx / cell + 500;
        const v = by / cell + 500;
        const fu = u - Math.floor(u);
        const fv = v - Math.floor(v);
        let c = base;
        let k = jit[((Math.floor(u) * 31 + Math.floor(v) * 17) & 4095)]!;
        if (kind === 'tiles' && (fu < 0.04 || fv < 0.04)) c = grout;
        if (kind === 'stripes') {
          if (fu < 0.5) c = mix(base, grout, 0.35);
          k = 1;
        }
        const o = (y * W + x) * 3;
        img[o] = c[0] * k;
        img[o + 1] = c[1] * k;
        img[o + 2] = c[2] * k;
      }
    }
  }
  if (kind === 'pieces') {
    // Blobs on a plain table, no board.
    const cam = makeCamera(U(0, 6), (U(40, 80) * Math.PI) / 180, 0, 560, 12, W / 2, H / 2);
    drawPieces(rng, img, cam, 12);
  }
  const rgba = degrade(cv, rng, img, meta);
  return { rgba, width: W, height: H, corners: null, label: `negative-${kind}`, seed, meta };
}
