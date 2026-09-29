import type { Params } from '../worker/protocol';
import { applyH, type Mat3, type Point } from '../geom/homography';
import { param, type ParamSpec } from './preprocess';
import type { BoardProfile } from './profile';

export const VERIFY_PARAMS: readonly ParamSpec[] = [
  { name: 'verifyAccept', min: 0.1, max: 0.9, step: 0.05, default: 0.4 },
  { name: 'verifyContrast', min: 0.5, max: 12, step: 0.5, default: 3 },
  /** With a locked profile: the contrast needed for a full score is this fraction of the profile's contrast. */
  { name: 'profileContrastFrac', min: 0, max: 1, step: 0.05, default: 0.3 },
];

/** Interleaved 3-channel 8-bit image (CLAHE-L, a*, b*). */
export interface Lab3 {
  data: ArrayLike<number>;
  width: number;
  height: number;
  /** Interleaved channels per pixel; default 3. Stride 1 holds CLAHE-L only (channel 0). */
  stride?: number;
}

export interface VerifyOptions {
  minContrast: number;
  /** Channels to score (default all of 0, 1, 2 that the image holds). */
  channels?: readonly number[];
  /** Observed surround of a locked profile; a candidate whose ring level contradicts it is penalised. */
  surround?: BoardProfile['surround'];
  /** Reporting only: another channel must beat channel 0 by this margin to be reported as the winner. */
  lBias?: number;
}

/** Verify options; with a locked profile only its verify channel is scored and the contrast floor follows it. */
export function verifyOptions(params: Params, profile?: BoardProfile | null): VerifyOptions {
  const minContrast = param(params, VERIFY_PARAMS, 'verifyContrast');
  if (!profile) return { minContrast };
  const frac = param(params, VERIFY_PARAMS, 'profileContrastFrac');
  return {
    minContrast: Math.max(minContrast, frac * Math.abs(profile.contrast)),
    channels: [profile.verifyChannel],
    surround: profile.surround,
  };
}

export interface VerifyResult {
  /** Final score in [0, 1]. */
  score: number;
  /** Alternation score of the 8x8 core in [0, 1] before the margin test. */
  alternation: number;
  /** Multiplier from the margin test (1 = margin does not continue the pattern). */
  ringFactor: number;
  channel: number;
  /** Per-cell agreement in [0, 1] (row-major 8x8), for the debug heat map. */
  cells: Float32Array;
  /** Signed pair differences median (for the winning channel), for debugging. */
  contrast: number;
  /** Median of the ring cells around the core on the winning channel, as a position between the dark (0) and
   *  light (1) square levels. NaN when not measurable. Used to classify the surround. */
  ringLevel: number;
}

// Sample offsets inside a cell: a grid over the outer part, skipping the centre where pieces stand.
const SAMPLES: Array<[number, number]> = (() => {
  const s: Array<[number, number]> = [];
  const n = 8;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const fx = 0.14 + (0.72 * i) / (n - 1);
      const fy = 0.14 + (0.72 * j) / (n - 1);
      if (Math.hypot(fx - 0.5, fy - 0.5) >= 0.3) s.push([fx, fy]);
    }
  }
  return s;
})();

const median = (a: Float32Array, n: number): number => {
  const s = a.subarray(0, n).sort();
  return n & 1 ? s[n >> 1]! : (s[n / 2 - 1]! + s[n / 2]!) / 2;
};

/** A 'dark' surround should sit clearly below the light squares, a 'bare' one clearly above the dark squares. */
const SURROUND_DARK_MAX = 0.7;
const SURROUND_BARE_MIN = 0.3;

const GRID = 10; // 8 cells + 1 margin cell on each side

/** Median value per cell (row-major 10x10 incl. margin) and channel; NaN where not visible. */
export function sampleCells(img: Lab3, hb: Mat3, channels: readonly number[] = [0, 1, 2]): Float32Array[] {
  const out = [new Float32Array(GRID * GRID), new Float32Array(GRID * GRID), new Float32Array(GRID * GRID)];
  const buf = [new Float32Array(SAMPLES.length), new Float32Array(SAMPLES.length), new Float32Array(SAMPLES.length)];
  const { data, width, height } = img;
  const stride = img.stride ?? 3;
  const chans = channels.filter((c) => c < stride);
  for (let j = 0; j < GRID; j++) {
    for (let i = 0; i < GRID; i++) {
      let n = 0;
      for (const [fx, fy] of SAMPLES) {
        const p = applyH(hb, [i - 1 + fx, j - 1 + fy]);
        const x = Math.round(p[0]);
        const y = Math.round(p[1]);
        if (!(x >= 0 && y >= 0 && x < width && y < height)) continue;
        const o = (y * width + x) * stride;
        for (const c of chans) buf[c]![n] = data[o + c]!;
        n++;
      }
      for (const c of chans) out[c]![j * GRID + i] = n >= 6 ? median(buf[c]!, n) : NaN;
    }
  }
  return out;
}

const cellAt = (v: Float32Array, i: number, j: number) => v[(j + 1) * GRID + (i + 1)]!;
const parity = (i: number, j: number) => ((i + j) & 1 ? -1 : 1);

function medianOf(vals: number[]): number {
  const s = Float32Array.from(vals).sort();
  const n = s.length;
  return n === 0 ? 0 : n & 1 ? s[n >> 1]! : (s[n / 2 - 1]! + s[n / 2]!) / 2;
}

interface ChannelScore {
  score: number;
  med: number;
  cells: Float32Array;
}

function alternationScore(v: Float32Array, minContrast: number): ChannelScore {
  const d: number[] = [];
  const pairs: Array<[number, number, number]> = []; // cell index a, b, d
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 8; i++) {
      const a = cellAt(v, i, j);
      if (i < 7) {
        const b = cellAt(v, i + 1, j);
        if (!Number.isNaN(a + b)) {
          const dd = (a - b) * parity(i, j);
          d.push(dd);
          pairs.push([j * 8 + i, j * 8 + i + 1, dd]);
        }
      }
      if (j < 7) {
        const b = cellAt(v, i, j + 1);
        if (!Number.isNaN(a + b)) {
          const dd = (a - b) * parity(i, j);
          d.push(dd);
          pairs.push([j * 8 + i, (j + 1) * 8 + i, dd]);
        }
      }
    }
  }
  const cells = new Float32Array(64);
  if (d.length < 80) return { score: 0, med: 0, cells };
  const med = medianOf(d);
  const sgn = med >= 0 ? 1 : -1;
  const thr = 0.5 * Math.abs(med);
  let agree = 0;
  const cnt = new Float32Array(64);
  const ok = new Float32Array(64);
  for (const [a, b, dd] of pairs) {
    const good = dd * sgn > thr;
    if (good) agree++;
    cnt[a]!++;
    cnt[b]!++;
    if (good) {
      ok[a]!++;
      ok[b]!++;
    }
  }
  for (let k = 0; k < 64; k++) cells[k] = cnt[k]! > 0 ? ok[k]! / cnt[k]! : 0;
  const g = Math.min(1, Math.abs(med) / minContrast);
  const f = Math.max(0, Math.min(1, (agree / d.length - 0.5) / 0.4));
  return { score: g * f, med, cells };
}

/** 1 when the ring around the core does not continue the alternation on any side, else lower. */
function ringFactor(v: Float32Array, med: number): number {
  if (Math.abs(med) < 1e-6) return 1;
  const sgn = med >= 0 ? 1 : -1;
  const thr = 0.5 * Math.abs(med);
  // Each side: ring cells and their (i, j) coordinates.
  const sides: Array<Array<[number, number]>> = [[], [], [], []];
  for (let k = 0; k < 8; k++) {
    sides[0]!.push([k, -1]);
    sides[1]!.push([k, 8]);
    sides[2]!.push([-1, k]);
    sides[3]!.push([8, k]);
  }
  let continuing = 0;
  for (const side of sides) {
    const d: number[] = [];
    for (let k = 0; k < 7; k++) {
      const [ai, aj] = side[k]!;
      const [bi, bj] = side[k + 1]!;
      const a = cellAt(v, ai, aj);
      const b = cellAt(v, bi, bj);
      if (!Number.isNaN(a + b)) d.push((a - b) * parity(ai, aj));
    }
    if (d.length >= 5 && medianOf(d) * sgn >= thr) continuing++;
  }
  return continuing === 0 ? 1 : continuing === 1 ? 0.4 : 0.2;
}

/** Multiplier < 1 when the ring around the core contradicts the surround observed on the locked board. */
function surroundFactor(surround: BoardProfile['surround'] | undefined, v: Float32Array): number {
  if (!surround || surround === 'frame') return 1;
  const r = ringLevel(v);
  if (Number.isNaN(r)) return 1;
  if (surround === 'dark') return r > SURROUND_DARK_MAX ? 0.7 : 1;
  return r < SURROUND_BARE_MIN ? 0.7 : 1;
}

/** Ring median level relative to the darker (0) and lighter (1) square medians of the core. */
function ringLevel(v: Float32Array): number {
  const even: number[] = [];
  const odd: number[] = [];
  for (let j = 0; j < 8; j++)
    for (let i = 0; i < 8; i++) {
      const a = cellAt(v, i, j);
      if (!Number.isNaN(a)) ((i + j) & 1 ? odd : even).push(a);
    }
  const ring: number[] = [];
  for (let k = 0; k < 8; k++)
    for (const [i, j] of [[k, -1], [k, 8], [-1, k], [8, k]] as const) {
      const a = cellAt(v, i, j);
      if (!Number.isNaN(a)) ring.push(a);
    }
  if (even.length < 8 || odd.length < 8 || ring.length < 8) return NaN;
  const me = medianOf(even);
  const mo = medianOf(odd);
  const lo = Math.min(me, mo);
  const hi = Math.max(me, mo);
  if (hi - lo < 1e-6) return NaN;
  return (medianOf(ring) - lo) / (hi - lo);
}

export function verifyBoard(img: Lab3, hb: Mat3, opt: VerifyOptions): VerifyResult {
  const scored = (opt.channels ?? [0, 1, 2]).filter((c) => c < (img.stride ?? 3));
  const chans = sampleCells(img, hb, scored);
  let best: VerifyResult = { score: 0, alternation: 0, ringFactor: 1, channel: scored[0] ?? 0, cells: new Float32Array(64), contrast: 0, ringLevel: NaN };
  for (const c of scored) {
    const a = alternationScore(chans[c]!, opt.minContrast);
    if (a.score <= 0) continue;
    const rf = ringFactor(chans[c]!, a.med);
    const score = a.score * rf * surroundFactor(opt.surround, chans[c]!);
    const bar = best.channel === 0 && best.score > 0 ? best.score + (opt.lBias ?? 0) : best.score;
    if (score > bar) best = { score, alternation: a.score, ringFactor: rf, channel: c, cells: a.cells, contrast: a.med, ringLevel: NaN };
  }
  if (best.score > 0) best.ringLevel = ringLevel(chans[best.channel]!);
  return best;
}

/** Image-space corners of the 8x8 area for a board->image homography (cell units). */
export function boardCorners(hb: Mat3): [Point, Point, Point, Point] {
  return [applyH(hb, [0, 0]), applyH(hb, [8, 0]), applyH(hb, [8, 8]), applyH(hb, [0, 8])];
}
