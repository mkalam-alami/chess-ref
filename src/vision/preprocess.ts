import type { Params } from '../worker/protocol';

// The OpenCV.js namespace type; the module is only loaded inside the worker.
export type CV = typeof import('@techstark/opencv-js');
type Mat = InstanceType<CV['Mat']>;

export interface ParamSpec {
  name: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

/** Tunables of this stage; the main thread registers them as debug sliders. */
export const PREPROCESS_PARAMS: readonly ParamSpec[] = [
  { name: 'claheClip', min: 0.5, max: 8, step: 0.5, default: 2 },
  { name: 'cannyPercentile', min: 80, max: 99, step: 1, default: 92 },
  { name: 'cannyFloor', min: 5, max: 80, step: 1, default: 20 },
  { name: 'cannyNoiseMult', min: 2, max: 12, step: 0.5, default: 5 },
  { name: 'edgeBlur', min: 0.4, max: 3, step: 0.1, default: 1.2 },
];

export function param(params: Params, spec: readonly ParamSpec[], name: string): number {
  const v = params[name];
  if (v !== undefined) return v;
  const s = spec.find((p) => p.name === name);
  if (!s) throw new Error(`Unknown param ${name}`);
  return s.default;
}

export interface PreprocessResult {
  /** CLAHE-equalised L channel (8U). */
  l: Mat;
  /** CLAHE-L, a*, b* interleaved (8UC3), for the checker verification. With an L-only profile this is the
   *  single-channel CLAHE-L image (stride 1). */
  labEq: Mat;
  /** Interleaved channel count of `labEq` (3, or 1 for the L-only fast path). */
  stride: 1 | 3;
  /** Max over L, a*, b* Sobel magnitudes, scaled to 8U for display. */
  gradient: Mat;
  /** Union of Canny edges over L, a*, b* (8U, 0/255). */
  edges: Mat;
  timings: Record<string, number>;
}

export interface PreprocessOptions {
  /** Subset of [0 L, 1 a*, 2 b*] for which gradient and edges are computed. Default: all three. When it is
   *  exactly [0], the Lab conversion is skipped (grayscale + CLAHE, single-channel output). */
  channels?: readonly number[];
}

/** Percentiles (0-100) of an 8-bit image via one histogram. */
function percentiles8(data: Uint8Array, pcts: readonly number[]): number[] {
  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i++) hist[data[i]!]!++;
  return pcts.map((pct) => {
    const target = (data.length * pct) / 100;
    let acc = 0;
    for (let v = 0; v < 256; v++) {
      acc += hist[v]!;
      if (acc >= target) return v;
    }
    return 255;
  });
}

/**
 * Lab conversion, CLAHE, blur, multi-channel gradient and Canny.
 * Owns its Mats and reuses them across frames; call dispose() when done.
 * Result Mats are valid until the next run().
 */
export class Preprocessor {
  private readonly rgb: Mat;
  private readonly lab: Mat;
  private readonly channels: InstanceType<CV['MatVector']>;
  private readonly chan: Mat;
  private readonly lEq: Mat;
  private readonly gx: Mat;
  private readonly gy: Mat;
  private readonly small: Mat;
  private smallSize: InstanceType<CV['Size']>;
  private readonly ax: Mat;
  private readonly ay: Mat;
  private readonly mag8: Mat;
  private readonly magMax: Mat;
  private readonly gradient: Mat;
  private readonly edgeCh: Mat;
  private readonly edges: Mat;
  private readonly labEq: Mat;
  private readonly clahe: InstanceType<CV['CLAHE']>;
  private ksize: InstanceType<CV['Size']>;

  constructor(private readonly cv: CV) {
    const m = () => new cv.Mat();
    this.rgb = m();
    this.lab = m();
    this.channels = new cv.MatVector();
    this.chan = m();
    this.lEq = m();
    this.gx = m();
    this.gy = m();
    this.small = m();
    this.smallSize = new cv.Size(0, 0);
    this.ax = m();
    this.ay = m();
    this.mag8 = m();
    this.magMax = m();
    this.gradient = m();
    this.edgeCh = m();
    this.edges = m();
    this.labEq = m();
    this.clahe = new cv.CLAHE(2, new cv.Size(8, 8));
    this.ksize = new cv.Size(5, 5);
  }

  run(rgba: Mat, params: Params, opts: PreprocessOptions = {}): PreprocessResult {
    const cv = this.cv;
    const timings: Record<string, number> = {};
    let t = performance.now();
    const lap = (name: string) => {
      const now = performance.now();
      timings[name] = now - t;
      t = now;
    };

    const chanList = opts.channels && opts.channels.length > 0 ? [...opts.channels].sort() : [0, 1, 2];
    const lOnly = chanList.length === 1 && chanList[0] === 0;
    this.clahe.setClipLimit(param(params, PREPROCESS_PARAMS, 'claheClip'));
    if (lOnly) {
      // Fast path: luminance only, no Lab conversion and no a*, b*.
      cv.cvtColor(rgba, this.chan, cv.COLOR_RGBA2GRAY);
      lap('lab');
      this.clahe.apply(this.chan, this.lEq);
    } else {
      cv.cvtColor(rgba, this.rgb, cv.COLOR_RGBA2RGB);
      cv.cvtColor(this.rgb, this.lab, cv.COLOR_RGB2Lab);
      cv.split(this.lab, this.channels);
      lap('lab');
      const l = this.channels.get(0);
      try {
        this.clahe.apply(l, this.lEq);
      } finally {
        l.delete();
      }
    }
    if (!lOnly) {
      // Interleave equalised L with the untouched a*, b* for the verification stage.
      const a = this.channels.get(1);
      const b = this.channels.get(2);
      const mv = new cv.MatVector();
      try {
        mv.push_back(this.lEq);
        mv.push_back(a);
        mv.push_back(b);
        cv.merge(mv, this.labEq);
      } finally {
        a.delete();
        b.delete();
        mv.delete();
      }
    }
    lap('clahe');

    const pct = param(params, PREPROCESS_PARAMS, 'cannyPercentile');
    const floor = param(params, PREPROCESS_PARAMS, 'cannyFloor');
    const noiseMult = param(params, PREPROCESS_PARAMS, 'cannyNoiseMult');
    const sigma = param(params, PREPROCESS_PARAMS, 'edgeBlur');
    const ks = Math.max(3, 2 * Math.round(1.5 * sigma) + 1);
    if (this.ksize.width !== ks) this.ksize = new cv.Size(ks, ks);
    if (this.smallSize.width !== (rgba.cols >> 1)) this.smallSize = new cv.Size(rgba.cols >> 1, rgba.rows >> 1);
    let first = true;
    let gradientMs = 0;
    let cannyMs = 0;
    for (const c of chanList) {
      let t0 = performance.now();
      if (c === 0) {
        this.lEq.copyTo(this.chan);
      } else {
        const ch = this.channels.get(c);
        try {
          ch.copyTo(this.chan);
        } finally {
          ch.delete();
        }
      }
      cv.GaussianBlur(this.chan, this.chan, this.ksize, sigma);
      // Gradient statistics on a half-resolution copy (4x cheaper). L1 magnitude, which is what Canny
      // thresholds, in quarter units so it fits 8 bits.
      cv.resize(this.chan, this.small, this.smallSize, 0, 0, cv.INTER_NEAREST);
      cv.Sobel(this.small, this.gx, cv.CV_16S, 1, 0, 3);
      cv.Sobel(this.small, this.gy, cv.CV_16S, 0, 1, 3);
      cv.convertScaleAbs(this.gx, this.ax, 0.25);
      cv.convertScaleAbs(this.gy, this.ay, 0.25);
      cv.add(this.ax, this.ay, this.mag8);
      if (first) this.mag8.copyTo(this.magMax);
      else cv.max(this.magMax, this.mag8, this.magMax);
      // Canny thresholds adapt to this channel's own gradient distribution. Never below a multiple of the
      // median gradient (the noise / texture level) so busy backgrounds do not pull the percentile
      // threshold down into the noise.
      const [pHigh, pMed] = percentiles8(this.mag8.data, [pct, 50]);
      const high = Math.max(pHigh! * 4, pMed! * 4 * noiseMult, floor);
      gradientMs += performance.now() - t0;

      t0 = performance.now();
      cv.Canny(this.chan, this.edgeCh, high * 0.4, high);
      if (first) this.edgeCh.copyTo(this.edges);
      else cv.bitwise_or(this.edges, this.edgeCh, this.edges);
      cannyMs += performance.now() - t0;
      first = false;
    }
    timings.gradient = gradientMs;
    timings.canny = cannyMs;
    t = performance.now();

    cv.resize(this.magMax, this.gradient, new cv.Size(rgba.cols, rgba.rows), 0, 0, cv.INTER_LINEAR);
    lap('gradientScale');

    return { l: this.lEq, labEq: lOnly ? this.lEq : this.labEq, stride: lOnly ? 1 : 3, gradient: this.gradient, edges: this.edges, timings };
  }

  dispose(): void {
    for (const m of [
      this.rgb, this.lab, this.channels, this.chan, this.lEq, this.gx, this.gy, this.small, this.ax, this.ay,
      this.mag8, this.magMax, this.gradient, this.edgeCh, this.edges, this.clahe, this.labEq,
    ]) {
      m.delete();
    }
  }
}
