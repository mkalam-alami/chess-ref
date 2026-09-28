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
  /** Max over L, a*, b* Sobel magnitudes, scaled to 8U for display. */
  gradient: Mat;
  /** Union of Canny edges over L, a*, b* (8U, 0/255). */
  edges: Mat;
  timings: Record<string, number>;
}

/** Percentile (0-100) of an 8-bit image via histogram. */
function percentile8(data: Uint8Array, pct: number): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i++) hist[data[i]!]!++;
  const target = (data.length * pct) / 100;
  let acc = 0;
  for (let v = 0; v < 256; v++) {
    acc += hist[v]!;
    if (acc >= target) return v;
  }
  return 255;
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
  private readonly mag: Mat;
  private readonly mag8: Mat;
  private readonly magMax: Mat;
  private readonly gradient: Mat;
  private readonly edgeCh: Mat;
  private readonly edges: Mat;
  private readonly clahe: InstanceType<CV['CLAHE']>;
  private readonly ksize: InstanceType<CV['Size']>;

  constructor(private readonly cv: CV) {
    const m = () => new cv.Mat();
    this.rgb = m();
    this.lab = m();
    this.channels = new cv.MatVector();
    this.chan = m();
    this.lEq = m();
    this.gx = m();
    this.gy = m();
    this.mag = m();
    this.mag8 = m();
    this.magMax = m();
    this.gradient = m();
    this.edgeCh = m();
    this.edges = m();
    this.clahe = new cv.CLAHE(2, new cv.Size(8, 8));
    this.ksize = new cv.Size(3, 3);
  }

  run(rgba: Mat, params: Params): PreprocessResult {
    const cv = this.cv;
    const timings: Record<string, number> = {};
    let t = performance.now();
    const lap = (name: string) => {
      const now = performance.now();
      timings[name] = now - t;
      t = now;
    };

    cv.cvtColor(rgba, this.rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(this.rgb, this.lab, cv.COLOR_RGB2Lab);
    cv.split(this.lab, this.channels);
    lap('lab');

    this.clahe.setClipLimit(param(params, PREPROCESS_PARAMS, 'claheClip'));
    const l = this.channels.get(0);
    try {
      this.clahe.apply(l, this.lEq);
    } finally {
      l.delete();
    }
    lap('clahe');

    const pct = param(params, PREPROCESS_PARAMS, 'cannyPercentile');
    const floor = param(params, PREPROCESS_PARAMS, 'cannyFloor');
    let first = true;
    let gradientMs = 0;
    let cannyMs = 0;
    for (let c = 0; c < 3; c++) {
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
      cv.GaussianBlur(this.chan, this.chan, this.ksize, 0);
      cv.Sobel(this.chan, this.gx, cv.CV_32F, 1, 0, 3);
      cv.Sobel(this.chan, this.gy, cv.CV_32F, 0, 1, 3);
      cv.magnitude(this.gx, this.gy, this.mag);
      if (first) this.mag.copyTo(this.magMax);
      else cv.max(this.magMax, this.mag, this.magMax);
      // Canny thresholds adapt to this channel's own gradient distribution.
      this.mag.convertTo(this.mag8, cv.CV_8U, 0.5);
      const high = Math.max(percentile8(this.mag8.data, pct) * 2, floor);
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

    this.magMax.convertTo(this.gradient, cv.CV_8U, 0.5);
    lap('gradientScale');

    return { l: this.lEq, gradient: this.gradient, edges: this.edges, timings };
  }

  dispose(): void {
    for (const m of [
      this.rgb, this.lab, this.channels, this.chan, this.lEq, this.gx, this.gy, this.mag,
      this.mag8, this.magMax, this.gradient, this.edgeCh, this.edges, this.clahe,
    ]) {
      m.delete();
    }
  }
}
