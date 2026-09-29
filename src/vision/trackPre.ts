import type { Params } from '../worker/protocol';
import { param, PREPROCESS_PARAMS, type CV } from './preprocess';

type Mat = InstanceType<CV['Mat']>;

export interface TrackPreResult {
  /** CLAHE-L, a*, b* interleaved (8UC3). */
  labEq: Mat;
  /** Interleaved channels in `labEq`: 3, or 1 (CLAHE-L only) for an L-only profile. */
  stride: 1 | 3;
  /** Canny edges (8U, 0/255) of the CLAHE-L channel. */
  edges: Mat;
  timings: Record<string, number>;
}

/**
 * Cut-down preprocessing for tracking: Lab + CLAHE for the checker verification and a single-channel (L) Canny map
 * for the edge polish. No multi-channel gradient and no per-channel Canny.
 * Owns its Mats; results are valid until the next run().
 */
export class TrackPreprocessor {
  private readonly rgb: Mat;
  private readonly lab: Mat;
  private readonly channels: InstanceType<CV['MatVector']>;
  private readonly lEq: Mat;
  private readonly chan: Mat;
  private readonly small: Mat;
  private readonly gx: Mat;
  private readonly gy: Mat;
  private readonly ax: Mat;
  private readonly ay: Mat;
  private readonly mag8: Mat;
  private readonly edges: Mat;
  private readonly labEq: Mat;
  private readonly clahe: InstanceType<CV['CLAHE']>;
  private ksize: InstanceType<CV['Size']>;
  private smallSize: InstanceType<CV['Size']>;

  constructor(private readonly cv: CV) {
    const m = () => new cv.Mat();
    this.rgb = m();
    this.lab = m();
    this.channels = new cv.MatVector();
    this.lEq = m();
    this.chan = m();
    this.small = m();
    this.gx = m();
    this.gy = m();
    this.ax = m();
    this.ay = m();
    this.mag8 = m();
    this.edges = m();
    this.labEq = m();
    this.clahe = new cv.CLAHE(2, new cv.Size(8, 8));
    this.ksize = new cv.Size(5, 5);
    this.smallSize = new cv.Size(0, 0);
  }

  /**
   * `roi` is the crop (in pixels of `rgba`) that contains the board plus a margin. `edges` is at the crop's size
   * (full resolution); `labEq` is at half the crop's size (the checker verification only needs cell medians).
   */
  run(rgba: Mat, params: Params, roi: { x: number; y: number; w: number; h: number }, lOnly = false): TrackPreResult {
    const cv = this.cv;
    const timings: Record<string, number> = {};
    let t = performance.now();
    const lap = (name: string) => {
      const now = performance.now();
      timings[name] = now - t;
      t = now;
    };
    const view = rgba.roi(new cv.Rect(roi.x, roi.y, roi.w, roi.h));
    try {
      // Edge map: raw grey, blurred, Canny with thresholds from a half-resolution gradient histogram.
      cv.cvtColor(view, this.chan, cv.COLOR_RGBA2GRAY);
      const pct = param(params, PREPROCESS_PARAMS, 'cannyPercentile');
      const floor = param(params, PREPROCESS_PARAMS, 'cannyFloor');
      const noiseMult = param(params, PREPROCESS_PARAMS, 'cannyNoiseMult');
      const sigma = param(params, PREPROCESS_PARAMS, 'edgeBlur');
      const ks = Math.max(3, 2 * Math.round(1.5 * sigma) + 1);
      if (this.ksize.width !== ks) this.ksize = new cv.Size(ks, ks);
      if (this.smallSize.width !== (roi.w >> 1) || this.smallSize.height !== (roi.h >> 1)) {
        this.smallSize = new cv.Size(roi.w >> 1, roi.h >> 1);
      }
      cv.GaussianBlur(this.chan, this.chan, this.ksize, sigma);
      cv.resize(this.chan, this.small, this.smallSize, 0, 0, cv.INTER_NEAREST);
      cv.Sobel(this.small, this.gx, cv.CV_16S, 1, 0, 3);
      cv.Sobel(this.small, this.gy, cv.CV_16S, 0, 1, 3);
      cv.convertScaleAbs(this.gx, this.ax, 0.25);
      cv.convertScaleAbs(this.gy, this.ay, 0.25);
      cv.add(this.ax, this.ay, this.mag8);
      const data = this.mag8.data;
      const hist = new Uint32Array(256);
      for (let i = 0; i < data.length; i++) hist[data[i]!]!++;
      const at = (p: number): number => {
        const target = (data.length * p) / 100;
        let acc = 0;
        for (let v = 0; v < 256; v++) {
          acc += hist[v]!;
          if (acc >= target) return v;
        }
        return 255;
      };
      const high = Math.max(at(pct) * 4, at(50) * 4 * noiseMult, floor);
      lap('trackGradient');
      cv.Canny(this.chan, this.edges, high * 0.4, high);
      lap('trackCanny');

      // Verification image: half resolution Lab with CLAHE on L.
      cv.resize(view, this.rgb, this.smallSize, 0, 0, cv.INTER_AREA);
      this.clahe.setClipLimit(param(params, PREPROCESS_PARAMS, 'claheClip'));
      if (lOnly) {
        cv.cvtColor(this.rgb, this.lab, cv.COLOR_RGBA2GRAY);
        this.clahe.apply(this.lab, this.lEq);
      } else {
        cv.cvtColor(this.rgb, this.rgb, cv.COLOR_RGBA2RGB);
        this.mergeLabEq();
      }
      lap('trackLab');
    } finally {
      view.delete();
    }
    return { labEq: lOnly ? this.lEq : this.labEq, stride: lOnly ? 1 : 3, edges: this.edges, timings };
  }

  /** Full-frame variant: full-resolution Lab + CLAHE-L (used for both verification and edges). */
  runFull(rgba: Mat, params: Params, lOnly = false): TrackPreResult {
    const cv = this.cv;
    const timings: Record<string, number> = {};
    let t = performance.now();
    const lap = (name: string) => {
      const now = performance.now();
      timings[name] = now - t;
      t = now;
    };
    this.clahe.setClipLimit(param(params, PREPROCESS_PARAMS, 'claheClip'));
    if (lOnly) {
      cv.cvtColor(rgba, this.lab, cv.COLOR_RGBA2GRAY);
      this.clahe.apply(this.lab, this.lEq);
    } else {
      cv.cvtColor(rgba, this.rgb, cv.COLOR_RGBA2RGB);
      this.mergeLabEq();
    }
    lap('clahe');
    const pct = param(params, PREPROCESS_PARAMS, 'cannyPercentile');
    const floor = param(params, PREPROCESS_PARAMS, 'cannyFloor');
    const noiseMult = param(params, PREPROCESS_PARAMS, 'cannyNoiseMult');
    const sigma = param(params, PREPROCESS_PARAMS, 'edgeBlur');
    const ks = Math.max(3, 2 * Math.round(1.5 * sigma) + 1);
    if (this.ksize.width !== ks) this.ksize = new cv.Size(ks, ks);
    if (this.smallSize.width !== (rgba.cols >> 1) || this.smallSize.height !== (rgba.rows >> 1)) {
      this.smallSize = new cv.Size(rgba.cols >> 1, rgba.rows >> 1);
    }
    cv.GaussianBlur(this.lEq, this.chan, this.ksize, sigma);
    cv.resize(this.chan, this.small, this.smallSize, 0, 0, cv.INTER_NEAREST);
    cv.Sobel(this.small, this.gx, cv.CV_16S, 1, 0, 3);
    cv.Sobel(this.small, this.gy, cv.CV_16S, 0, 1, 3);
    cv.convertScaleAbs(this.gx, this.ax, 0.25);
    cv.convertScaleAbs(this.gy, this.ay, 0.25);
    cv.add(this.ax, this.ay, this.mag8);
    const data = this.mag8.data;
    const hist = new Uint32Array(256);
    for (let i = 0; i < data.length; i++) hist[data[i]!]!++;
    const at = (p: number): number => {
      const target = (data.length * p) / 100;
      let acc = 0;
      for (let v = 0; v < 256; v++) {
        acc += hist[v]!;
        if (acc >= target) return v;
      }
      return 255;
    };
    const high = Math.max(at(pct) * 4, at(50) * 4 * noiseMult, floor);
    lap('trackGradient');
    cv.Canny(this.chan, this.edges, high * 0.4, high);
    lap('trackCanny');
    return { labEq: lOnly ? this.lEq : this.labEq, stride: lOnly ? 1 : 3, edges: this.edges, timings };
  }

  /** this.rgb (RGB) -> Lab, CLAHE on L, merged into labEq. */
  private mergeLabEq(): void {
    const cv = this.cv;
    cv.cvtColor(this.rgb, this.lab, cv.COLOR_RGB2Lab);
    cv.split(this.lab, this.channels);
    const l = this.channels.get(0);
    try {
      this.clahe.apply(l, this.lEq);
    } finally {
      l.delete();
    }
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

  dispose(): void {
    for (const m of [
      this.rgb, this.lab, this.channels, this.lEq, this.chan, this.small, this.gx, this.gy, this.ax, this.ay,
      this.mag8, this.edges, this.labEq, this.clahe,
    ]) {
      m.delete();
    }
  }
}
