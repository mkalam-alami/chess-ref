import type { Mat3 } from '../geom/homography';
import { Detector, TRACK_PARAMS, type DetectOptions, type DetectResult } from '../vision/detector';
import { param } from '../vision/preprocess';
import type { Params } from './protocol';

type Input = Parameters<Detector['detect']>[0];

export interface SessionResult extends DetectResult {
  mode: 'full' | 'tracking';
  /** Why a full detection ran instead of (or after a failed) tracking; absent for tracked frames. */
  fullReason?: 'no-lock' | 'disabled' | 'periodic' | 'age' | 'track-failed' | 'resized';
}

/**
 * The tracking state machine: keeps the last accepted board homography, tries the cheap Detector.track() first and
 * falls back to a full Detector.detect() on failure, every `trackFullEvery` frames, or once the last full
 * detection is older than `trackMaxMs`. Pure logic (no worker globals) so tests drive the same code as the worker.
 */
export class TrackingSession {
  private hb: Mat3 | null = null;
  private lastFullAt = 0;
  private sinceFull = 0;
  private size = '';

  constructor(private readonly detector: Detector) {}

  reset(): void {
    this.hb = null;
    this.sinceFull = 0;
    this.size = '';
  }

  get locked(): boolean {
    return this.hb !== null;
  }

  process(input: Input, width: number, height: number, params: Params, opts: DetectOptions = {}, nowMs = performance.now()): SessionResult {
    const size = `${width}x${height}`;
    let reason: SessionResult['fullReason'] = 'no-lock';
    let failedTrack: DetectResult | null = null;
    if (size !== this.size) {
      this.hb = null;
      this.size = size;
      reason = 'resized';
    }
    const every = param(params, TRACK_PARAMS, 'trackFullEvery');
    const maxMs = param(params, TRACK_PARAMS, 'trackMaxMs');
    if (this.hb) {
      if (param(params, TRACK_PARAMS, 'tracking') <= 0) reason = 'disabled';
      else if (every > 0 && this.sinceFull >= every) reason = 'periodic';
      else if (nowMs - this.lastFullAt > maxMs) reason = 'age';
      else {
        const tr = this.detector.track(input, params, this.hb, opts);
        if (tr.corners && tr.hb) {
          this.hb = tr.hb;
          this.sinceFull++;
          return { ...tr, mode: 'tracking' };
        }
        failedTrack = tr;
        reason = 'track-failed';
      }
    }
    const full = this.detector.detect(input, params, opts);
    if (failedTrack) full.timings.trackFailed = failedTrack.timings.total ?? 0;
    if (full.corners && full.hb) {
      this.hb = full.hb;
      this.lastFullAt = nowMs;
      this.sinceFull = 0;
    } else {
      this.hb = null;
    }
    return { ...full, mode: 'full', fullReason: reason };
  }
}
