import { hbCorners, orientHomography } from '../geom/cornerOrder';
import type { Mat3 } from '../geom/homography';
import { Detector, TRACK_PARAMS, type DetectOptions, type DetectResult } from '../vision/detector';
import { param } from '../vision/preprocess';
import type { Corners, Params } from './protocol';

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
 *
 * Orientation: a full re-detection may return `hb` in any of the board's 8 dihedral labellings. Every accepted
 * `hb` is relabelled (orientHomography) to the one closest to the previous frame's, so board cell (i, j) keeps
 * referring to the same physical square while the lock holds, and the returned `corners[k]` are exactly `hb`
 * applied to board (0,0), (8,0), (8,8), (0,8). The orientation reference (last accepted `hb` and its time) is kept
 * separately from the tracking lock so it survives short board losses: a re-detection after a lost frame keeps the
 * same labelling. It is dropped on reset(), a frame-size change, or once older than `trackOrientMs`.
 */
export class TrackingSession {
  private hb: Mat3 | null = null;
  /** Orientation reference: last accepted hb and when it was accepted; survives board loss (see class doc). */
  private orientRef: Mat3 | null = null;
  private orientAt = 0;
  private lastFullAt = 0;
  private sinceFull = 0;
  private size = '';

  constructor(private readonly detector: Detector) {}

  reset(): void {
    this.hb = null;
    this.orientRef = null;
    this.sinceFull = 0;
    this.size = '';
  }

  private accept(hb: Mat3, nowMs: number): Mat3 {
    this.hb = hb;
    this.orientRef = hb;
    this.orientAt = nowMs;
    return hb;
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
      this.orientRef = null;
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
          const hb = this.accept(orientHomography(tr.hb, this.hb), nowMs);
          this.sinceFull++;
          return { ...tr, hb, corners: hbCorners(hb) as Corners, mode: 'tracking' };
        }
        failedTrack = tr;
        reason = 'track-failed';
      }
    }
    const full = this.detector.detect(input, params, opts);
    if (failedTrack) full.timings.trackFailed = failedTrack.timings.total ?? 0;
    if (full.corners && full.hb) {
      const orientMs = param(params, TRACK_PARAMS, 'trackOrientMs');
      const ref = this.hb ?? (this.orientRef && nowMs - this.orientAt <= orientMs ? this.orientRef : null);
      const hb = this.accept(orientHomography(full.hb, ref), nowMs);
      this.lastFullAt = nowMs;
      this.sinceFull = 0;
      return { ...full, hb, corners: hbCorners(hb) as Corners, mode: 'full', fullReason: reason };
    }
    this.hb = null;
    return { ...full, mode: 'full', fullReason: reason };
  }
}
