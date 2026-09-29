import type { Params } from '../worker/protocol';
import type { DetectResult } from './detector';

/**
 * What kind of board the camera is looking at, learned from confident detections. It lets the pipeline compute
 * only what that board needs (performance) and judge the checker verification against the observed contrast.
 */
export interface BoardProfile {
  /** Subset of [0 L, 1 a*, 2 b*] for which gradient and edges are computed. */
  channels: number[];
  /** Channel on which the checker alternation is verified. */
  verifyChannel: number;
  /** Signed median pair contrast observed on `verifyChannel` (channel units). */
  contrast: number;
  /** What surrounds the 8x8 squares: dark (black ring), frame (neutral/coloured frame) or bare (light margin). */
  surround: 'dark' | 'frame' | 'bare';
}

export const PROFILE_PARAMS = [
  { name: 'profileLockCount', min: 1, max: 10, step: 1, default: 3 },
  { name: 'profileUnlockMs', min: 1000, max: 30000, step: 500, default: 5000 },
] as const;

const paramOf = (params: Params, name: (typeof PROFILE_PARAMS)[number]['name']): number =>
  params[name] ?? PROFILE_PARAMS.find((p) => p.name === name)!.default;

/** Ring level thresholds (0 = dark squares, 1 = light squares). */
const DARK_BELOW = 0.45;
const BARE_ABOVE = 0.75;

export function classifySurround(ringLevel: number | undefined): BoardProfile['surround'] {
  if (ringLevel === undefined || Number.isNaN(ringLevel)) return 'frame';
  return ringLevel < DARK_BELOW ? 'dark' : ringLevel > BARE_ABOVE ? 'bare' : 'frame';
}

/** Profile implied by one detection, or null when the detection carries no verification data. */
export function inferProfile(result: DetectResult): BoardProfile | null {
  const v = result.verify;
  if (!result.corners || !v || !Number.isFinite(v.contrast) || v.contrast === 0) return null;
  const channels = v.channel === 0 ? [0] : [0, v.channel];
  return { channels, verifyChannel: v.channel, contrast: v.contrast, surround: classifySurround(v.ringLevel) };
}

export function describeProfile(p: BoardProfile | null): string {
  if (!p) return 'auto (unlocked)';
  const chroma = p.channels.filter((c) => c > 0).map((c) => (c === 1 ? 'a*' : 'b*'));
  const ch = chroma.length === 0 ? 'L-only' : `L+${chroma.join('+')}`;
  return `${ch}, ${p.surround} ring, contrast ${Math.abs(p.contrast).toFixed(1)}`;
}

/**
 * Locks a BoardProfile after `profileLockCount` consecutive confident detections that agree on the verify
 * channel, and drops it after `profileUnlockMs` without an accepted detection so that a different board can be
 * picked up.
 */
export class ProfileLock {
  private locked: BoardProfile | null = null;
  private streak: BoardProfile[] = [];
  private lastAccepted = 0;

  get profile(): BoardProfile | null {
    return this.locked;
  }

  /** Feed one detection (`null` = no board). Returns the profile now in force. */
  update(result: DetectResult | null, nowMs: number, params: Params = {}): BoardProfile | null {
    const accepted = result?.corners ? result : null;
    if (accepted) this.lastAccepted = nowMs;
    else if (this.locked && nowMs - this.lastAccepted > paramOf(params, 'profileUnlockMs')) this.reset();
    if (this.locked) return this.locked;
    const p = accepted ? inferProfile(accepted) : null;
    if (!p) {
      this.streak = [];
      return null;
    }
    if (this.streak.length && this.streak[0]!.verifyChannel !== p.verifyChannel) this.streak = [];
    this.streak.push(p);
    if (this.streak.length >= paramOf(params, 'profileLockCount')) {
      // Lock the weakest observed contrast so that later dimmer frames are not judged against a peak.
      const contrasts = this.streak.map((s) => Math.abs(s.contrast)).sort((a, b) => a - b);
      const last = this.streak[this.streak.length - 1]!;
      const surround = this.streak.every((s) => s.surround === last.surround) ? last.surround : 'frame';
      this.locked = { ...last, contrast: Math.sign(last.contrast) * contrasts[0]!, surround };
      this.streak = [];
    }
    return this.locked;
  }

  reset(): void {
    this.locked = null;
    this.streak = [];
    this.lastAccepted = 0;
  }
}
