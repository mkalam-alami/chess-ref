import type { GameEvent } from '../game/types';

/**
 * Move sounds, synthesised with the Web Audio API (no audio files: nothing to license or download).
 * - move: a piece set down on a wooden board, a short band-passed noise burst (the contact) over a low damped
 *   resonance (the board), about 110 ms;
 * - capture: the same knock, doubled (the captured piece taken off, the capturing one set down);
 * - click: a small, quiet tick for corrections (revisions, takebacks, undo).
 * Mobile browsers only allow audio after a user gesture: `unlock()` must be called from one (the Start tap). Every
 * failure is silent.
 */
export type SoundKind = 'move' | 'capture' | 'click';

/** One sound for a batch of game events: a move (a capture if any new ply captures) wins over a correction click. */
export function soundFor(events: readonly GameEvent[]): SoundKind | null {
  let kind: SoundKind | null = null;
  for (const e of events) {
    if (e.type === 'move') {
      if (e.ply.san.includes('x')) return 'capture';
      kind = 'move';
    } else if ((e.type === 'revised' || e.type === 'undone') && kind === null) {
      kind = 'click';
    }
  }
  return kind;
}

/** Minimum time between two sounds (s): a burst of events never stacks knocks. */
export const MIN_GAP_S = 0.06;
/** Master volume (0..1). */
export const VOLUME = 0.8;

/** The parts of AudioContext used here (a mock in tests). */
export type AudioLike = Pick<
  BaseAudioContext,
  'currentTime' | 'sampleRate' | 'state' | 'destination' | 'createBuffer' | 'createBufferSource' | 'createBiquadFilter' | 'createGain' | 'createOscillator'
> & { resume(): Promise<void> };

function defaultContext(): AudioLike | null {
  const g = globalThis as { AudioContext?: typeof AudioContext; webkitAudioContext?: typeof AudioContext };
  const C = g.AudioContext ?? g.webkitAudioContext;
  return C ? new C() : null;
}

export class SoundPlayer {
  private ctx: AudioLike | null = null;
  private noise: AudioBuffer | null = null;
  private lastAt = -Infinity;

  constructor(
    public muted = false,
    private create: () => AudioLike | null = defaultContext,
  ) {}

  /** Creates / resumes the audio context; call from a user gesture. */
  unlock(): void {
    try {
      this.ctx ??= this.create();
      if (this.ctx && this.ctx.state !== 'running') this.ctx.resume().catch(() => {});
    } catch {
      this.ctx = null;
    }
  }

  /** Plays the sound for a batch of game events (at most one). Returns what was played, if anything. */
  playFor(events: readonly GameEvent[]): SoundKind | null {
    const kind = soundFor(events);
    return kind && this.play(kind) ? kind : null;
  }

  /** Plays a sound unless muted, not unlocked, suspended, or too soon after the previous one. */
  play(kind: SoundKind): boolean {
    const ctx = this.ctx;
    if (this.muted || !ctx) return false;
    try {
      if (ctx.state !== 'running') {
        ctx.resume().catch(() => {});
        return false;
      }
      const t = ctx.currentTime;
      if (t - this.lastAt < MIN_GAP_S) return false;
      this.lastAt = t;
      const out = ctx.createGain();
      out.gain.value = VOLUME;
      out.connect(ctx.destination);
      const t0 = t + 0.005;
      if (kind === 'click') {
        this.click(ctx, out, t0);
      } else {
        this.knock(ctx, out, t0, 1);
        if (kind === 'capture') this.knock(ctx, out, t0 + 0.07, 0.8, 1.15);
      }
      return true;
    } catch {
      return false;
    }
  }

  private noiseBuffer(ctx: AudioLike): AudioBuffer {
    if (this.noise && this.noise.sampleRate === ctx.sampleRate) return this.noise;
    const n = Math.round(ctx.sampleRate * 0.15);
    const buf = ctx.createBuffer(1, n, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let k = 0; k < n; k++) d[k] = Math.random() * 2 - 1;
    this.noise = buf;
    return buf;
  }

  /** Decaying envelope on a new gain node: attack to `peak` in 2 ms, exponential decay to silence at t + dur. */
  private env(ctx: AudioLike, out: AudioNode, t: number, peak: number, dur: number): GainNode {
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    g.connect(out);
    return g;
  }

  private noiseBurst(ctx: AudioLike, out: AudioNode, t: number, type: BiquadFilterType, freq: number, q: number, peak: number, dur: number): void {
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuffer(ctx);
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    src.connect(f);
    f.connect(this.env(ctx, out, t, peak, dur));
    src.start(t);
    src.stop(t + dur + 0.01);
  }

  private tone(ctx: AudioLike, out: AudioNode, t: number, type: OscillatorType, f0: number, f1: number, peak: number, dur: number): void {
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    o.connect(this.env(ctx, out, t, peak, dur));
    o.start(t);
    o.stop(t + dur + 0.01);
  }

  /** A piece set down on wood: contact noise, a hollow mid knock and the board's low thump. `pitch` scales it. */
  private knock(ctx: AudioLike, out: AudioNode, t: number, level: number, pitch = 1): void {
    this.noiseBurst(ctx, out, t, 'bandpass', 1500 * pitch, 1.2, 0.5 * level, 0.05);
    this.tone(ctx, out, t, 'triangle', 620 * pitch, 480 * pitch, 0.18 * level, 0.045);
    this.tone(ctx, out, t, 'sine', 210 * pitch, 130 * pitch, 0.6 * level, 0.11);
  }

  private click(ctx: AudioLike, out: AudioNode, t: number): void {
    this.noiseBurst(ctx, out, t, 'highpass', 2800, 0.7, 0.15, 0.02);
    this.tone(ctx, out, t, 'sine', 1300, 1100, 0.06, 0.025);
  }
}
