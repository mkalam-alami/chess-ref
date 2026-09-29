import { describe, expect, it, vi } from 'vitest';
import type { GameEvent, PlyInfo } from '../src/game/types';
import { MIN_GAP_S, SoundPlayer, soundFor, type AudioLike } from '../src/ui/sound';
import { loadMuted, MUTED_KEY, saveMuted } from '../src/ui/storage';

const ply = (san: string): PlyInfo => ({ san, from: 0, to: 0, tentative: true });
const move = (san: string): GameEvent => ({ type: 'move', ply: ply(san) });

/** A minimal AudioContext mock that counts the scheduled sources. */
class FakeAudio {
  currentTime = 0;
  sampleRate = 8000;
  state: AudioContextState = 'running';
  destination = {} as AudioDestinationNode;
  started = 0;
  resume = vi.fn(async () => {
    this.state = 'running';
  });
  private param = () => ({ value: 0, setValueAtTime() {}, exponentialRampToValueAtTime() {}, linearRampToValueAtTime() {} });
  private node = () => ({ connect: () => undefined });
  private source = () => ({ ...this.node(), start: () => void this.started++, stop() {} });
  createGain = () => ({ ...this.node(), gain: this.param() });
  createBiquadFilter = () => ({ ...this.node(), type: 'lowpass', frequency: this.param(), Q: this.param() });
  createOscillator = () => ({ ...this.source(), type: 'sine', frequency: this.param() });
  createBufferSource = () => ({ ...this.source(), buffer: null });
  createBuffer = (_c: number, n: number, sampleRate: number) => {
    const d = new Float32Array(n);
    return { sampleRate, getChannelData: () => d };
  };
}

function player(muted = false) {
  const fake = new FakeAudio();
  const create = vi.fn(() => fake as unknown as AudioLike);
  return { fake, create, p: new SoundPlayer(muted, create) };
}

describe('soundFor', () => {
  it('picks one sound per batch of events', () => {
    expect(soundFor([])).toBeNull();
    expect(soundFor([{ type: 'started' }, { type: 'desync' }])).toBeNull();
    expect(soundFor([move('e4')])).toBe('move');
    expect(soundFor([move('e4'), move('e5'), move('Nf3')])).toBe('move');
    expect(soundFor([move('e4'), move('exd5')])).toBe('capture');
    expect(soundFor([{ type: 'revised', fromPly: 1, plies: [] }])).toBe('click');
    expect(soundFor([{ type: 'undone' }])).toBe('click');
    // A move wins over a correction in the same batch.
    expect(soundFor([{ type: 'revised', fromPly: 1, plies: [ply('d4')] }, move('d5')])).toBe('move');
  });
});

describe('SoundPlayer', () => {
  it('stays silent until unlocked, then plays one sound per batch', () => {
    const { fake, create, p } = player();
    expect(p.playFor([move('e4')])).toBeNull();
    expect(create).not.toHaveBeenCalled();
    p.unlock();
    p.unlock();
    expect(create).toHaveBeenCalledTimes(1);
    expect(p.playFor([move('e4'), move('e5')])).toBe('move');
    expect(fake.started).toBeGreaterThan(0);
  });

  it('does not stack sounds closer than MIN_GAP_S', () => {
    const { fake, p } = player();
    p.unlock();
    expect(p.play('move')).toBe(true);
    expect(p.play('click')).toBe(false);
    fake.currentTime += MIN_GAP_S;
    expect(p.play('click')).toBe(true);
  });

  it('respects mute', () => {
    const { fake, p } = player(true);
    p.unlock();
    expect(p.play('capture')).toBe(false);
    expect(fake.started).toBe(0);
    p.muted = false;
    expect(p.play('capture')).toBe(true);
  });

  it('resumes a suspended context instead of playing', () => {
    const { fake, p } = player();
    fake.state = 'suspended';
    p.unlock();
    expect(fake.resume).toHaveBeenCalled();
    fake.state = 'suspended';
    expect(p.play('move')).toBe(false);
  });

  it('fails silently without Web Audio', () => {
    const none = new SoundPlayer(false, () => null);
    none.unlock();
    expect(none.play('move')).toBe(false);
    const broken = new SoundPlayer(false, () => {
      throw new Error('no audio');
    });
    expect(() => broken.unlock()).not.toThrow();
    expect(broken.play('move')).toBe(false);
    // The default factory in an environment without AudioContext (node).
    const def = new SoundPlayer();
    expect(() => def.unlock()).not.toThrow();
    expect(def.play('click')).toBe(false);
  });
});

describe('mute persistence', () => {
  it('round-trips and defaults to sound on', () => {
    const map = new Map<string, string>();
    const s = { getItem: (k: string) => map.get(k) ?? null, setItem: (k: string, v: string) => void map.set(k, v), removeItem: (k: string) => void map.delete(k) };
    expect(loadMuted(s)).toBe(false);
    saveMuted(true, s);
    expect(map.get(MUTED_KEY)).toBe('1');
    expect(loadMuted(s)).toBe(true);
    saveMuted(false, s);
    expect(loadMuted(s)).toBe(false);
    expect(loadMuted(null)).toBe(false);
  });
});
