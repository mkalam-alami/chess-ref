import { describe, expect, it } from 'vitest';
import { Chess } from 'chess.js';
import { gridOf, sqIndex, START_GRID } from '../src/game/board';
import { GameTracker, type GameParams } from '../src/game/game';
import type { GameEvent, Observation } from '../src/game/types';
import { OCC_EMPTY, OCC_WHITE } from '../src/worker/protocol';

// ---------------------------------------------------------------------------------------------------------------
// Synthetic observations

/** Deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface ObsOpts {
  /** Squares hidden from the camera: low vis, near-uniform (flattened) logLik. */
  occluded?: readonly string[];
  /** Per-square class probabilities overriding the grid (vis stays high). */
  probs?: Record<string, [number, number, number]>;
  stable?: boolean;
  /** Board not wholly in view (vision then also reports stable = false). */
  framed?: boolean;
  oriented?: boolean;
  calibration?: Observation['calibration'];
}

const FRAME_MS = 33;

/** One frame showing `grid`: p(true class) ≈ 0.9 ± noise, 2 % of squares misread, occluded squares flattened. */
function observation(grid: Uint8Array, t: number, rand: () => number, o: ObsOpts = {}): Observation {
  const logLik = new Float32Array(192);
  const vis = new Float32Array(64);
  const occluded = new Set((o.occluded ?? []).map(sqIndex));
  for (let sq = 0; sq < 64; sq++) {
    const k = grid[sq]!;
    let p = [0, 0, 0];
    const over = Object.entries(o.probs ?? {}).find(([name]) => sqIndex(name) === sq);
    if (over) {
      p = over[1].slice();
      vis[sq] = 0.9;
    } else if (occluded.has(sq)) {
      p = [1 / 3 + 0.02 * (rand() - 0.5), 1 / 3 + 0.02 * (rand() - 0.5), 1 / 3];
      vis[sq] = 0.15;
    } else {
      vis[sq] = 0.7 + 0.3 * rand();
      const g = Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
      const pk = Math.min(0.99, Math.max(0.55, 0.9 + 0.05 * g));
      const u = 0.3 + 0.4 * rand();
      const a = (k + 1) % 3;
      const b = (k + 2) % 3;
      p[k] = pk;
      p[a] = (1 - pk) * u;
      p[b] = (1 - pk) * (1 - u);
      if (rand() < 0.02) {
        const w = rand() < 0.5 ? a : b;
        p[w] = 0.7;
        p[k] = 0.2;
        p[w === a ? b : a] = 0.1;
      }
    }
    const s = p[0]! + p[1]! + p[2]!;
    for (let c = 0; c < 3; c++) logLik[sq * 3 + c] = Math.log(p[c]! / s);
  }
  return { t, logLik, vis, oriented: o.oriented ?? true, calibration: o.calibration ?? 'calibrated', stable: o.stable ?? true, framed: o.framed ?? true };
}

/** Hand over the board: unstable frames with garbage evidence (or strong evidence for `grid`, to check it is ignored). */
function handObservation(t: number, rand: () => number, grid?: Uint8Array): Observation {
  if (grid) return { ...observation(grid, t, rand), stable: false };
  const logLik = new Float32Array(192);
  const vis = new Float32Array(64);
  for (let sq = 0; sq < 64; sq++) {
    vis[sq] = rand();
    const p = [rand() + 0.01, rand() + 0.01, rand() + 0.01];
    const s = p[0]! + p[1]! + p[2]!;
    for (let c = 0; c < 3; c++) logLik[sq * 3 + c] = Math.log(p[c]! / s);
  }
  return { t, logLik, vis, oriented: true, calibration: 'calibrated', stable: false, framed: true };
}

/** Occupancy after a SAN line from the start. */
function gridAfter(...sans: string[]): Uint8Array {
  const c = new Chess();
  for (const s of sans) c.move(s);
  return gridOf(c);
}

/** Drives a tracker with a frame clock and collects events. */
class Sim {
  t = 1000;
  events: GameEvent[] = [];
  readonly rand: () => number;
  readonly game: GameTracker;

  constructor(
    params: Partial<GameParams> = {},
    seed = 1,
    readonly frameMs = FRAME_MS,
  ) {
    this.game = new GameTracker(params);
    this.rand = rng(seed);
  }

  /** Feeds frames for `ms`; returns the events of this call. */
  feed(ms: number, make: (t: number) => Observation): GameEvent[] {
    const out: GameEvent[] = [];
    for (const end = this.t + ms; this.t < end; this.t += this.frameMs) out.push(...this.game.observe(make(this.t)));
    this.events.push(...out);
    return out;
  }

  show(grid: Uint8Array, ms: number, o: ObsOpts = {}): GameEvent[] {
    return this.feed(ms, (t) => observation(grid, t, this.rand, o));
  }

  hand(ms: number, grid?: Uint8Array): GameEvent[] {
    return this.feed(ms, (t) => handObservation(t, this.rand, grid));
  }

  /** Lock-in on the start position. */
  start(): void {
    const ev = this.show(START_GRID, 1500);
    expect(ev.map((e) => e.type)).toEqual(['started']);
  }

  /** Plays a SAN line physically: hand, then the new position held for `holdMs`. */
  play(line: string[], holdMs = 1000): void {
    const done = this.sans();
    for (let i = 0; i < line.length; i++) {
      this.hand(300);
      this.show(gridAfter(...done, ...line.slice(0, i + 1)), holdMs);
    }
  }

  sans(): string[] {
    return this.game.snapshot().plies.map((p) => p.san);
  }
}

const count = (ev: GameEvent[], type: GameEvent['type']) => ev.filter((e) => e.type === type).length;

// ---------------------------------------------------------------------------------------------------------------

describe('GameTracker lock-in', () => {
  it('locks in on a confident, calibrated, oriented start position, despite a few occluded squares', () => {
    const sim = new Sim();
    expect(sim.game.snapshot().state).toBe('waiting');
    expect(sim.game.snapshot().grid).toBeNull();
    const ev = sim.show(START_GRID, 1500, { occluded: ['d8', 'e8', 'f7', 'c7'] });
    expect(ev).toEqual([{ type: 'started' }]);
    const s = sim.game.snapshot();
    expect(s.state).toBe('playing');
    expect(s.fen).toBe(new Chess().fen());
    expect(s.pieces[sqIndex('e1')]).toBe('wK');
    expect(s.pieces[sqIndex('d8')]).toBe('bQ');
    expect(s.pieces[sqIndex('e4')]).toBeNull();
    expect(Array.from(s.grid!)).toEqual(Array.from(START_GRID));
  });

  it('needs about a second of stable frames', () => {
    const sim = new Sim();
    expect(sim.show(START_GRID, 600)).toEqual([]);
    sim.hand(100);
    expect(sim.show(START_GRID, 600)).toEqual([]);
    expect(sim.show(START_GRID, 600)).toEqual([{ type: 'started' }]);
  });

  it('does not lock in on fallback calibration, unoriented frames, or a wrong setup', () => {
    for (const o of [{ calibration: 'fallback' as const }, { calibration: 'start' as const }, { oriented: false }]) {
      const sim = new Sim();
      expect(sim.show(START_GRID, 3000, o)).toEqual([]);
      expect(sim.game.snapshot().state).toBe('waiting');
    }
    const missing = START_GRID.slice();
    missing[sqIndex('e2')] = OCC_EMPTY;
    const sim = new Sim();
    expect(sim.show(missing, 3000)).toEqual([]);
    const colour = START_GRID.slice();
    colour[sqIndex('a8')] = OCC_WHITE;
    expect(sim.show(colour, 3000)).toEqual([]);
  });
});

describe('GameTracker moves', () => {
  // Captures, en passant (3. exf6), promotion with capture (5. gxh8=Q), queenside castling for black, kingside for white.
  const GAME = ['e4', 'd5', 'e5', 'f5', 'exf6', 'Nc6', 'fxg7', 'Be6', 'gxh8=Q', 'Qd6', 'Nf3', 'O-O-O', 'Bc4', 'dxc4', 'O-O'];

  it('follows a noisy game with captures, en passant, promotion and both castlings; PGN re-imports', () => {
    const sim = new Sim({}, 7);
    sim.start();
    sim.play(GAME);
    expect(sim.sans()).toEqual(GAME);
    expect(count(sim.events, 'move')).toBe(GAME.length);
    expect(count(sim.events, 'revised')).toBe(0);
    expect(count(sim.events, 'desync')).toBe(0);
    const s = sim.game.snapshot();
    expect(s.plies.map((p) => p.tentative)).toEqual(GAME.map((_, i) => i === GAME.length - 1));
    expect(s.lastMove).toEqual({ from: sqIndex('e1'), to: sqIndex('g1') });
    expect(s.pieces[sqIndex('h8')]).toBe('wQ');
    expect(s.pieces[sqIndex('c8')]).toBe('bK');
    expect(s.pieces[sqIndex('d8')]).toBe('bR');
    expect(s.pieces[sqIndex('f6')]).toBeNull();
    expect(s.plies[8]!.promotion).toBe('q');
    expect(s.top.length).toBe(3);
    expect(s.top[0]!.line).toBe('O-O');

    const pgn = sim.game.pgn();
    expect(pgn).toMatch(/\[Date "\d{4}\.\d{2}\.\d{2}"\]/);
    for (const tag of ['Event', 'Site', 'Round', 'White', 'Black']) expect(pgn).toContain(`[${tag} "?"]`);
    expect(pgn).toContain('[Result "*"]');
    const back = new Chess();
    back.loadPgn(pgn);
    expect(back.history()).toEqual(GAME);
    expect(sim.game.pgn({ White: 'Alice' })).toContain('[White "Alice"]');

    // Promotion piece: not observable, queen by default, switchable when legal.
    expect(sim.game.setPromotion(0, 'n')).toBe(false);
    expect(sim.game.setPromotion(8, 'n')).toBe(true);
    const after = sim.game.snapshot();
    expect(after.plies[8]!.san).toBe('gxh8=N');
    expect(after.pieces[sqIndex('h8')]).toBe('wN');
    const back2 = new Chess();
    back2.loadPgn(sim.game.pgn());
    expect(back2.history()[8]).toBe('gxh8=N');
    // Tracking continues on the replayed game.
    sim.play(['Kb8']);
    expect(sim.sans().slice(-2)).toEqual(['O-O', 'Kb8']);
  });

  it('commits nothing during hand frames, even when they look like a move', () => {
    const sim = new Sim({}, 3);
    sim.start();
    expect(sim.hand(2000, gridAfter('e4'))).toEqual([]);
    expect(sim.hand(1000)).toEqual([]);
    expect(sim.show(START_GRID, 1000)).toEqual([]);
    // Interleaved hand frames break the dwell and the fast-path streak, so a move needs consecutive clean frames: with
    // single stable frames between hand frames, neither the hold nor the 2-frame fast path can ever complete. (This
    // used to show 300 ms glimpses against the 400 ms hold; a clean 300 ms is now enough evidence to commit on its own.)
    for (let i = 0; i < 20; i++) {
      expect(sim.show(gridAfter('e4'), FRAME_MS)).toEqual([]);
      sim.hand(100);
    }
    expect(sim.show(gridAfter('e4'), 1000).map((e) => e.type)).toEqual(['move']);
  });

  it('neither locks in nor commits while the board is cropped (framed = false, so stable = false)', () => {
    // Cropped board: the far edge is out of view (flattened evidence), the rest reads cleanly.
    const cropped: ObsOpts = { stable: false, framed: false, occluded: ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'] };
    const sim = new Sim({}, 4);
    expect(sim.show(START_GRID, 3000, cropped)).toEqual([]);
    expect(sim.game.snapshot().state).toBe('waiting');
    sim.start();
    // A position one move on, held well past the dwell time on cropped frames: no commit.
    expect(sim.show(gridAfter('e4'), 3000, cropped)).toEqual([]);
    expect(sim.sans()).toEqual([]);
    // Back in view: the move is committed from clean frames.
    expect(sim.show(gridAfter('e4'), 1000).map((e) => e.type)).toEqual(['move']);
    expect(sim.sans()).toEqual(['e4']);
  });

  it('follows a physical takeback (tentative ply) and reports it as a revision', () => {
    const sim = new Sim({}, 4);
    sim.start();
    sim.play(['e4']);
    sim.hand(300);
    const ev = sim.show(START_GRID, 1500);
    expect(ev).toEqual([{ type: 'revised', fromPly: 0, plies: [] }]);
    expect(sim.sans()).toEqual([]);
    sim.play(['e4', 'e5']);
    sim.hand(300);
    expect(sim.show(gridAfter('e4'), 1500)).toEqual([{ type: 'revised', fromPly: 1, plies: [] }]);
    sim.play(['c5']);
    expect(sim.sans()).toEqual(['e4', 'c5']);
    expect(count(sim.events, 'newGame')).toBe(0);
  });

  it('manual undo pops the last ply', () => {
    const sim = new Sim({}, 5);
    sim.start();
    sim.play(['e4', 'e5', 'Nf3']);
    expect(sim.game.undo()).toEqual([{ type: 'undone' }]);
    expect(sim.sans()).toEqual(['e4', 'e5']);
    // The board still shows Nf3: the lattice commits it again.
    sim.show(gridAfter('e4', 'e5', 'Nf3'), 1000);
    expect(sim.sans()).toEqual(['e4', 'e5', 'Nf3']);
  });

  it('ends on mate', () => {
    const sim = new Sim({}, 6);
    sim.start();
    sim.play(['f3', 'e5', 'g4', 'Qh4#']);
    const s = sim.game.snapshot();
    expect(s.state).toBe('over');
    expect(s.result).toBe('0-1');
    expect(s.check).toBe(true);
    expect(sim.game.pgn()).toContain('[Result "0-1"]');
  });
});

describe('GameTracker corrections', () => {
  /** e4 committed from misleading evidence: the d-pawn squares hidden, the e-pawn squares misread. */
  function wrongE4(sim: Sim): void {
    sim.hand(300);
    const ev = sim.show(gridAfter('e4'), 1000, { occluded: ['d2', 'd4'] });
    expect(ev.map((e) => e.type)).toEqual(['move']);
    expect(sim.sans()).toEqual(['e4']);
  }

  it('revises e4 into d4 once d4 is revealed', () => {
    const sim = new Sim({}, 11);
    sim.start();
    wrongE4(sim);
    const ev = sim.show(gridAfter('d4'), 1500);
    expect(ev.length).toBe(1);
    const e = ev[0]!;
    expect(e.type).toBe('revised');
    if (e.type !== 'revised') return;
    expect(e.fromPly).toBe(0);
    expect(e.plies.map((p) => p.san)).toEqual(['d4']);
    expect(e.plies[0]!.correctedAt).toBeGreaterThan(0);
    expect(sim.sans()).toEqual(['d4']);
    expect(sim.game.snapshot().plies[0]!.tentative).toBe(true);
  });

  it('revises e4 into [d4, e5] when black replied before d4 was seen', () => {
    const sim = new Sim({}, 12);
    sim.start();
    wrongE4(sim);
    sim.hand(400);
    const ev = sim.show(gridAfter('d4', 'e5'), 2000);
    expect(ev.map((e) => e.type)).toEqual(['revised']);
    expect(sim.sans()).toEqual(['d4', 'e5']);
    const s = sim.game.snapshot();
    expect(s.plies[0]!.correctedAt).toBeDefined();
    expect(s.plies[1]!.correctedAt).toBeUndefined();
    expect(s.plies.map((p) => p.tentative)).toEqual([false, true]);
    sim.play(['c4']);
    expect(sim.sans()).toEqual(['d4', 'e5', 'c4']);
  });

  it('K = 2 revises two plies back, including with a third ply on top', () => {
    for (const reply of [[], ['Nf3']]) {
      const sim = new Sim({ revisionDepth: 2 }, 13);
      sim.start();
      wrongE4(sim);
      sim.hand(300);
      sim.show(gridAfter('e4', 'e5'), 1000, { occluded: ['d2', 'd4'] });
      expect(sim.sans()).toEqual(['e4', 'e5']);
      expect(sim.game.snapshot().plies.map((p) => p.tentative)).toEqual([true, true]);
      sim.hand(300);
      const ev = sim.show(gridAfter('d4', 'e5', ...reply), 2000);
      expect(ev.map((e) => e.type)).toEqual(['revised']);
      const e = ev[0]!;
      if (e.type === 'revised') expect(e.fromPly).toBe(0);
      expect(sim.sans()).toEqual(['d4', 'e5', ...reply]);
      expect(sim.game.snapshot().plies[0]!.correctedAt).toBeDefined();
    }
    // With K = 1 the same late evidence can't revise the (now final) first ply (the best it can do is add d4 on top).
    const k1 = new Sim({}, 13);
    k1.start();
    wrongE4(k1);
    k1.hand(300);
    k1.show(gridAfter('e4', 'e5'), 1000, { occluded: ['d2', 'd4'] });
    k1.hand(300);
    k1.show(gridAfter('d4', 'e5'), 2000);
    expect(k1.sans().slice(0, 2)).toEqual(['e4', 'e5']);
  });

  /** Moderate evidence for e4 (phase 'e') or d4 (phase 'd'): each cell 0.65 vs 0.24. */
  function lean(which: 'e' | 'd'): Record<string, [number, number, number]> {
    const hi = 0.65;
    const lo = 0.24;
    const rest = 1 - hi - lo;
    const emptyish: [number, number, number] = [hi, lo, rest];
    const whiteish: [number, number, number] = [lo, hi, rest];
    return which === 'e' ? { e2: emptyish, e4: whiteish, d2: whiteish, d4: emptyish } : { e2: whiteish, e4: emptyish, d2: emptyish, d4: whiteish };
  }

  it('anti-flicker: alternating near-equal hypotheses switch at most once', () => {
    const run = (params: Partial<GameParams>) => {
      const sim = new Sim(params, 21);
      sim.start();
      for (const phase of ['e', 'd', 'e', 'd'] as const) sim.show(START_GRID, 1500, { probs: lean(phase) });
      return sim;
    };
    const sim = run({});
    expect(count(sim.events, 'move')).toBe(1);
    expect(count(sim.events, 'revised')).toBe(1);
    expect(sim.sans()).toEqual(['d4']);
    // Without the revision cooldown the same evidence ping-pongs.
    expect(count(run({ revisionCooldownMs: 0 }).events, 'revised')).toBeGreaterThan(1);
  });

  it('anti-flicker: randomly interleaved near-equal evidence commits at most one hypothesis', () => {
    const sim = new Sim({}, 22);
    sim.start();
    const r = rng(99);
    sim.feed(6000, (t) => observation(START_GRID, t, sim.rand, { probs: lean(r() < 0.5 ? 'e' : 'd') }));
    expect(count(sim.events, 'move') + count(sim.events, 'revised')).toBeLessThanOrEqual(1);
  });
});

describe('GameTracker desync and new game', () => {
  it('raises desync on an unexplained board and resyncs when it matches again', () => {
    const sim = new Sim({}, 31);
    sim.start();
    sim.play(['e4', 'e5']);
    const bad = gridAfter('e4', 'e5');
    for (const sq of ['a1', 'b1', 'h8', 'g8']) bad[sqIndex(sq)] = OCC_EMPTY;
    sim.hand(300);
    expect(sim.show(bad, 2500)).toEqual([]);
    expect(sim.show(bad, 1500)).toEqual([{ type: 'desync' }]);
    expect(sim.game.snapshot().state).toBe('desync');
    expect(sim.show(bad, 1000)).toEqual([]);
    sim.hand(300);
    expect(sim.show(gridAfter('e4', 'e5'), 1500)).toEqual([{ type: 'resynced' }]);
    expect(sim.game.snapshot().state).toBe('playing');
    sim.play(['Nf3']);
    expect(sim.sans()).toEqual(['e4', 'e5', 'Nf3']);
  });

  it('continueAfterDesync resumes play', () => {
    const sim = new Sim({}, 32);
    sim.start();
    sim.play(['d4']);
    const bad = gridAfter('d4');
    for (const sq of ['a8', 'b8', 'c8', 'h1']) bad[sqIndex(sq)] = OCC_EMPTY;
    sim.show(bad, 4000);
    expect(sim.game.state).toBe('desync');
    sim.game.continueAfterDesync();
    expect(sim.game.state).toBe('playing');
  });

  it('starts a new game automatically on a start position held after moves; load restores the old one', () => {
    const sim = new Sim({}, 41);
    sim.start();
    sim.play(['e4', 'e5', 'Nf3']);
    sim.hand(1000);
    const ev = sim.show(START_GRID, 3000);
    expect(ev.map((e) => e.type)).toEqual(['newGame']);
    const e = ev[0]!;
    if (e.type !== 'newGame') return;
    expect(e.previousPgn).toContain('1. e4 e5 2. Nf3');
    expect(sim.sans()).toEqual([]);
    expect(sim.game.state).toBe('playing');
    sim.play(['d4']);
    expect(sim.sans()).toEqual(['d4']);
    // "Undo" of the automatic new game.
    expect(sim.game.load(e.previousPgn)).toBe(true);
    expect(sim.sans()).toEqual(['e4', 'e5', 'Nf3']);
  });

  it('does not start a new game after a mere transposition back to the start grid', () => {
    const sim = new Sim({}, 42);
    sim.start();
    sim.play(['Nf3', 'Nf6', 'Ng1', 'Ng8']);
    sim.show(START_GRID, 3000);
    expect(sim.sans()).toEqual(['Nf3', 'Nf6', 'Ng1', 'Ng8']);
    expect(count(sim.events, 'newGame')).toBe(0);
  });
});

describe('GameTracker persistence', () => {
  it('round-trips save/load and rejects garbage without changing state', () => {
    const sim = new Sim({}, 51);
    sim.start();
    sim.play(['e4', 'c5', 'Nf3']);
    const saved = sim.game.save();
    const other = new GameTracker();
    expect(other.load(saved)).toBe(true);
    const a = sim.game.snapshot();
    const b = other.snapshot();
    expect(b.fen).toBe(a.fen);
    expect(b.plies.map((p) => [p.san, p.tentative])).toEqual(a.plies.map((p) => [p.san, p.tentative]));
    expect(b.state).toBe('playing');

    const before = JSON.stringify(other.snapshot());
    for (const junk of ['', 'garbage', '{}', 'null', '42', '[1,2]', '{"format":"chess-ref-game","moves":["e2e5"]}', '{"format":"chess-ref-game","moves":[1]}', '1. e4 e9', '{"a":1}']) {
      expect(other.load(junk)).toBe(false);
    }
    expect(JSON.stringify(other.snapshot())).toBe(before);

    // A waiting tracker saves as waiting.
    const w = new GameTracker();
    expect(other.load(w.save())).toBe(true);
    expect(other.state).toBe('waiting');
    // The restored game keeps tracking.
    const sim2 = new Sim({}, 52);
    expect(sim2.game.load(saved)).toBe(true);
    sim2.play(['d6']);
    expect(sim2.sans()).toEqual(['e4', 'c5', 'Nf3', 'd6']);
  });
});

describe('GameTracker performance', () => {
  it('K = 2 stays within a few ms per frame over 2000 frames', () => {
    const line = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', 'O-O', 'Be7', 'Re1', 'b5', 'Bb3', 'd6', 'c3', 'O-O', 'h3', 'Nb8', 'd4', 'Nbd7'];
    const rand = rng(77);
    const frames: Observation[] = [];
    let t = 0;
    const push = (n: number, f: () => Observation) => {
      for (let i = 0; i < n; i++, t += FRAME_MS) frames.push(f());
    };
    push(45, () => observation(START_GRID, t, rand));
    for (let i = 0; frames.length < 2000; i = (i + 1) % line.length) {
      const g = gridAfter(...line.slice(0, i + 1));
      push(8, () => handObservation(t, rand));
      push(30, () => observation(g, t, rand, { occluded: rand() < 0.3 ? ['c7', 'f7'] : [] }));
      if (i === line.length - 1) push(90, () => observation(START_GRID, t, rand));
    }
    frames.length = 2000;
    for (const K of [1, 2]) {
      const game = new GameTracker({ revisionDepth: K });
      const times: number[] = [];
      let moves = 0;
      const t0 = performance.now();
      for (const o of frames) {
        const a = performance.now();
        moves += game.observe(o).filter((e) => e.type === 'move').length;
        times.push(performance.now() - a);
      }
      const avg = (performance.now() - t0) / frames.length;
      times.sort((x, y) => x - y);
      const q = (f: number) => times[Math.floor(f * (times.length - 1))]!.toFixed(3);
      console.log(`GameTracker K=${K}: ${avg.toFixed(3)} ms/frame avg, median ${q(0.5)}, p95 ${q(0.95)}, worst ${q(1)} ms, ${moves} moves`);
      expect(moves).toBeGreaterThan(40);
      expect(avg).toBeLessThan(K === 1 ? 2 : 5);
    }
  });
});


describe('GameTracker latency (10 detections/s)', () => {
  const MS = 100;
  interface Scenario {
    name: string;
    /** Committed line before the move. */
    before: string[];
    move: string;
    occluded?: string[];
  }
  const SCENARIOS: Scenario[] = [
    { name: 'clean move', before: [], move: 'e4' },
    { name: 'occluded far rank', before: ['e4'], move: 'e5', occluded: ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 'e7'] },
    { name: 'capture', before: ['e4', 'd5'], move: 'exd5' },
    { name: 'castling', before: ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5'], move: 'O-O' },
  ];

  /** A saved game (GameTracker.save format) at the end of a SAN line: skips the lock-in. */
  function savedGame(line: string[]): string {
    const c = new Chess();
    for (const m of line) c.move(m);
    return JSON.stringify({ format: 'chess-ref-game', v: 1, state: 'playing', moves: c.history({ verbose: true }).map((m) => m.lan) });
  }

  /** The pre-change tuning (400 ms hold, no fast path, hands fade evidence like stable frames), for comparison. */
  const BEFORE: Partial<GameParams> = { moveHoldMs: 400, fastMargin: Infinity, unstableDecay: 0.85 };

  /** A tracker at `s.before`, with that position observed for 2 s, then `unstable` hand/settle frames. */
  function setup(s: Scenario, seed: number, unstable: number, params: Partial<GameParams> = {}): Sim {
    const sim = new Sim(params, seed, MS);
    expect(sim.game.load(savedGame(s.before))).toBe(true);
    sim.show(gridAfter(...s.before), 2000, { occluded: s.occluded });
    expect(sim.sans()).toEqual(s.before);
    expect(sim.hand(unstable * MS)).toEqual([]);
    return sim;
  }

  /**
   * Stable frames from the hand leaving (after `unstable` hand/settle frames) to the commit of `s.move`, the frame of
   * the commit included; Infinity when it doesn't commit within 3 s. Checks that nothing else is committed.
   */
  function latency(s: Scenario, seed: number, unstable = 6, params: Partial<GameParams> = {}, o: ObsOpts = {}): { frames: number; sim: Sim } {
    const sim = setup(s, seed, unstable, params);
    const target = gridAfter(...s.before, s.move);
    for (let f = 1; f <= 30; f++) {
      const ev = sim.show(target, MS, { occluded: s.occluded, ...o });
      if (ev.length) {
        expect(ev.map((e) => e.type)).toEqual(['move']);
        expect(sim.sans()).toEqual([...s.before, s.move]);
        return { frames: f, sim };
      }
    }
    return { frames: Infinity, sim };
  }

  const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  /** Hand + settle frames before the move is seen (400 ms settle alone is 4). */
  const HANDS = [4, 6, 10];
  const WEAK: ObsOpts = { probs: { e2: [0.65, 0.24, 0.11], e4: [0.24, 0.65, 0.11] } };

  it('commits a clean move within 3 stable frames after the hand; reports before/after latency', () => {
    const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1]!;
    for (const s of SCENARIOS) {
      for (const n of HANDS) {
        const after = SEEDS.map((seed) => latency(s, seed, n).frames);
        const before = SEEDS.map((seed) => latency(s, seed, n, BEFORE).frames);
        console.log(`latency ${s.name}, ${n} unstable frames: before median ${med(before)} max ${Math.max(...before)} [${before.join(' ')}], after median ${med(after)} max ${Math.max(...after)} [${after.join(' ')}] stable frames`);
        expect(Math.max(...after)).toBeLessThan(Infinity);
        expect(med(after)).toBeLessThan(med(before));
        if (s.name !== 'occluded far rank') {
          // Clean evidence: 3 stable frames at most, except when the synthetic 2 % per-cell misread hits one of the
          // move's cells on a deciding frame (that frame isn't decisive, and the fast streak restarts).
          expect(med(after)).toBeLessThanOrEqual(3);
          expect(after.filter((f) => f <= 3).length).toBeGreaterThanOrEqual(SEEDS.length - 2);
          expect(Math.max(...after)).toBeLessThanOrEqual(5);
        }
      }
    }
  });

  it('a marginal move (weak contrast on its cells) still commits, through the hold', () => {
    for (const seed of SEEDS.slice(0, 4)) {
      const { frames, sim } = latency(SCENARIOS[0]!, seed, 6, {}, WEAK);
      expect(frames).toBeGreaterThan(3);
      expect(frames).toBeLessThan(Infinity);
      const tm = sim.game.snapshot().lastTiming!;
      expect(tm.kind).toBe('advance');
      expect(tm.committedAt - tm.favouriteAt).toBeGreaterThanOrEqual(sim.game.params.moveHoldMs);
    }
  });

  it('near-equal alternating evidence switches at most once', () => {
    const lean = (which: 'e' | 'd'): Record<string, [number, number, number]> => {
      const em: [number, number, number] = [0.65, 0.24, 0.11];
      const wh: [number, number, number] = [0.24, 0.65, 0.11];
      return which === 'e' ? { e2: em, e4: wh, d2: wh, d4: em } : { e2: wh, e4: em, d2: em, d4: wh };
    };
    for (const seed of [1, 2, 3]) {
      const sim = new Sim({}, seed, MS);
      expect(sim.game.load(savedGame([]))).toBe(true);
      sim.show(START_GRID, 2000);
      const r = rng(seed * 17);
      sim.feed(10000, (t) => observation(START_GRID, t, sim.rand, { probs: lean(r() < 0.5 ? 'e' : 'd') }));
      expect(count(sim.events, 'move') + count(sim.events, 'revised')).toBeLessThanOrEqual(1);
      const alt = new Sim({}, seed, MS);
      expect(alt.game.load(savedGame([]))).toBe(true);
      alt.show(START_GRID, 2000);
      for (const phase of ['e', 'd', 'e', 'd'] as const) alt.show(START_GRID, 2000, { probs: lean(phase) });
      expect(count(alt.events, 'move')).toBe(1);
      expect(count(alt.events, 'revised')).toBeLessThanOrEqual(1);
    }
  });

  it('commits nothing during unstable frames, however clear they look', () => {
    for (const seed of [1, 2]) {
      const sim = setup(SCENARIOS[0]!, seed, 0);
      expect(sim.hand(3000, gridAfter('e4'))).toEqual([]);
      expect(sim.game.snapshot().pending).toBeNull();
      // Single clean frames between hand frames: no 2-frame streak, no hold.
      for (let i = 0; i < 20; i++) {
        expect(sim.show(gridAfter('e4'), MS)).toEqual([]);
        expect(sim.hand(2 * MS, gridAfter('e4'))).toEqual([]);
      }
      expect(sim.show(gridAfter('e4'), 3 * MS).map((e) => e.type)).toEqual(['move']);
    }
  });

  it('snapshot().pending follows the challenger until the commit', () => {
    const sim = setup(SCENARIOS[0]!, 3, 6);
    expect(sim.game.snapshot().pending).toBeNull();
    const progress: number[] = [];
    for (let f = 0; f < 40 && sim.sans().length === 0; f++) {
      sim.show(gridAfter('e4'), MS, WEAK);
      const p = sim.game.snapshot().pending;
      if (sim.sans().length) expect(p).toBeNull();
      else if (p) {
        expect(p.kind).toBe('advance');
        expect(p.fromPly).toBe(0);
        expect(p.plies).toEqual([{ san: 'e4', from: sqIndex('e2'), to: sqIndex('e4'), tentative: true }]);
        progress.push(p.progress);
      }
    }
    expect(sim.sans()).toEqual(['e4']);
    expect(progress.length).toBeGreaterThan(2);
    for (const x of progress) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
    expect(progress[progress.length - 1]!).toBeGreaterThan(progress[0]!);
    // Incumbent favoured: no pending change.
    sim.show(gridAfter('e4'), 1000);
    expect(sim.game.snapshot().pending).toBeNull();
    // A takeback in progress: a shorter line (nothing replaces ply 0).
    sim.hand(600);
    let seen = false;
    for (let f = 0; f < 30 && sim.sans().length === 1; f++) {
      sim.show(START_GRID, MS);
      const p = sim.game.snapshot().pending;
      if (p && sim.sans().length === 1) {
        expect(p.kind).toBe('takeback');
        expect(p.fromPly).toBe(0);
        expect(p.plies).toEqual([]);
        seen = true;
      }
    }
    expect(seen).toBe(true);
    expect(sim.sans()).toEqual([]);
    expect(sim.game.snapshot().lastTiming!.kind).toBe('takeback');
  });

  it('snapshot().lastTiming records the hand, the favourite and the commit', () => {
    const s = SCENARIOS[2]!;
    const sim = setup(s, 5, 6);
    const handEnd = sim.t - MS;
    const firstStable = sim.t;
    const after = gridAfter(...s.before, s.move);
    sim.show(after, 1000);
    const tm = sim.game.snapshot().lastTiming!;
    expect(tm.kind).toBe('advance');
    expect(tm.unstableEndAt).toBe(handEnd);
    expect(tm.stableAt).toBe(firstStable);
    expect(tm.favouriteAt).toBeGreaterThanOrEqual(firstStable);
    expect(tm.committedAt).toBeGreaterThanOrEqual(tm.favouriteAt);
    expect(tm.committedAt - firstStable).toBeLessThanOrEqual(2 * MS);
    // Kept through later observations, including hands.
    sim.hand(500);
    sim.show(after, 1000);
    expect(sim.game.snapshot().lastTiming).toEqual(tm);
    // A commit without a preceding hand has no hand times.
    const noHand = setup(SCENARIOS[0]!, 6, 0);
    expect(noHand.game.snapshot().lastTiming).toBeNull();
    noHand.show(gridAfter('e4'), 1000);
    const t2 = noHand.game.snapshot().lastTiming!;
    expect(t2.stableAt).toBeNull();
    expect(t2.unstableEndAt).toBeNull();
  });

  it('snapshot() is cheap enough to call on every frame', () => {
    const line = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', 'O-O', 'Be7'];
    const sim = new Sim({}, 9, MS);
    expect(sim.game.load(savedGame(line))).toBe(true);
    sim.show(gridAfter(...line, 'Re1'), 200);
    const n = 10000;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) sim.game.snapshot();
    const us = ((performance.now() - t0) / n) * 1000;
    console.log(`snapshot(): ${us.toFixed(1)} µs/call`);
    expect(us).toBeLessThan(100);
  });
});
