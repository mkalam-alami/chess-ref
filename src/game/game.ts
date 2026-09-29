import { Chess, type Move } from 'chess.js';
import { gridOf, piecesOf, sqIndex, START_FEN, START_GRID, uciMove } from './board';
import { Lattice, type EvidenceFrame, type HNode } from './lattice';
import type { ChangeTiming, GameEvent, GameSnapshot, GameState, Observation, PendingChange, PlyInfo } from './types';

/**
 * Tunables of the game layer (thresholds in summed log-likelihood units, times in ms of observation clock).
 *
 * Scale: a cell seen confidently (p ≈ 0.9) that a hypothesis gets wrong costs it about 2.5–3.5 per frame; with the
 * leaky sum (decay 0.85) a sustained difference is worth about 6.7× its per-frame value. A clean quiet move (2 cells)
 * therefore builds a margin of ~5–7 per frame, ~35–45 in steady state; a single-cell glitch can't be explained by any
 * legal move (every move changes at least 2 cells), so it never raises a hypothesis by itself.
 *
 * Latency (at ~10 frames/s): a change commits either
 * - on the **hold path**: margin τ over the incumbent (and τ_amb over the runner-up) held continuously for t_hold, which
 *   is what marginal evidence (occluded cells, weak contrast) goes through; or
 * - on the **fast path**: the same margins, plus `fastFrames` consecutive stable frames on each of which that frame's
 *   own evidence favours the challenger over the incumbent and over the runner-up by `fastMargin` (scaled like τ for
 *   revisions, takebacks and cooldown). A clean move (~5–7 per frame) qualifies; weak or contradictory evidence (the
 *   near-equal flicker case, ~2 per frame) never does, so the dwell time only guards the cases it was made for.
 * Wrong fast commits are what the revision window is for. Evidence from before a hand fades by `unstableDecay` per
 * unstable frame, so a move made under the hand doesn't first have to pay back the stale lead of the old position.
 */
export interface GameParams {
  /** Revision window: number of most recent plies that stay correctable (1 by default, 2 supported). */
  revisionDepth: number;
  /** Beam width B: nodes per inner level of the lattice whose children are expanded. */
  beamWidth: number;
  /** Leaky-sum factor λ per stable frame. */
  decay: number;
  /**
   * Fading of every score per unstable frame (hand, freeze, dropped). Evidence from before a hand describes a board
   * that may have changed under it; with 0.5, 4 unstable frames (the vision settle alone) keep ~6 % of it.
   */
  unstableDecay: number;
  /** Prior cost per ply of edit distance to the committed line (shorter explanations win ties). */
  plyPenalty: number;
  /** Incumbent (committed tip) bonus β: challengers must beat incumbent + β. */
  incumbentBonus: number;
  /** τ_move: margin over the incumbent for an advance (new plies on top of the tip). */
  moveMargin: number;
  /** t_hold: an advance must stay the qualified argmax this long (stable frames, continuously). */
  moveHoldMs: number;
  /**
   * Fast path: per-frame evidence (one frame's log-likelihood difference) the challenger needs over the incumbent and
   * over the runner-up, on `fastFrames` consecutive stable frames, to commit without the hold (advance; scaled by
   * τ_revise / τ_move for revisions and takebacks, and doubled again under cooldown). Infinity disables the fast path.
   */
  fastMargin: number;
  /** Fast path: consecutive stable frames of decisive evidence (the commit frame included). */
  fastFrames: number;
  /** τ_revise: margin for a revision (the line diverges inside the window) or a takeback (a shorter line). */
  reviseMargin: number;
  /** t_revise: dwell for revisions and takebacks. */
  reviseHoldMs: number;
  /** τ_amb: margin over the best real alternative (different grid, not an extension of the challenger). */
  ambiguityMargin: number;
  /** A corrected ply can't be changed again for this long unless the margin exceeds 2·τ_revise. */
  revisionCooldownMs: number;
  /** Frames of evidence kept for replaying scores into newly created lattice nodes. */
  replayFrames: number;
  /** Lattice nodes expanded per frame at most (bounds the move-generation spikes; the beam catches up next frames). */
  maxExpansionsPerFrame: number;
  /** Frames out of the beam before an expanded node's children are dropped. */
  collapseAfterFrames: number;
  /** Cells with a smaller visible fraction are ignored by lock-in, new-game and desync checks. */
  visMin: number;
  /** Lock-in: min log-odds (leaky mean) of the start class over the best other class, on every visible cell. */
  lockMargin: number;
  /** Lock-in: at least this many visible cells. */
  lockMinVisible: number;
  /** Lock-in: the start position must hold this long over stable frames. */
  lockHoldMs: number;
  /** New game: a confident start position (not explained by the lattice) held this long after ≥ 1 ply. */
  newGameHoldMs: number;
  /** Desync: a visible cell is unexplained when the observed class beats the best node's class by this log-odds. */
  desyncCellMargin: number;
  /** Desync: at least this many unexplained cells ... */
  desyncCells: number;
  /** ... for this long. */
  desyncMs: number;
  /** Resync: at most this many unexplained cells ... */
  resyncCells: number;
  /** ... for this long. */
  resyncMs: number;
}

export const DEFAULT_GAME_PARAMS: GameParams = {
  revisionDepth: 1,
  beamWidth: 6,
  decay: 0.85,
  unstableDecay: 0.5,
  plyPenalty: 1,
  incumbentBonus: 2,
  moveMargin: 8,
  moveHoldMs: 250,
  fastMargin: 4,
  fastFrames: 2,
  reviseMargin: 16,
  reviseHoldMs: 800,
  ambiguityMargin: 4,
  revisionCooldownMs: 3000,
  replayFrames: 48,
  collapseAfterFrames: 30,
  maxExpansionsPerFrame: 1,
  visMin: 0.5,
  lockMargin: 0.5,
  lockMinVisible: 32,
  lockHoldMs: 1000,
  newGameHoldMs: 2000,
  desyncCellMargin: 1.5,
  desyncCells: 3,
  desyncMs: 3000,
  resyncCells: 1,
  resyncMs: 500,
};

const SAVE_FORMAT = 'chess-ref-game';

/** PGN date of today (local time), YYYY.MM.DD. */
function pgnDate(): string {
  const d = new Date();
  return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, '0')}.${String(d.getDate()).padStart(2, '0')}`;
}

function resultOf(chess: Chess): string {
  if (chess.isCheckmate()) return chess.turn() === 'w' ? '0-1' : '1-0';
  if (chess.isDraw() || chess.isStalemate()) return '1/2-1/2';
  return '*';
}

function plyOf(m: Move, tentative: boolean): PlyInfo {
  const p: PlyInfo = { san: m.san, from: sqIndex(m.from), to: sqIndex(m.to), tentative };
  if (m.promotion) p.promotion = m.promotion;
  return p;
}

/** Replays UCI moves from the start; null when one is illegal or malformed. */
function replay(moves: readonly string[]): { chess: Chess; played: Move[] } | null {
  const chess = new Chess();
  const played: Move[] = [];
  for (const u of moves) {
    if (typeof u !== 'string' || !/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(u)) return null;
    try {
      played.push(chess.move(uciMove(u)));
    } catch {
      return null;
    }
  }
  return { chess, played };
}

/**
 * Turns occupancy observations into a legal game from the standard starting position.
 *
 * Pipeline per oriented frame (see docs/PLAN-game.md, "Game layer"):
 * - waiting: lock in once the starting position is confidently observed over `lockHoldMs` of stable frames on a
 *   tracker calibrated on it ('setup' / 'calibrated');
 * - playing: every stable frame updates the revisable move lattice (hypotheses = move sequences of 0..K+1 plies from the
 *   anchor, K = revisionDepth plies behind the tip); the argmax replaces the committed window when it beats the
 *   incumbent (advance: τ_move / t_hold; revision or takeback: τ_revise / t_revise, with per-ply cooldown) and the best
 *   real alternative (τ_amb), continuously for the dwell time. Unstable frames only decay scores and reset dwell.
 * - the oldest tentative ply becomes final once the window exceeds K; desync / new-game detection run alongside.
 *
 * Time is always the observation clock (`obs.t`), never the wall clock, so replays are deterministic.
 */
export class GameTracker {
  readonly params: GameParams;

  private st: GameState = 'waiting';
  /** Game at the committed tip. */
  private chess = new Chess();
  private uci: string[] = [];
  private plies: PlyInfo[] = [];
  /** Plies [0, finalCount) are final; the rest (at most revisionDepth) form the revision window. */
  private finalCount = 0;
  private lattice: Lattice | null = null;

  /** Frame counter over oriented frames (stable or not): the clock of the leaky sums. */
  private tick = 0;
  private frames: EvidenceFrame[] = [];
  /** Leaky mean of logLik per square and class (stable frames), for lock-in, new-game and desync checks. */
  private mean = new Float32Array(192);

  private lockSince: number | null = null;
  private newGameSince: number | null = null;
  private desyncSince: number | null = null;
  private resyncSince: number | null = null;
  private dwellKey: string | null = null;
  private dwellSince = 0;
  /** Fast path: challenger with decisive per-frame evidence on the last `fastCount` consecutive stable frames. */
  private fastKey: string | null = null;
  private fastCount = 0;
  /** Decay clock of the leaky sums (see EvidenceFrame.age). */
  private age = 0;
  /** Challenger leading the incumbent but not committed (null otherwise), with its progress; plies built on demand. */
  private pendingNode: HNode | null = null;
  private pendingProgress = 0;
  /** Timing trail (observation clock) for ChangeTiming. */
  private unstableAt: number | null = null;
  private stableAt: number | null = null;
  private favKey: string | null = null;
  private favSince = 0;
  private timing: ChangeTiming | null = null;
  /** snapshot() parts that only change with the committed game (dropped by rebuild / resetGame). */
  private cache: (Pick<GameSnapshot, 'fen' | 'turn' | 'check' | 'result' | 'plies' | 'pieces' | 'lastMove' | 'grid'> & { waiting: boolean }) | null = null;

  constructor(params: Partial<GameParams> = {}) {
    this.params = { ...DEFAULT_GAME_PARAMS, ...params };
    if (!(this.params.revisionDepth >= 0)) this.params.revisionDepth = 0;
    this.params.revisionDepth = Math.floor(this.params.revisionDepth);
  }

  /** Current state (same as snapshot().state). */
  get state(): GameState {
    return this.st;
  }

  /** Occupancy of the committed tip (chess square order), null while waiting. This is the PositionHint grid. */
  expectedGrid(): Uint8Array | null {
    return this.st === 'waiting' ? null : gridOf(this.chess);
  }

  /** Feeds one frame of evidence; returns what changed. */
  observe(obs: Observation): GameEvent[] {
    if (!obs.oriented || obs.logLik.length < 192 || obs.vis.length < 64) return [];
    const p = this.params;
    this.tick++;
    let stable = obs.stable;
    if (stable) for (let i = 0; i < 192; i++) if (!Number.isFinite(obs.logLik[i]!)) stable = false;
    if (!stable) {
      this.age += this.unstableAge();
      this.lattice?.decayOnly();
      this.dwellKey = this.fastKey = this.favKey = null;
      this.pendingNode = null;
      this.lockSince = null;
      this.newGameSince = null;
      this.unstableAt = obs.t;
      this.stableAt = null;
      return [];
    }
    this.age += 1;
    if (this.unstableAt !== null) this.stableAt ??= obs.t;
    this.pendingNode = null;

    const L = obs.logLik;
    const lam = p.decay;
    for (let i = 0; i < 192; i++) this.mean[i] = lam * this.mean[i]! + (1 - lam) * L[i]!;
    this.frames.push({ logLik: L.slice(0, 192), tick: this.tick, age: this.age });
    while (this.frames.length && this.tick - this.frames[0]!.tick >= p.replayFrames) this.frames.shift();

    if (this.st === 'waiting') {
      // 'setup': vision accepted a starting position and fitted its (provisional) models on it; 'calibrated' only
      // appears with a position hint, i.e. for a frame or two after a new game.
      if ((obs.calibration === 'setup' || obs.calibration === 'calibrated') && this.startConfident(obs.vis)) {
        this.lockSince ??= obs.t;
        if (obs.t - this.lockSince >= p.lockHoldMs) {
          this.resetGame();
          this.st = 'playing';
          this.rebuild();
          return [{ type: 'started' }];
        }
      } else this.lockSince = null;
      return [];
    }

    const lat = this.lattice!;
    lat.setFrames(this.frames, this.age);
    lat.update(L);
    lat.refreshEff();
    lat.maintain(this.tick);
    const { best, runnerUp } = lat.rank();

    // New game: the start position, which the lattice can't explain, held after at least one ply.
    if (this.plies.length > 0 && !gridEq(best.grid, START_GRID) && !gridEq(lat.incumbent.grid, START_GRID) && this.startConfident(obs.vis)) {
      this.newGameSince ??= obs.t;
      if (obs.t - this.newGameSince >= p.newGameHoldMs) {
        const previousPgn = this.pgn();
        this.resetGame();
        this.st = 'playing';
        this.rebuild();
        return [{ type: 'newGame', previousPgn }];
      }
    } else this.newGameSince = null;

    // Desync / resync, judged on the best hypothesis (a move being played is explained by its challenger).
    const events: GameEvent[] = [];
    if (this.st !== 'over') {
      const bad = this.unexplained(best.grid, obs.vis);
      if (this.st === 'desync') {
        if (bad <= p.resyncCells) {
          this.resyncSince ??= obs.t;
          if (obs.t - this.resyncSince >= p.resyncMs) {
            this.st = 'playing';
            this.desyncSince = this.resyncSince = null;
            this.dwellKey = this.fastKey = null;
            events.push({ type: 'resynced' });
          }
        } else this.resyncSince = null;
      } else if (bad >= p.desyncCells) {
        this.desyncSince ??= obs.t;
        if (obs.t - this.desyncSince >= p.desyncMs) {
          this.st = 'desync';
          this.resyncSince = null;
          this.dwellKey = this.fastKey = null;
          return [{ type: 'desync' }];
        }
      } else this.desyncSince = null;
    }
    if (this.st === 'desync') return events;

    // Commit decision.
    const inc = lat.incumbent;
    if (best === inc) {
      this.dwellKey = this.fastKey = this.favKey = null;
      return events;
    }
    if (this.favKey !== best.key) {
      this.favKey = best.key;
      this.favSince = obs.t;
    }
    const margin = best.eff - inc.eff;
    const amb = runnerUp ? best.eff - runnerUp.eff : Infinity;
    const advance = best.kind === 'advance';
    let need = advance ? p.moveMargin : p.reviseMargin;
    if (!advance && this.inCooldown(best.common, obs.t)) need = Math.max(need, 2 * p.reviseMargin + 1e-9);
    const hold = advance ? p.moveHoldMs : p.reviseHoldMs;

    // Fast-path streak: this frame's own evidence is decisive against the incumbent and the runner-up.
    const fastNeed = (p.fastMargin * need) / p.moveMargin;
    const frameLead = Math.min(best.gain - inc.gain, runnerUp ? best.gain - runnerUp.gain : Infinity);
    if (frameLead >= fastNeed) {
      if (this.fastKey !== best.key) {
        this.fastKey = best.key;
        this.fastCount = 0;
      }
      this.fastCount++;
    } else this.fastKey = null;

    if (margin < need || amb < p.ambiguityMargin) {
      this.dwellKey = null;
      this.setPending(best, margin / need);
      return events;
    }
    if (this.dwellKey !== best.key) {
      this.dwellKey = best.key;
      this.dwellSince = obs.t;
    }
    const fast = this.fastKey === best.key && this.fastCount >= p.fastFrames;
    if (!fast && obs.t - this.dwellSince < hold) {
      this.setPending(best, Math.max(margin / need, (obs.t - this.dwellSince) / hold));
      return events;
    }
    events.push(...this.commit(best, obs.t));
    return events;
  }

  /**
   * Current state for display. Cheap enough to call on every frame: what only changes on commits (position, plies,
   * pieces, grid) is computed once per change and copied; only `pending` and `top` are built per call.
   */
  snapshot(): GameSnapshot {
    const waiting = this.st === 'waiting';
    let c = this.cache;
    if (!c || c.waiting !== waiting) {
      const last = this.plies[this.plies.length - 1];
      c = this.cache = {
        waiting,
        fen: this.chess.fen(),
        turn: this.chess.turn(),
        check: this.chess.inCheck(),
        result: resultOf(this.chess),
        plies: this.plies.map((q) => ({ ...q })),
        pieces: waiting ? new Array(64).fill(null) : piecesOf(this.chess),
        lastMove: last ? { from: last.from, to: last.to } : null,
        grid: waiting ? null : gridOf(this.chess),
      };
    }
    return {
      state: this.st,
      fen: c.fen,
      turn: c.turn,
      check: c.check,
      result: c.result,
      plies: c.plies.map((q) => ({ ...q })),
      pieces: c.pieces.slice(),
      lastMove: c.lastMove ? { ...c.lastMove } : null,
      grid: c.grid ? c.grid.slice() : null,
      pending: this.pending(),
      lastTiming: this.timing ? { ...this.timing } : null,
      top: this.lattice && !waiting ? this.lattice.top(3).map((n) => ({ line: n.line(), score: n.eff })) : [],
    };
  }

  /** Manual undo of the last ply (sidebar button). */
  undo(): GameEvent[] {
    if (this.st === 'waiting' || this.plies.length === 0) return [];
    this.chess.undo();
    this.uci.pop();
    this.plies.pop();
    this.finalCount = Math.min(this.finalCount, this.plies.length);
    this.st = this.chess.isGameOver() ? 'over' : 'playing';
    this.resetTimers();
    this.rebuild();
    return [{ type: 'undone' }];
  }

  /** Drops the game and waits for the starting position again. */
  newGame(): void {
    this.resetGame();
    this.timing = null;
    this.st = 'waiting';
    this.lattice = null;
  }

  /** Accepts a desync: resume inference from the current game position. */
  continueAfterDesync(): void {
    if (this.st !== 'desync') return;
    this.st = this.chess.isGameOver() ? 'over' : 'playing';
    this.resetTimers();
  }

  /** Changes the promotion piece of ply `ply` (0-based); false when that ply is not a promotion or the change is illegal. */
  setPromotion(ply: number, piece: 'q' | 'r' | 'b' | 'n'): boolean {
    const old = this.plies[ply];
    if (!old || !old.promotion || !'qrbn'.includes(piece) || piece.length !== 1) return false;
    if (old.promotion === piece) return true;
    const moves = this.uci.slice();
    moves[ply] = moves[ply]!.slice(0, 4) + piece;
    const r = replay(moves);
    if (!r) return false;
    this.chess = r.chess;
    this.uci = moves;
    this.plies = r.played.map((m, i) => {
      const q = plyOf(m, this.plies[i]!.tentative);
      const c = this.plies[i]!.correctedAt;
      if (c !== undefined) q.correctedAt = c;
      return q;
    });
    if (this.st !== 'waiting' && this.st !== 'desync') this.st = this.chess.isGameOver() ? 'over' : 'playing';
    this.resetTimers();
    this.rebuild();
    return true;
  }

  /** PGN of the game (Seven Tag Roster; extra headers override). */
  pgn(headers: Record<string, string> = {}): string {
    const r = replay(this.uci) ?? { chess: new Chess(), played: [] };
    const tags: Record<string, string> = { Event: '?', Site: '?', Date: pgnDate(), Round: '?', White: '?', Black: '?', Result: resultOf(r.chess), ...headers };
    for (const [k, v] of Object.entries(tags)) r.chess.setHeader(k, v);
    return r.chess.pgn();
  }

  /** Serialised game for localStorage. */
  save(): string {
    return JSON.stringify({ format: SAVE_FORMAT, v: 1, state: this.st === 'waiting' ? 'waiting' : 'playing', moves: this.uci, finalCount: this.finalCount, pgn: this.pgn() });
  }

  /**
   * Restores a game saved by `save()`, or a plain PGN (e.g. `newGame`'s `previousPgn`, to undo an automatic new
   * game); false (and state unchanged) when invalid. The restored game is 'playing' (or 'over'), without lock-in.
   */
  load(saved: string): boolean {
    if (typeof saved !== 'string') return false;
    let moves: string[] | null = null;
    let finalCount: number | null = null;
    let waiting = false;
    let json: unknown;
    try {
      json = JSON.parse(saved);
    } catch {
      json = undefined;
    }
    if (json !== undefined) {
      if (!json || typeof json !== 'object') return false;
      const o = json as Record<string, unknown>;
      if (o.format !== SAVE_FORMAT || !Array.isArray(o.moves)) return false;
      moves = o.moves as string[];
      waiting = o.state === 'waiting';
      if (typeof o.finalCount === 'number' && Number.isFinite(o.finalCount)) finalCount = o.finalCount;
    } else {
      moves = pgnMoves(saved);
      if (!moves) return false;
    }
    const r = replay(moves);
    if (!r) return false;
    if (waiting) {
      this.newGame();
      return true;
    }
    const n = moves.length;
    const fc = Math.max(0, n - this.params.revisionDepth, Math.min(n, Math.floor(finalCount ?? n)));
    this.chess = r.chess;
    this.uci = moves.slice();
    this.finalCount = fc;
    this.plies = r.played.map((m, i) => plyOf(m, i >= fc));
    this.st = this.chess.isGameOver() ? 'over' : 'playing';
    this.frames = [];
    this.resetTimers();
    this.rebuild();
    return true;
  }

  // -------------------------------------------------------------------------------------------------------------

  private resetGame(): void {
    this.chess = new Chess();
    this.uci = [];
    this.plies = [];
    this.finalCount = 0;
    this.cache = null;
    this.resetTimers();
  }

  private resetTimers(): void {
    this.lockSince = this.newGameSince = this.desyncSince = this.resyncSince = null;
    this.dwellKey = this.fastKey = this.favKey = null;
    this.pendingNode = null;
  }

  private setPending(node: HNode, progress: number): void {
    this.pendingNode = node;
    this.pendingProgress = Number.isFinite(progress) ? Math.min(1, Math.max(0, progress)) : 1;
  }

  /** Decay-clock advance of one unstable frame: decay^this = unstableDecay. */
  private unstableAge(): number {
    const lam = this.params.decay;
    const u = Math.min(1, Math.max(1e-6, this.params.unstableDecay));
    return lam > 0 && lam < 1 ? Math.log(u) / Math.log(lam) : 1;
  }

  /** The pending change (see GameSnapshot.pending), built on demand from the stored challenger. */
  private pending(): PendingChange | null {
    const n = this.pendingNode;
    if (!n || n.kind === 'incumbent' || this.st === 'waiting' || this.st === 'desync') return null;
    const plies: PlyInfo[] = [];
    for (let m: HNode | null = n; m && m.spec && m.depth > n.common; m = m.parent) {
      const q: PlyInfo = { san: m.spec.san, from: m.spec.from, to: m.spec.to, tentative: true };
      if (m.spec.promotion) q.promotion = m.spec.promotion;
      plies.push(q);
    }
    return { kind: n.kind, fromPly: this.finalCount + n.common, plies: plies.reverse(), progress: this.pendingProgress };
  }

  /** Rebuilds the lattice at the current anchor; scores come back from the replay buffer. */
  private rebuild(): void {
    // Anchor position: step the tip back over the window and forward again (cheaper than a replay from the start).
    const window = this.uci.slice(this.finalCount);
    for (let i = 0; i < window.length; i++) this.chess.undo();
    const anchorFen = this.chess.fen();
    const anchorGrid = gridOf(this.chess);
    for (const u of window) this.chess.move(uciMove(u));
    const cfg = {
      maxDepth: this.params.revisionDepth + 1,
      beam: Math.max(1, this.params.beamWidth),
      decay: this.params.decay,
      unstableDecay: Math.min(1, Math.max(1e-6, this.params.unstableDecay)),
      plyPenalty: this.params.plyPenalty,
      incumbentBonus: this.params.incumbentBonus,
      collapseAfter: this.params.collapseAfterFrames,
      maxExpansions: Math.max(1, this.params.maxExpansionsPerFrame),
    };
    this.lattice = new Lattice(anchorFen, anchorGrid, window, cfg, this.frames, this.tick, this.age);
    this.cache = null;
    this.dwellKey = this.fastKey = this.favKey = null;
    this.pendingNode = null;
  }

  /** Whether a window ply from index `finalCount + from` on was corrected less than the cooldown ago. */
  private inCooldown(from: number, t: number): boolean {
    for (let i = this.finalCount + from; i < this.plies.length; i++) {
      const c = this.plies[i]!.correctedAt;
      if (c !== undefined && t - c < this.params.revisionCooldownMs) return true;
    }
    return false;
  }

  /** Switches the committed window to `node`'s line at once, finalises, re-roots, and reports it. */
  private commit(node: HNode, t: number): GameEvent[] {
    if (node.kind !== 'incumbent') {
      this.timing = { kind: node.kind, stableAt: this.stableAt, unstableEndAt: this.unstableAt, favouriteAt: this.favKey === node.key ? this.favSince : t, committedAt: t };
    }
    const oldWin = this.uci.slice(this.finalCount);
    const newWin = node.moves();
    const common = node.common;
    const oldPlies = this.plies.splice(this.finalCount);
    this.uci.length = this.finalCount;
    for (let i = 0; i < oldWin.length; i++) this.chess.undo();
    for (let i = 0; i < newWin.length; i++) {
      const m = this.chess.move(uciMove(newWin[i]!));
      let ply: PlyInfo;
      if (i < common) ply = oldPlies[i]!;
      else {
        ply = plyOf(m, true);
        if (i < oldWin.length) ply.correctedAt = t;
      }
      this.plies.push(ply);
      this.uci.push(newWin[i]!);
    }
    const firstChanged = this.finalCount + common;
    while (this.plies.length - this.finalCount > this.params.revisionDepth) this.finalCount++;
    this.plies.forEach((q, i) => (q.tentative = i >= this.finalCount));
    this.st = this.chess.isGameOver() ? 'over' : 'playing';
    this.resetTimers();
    this.rebuild();
    if (node.kind === 'advance') return this.plies.slice(firstChanged).map((q) => ({ type: 'move', ply: { ...q } }));
    return [{ type: 'revised', fromPly: firstChanged, plies: this.plies.slice(firstChanged).map((q) => ({ ...q })) }];
  }

  /** Whether every visible square confidently shows the starting position (leaky means), with enough squares visible. */
  private startConfident(vis: Float32Array): boolean {
    const p = this.params;
    let visible = 0;
    for (let sq = 0; sq < 64; sq++) {
      if (vis[sq]! < p.visMin) continue;
      visible++;
      const k = START_GRID[sq]!;
      const m = this.mean;
      const own = m[sq * 3 + k]!;
      const other = Math.max(m[sq * 3 + ((k + 1) % 3)]!, m[sq * 3 + ((k + 2) % 3)]!);
      if (own - other < p.lockMargin) return false;
    }
    return visible >= p.lockMinVisible;
  }

  /** Visible squares whose observed class (leaky mean) beats `grid`'s class by desyncCellMargin. */
  private unexplained(grid: Uint8Array, vis: Float32Array): number {
    const p = this.params;
    const m = this.mean;
    let n = 0;
    for (let sq = 0; sq < 64; sq++) {
      if (vis[sq]! < p.visMin) continue;
      const g = m[sq * 3 + grid[sq]!]!;
      const best = Math.max(m[sq * 3]!, m[sq * 3 + 1]!, m[sq * 3 + 2]!);
      if (best - g >= p.desyncCellMargin) n++;
    }
    return n;
  }
}

function gridEq(a: Uint8Array, b: Uint8Array): boolean {
  for (let i = 0; i < 64; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** UCI moves of a PGN from the standard start; null when it doesn't parse or uses another start position. */
function pgnMoves(text: string): string[] | null {
  const t = text.trim();
  if (!/^(\[|1\.)/.test(t)) return null;
  const c = new Chess();
  try {
    c.loadPgn(t);
  } catch {
    return null;
  }
  const fen = c.getHeaders()['FEN'];
  if (fen && fen !== START_FEN) return null;
  return c.history({ verbose: true }).map((m) => m.lan);
}
