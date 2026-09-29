import { applyEdits, legalMoves, type MoveSpec } from './board';

/**
 * One stable frame of evidence kept for replay: when the lattice creates a node (beam expansion, re-rooting after a
 * commit, rebuild after undo / load / promotion change), its score is recomputed from these frames, so a node's
 * evidence does not depend on when it happened to be created (e.g. the d2/d4 cells seen while e4 was being committed
 * still count for `[d4]` after the commit).
 */
export interface EvidenceFrame {
  /** Copy of Observation.logLik (64×3, chess square order). */
  logLik: Float32Array;
  /** Frame counter (stable and unstable frames) when it was observed (replay-buffer eviction). */
  tick: number;
  /**
   * Decay clock when it was observed: +1 per stable frame, +log(unstableDecay)/log(decay) per unstable frame, so that
   * weight = decay^(age now - this) matches the per-frame fading of the live scores.
   */
  age: number;
}

/** Weight of HNode.tie in eff: only decides between grids that the evidence can't tell apart. */
const TIE_WEIGHT = 0.01;

/** How a hypothesis relates to the committed window. */
export type NodeKind = 'incumbent' | 'advance' | 'revise' | 'takeback';

export interface LatticeConfig {
  /** Deepest hypothesis, in plies from the anchor (revisionDepth + 1). */
  maxDepth: number;
  /** Nodes per level (except the full first level and the last) whose children are expanded. */
  beam: number;
  /** Leaky-sum factor per frame. */
  decay: number;
  /** Fading of every score per unstable frame (hand, freeze, dropped frame). */
  unstableDecay: number;
  /** Prior cost per ply of edit distance between a hypothesis and the committed window. */
  plyPenalty: number;
  /** Hysteresis bonus of the incumbent (the committed tip). */
  incumbentBonus: number;
  /**
   * Beam expansions per frame (best first; the rest follow on the next frames). Move generation is the only costly
   * step (chess.js builds every child's FEN, ~0.5–1 ms per position), so this bounds the per-frame spikes.
   */
  maxExpansions: number;
  /** An expanded node that has left the beam for this many frames loses its children (they are replayed if needed). */
  collapseAfter: number;
}

/**
 * A hypothesis: a move sequence from the anchor. Its occupancy grid is stored in full (for desync / new-game checks)
 * and as the list of cells that differ from the anchor grid, which is all the per-frame scoring needs: scores are
 * relative to the anchor (the anchor's own log-likelihood is common to every node, so it cancels in comparisons).
 */
export class HNode {
  children: HNode[] | null = null;
  /** Leaky sum of per-frame log-likelihood relative to the anchor. */
  score = 0;
  /** This frame's own contribution to `score` (0 after an unstable frame): the evidence of a single frame. */
  gain = 0;
  /** score - prior + incumbent bonus, refreshed every frame. */
  eff = 0;
  /** Last frame this node was in the beam (or on the committed path). */
  lastWanted = 0;
  /** Plies shared with the committed window (from the anchor). */
  common = 0;
  /** Edit distance to the committed window, in plies. */
  cost = 0;
  kind: NodeKind = 'incumbent';
  /**
   * Tie-break among equal grids: off-path plies whose moving piece differs from the committed ply at the same index.
   * Occupancy can't order independent moves (`[d4, e5, Nf3]` vs `[Nf3, e5, d4]`), so a correction keeps the committed
   * piece where it can (a misread pawn move was more likely another pawn move).
   */
  tie = 0;

  constructor(
    readonly parent: HNode | null,
    /** Move from the parent, null for the anchor. */
    readonly spec: MoveSpec | null,
    readonly depth: number,
    /** FEN of this node's position. */
    readonly fen: string,
    readonly grid: Uint8Array,
    /** logLik indices (sq * 3 + class) of this grid and of the anchor grid, over the cells where they differ. */
    readonly idxN: Int32Array,
    readonly idxA: Int32Array,
    /** Canonical form of the differing cells: equal signatures <=> equal grids (transpositions, promotion pieces). */
    readonly sig: string,
    /** Move sequence from the anchor, UCI joined by spaces ('' for the anchor). */
    readonly key: string,
  ) {}

  /** UCI moves from the anchor. */
  moves(): string[] {
    const out: string[] = [];
    for (let n: HNode | null = this; n && n.spec; n = n.parent) out.push(n.spec.uci);
    return out.reverse();
  }

  /** SAN moves from the anchor joined by spaces ('' for the anchor). */
  line(): string {
    const out: string[] = [];
    for (let n: HNode | null = this; n && n.spec; n = n.parent) out.push(n.spec.san);
    return out.reverse().join(' ');
  }

  /** Whether `this` lies in the subtree of `a` (a node is its own descendant). */
  descendsFrom(a: HNode): boolean {
    let n: HNode | null = this;
    while (n && n.depth > a.depth) n = n.parent;
    return n === a;
  }
}

/**
 * The revisable move lattice: a tree of hypotheses rooted at the anchor (the position before the revision window),
 * fully expanded at the first level, beam-expanded below, always including the committed path. Scores are leaky sums of
 * log-likelihood; decisions (thresholds, dwell, cooldown) belong to the GameTracker.
 */
export class Lattice {
  readonly root: HNode;
  /** The committed tip. */
  readonly incumbent: HNode;
  /** Every node of the tree (root included), in creation order. */
  nodes: HNode[] = [];
  private readonly anchorGrid: Uint8Array;
  /** Moving piece of each committed window ply. */
  private readonly pathPieces: string[] = [];
  private weights: Float64Array = new Float64Array(0);
  private weightsAge = -1;
  private framesSeen: readonly EvidenceFrame[] = [];
  /** Decay clock of the current frame (see EvidenceFrame.age). */
  private age = 0;

  /**
   * @param path the committed window (UCI moves from the anchor); every prefix must be legal.
   * @param frames evidence to replay into new nodes, with `tick` the current frame counter and `age` the decay clock.
   */
  constructor(anchorFen: string, anchorGrid: Uint8Array, readonly path: readonly string[], readonly cfg: LatticeConfig, frames: readonly EvidenceFrame[], tick: number, age: number) {
    this.anchorGrid = anchorGrid;
    for (let i = 0, fen = anchorFen; i < path.length; i++) {
      const spec = legalMoves(fen).find((m) => m.uci === path[i]);
      if (!spec) throw new Error(`lattice: committed move ${path[i]} is not legal`);
      this.pathPieces.push(spec.piece);
      fen = spec.after;
    }
    this.root = new HNode(null, null, 0, anchorFen, anchorGrid, new Int32Array(0), new Int32Array(0), '', '');
    this.classify(this.root);
    this.nodes.push(this.root);
    this.setFrames(frames, age);
    let n = this.root;
    for (let d = 0; ; d++) {
      n.lastWanted = tick;
      if (d < cfg.maxDepth) this.expand(n, tick);
      if (d === path.length) break;
      const next = n.children?.find((c) => c.spec!.uci === path[d]);
      if (!next) throw new Error(`lattice: committed move ${path[d]} is not legal`);
      n = next;
    }
    this.incumbent = n;
    this.refreshEff();
  }

  /** Evidence used to initialise nodes created from now on (the tracker's replay buffer), `age` the current decay clock. */
  setFrames(frames: readonly EvidenceFrame[], age: number): void {
    this.framesSeen = frames;
    this.age = age;
    if (age !== this.weightsAge || this.weights.length !== frames.length) {
      this.weights = new Float64Array(frames.length);
      for (let i = 0; i < frames.length; i++) this.weights[i] = Math.pow(this.cfg.decay, age - frames[i]!.age);
      this.weightsAge = age;
    }
  }

  /** One stable frame: leaky update of every node. Only the cells differing from the anchor are summed. */
  update(logLik: Float32Array): void {
    const lam = this.cfg.decay;
    const nodes = this.nodes;
    for (let k = 0; k < nodes.length; k++) {
      const n = nodes[k]!;
      const a = n.idxN;
      const b = n.idxA;
      let g = 0;
      for (let i = 0; i < a.length; i++) g += logLik[a[i]!]! - logLik[b[i]!]!;
      n.gain = g;
      n.score = n.score * lam + g;
    }
  }

  /** One unstable frame: evidence only fades (by `unstableDecay`). */
  decayOnly(): void {
    const u = this.cfg.unstableDecay;
    for (const n of this.nodes) {
      n.score *= u;
      n.gain = 0;
    }
  }

  refreshEff(): void {
    const { plyPenalty, incumbentBonus } = this.cfg;
    for (const n of this.nodes) n.eff = n.score - plyPenalty * n.cost - TIE_WEIGHT * n.tie + (n === this.incumbent ? incumbentBonus : 0);
  }

  /**
   * Beam maintenance after a frame's update: the best `beam` nodes of every inner level get their children (created
   * with replayed scores, at most `maxExpansions` per call); nodes out of the beam for `collapseAfter` frames lose theirs. Call refreshEff() first.
   */
  maintain(tick: number): void {
    const { maxDepth, beam, collapseAfter } = this.cfg;
    let budget = this.cfg.maxExpansions;
    for (let d = 1; d < maxDepth; d++) {
      const top: HNode[] = [];
      for (const n of this.nodes) {
        if (n.depth !== d) continue;
        if (top.length < beam) top.push(n);
        else if (n.eff > top[beam - 1]!.eff) top[beam - 1] = n;
        else continue;
        for (let i = top.length - 1; i > 0 && top[i]!.eff > top[i - 1]!.eff; i--) [top[i], top[i - 1]] = [top[i - 1]!, top[i]!];
      }
      for (const n of top) {
        n.lastWanted = tick;
        if (!n.children && budget > 0) {
          this.expand(n, tick);
          budget--;
        }
      }
    }
    if (tick % 8 === 0) {
      let collapsed = false;
      for (const n of this.nodes) {
        if (n.children && !this.onPath(n) && tick - n.lastWanted > collapseAfter) {
          n.children = null;
          collapsed = true;
        }
      }
      if (collapsed) {
        const out: HNode[] = [];
        const stack = [this.root];
        while (stack.length) {
          const n = stack.pop()!;
          out.push(n);
          if (n.children) for (const c of n.children) stack.push(c);
        }
        this.nodes = out;
      }
    }
  }

  /**
   * Best node by eff, and the best real alternative to it: a different grid, not an extension of it, and no more
   * costly (edit distance to the committed line) than it. A costlier alternative that is near-equal on evidence loses
   * on the prior anyway (e.g. `[d4, e5, e4]` vs `[e4, e5]` when d2/d4 are hidden); it must not block the cheaper change.
   */
  rank(): { best: HNode; runnerUp: HNode | null } {
    let best = this.root;
    for (const n of this.nodes) if (n.eff > best.eff) best = n;
    let runnerUp: HNode | null = null;
    for (const n of this.nodes) {
      if (n === best || n.cost > best.cost || (runnerUp && n.eff <= runnerUp.eff) || n.sig === best.sig || n.descendsFrom(best)) continue;
      runnerUp = n;
    }
    return { best, runnerUp };
  }

  /** The `k` best nodes by eff, best first (debug display; one pass, no sort of the whole tree). */
  top(k: number): HNode[] {
    const out: HNode[] = [];
    for (const n of this.nodes) {
      if (out.length < k) out.push(n);
      else if (k > 0 && n.eff > out[k - 1]!.eff) out[k - 1] = n;
      else continue;
      for (let i = out.length - 1; i > 0 && out[i]!.eff > out[i - 1]!.eff; i--) [out[i], out[i - 1]] = [out[i - 1]!, out[i]!];
    }
    return out;
  }

  private onPath(n: HNode): boolean {
    return n.common === n.depth && n.depth <= this.path.length;
  }

  private classify(n: HNode): void {
    const p = this.path.length;
    n.cost = p - n.common + (n.depth - n.common);
    if (n.common === p) n.kind = n.depth === p ? 'incumbent' : 'advance';
    else n.kind = n.common === n.depth ? 'takeback' : 'revise';
  }

  private expand(n: HNode, tick: number): void {
    this.setFrames(this.framesSeen, this.age);
    const onPath = this.onPath(n);
    const pathMove = onPath && n.depth < this.path.length ? this.path[n.depth]! : null;
    const children: HNode[] = [];
    for (const spec of legalMoves(n.fen)) {
      if (spec.promotion) {
        // The four promotion pieces give the same grid: keep one (the committed one on the committed path, else queen).
        const want = pathMove && pathMove.length > 4 && pathMove.startsWith(spec.uci.slice(0, 4)) ? pathMove[4] : 'q';
        if (spec.promotion !== want) continue;
      }
      const grid = applyEdits(n.grid, spec);
      const cells: number[] = [];
      for (let c = 0; c < 64; c++) if (grid[c] !== this.anchorGrid[c]) cells.push(c);
      const idxN = new Int32Array(cells.length);
      const idxA = new Int32Array(cells.length);
      let sig = '';
      for (let i = 0; i < cells.length; i++) {
        const c = cells[i]!;
        idxN[i] = c * 3 + grid[c]!;
        idxA[i] = c * 3 + this.anchorGrid[c]!;
        sig += String.fromCharCode(65 + idxN[i]!);
      }
      const child = new HNode(n, spec, n.depth + 1, spec.after, grid, idxN, idxA, sig, n.key ? `${n.key} ${spec.uci}` : spec.uci);
      child.common = onPath && spec.uci === pathMove ? n.depth + 1 : n.common;
      child.tie = n.tie + (child.common < child.depth && n.depth < this.path.length && spec.piece !== this.pathPieces[n.depth] ? 1 : 0);
      this.classify(child);
      this.replay(child);
      child.eff = child.score - this.cfg.plyPenalty * child.cost - TIE_WEIGHT * child.tie;
      child.lastWanted = tick;
      children.push(child);
      this.nodes.push(child);
    }
    n.children = children;
  }

  /** Sets the score (and gain) a node would have if it had been scored on every frame of the replay buffer. */
  private replay(n: HNode): void {
    const { idxN, idxA } = n;
    let s = 0;
    let g = 0;
    const frames = this.framesSeen;
    for (let f = 0; f < frames.length; f++) {
      const L = frames[f]!.logLik;
      let d = 0;
      for (let i = 0; i < idxN.length; i++) d += L[idxN[i]!]! - L[idxA[i]!]!;
      s += this.weights[f]! * d;
      if (frames[f]!.age === this.age) g = d;
    }
    n.score = s;
    n.gain = g;
  }
}
