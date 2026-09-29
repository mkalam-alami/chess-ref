# Milestone 9: Chess game layer (move inference from occupancy)

## Context
Milestones 1–8 give us a tracked board homography and per-square occupancy (empty / white / black). The occupancy step already outputs per-cell class log-likelihoods (`OccupancyResult.logLik`, 64×3). It was built to score legal positions, but nothing uses it for that yet. Next steps:
- Add a **game layer** that starts from the standard starting position and turns the occupancy evidence into **legal moves**. Only candidate positions that a legal move can reach get scored, so the piece count and move rules act as a strong filter on noisy occupancy.
- Feed the game's position back to vision as a prior, so occupancy itself gets better.
- In play mode (debug panel hidden), draw small piece icons instead of dots. Add a thin sidebar with the move list and a "Copy PGN" button.

We use existing libraries; we don't write our own:
- **Rules:** `chess.js` (BSD-2, TypeScript types, no DOM). It covers legal move generation, SAN, FEN, PGN, undo, check and game-over. We need no search engine (e.g. Stockfish). The job is recognition, not playing.
- **Piece icons:** the "cburnett" SVG set (Wikimedia/lichess, BSD/GPL/GFDL tri-licence, so we can use it under BSD). We vendor the 12 SVGs into `src/assets/pieces/` with an attribution file. That's about 20 KB and needs no runtime network.

## Module boundaries
```
src/vision/   worker only. OpenCV, detection, occupancy. Knows nothing about chess rules.
src/game/     NEW, pure TS (no DOM, no cv). chess.js wrapper, move inference, PGN. Runs in Node tests.
src/worker/   glue: vision → Observation messages.
src/ui/       main thread: overlay pieces, move sidebar, debug panel (main.ts, overlay.ts and panel.ts move here or stay put)
```
**The hook (vision → game):** the worker emits one `Observation` per processed frame:
```ts
interface Observation {
  t: number;
  /** 64×3 log-probs (empty/white/black), in CHESS square order a1..h8 once orientation is known, else board-cell order */
  logLik: Float32Array;
  oriented: boolean;           // vision knows which cell is a1
  stable: boolean;             // not frozen (no hand / mass change / recent loss), not dropped
  vis: Float32Array;           // per-square visible fraction (lets the game ignore hidden squares when needed)
}
```
**Feedback (game → vision):** `PositionHint { grid: Uint8Array /*64, a1..h8*/ }`. Vision uses this as its committed grid, i.e. as the occluder prior, the illumination fit labels and the model-learning labels. In game mode it replaces `OccupancyFilter`'s own hysteresis. Without a game (free mode / debug), the filter keeps working as it does today.

**Where the game runs (decided): the main thread.** The worker stays pure vision. The observation is about 1 KB per frame, and the hint goes back as a message (one frame of latency, which is harmless). The UI owns game commands (undo, new game, copy, persistence) with no round trip. `src/game` is pure, so the Node bench can still wire vision and game together directly.

## Orientation (vision's job, done before the game starts)
The game should never see `hb` cell coordinates. Vision maps board cells to chess squares:
- **Rank axis and white side:** already found by `tryBootstrap` / `testStart` (`startGrid(axis, side)`, lightness `dL`) in `src/vision/occupancy.ts`.
- **File direction (the mirror ambiguity):** the camera is always above the board, so the handedness of `hb` (the sign of the board frame's orientation seen from above, via `cameraFromHomography`) fixes which way a→h runs. As a cross-check, a1 must be a dark square (square parity from the verification contrast). If the two disagree, don't orient; keep waiting.
- **Across full re-detections**, `dihedralRemap` already permutes the tracker state. The orientation map is permuted the same way.
- **After a long board loss** (when the tracker and profile reset), the orientation is lost. The game still holds the position, so vision re-orients by scoring the observed logLik against the game's expected grid under the 8 dihedral maps, and takes the best one if it wins by a margin. Mid-game positions are rarely symmetric. The class models are also kept across losses; only "Reset board" drops them.

## Game layer (`src/game/`)
- `GameTracker` (pure):
  - `observe(obs) → GameEvent[]`: `started`, `move{san, from, to, fen}`, `undone`, `desync`.
  - `undo()`, `newGame()`, `pgn()`, `expectedGrid()`, `state`: `waiting` | `playing` | `desync` | `over`.
- **Lock-in (`waiting` → `playing`):**
  - Conditions: vision is oriented and calibrated from the start position (not the unsupervised fallback), and over about 1 s of stable frames the start grid's joint log-likelihood beats every single-cell deviation by margin τ. In practice: every visible cell of the start grid has p ≥ p_min.
  - Then set up chess.js at the standard FEN and emit `started`.
- **Move inference: a revisable move lattice.**
  - **Committed plies and the anchor.** The game history splits in two:
    - **final plies**, which never change;
    - a **revision window** of the last `K` committed plies, which stay tentative. `K = 1` at first; the structure supports `K = 2` from day one (see below).
    - The **anchor** is the position at the start of the window, i.e. `K` plies back from the current position (the tip).
  - **Hypotheses** are move sequences from the anchor, of length `0..K+1`. For `K = 1` they are:
    - `[]`: a takeback to the anchor;
    - `[m]` for every legal `m`: the current tip and all its siblings (e.g. `e4` and `d4`);
    - `[m, m']`: a new move played on top of the tip, *or on top of a sibling*. So "it was actually d4, and black has already answered e5" is a single hypothesis, `[d4, e5]`.
    - Each hypothesis maps to a 64-cell occupancy grid. The grid is computed with chess.js `move`/`undo` and cached per node.
  - **Tree with beam pruning.** Hypotheses are nodes of a tree rooted at the anchor.
    - The first level is fully expanded (about 40 moves).
    - Deeper levels expand only the children of the best `B` nodes (B ≈ 6) plus the incumbent path.
    - This keeps `K = 2` (sequences of up to 3 plies) at a few hundred nodes rather than 64k. Moving from `K = 1` to `K = 2` is a parameter change, not a redesign.
  - **Scoring.** On each stable frame, every node gets `score_n = λ·score_n + Σ_c logLik[c, grid_n(c)]` (λ ≈ 0.85). The sum only covers cells where `grid_n` differs from the incumbent's grid, so it is cheap. Each node also gets a small per-ply prior penalty, so shorter explanations are preferred.
    - Scores are keyed by the node's move sequence, not by the current position. Evidence gathered *before* a commit therefore still counts for siblings afterwards, e.g. the d2/d4 cells seen while e4 was being committed.
    - Unstable frames (hand, freeze, dropped) are skipped. Scores are not reset, only decayed.
  - **Incumbent.** The incumbent is the node on the committed path, i.e. the current tip.
  - **Three kinds of change**, each with its own thresholds:
    - **Advance** (a child of the tip): margin `τ_move` over the incumbent, held for `t_hold` ≈ 400 ms of stable frames.
    - **Revise** (any node whose path leaves the committed one *inside* the window, e.g. `[d4]` or `[d4, e5]` replacing `[e4]` / `[e4, e5]`): a stricter margin `τ_revise` ≈ 2×`τ_move`, and held for `t_revise` ≈ 800 ms.
    - **Takeback** (a shorter path): same thresholds as revise.
  - **Anti-flicker**:
    1. **Hysteresis.** The incumbent gets a bonus `β`, and the challenger must beat *incumbent + β*.
    2. **Dwell time.** The challenger must stay the argmax continuously for the whole hold time; any lapse restarts the timer.
    3. **Revision cooldown.** After a ply is revised, it can't be revised again for about 3 s unless the margin exceeds `2·τ_revise`. Two near-equal hypotheses can't ping-pong.
    4. **Ambiguity guard.** A change also needs a margin `τ_amb` over the runner-up, not only over the incumbent.
    5. **One change per decision.** The whole lattice switches to the winning path at once. The UI never shows intermediate states.
  - **Finalising.** When a new ply is committed and the window exceeds `K`, the oldest tentative ply becomes final. The anchor moves forward and the tree is re-rooted: subtrees that survive keep their scores. On every change, the tip grid is re-sent as the `PositionHint`.
  - **Deeper recovery (both moves played while the board was hidden).** Covered by the `[m, m']` level: 2 new plies on top of an unchanged tip is a leaf of the `K = 1` tree when the anchor is the tip itself. The tree is built from the anchor, so this needs no special case.
  - **Desync:** if nothing in the tree explains the board (the best node leaves many confident cells unexplained) for T seconds, raise `desync` (UI: "Board doesn't match: Undo / Continue"). Don't guess.
  - **UI for tentative moves:** moves inside the revision window are shown lighter or italic in the sidebar. A revised move briefly flashes "corrected". Copy PGN includes tentative moves, since they are the best current guess.
- **Which ambiguities occupancy can't resolve:**
  - Every legal move gives a distinct occupancy grid; the origin square always differs. chess.js tracks piece types from the start position.
  - **Exception: the promotion piece.** Default to a queen; in the sidebar, tapping that move switches the piece (Q/R/B/N).
  - Piece count and colour are enforced by legality, so a spurious "extra piece" or a colour flip on a dark square can never be committed.

## Vision changes (`src/vision/occupancy.ts`, `src/worker/*`)
- `OccupancyTracker`:
  - Expose the orientation map: board cell → square.
  - Add `setPosition(grid)` (the hint), which bypasses `OccupancyFilter` in game mode.
  - Keep the models across board loss.
  - Emit `vis` alongside `logLik`.
- `protocol.ts`: add `observation?: Observation` to `ResultMessage`, and new main→worker messages `positionHint` and `gameMode` (on/off).
- **Learning guard:** `learn()` only uses cells where the hint and the frame's own confident classification agree. Without that guard, one wrong engine commit would teach the models wrong labels (drift).

## UI
- **Play mode** = debug panel hidden. Before lock-in, show the quad and a status chip: "Set up the starting position". After lock-in, draw a piece icon per occupied square, from `game.board()` (the engine's position, not the raw occupancy).
  - The icon is screen-upright and scaled by the projected cell size (the same size maths as `occupancyDots` in `src/overlay.ts`). It is anchored slightly toward the square's base, faint, and fades with the quad.
  - The squares of the last move are tinted.
  - Icons are preloaded as `HTMLImageElement` from the vendored SVGs.
- **Debug mode** keeps today's dots unchanged, plus the game state and the top-3 hypothesis scores in the panel.
- **Sidebar** (new `src/ui/moves.ts`, markup in `index.html`):
  - A thin (about 140 px) translucent column on the right in landscape. It is collapsible.
  - It shows move pairs (`1. e4 e5`) and auto-scrolls. The last move is highlighted.
  - Buttons: **Copy PGN** (`navigator.clipboard.writeText`, with a Seven Tag Roster where `Date` is today and the other tags are `?`), **Undo** and **New game**.
  - A status line shows: waiting / White to move / check / mate / desync.
- **Persistence:** the game PGN goes to `localStorage` (inside try/catch), so a reload or camera switch doesn't lose it.

## Main challenges (and mitigations)
| Challenge | Mitigation |
|---|---|
| Mid-move states: a lifted piece, a capture removed before the capturing piece lands, a hand hovering | Only stable frames count; a hypothesis must win for 300–500 ms; `stay` stays a candidate |
| Oblique occlusion: a far-rank move hidden behind tall pieces | logLik is already flattened by visibility; the move commits later, once its cells are seen, or via depth 2 |
| Touch-move and takebacks | Shorter paths in the lattice are hypotheses; the UI has Undo |
| A wrong commit (e4 instead of d4) | Revision window: sibling paths keep being scored and replace the tip under stricter thresholds |
| Flicker between near-equal hypotheses | Incumbent bonus, dwell time, revision cooldown, runner-up margin |
| Wrong commit feeding back into vision | Learning guard; Undo re-sends the hint; desync state |
| Orientation lost after board loss | Re-orient against the game's expected grid (8 dihedral maps) |
| Mirror ambiguity at lock-in | Handedness from the camera pose, plus the dark-a1 check |
| Promotion piece not observable | Queen by default; tap to change in the sidebar |
| Starting position set up wrong (e.g. king/queen swapped) | Not detectable (types aren't observed). Documented limitation |
| New game mid-session | Decided: if a confident start position holds for about 2 s after at least one move, save the current PGN (the last N games in `localStorage`), start a new game and show a toast with "Undo". The sidebar also has a "New game" button |

## Phasing
1. **Hook and orientation.** Observation/hint protocol and the orientation map in vision. No UI change; the existing tests stay green.
2. **`src/game` with chess.js.** Lock-in, the revisable lattice with `K = 1` (beam, advance / revise / takeback, anti-flicker), PGN. Unit tests on synthetic logLik sequences:
   - noise, hand frames and occluded cells;
   - castling, en passant and promotion;
   - **e4 committed from occluded evidence, then d4 revealed**: it gets revised, including after black has replied (`[d4, e5]`);
   - **two alternating near-equal hypotheses**: at most one switch;
   - **`K = 2` parametrised test**: revising two plies back.
3. **Feedback loop.** Hint → occupancy prior and learning guard; re-orientation after loss.
4. **UI.** Vendored icons, piece rendering in play mode, sidebar with Copy PGN / Undo / New game, persistence.
5. **Desync UX; evaluate `K = 2`** on the bench (wrong-commit rate vs latency) before making it the default.

## Verification
- `npm test`: new `tests/game.test.ts`, which feeds `GameTracker` synthetic observations built from the grids of a known game plus noise. It asserts that the SAN sequence and PGN match, that no move is committed during "hand" frames, and that a takeback undoes.
- **Synthetic end-to-end in Node:** extend `tests/synth/occBench.ts`, which already plays `GAME` moves with `makeBoardSample`. Wire vision → game and report the move accuracy and the commit latency (in frames) per elevation and palette. Target: no wrong commits.
- `npm run typecheck`, `npm run build`, and `npm run e2e` (assert that the sidebar exists, and that `data-game` reports `waiting` or `playing`).
- **Manual check on Android** (Pages URL): set up the starting position, check the lock-in, play about 10 moves including a capture and castling, pass a hand over the board, take back a move, then use Copy PGN and paste it into lichess to check it imports.
