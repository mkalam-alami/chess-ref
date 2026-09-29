# Milestone 8: Square occupancy (empty / white / black)

## Context
The board detector (milestones 1–7) outputs a verified board→image homography `hb` (cell units, 0..8) at each frame. The next phase reads the pieces. We deliberately skip piece-type recognition: each of the 64 squares only gets **empty / white / black**. A chess engine will later turn the sequence of occupancy grids into moves, starting from the initial position. For debugging, the overlay draws one dot per square: a small black dot for empty, a white dot for a white piece, a green dot for a black piece.

## Challenges and design issues

### 1. Pieces are 3D, but the homography only covers the board plane
This is the biggest issue. In an oblique view, a piece standing on a square rises in the image and covers the square(s) *behind* it. Tall pieces on ranks 1–2 hide much of ranks 3–4 (fixture `04-side-low-oblique` shows this). Two naive approaches both fail:
- **Sampling the whole cell** mixes in the piece from the square in front.
- **Sampling only the cell centre** is exactly what `verify.ts` avoids. On an empty square the centre is fine. On an occupied square it shows the piece, but it may also show a piece from the square in front.

**Proposal: a camera-aware sampling footprint.**
- Recover an approximate camera pose from `hb`: estimate the focal length from the square-board constraint (the principal point is the image centre and pixels are square). If the estimate is degenerate, as near overhead, fall back to a nominal phone FOV of about 65°.
- Then `K⁻¹·hb = [r1 r2 t]` and `r3 = r1×r2`. With this we can project a point at height *z* above a square.
- For each square, the evidence region is the **base footprint**: a disc of radius about 0.3 cells at z≈0, plus a short "stem" up to z≈0.4 cells. Only the part of the stem that no nearer square's piece could cover is kept.
- Pixels that nearer squares could cover are down-weighted instead of ignored. Each cell's confidence reflects how much of its footprint is still visible.
- **Decision (user): no 4th state; drop low-confidence evidence.**
  - **Per cell:** if a cell's confidence is below `occCellMin`, this frame does not update it. The filter keeps its last committed state (initially the starting position).
  - **Per frame:** if too many cells are low-confidence (more than `occMaxLowCells`), or the mean confidence is below `occFrameMin`, drop the whole frame's occupancy.
    - The worker sends `occupancy: null` for that frame.
    - The filter is not updated.
    - The overlay keeps drawing the last committed grid, which holds and fades with the quad.
  - The output contract stays 3 states. The debug panel shows the count of dropped cells and frames, so this can be tuned.

### 2. Piece colour vs square colour
The hard cases are a black piece on a dark square and a white piece on a light square. The owner's printed board is black/light-grey, so a black piece on a black square is close to invisible in plain luminance. Mitigations:
- **Compare per parity, not absolutely.**
  - `verify.ts` already knows the dark-square and light-square levels (`sampleCells` gives the per-cell medians).
  - Build an "empty square" model per parity: the mean and spread in L, a\*, b\*, using the squares that are confidently empty.
  - An occupied square is one that **deviates** from its parity's model.
- **Use more than mean colour.** Pieces add *structure*: outline edges, specular highlights, shading gradients and a bimodal histogram. Per cell, compute:
  - the colour distance from the parity model;
  - the gradient/edge energy inside the footprint (we already have `pre.gradient`);
  - the percentile spread (p90 − p10).

  An empty printed or vinyl square is flat. Wood grain is the exception, and the empty-square model absorbs it.
- **Use a piece-colour model, not the board's.** White and black pieces are not the same colours as the light and dark squares (for example boxwood/ebony on a brown board). Classify occupied cells into 2 clusters by piece colour:
  - The cluster centres are initialised from the starting position (see 4).
  - After that they are updated slowly from confidently classified cells.

### 3. Which image data to sample
- The Lab3 image from `Preprocessor` is CLAHE-equalised. CLAHE is good for edges but distorts absolute levels locally, and levels are what piece colour depends on.
- Once an L-only profile is locked, a\*/b\* aren't computed at all.
- **Proposal:** occupancy samples the raw frame (`s.rgba` in the worker). It converts only the sampled pixels to Lab, which is cheap (≈64 cells × ~40 samples). This keeps occupancy separate from the profile optimisation.
- Frames are downscaled to 640 px on the long side. Far ranks in a low oblique view are about 12–20 px per cell. That is enough for occupancy, but the parallax model from issue 1 matters more there.

### 4. Board orientation (4-fold symmetry)
Today the corner order is only stabilised on the **main thread** (`stabiliseCorners`, `repairQuad` in `src/geom/cornerOrder.ts`). The worker's `hb` can come back rotated by 90°/180° after each full re-detection. That creates two problems:
- **Temporal filtering in the worker breaks** unless the grid is indexed in a stable frame. **Proposal:** stabilise the orientation of `hb` in the worker (`TrackingSession`). Choose the dihedral relabelling of the new `hb` whose corners are closest to the previous ones. Then the occupancy grid and the corners share one frame.
- **Absolute orientation.** For the engine we need to know which edge is rank 1. The starting position resolves this: the edge with 16 white pieces is rank 1. Then check that a1 (the near-left corner) is a dark square. This is also the natural moment to **calibrate** the piece-colour clusters (issue 2) and the empty-square models: ranks 3–6 are known empty.
- For the debug dots alone, absolute orientation isn't needed. But the grid must be rotated consistently with any corner permutation the main thread applies. Simplest fix: move `repairQuad`/`stabiliseCorners` into the worker, so that `corners[k]` always maps to board corner k.

### 5. Temporal stability and hands
- Single-frame classification will flicker. **Proposal:** keep a per-cell log-odds accumulator with hysteresis. A cell changes state only after N consistent frames.
- **Hands over the board** (during moves) produce large, skin-coloured, fast-changing blobs. Tracking may also fail for a frame or two. Proposal:
  - Freeze updates when many cells change at once, when skin chroma is detected on the board, or when detection or tracking fails.
  - Only commit a new grid after the board has been stable for about 300–500 ms.
  - Later the engine can reject grids that don't correspond to a legal move.
- **Scope choice for now:** report both the raw per-frame grid and the smoothed grid, but draw only the smoothed one. The engine milestone owns "stable position" semantics.

### 6. Shadows and glare
- Pieces cast shadows onto neighbouring light squares, which can look like a dark piece. Glare on vinyl can look like a white piece.
- Both are low-texture and have soft edges. The edge-energy feature and the footprint geometry help: a shadow lies away from the light source, not at a piece base.
- We'll tune this on real clips, and expect residual errors here.

### 7. Coupling with detection and performance
- Occupancy runs only when a board is accepted (full or tracking), after `hb` is final. The target is under 3 ms per frame.
- It must not change detection results. The existing detection tests must stay green.
- Later we could feed occupancy back (for example, re-weighting verification samples on occupied cells), but that is out of scope.

### 8. Ground truth for testing
- **Real photos:** all 5 positive fixtures show the starting position. Add to `tests/fixtures/real/corners.json` which corner is on the white side, so that each photo has a known 64-cell ground truth. Occluded squares in `04` get the value "don't care".
- **Synthetic:** `tests/synth/generate.ts` `drawPieces` already places pieces on cells with a light/dark colour, but it doesn't record them. Return per-cell ground truth in `meta`. Add a `start` pieces mode (ranks 1–2 light, 7–8 dark, with realistic heights) and a random mid-game mode.
- Record a few video clips of real play for manual tuning, using the existing "Load file" input.

## Proposed implementation

### Worker / vision
- **New `src/vision/occupancy.ts`:**
  - `cameraFromHomography(hb, w, h)`: returns the focal estimate and pose, plus `projectHeight(i, j, z)`.
  - `cellFootprints(pose)`: for each cell, returns sample points with weights, including the occlusion weight from nearer cells.
  - `sampleOccupancyFeatures(rgba, gradient, footprints)`: returns per-cell Lab stats, edge energy and spread.
  - `classifyOccupancy(features, model)`: returns `Uint8Array(64)` of `0 empty, 1 white, 2 black`, plus a `Float32Array(64)` confidence per cell. The confidence combines the classifier margin and the visible fraction of the footprint.
  - `OccupancyModel`: holds the per-parity empty models and the two piece-colour clusters. It has two initialisation paths: bootstrap from the starting position (auto-detected once 32 occupied cells split 16/16 across opposite edges), or unsupervised fallback (empty = closest to the parity model, then 2-means on the occupied cells).
  - `OccupancyFilter`: per-cell hysteresis. It skips cells with low confidence, drops frames whose confidence is too low, and freezes on mass change (hands). It returns the committed grid.
  - `OCCUPANCY_PARAMS` (added to `ALL_PARAMS` in `src/vision/detector.ts`, so they become debug sliders).
- **`src/worker/tracker.ts`:** keep `hb` in a stable dihedral orientation across frames.
- **`src/worker/detector.worker.ts`:** after `session.process`, run occupancy when `det.hb` is set. Hold the model and filter next to `profileLock`, and reset them on `reset`/`resetProfile`.
- **`src/worker/protocol.ts`:**
  - Add `occupancy?: Uint8Array` (64 entries, row-major in `hb` cell coordinates, aligned so that `corners[0]` is the board-space corner at (0,0)) and `occupancyConf?`.
  - Add a debug view `'occupancy'` that shows the footprints and features.

### Main thread
- Corner ordering moves into the worker (or the main thread derives the dihedral permutation it applied and permutes the grid the same way).
- **`src/overlay.ts`:**
  - Store the grid with the held quad.
  - In `draw`, compute H from the smoothed screen corners with `homographyFrom4` (`src/geom/homography.ts`), and place dots at the cell centres (optionally the base point).
  - Dot radius scales with the local cell size: small for empty (black dot), larger for white (white dot) and black (green dot), with a thin dark outline so that white dots show on light squares.
  - When `occupancy` is null, keep the last grid.
  - Dots fade with the quad (`quadAlpha`).
- **`src/debug/panel.ts`:** add occupancy counts (white/black/empty), the number of low-confidence cells, the frame-drop rate, the bootstrap state and the occupancy timing.

### Tests
- `tests/occupancy.test.ts`: unit tests for pose from a synthetic `hb` (focal recovery on known cameras), footprint/occlusion geometry, and filter hysteresis.
- Synthetic bench: occupancy accuracy per mode (start / mid-game, elevation, palette). - Target ≥ 97% on committed cells.
- Also report the frame-drop rate. Low oblique views may drop, but must never commit a wrong grid.
- Real photos: starting-position accuracy on the 5 fixtures.
- Also update `tests/overlay.test.ts` for dot placement.

## Verification
- `npm test`, `npm run typecheck`, `npm run build`, and `npm run e2e`. In the e2e smoke test, also assert that the result message carries a 64-cell grid.
- Run `npm run bench` (the synthetic bench) before and after the change: detection accuracy and timing must not regress, and occupancy accuracy must meet the target.
- Manual check on an Android phone, on the Pages URL: starting position (overhead and seated view), then a few moves, with hands passing over the board.

## Setup phase (provisional calibration before lock-in)
Live use showed trackers stuck in a wrong calibration: a weak two-frame start test had passed on the wrong thing (pieces still being placed, an earlier view), and nothing ever revisited the models. So calibration is now **provisional until the game locks in**. `OccupancyTracker` (`src/vision/occupancy.ts`) behaves as follows:
- **Start test on every frame.** While no position hint is set, every framed frame runs the model-free start test (`testStart`) on both rank axes.
  - The test is trimmed: `occSetupTrim` odd cells per group do not count.
  - A pass (`occSetupScore`, `occSetupGap`, `occSetupDL`) adds 1 to that placement's leaky evidence, which decays by `occSetupDecay` per frame.
  - A placement is accepted at `occSetupAccept` with a lead of 1 over the others, which means 2 consecutive passes.
  - It is not accepted when a fresh fit on it still contradicts it on more than `occSetupBaseMax` cells. This covers mid-game boards that pass the trimmed test.
  - The state is then **`setup`**.
- **Recalibration.** When the current models contradict the accepted start on more than `occSetupRecal` visible cells, on a frame that passes the test, a fresh start fit is tried. It is adopted when it explains the frame clearly better. The origin of the bad models does not matter.
  - A recalibration re-seeds the filter's committed grid.
  - It unverifies an orientation that puts the start elsewhere.
- **Cumulative calibration.** A frame is accumulated only when it passes the consistency guard: at most `occSetupGuard` contradicting cells (plus up to as many as the board's own fit cannot avoid) and at most that many outliers, not dropped, not frozen. Frames under a hand are frozen, so they are skipped.
  - The per-frame models of the last `occSetupFrames` frames that passed are pooled: the median of the means, plus the median variance and the squared spread between frames.
  - After `occSetupDrop` consecutive settled frames that fail the guard, the start is withdrawn and the state goes back to `start`. The models are kept.
- **Optimism.** While a start is accepted:
  - `logLik` gets a log-prior of `occSetupPrior` nats towards the start class.
  - While frames agree with the start, the committed grid is the start grid.
  - `raw` / `conf` stay the frame's own evidence.
  - The game's `startConfident` check still gates lock-in, which accepts `setup` or `calibrated`.
- **Lock-in and reset.** Once the hint arrives, the state is `calibrated` and the models are final. `learn()` then adapts them with the hint as labels. Clearing the hint (a new game) returns to the provisional regime.
- **Fallback.** `fitUnsupervised` only runs while no start was ever accepted, after `occFallbackMs`. A later accepted start replaces its models.
- **`learn()` drift fix.** Before the fix, repeating one static frame of fixture 04 degraded raw from 63 to 57/64. The root cause was that learning selected cells by the classifier's own confident agreement (conf ≥ 0.6). That drops each model's tail towards the other classes, so the variance estimated from the remaining cells is too small. The tighter model then excludes more tail cells, a positive feedback that collapsed the variances to their floors.
  - `learn()` now uses the hint's labels only, not the tracker's own committed grid.
  - It keeps a cell when the label's model explains it (zdist < `occDeviation`) and the frame does not confidently read it as another class.
  - Before lock-in it does not run: the setup accumulation replaces it.
