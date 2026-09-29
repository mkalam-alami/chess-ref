# Status: multi-board dataset + occupancy calibration (resume notes)

Companion to [PLAN-multiboard.md](PLAN-multiboard.md). Paused on 2026-09-29 after Wave 3's first prototype (D5), for budget reasons. D5 landed in Wave 4 (`5e83163`); it is the only `src/` change so far. The baseline below predates it.

## What landed on main

| Commit | What |
|---|---|
| `2b4a614` | Real-photo occupancy bench, `BENCH_ONLY=realocc npm run bench` (`tests/synth/realOccBench.ts`; env vars documented at the top of the file). Holdout images are excluded unless `REALOCC_HOLDOUT=1`. |
| `f79d329` | Annotation tooling, `npm run annot` (`tests/tools/`). Workflow in `tests/fixtures/web/ANNOTATION.md`. |
| `5f88243` | `tests/fixtures/private/` is gitignored. |
| `b0bcc2e` | The bench skips images whose `SOURCES.json` split is `"excluded"`. |
| `1a21e05` | Per-cell classification margin and start-test headroom in the bench. |
| `5e83163` | **D5 landed:** detection retry with a lower Canny percentile (`retryCannyPercentile`, default 85, 0 disables) in `src/vision/detector.ts`, plus a regression test in `tests/detect.test.ts`. |

## Dataset (private, not in git)

- The image sourcing agent (Wave 1 A) was stopped. The user supplied their own photos **for private use only**. They live in `tests/fixtures/private/` (gitignored), together with `SOURCES.json`, `corners.json` and the annotators' working files. **Never commit them or anything derived from them** (crops, overlays, per-image results).
- The container is ephemeral. If `tests/fixtures/private/` is gone when resuming, ask the user for the photos, `corners.json` and `SOURCES.json` again. They were sent a copy of the two JSON files; without them the annotation step has to be redone (see ANNOTATION.md).
- 17 photos supplied, 12 usable. Excluded: 03 and 06 (pieces off the board, likely AI-generated), 05 and 09 (mid-game), 14 (pieces rotated 90° relative to the board, so a1 is light).
- **Split:**
  - Tune (8): 01 marble, 04 walnut top-down, 08 burl + chrome pieces, 10 carved redwood top-down, 11 green vinyl top-down, 13 wall demo board, 15 small folding board, 17 wooden board on a dock in hard sun.
  - Holdout (4): 02, 07, 12, 16. Never run them until the final scoring. The user doesn't need a holdout "before" score; score it once on the final commit.
- **Corners:** two annotators, cross-checked. All corners are under 1% of the board diagonal, and most are under 0.4%. The user reviewed the grid overlays and approved them.

## Baseline (before any `src/` change)

Raw cell accuracy at 640 px, starting position. The "GT corners" column uses the annotated corners, which isolates occupancy from detection.

| Image | Detected | Detected corners | GT corners | Notes |
|---|---|---|---|---|
| orig 01–03 | yes | 100 | 100 | min margin only 1.8–4.0 |
| orig 04 | yes | 96.9 | 98.4 | |
| orig 05 | yes | 93.8 | 85.9 | start test 0.72, below the 0.75 threshold → fallback |
| 01 marble | **no** | – | 87.5 | empty light squares read as white |
| 04 walnut | yes | 100 | 100 | |
| 08 burl | yes | 96.9 | 96.9 | 2 black pieces with gold collars read as white |
| 10 redwood | yes | 100 | 100 | |
| 11 green vinyl | yes | 100 | 100 | |
| 13 wall board | yes | 95.3 | 96.9 | |
| 15 folding | **no** | – | 98.4 | |
| 17 dock | **no** | – | 85.9 | start test 0.71 → fallback; shadows, occlusion |

- **Tune aggregate:** 5/8 detected. End-to-end accuracy is 61.5%; with GT corners it is 95.7%.
- **Caveat:** raw accuracy is scored on the same frame the calibration learns from, so it is flattering. Use the margin metric (`1a21e05`) to see progress.
- **Weak spots by margin:** empty dark squares, and black pieces on dark squares. Top-down boards have large margins.
- **Where the baseline JSON lives:** the gitignored `tests/synth/realocc-out/`. It is lost with the container; regenerate it by running the bench on the commit before any `src/` change.

## Wave 3 status

- **D5, detection recall:** **landed** in `5e83163` (diff below kept for reference). Realocc bench re-run on the landed commit: orig 5/5 detected (unchanged), `neg-01` no detection, tune 6/8 (01 marble now detected at 0.32% error; 15 and 17 still undetected), tune end-to-end 72.1%. Synth bench not re-run; the prototype's numbers below stand. The every-other-frame retry was not implemented.
- **D1** (relative/stem features), **D3** (glare/shadow), **D2** (chroma white/black split): not started; cancelled for budget.
- **D4** (appearance presets): deprioritised by the user.

### D5 findings

- **01 marble:** a vanishing-point failure, not a colour problem. The adaptive Canny threshold follows the whole image, so the wood grain in the surround pushes one grid-line family below the threshold, and the vanishing points lock onto furniture lines. Fixed by a retry with a lower threshold.
- **15 folding:** fails the checker verification even at the GT corners (score 0.31 < 0.4). A side view where the pieces hide half the cells. Fixing it needs an occlusion-tolerant verification, which carries false-positive risk.
- **17 dock:** almost no edges on the grid lines (glitter, planks, hard shadows), and checker verification is about 0.25 even at the GT corners. Fixing it needs regional edge thresholds and a shadow-robust verification.
- **Colour-based corner detection** would not have helped any of the three.

### D5 results (prototype; confirmed on the realocc bench after landing)

- **Tune:** 6/8 detected; end-to-end accuracy 61.5% → 72.1%. 01 marble: 0.32% mean corner error, 84.4% cells.
- **Regressions checked:** original 5 photos unchanged, still no false positive on `neg-01`, `npm test` and `npm run typecheck` pass.
- **Synth bench:**
  - recall 86.1% → 88.9%
  - wrong detections 13 → 15 (of 216)
  - false positives 0 → 0
- **Cost:** frames without a board take about 2× the detection time (roughly 37 → 65 ms on the synthetic negatives). Frames with a board cost nothing extra. Option: run the retry only every other frame.

```diff
diff --git a/src/vision/detector.ts b/src/vision/detector.ts
@@ export const VANISHING_PARAMS: readonly ParamSpec[] = [
   { name: 'vpMinSeparation', min: 5, max: 60, step: 1, default: 20 },
 ];
 
+export const RETRY_PARAMS: readonly ParamSpec[] = [
+  /**
+   * When a full detection finds no board, retry once with this (lower) Canny percentile; 0 disables. The adaptive
+   * Canny threshold follows the whole image's gradient distribution, so a busy surround (wood grain, reflections)
+   * can push a board's weaker grid lines (low-contrast squares, one rank direction foreshortened) below it and the
+   * vanishing points lock onto clutter instead. Costs a second pass on frames without a board only.
+   */
+  { name: 'retryCannyPercentile', min: 0, max: 99, step: 1, default: 85 },
+];
+
@@ export const ALL_PARAMS: readonly ParamSpec[] = [
   ...VANISHING_PARAMS,
+  ...RETRY_PARAMS,
   ...GRID_PARAMS,
@@ export class Detector {
   /** Accepts an RGBA Mat (not owned) or ImageData. */
   detect(input: Mat | ImageData, params: Params, opts: DetectOptions = {}): DetectResult {
+    const first = this.detectOnce(input, params, opts);
+    if (first.corners) return first;
+    const retry = param(params, RETRY_PARAMS, 'retryCannyPercentile');
+    if (retry <= 0 || retry >= param(params, PREPROCESS_PARAMS, 'cannyPercentile')) return first;
+    const second = this.detectOnce(input, { ...params, cannyPercentile: retry }, opts);
+    for (const [k, v] of Object.entries(first.timings)) second.timings[k] = (second.timings[k] ?? 0) + v;
+    second.timings.retry = 1;
+    return second;
+  }
+
+  private detectOnce(input: Mat | ImageData, params: Params, opts: DetectOptions): DetectResult {
     const cv = this.cv;
```

## Next steps when resuming

1. **Wave 3, remaining prototypes:** D1 (symptom 1, black on dark / light squares read as white), D3 (17's shadows), then D2. Each runs in an isolated worktree, reports bench numbers and margins only, and uses a strict time budget. Only the integrator edits `src/vision/occupancy.ts`.
2. **Calibration:** the start-test score is just under the threshold on orig 05 (0.72) and tune 17 (0.71). Worth a look alongside D1/D3.
3. **Optional:** run the D5 retry only every other frame to halve its cost on frames without a board.
4. **Final:** score the holdout once, with `REALOCC_HOLDOUT=only`, on the final commit.

## Process rules (from the user)

- Work and push on `main`.
- The orchestrator delegates the work to sub-agents.
- The private photos are never committed.
- The holdout is scored only at the end.
- Give every agent a time budget: the user's budget is limited.
