# Plan: multi-board dataset + occupancy calibration improvements

## Context
Board localisation works, but occupancy classification (`src/vision/occupancy.ts`) is weak on boards other than the owner's black/light-grey printed board. Three symptoms: square colour mistaken for piece colour, both piece colours classified as one, empty squares read as occupied under glare or shadow. All real-photo tests use 5 photos of that one board (`tests/fixtures/real/`). Goal: add 10+ free-licence starting-position photos covering different materials and colours, measure against them, and improve bootstrap/calibration. This may add appearance presets ("board profiles") per colour set.

## Current state (from exploration)
- Calibration: `testStart` (occupancy.ts:613) finds the starting position by checking two things: the middle 4 ranks fit one empty model per square colour, and the outer ranks deviate from it by zdist ≥ 16. It then needs |ΔL| ≥ 6 between the two edge groups (the lighter group is white). `fitModels`/`fitIllum` learn 6 diagonal Gaussians, one per (empty/white/black) × (light/dark square), over 5 features: mean L*, a*, b*, L-spread, gradient.
- Fallback `fitUnsupervised` (675) splits white from black with a 2-means on **L\* only**.
- `BoardProfile` (`src/vision/profile.ts`) is used by detection only and inferred automatically (channels, contrast, surround). No presets exist and nothing feeds occupancy.
- Bench: `npm run bench` (`tests/synth/bench.ts`, `occBench.ts`). Real-photo occupancy is measured only in `tests/occupancy.test.ts:377-439`. Ground truth comes from `corners.json` (4 corners + `whiteEdge`) via `gtStart`/`alignGt`. For starting-position photos, 4 corners + whiteEdge give full per-cell ground truth for free.

## Likely root causes, by symptom
1. **Square colour vs piece colour.** Features are absolute Lab at z=0, dominated by the square underneath. The stem samples at z=0.2/0.4 have weight 0.7 but never dominate. A black piece on a dark square, or a boxwood piece on a maple square, differ little in mean L. → Use features relative to the cell's own empty model (ΔLab from the expected empty value). Weight the stem/top samples more. Consider a height/parallax cue: a piece hides the neighbouring square in the viewing direction, glare does not.
2. **All pieces one colour.** The start test requires |ΔL| ≥ 6, and the fallback splits on L only. That fails on red/cream, natural/stained wood and similar sets. → Split on the Lab direction that best separates the two edge groups (LDA-like, including chroma). Keep the "lighter = white" rule, but use a combined contrast measure.
3. **Empty read as occupied under light.** Specular highlights raise the L-spread and gradient features. Illumination gains are clamped to exp(±1.5), and wood grain looks like texture. → Mask saturated pixels, use trimmed statistics, rely more on chroma where shadows leave it unchanged, and apply per-profile feature floors.
4. **Profiles (decided: auto + manual override).** Add `AppearancePreset` priors (square colours, piece colours, which channels separate them, feature floors), e.g. brown wood, green/white vinyl, blue vinyl, black/grey printed, marble/glass. During calibration, pick a preset automatically by matching the sampled empty-square colours, then let the learned models refine it. A picker in the debug/settings panel (`src/main.ts`, sent to the worker through `src/worker/protocol.ts`) can force a preset, with "Auto" as the default. The detection `BoardProfile` could take its channels from the preset when squares alternate in chroma.

## Key challenges
- **Image access.** At planning time the egress policy blocked the image hosts. The user is switching the environment to broader network access. Agent A first re-probes Wikimedia, Unsplash, Pexels and Openverse, and stops to report if they are still blocked.
- **Licensing.** Only CC0, CC-BY or CC-BY-SA, or public domain. Record author, URL and licence per image in `SOURCES.json`. Downscale to about 1600 px long side to keep the repo small.
- **Annotation accuracy.** Agents can view images, but corners must be accurate to about 1% of the diagonal. Approach: the detector proposes corners, a crop tool zooms on each corner, and an agent confirms or adjusts. Then a second agent cross-checks, plus a rendered grid overlay PNG for the user to eyeball. whiteEdge is visual.
- **Photo scope (decided).** Real photos only, no renders. Both oblique and top-down views are allowed, and the full 8×8 area must be visible. Starting position only; mid-game photos would need 64-cell labels, so they are out of scope. Top-down shots stress the camera estimate in `cameraFromHomography` and its FOV fallback, and pieces have no visible side "stem" in them. Tag each image with its view (`oblique`/`topdown`) so the bench reports each group separately.
- **Overfitting.** Split the new images into tune (~2/3) and a holdout (~1/3) that is only scored at the end. The existing 5 photos and the synth bench must not regress.
- **Parallel work on main.** Several agents editing `occupancy.ts` at once would conflict. Experiments run in parallel as prototypes and report with bench numbers. One integrator agent lands changes on main one after another.

## Execution (delegated, parallel where possible)
**Wave 1 (parallel)**
- A: Source and download 10–15 images. Diversity matrix: light/dark wood, green vinyl, blue vinyl, marble or glass, red/black, plastic Staunton, boxwood/ebony, coloured sets, window light, overhead glare, dim. Write `tests/fixtures/web/` + `SOURCES.json`.
- B: Build a real-photo occupancy bench (`BENCH_ONLY=realocc`) over any fixture directory. Report per-image detection OK and corner error, bootstrap state (start/fallback/none), orientation, and a confusion matrix split by square colour (E/W/B × light/dark). Dump overlay PNGs. Run the baseline on the existing 5 photos.
**Wave 2**
- C1/C2: Annotate corners + whiteEdge (images split between two agents, each cross-checking the other's). Commit, then run the baseline over all images and categorise failures by symptom.
**Wave 3 (parallel prototypes, report only)**
- D1: relative/stem features (symptom 1).
- D2: chroma-aware white/black split in `testStart`/`fitUnsupervised` (symptom 2).
- D3: glare/shadow robustness (symptom 3).
- D4: appearance presets + auto-selection (profiles). Also detection recall on the new boards if baseline shows failures.
**Wave 4:** Integrator lands the winning changes one at a time on main, re-running `npm test`, `npm run typecheck` and the full bench each time. Adds regression tests: per-image thresholds on tune images, and an aggregate threshold on holdout.

All agents commit and push directly to `main`, as the user and AGENTS.md require. Only the integrator touches `src/vision/occupancy.ts`. Wave 1 agents work on separate files (fixtures vs bench), so they don't conflict.

## Verification
- `npm test`, `npm run typecheck`, `npm run build` green.
- `BENCH_ONLY=realocc npm run bench` before/after table per image. Targets: bootstrap succeeds on ≥ 80% of detected boards; raw cell accuracy ≥ 95% on tune and ≥ 90% on holdout; no regression on the original 5 photos or the synth occupancy bench.
- Overlay PNGs for the user to review visually.

## Handoff notes (decisions already made with the user)
- Image source: broader network access (the user widens the environment's policy). If the hosts are still blocked, stop and report; don't fall back to other sources silently.
- Profiles: chosen automatically, plus a manual override picker (default "Auto").
- Photo scope: real photos only, oblique and top-down, starting position, full board visible. No renders, no mid-game.
- Every agent works and pushes on `main` (AGENTS.md). Only the integrator edits `src/vision/occupancy.ts`.
- The orchestrating session doesn't implement anything itself: it delegates to sub-agents and runs independent waves in parallel.
