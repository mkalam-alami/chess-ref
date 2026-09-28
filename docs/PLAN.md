# PoC: Live chessboard detection in the browser (GitHub Pages)

## Context
The repo `chess-ref` is empty. The goal is a proof of concept of a static web page, deployed to GitHub Pages, that:
- opens the phone's rear camera,
- plays the camera stream full screen,
- detects the **8x8 playing area** of a physical chessboard,
- draws the detected quadrilateral on top of the video.

Everything runs in the browser. There is no server.

Constraints confirmed with the user:
- **Camera pose:** any angle, from overhead down to an oblique view from the player's seat. Handheld or on a stand.
- **Boards:** wooden (brown/tan) and vinyl tournament (green/cream). Both have low luminance contrast. Pieces may be on the board.
- **Overlay:** only the outer quad of the 64 squares.
- **Lost track:** hold the last good quad for about 500 ms, then fade it out.
- **Stack:** OpenCV.js (wasm), Vite + TypeScript, Android Chrome only.
- **Debug:** a toggleable debug/tuning panel.

## Core idea
The detector works in two steps:
1. **Propose candidates from geometry.** Find straight lines, group them into the two grid directions, and fit a regularly spaced grid.
2. **Verify with the checker pattern.** Keep a candidate only if the squares alternate dark/light the way a chessboard does.

The verification step is the user's "alternating squares contrast" idea. It makes the detector robust because it does not rely on absolute brightness: it only compares each square with its neighbours, and it uses medians so squares covered by pieces don't break the score.

## Pipeline (runs in a Web Worker)
1. **Frame capture (main thread).** Take a frame with `createImageBitmap(video)`, downscale it so the long side is about 640 px, and transfer it to the worker. Only one frame is in flight at a time; if the worker is busy, frames are dropped.
2. **Preprocessing.**
   - Convert RGB to Lab.
   - Apply CLAHE to the L channel. This evens out uneven lighting and shadows.
   - Build a **multi-channel gradient**: per pixel, take the max of the Sobel gradients on L, a\* and b\*. The a\* channel is what makes green/cream vinyl boards visible; plain grayscale barely shows them.
   - Apply a light bilateral or median blur first, to suppress wood grain.
3. **Edges and lines.**
   - Run Canny with thresholds set automatically from gradient percentiles, so they adapt to lighting.
   - Run `HoughLinesP`.
   - Merge collinear segments, because pieces break grid lines into pieces.
   - Drop short segments.
4. **Two line families via vanishing points.**
   - Use RANSAC over pairs of segments to find the dominant vanishing point (VP), in homogeneous coordinates so that an overhead view (VP at infinity) works too.
   - Remove its inliers and find the second VP.
   - This handles overhead and oblique views the same way. Grouping lines by angle alone would fail under strong perspective.
5. **Affine rectification.**
   - The line joining the two VPs is the image of the "line at infinity".
   - Mapping it back to infinity makes each family parallel. Then rotate so the families are horizontal and vertical.
   - In the rectified space, the grid lines of each family are evenly spaced, because affine maps preserve spacing ratios along parallel lines.
6. **Comb fit per axis.**
   - Project the lines of each family onto the perpendicular axis.
   - Search for a period and offset that best explain the observed positions with 9 evenly spaced "teeth", allowing for missing lines. Score the fit on line support plus the gradient profile.
   - Keep the top-K hypotheses per axis. There can be ambiguity: an extra line from the board border, or a fit that is off by one square.
7. **Checker verification and selection.** For each combination of axis hypotheses (and ±1-cell shifts):
   - Build the homography H (board → image).
   - Warp to a 256×256 board image.
   - Compute a **robust alternation score**: the median over cells of the signed contrast with each neighbour, compared against the expected checker parity.
   - Add a small bonus if the ring just outside the 8x8 area is *not* checkered.
   - Pick the best combination. Accept it only if the score is above a threshold.
8. **Refinement.**
   - Around each of the 49 predicted inner corners, look for an X-corner (saddle point) and refine it with `cornerSubPix`.
   - Re-fit H with `findHomography(RANSAC)`. Corners hidden by pieces become outliers and are ignored.
9. **Tracking mode.**
   - When the previous frame was accepted, skip steps 3–6 and go straight to refinement around the corners predicted from the previous H.
   - Fall back to full detection if the verification score drops, or once per second.
   - This makes the loop faster and the overlay steadier.
10. **Output.** The worker returns four corners in frame coordinates, a confidence score, and optional debug images.

## Main thread: display and overlay
- **Camera:**
  - Request the rear camera at about 1280×720 with `getUserMedia({video:{facingMode:'environment', width:{ideal:1280}, height:{ideal:720}}})`.
  - Put the stream in a `<video playsinline muted>` with `object-fit: cover` covering the whole viewport.
- **Full screen:** a "Start" button calls `requestFullscreen()` and then `screen.orientation.lock('landscape')`. The button also serves as the user gesture that the camera permission prompt needs.
- **Overlay canvas:**
  - A full-screen canvas on top of the video, redrawn with `requestAnimationFrame`.
  - Map frame coordinates to screen coordinates, taking the `object-fit: cover` scale and crop into account.
- **Temporal filtering:**
  - Apply a One-Euro filter per corner.
  - Keep the corner order stable: the board has 4-fold symmetry, so rotate the new corner order to the one closest to the previous frame.
  - After a detection miss, hold the quad for 500 ms, then fade its alpha to zero.
- **Debug panel** (toggled with a three-finger tap or a small button):
  - FPS, detection time per stage, confidence, and mode (full or tracking).
  - A view selector: raw, gradient, edges, line families, rectified board, verification heat map.
  - Sliders for key thresholds.
  - A "load image/video file" input, so the pipeline can be tested on recorded clips without holding a board.

## Difficulties and mitigations
| Difficulty | Mitigation |
|---|---|
| Pieces hide lines and inner corners, and tall pieces hide more squares when the view is oblique | Merge collinear segments; allow missing teeth in the comb fit; use median-based verification; use RANSAC so hidden corners become outliers |
| Low contrast on wood and vinyl | Use the Lab chroma channels in the gradient, CLAHE, and automatic Canny thresholds |
| Wood grain and table texture create spurious edges | Blur before edge detection; require a minimum segment length; require support from a vanishing point |
| Shadows, glare on vinyl, uneven light | Verification compares each square with its neighbours, not with a fixed brightness, so local changes in illumination cancel out |
| Other straight lines (board frame, table, tiles, a laptop) | These can fool the line steps, but they fail checker verification; the "not checkered outside" bonus rejects fits that include the frame |
| Strong perspective: far rows get only a few pixels | Process at 640 px (adjustable in debug); refine corners at full-frame resolution if needed |
| Overhead view: vanishing points at infinity | Do all geometry in homogeneous coordinates |
| 90° symmetry, so no fixed corner order | Keep corner order consistent over time (orientation to a1 is out of scope) |
| OpenCV.js is about 9 MB; mid-range phones are slow | Load it in the worker behind a loading indicator (cached after the first visit); process at low resolution; use tracking mode; drop frames; free every `cv.Mat` explicitly to avoid wasm memory leaks |
| The overlay lags when the phone moves | Show the quad from the most recent result, smooth it with the One-Euro filter, and rely on tracking mode for lower latency |
| Autofocus hunting blurs frames | Request `focusMode: 'continuous'` via `applyConstraints` when available; the 500 ms hold hides short blurs |

## Project structure
```
index.html
vite.config.ts                 # base: '/chess-ref/'
src/main.ts                    # boot, start button, fullscreen, render loop
src/camera.ts                  # getUserMedia, frame grabbing, file-source fallback
src/overlay.ts                 # cover-mapping, quad drawing, fade
src/debug/panel.ts
src/worker/detector.worker.ts  # loads OpenCV, runs pipeline, message protocol
src/vision/preprocess.ts
src/vision/lines.ts            # Hough + segment merging
src/vision/vanishing.ts        # RANSAC VPs, rectification
src/vision/grid.ts             # comb fit, hypothesis generation
src/vision/verify.ts           # checker alternation score
src/vision/refine.ts           # X-corner + cornerSubPix + findHomography
src/geom/homography.ts         # pure-TS helpers (apply H, homogeneous ops)
src/geom/oneEuro.ts
tests/                         # vitest unit tests + synthetic board generator
.github/workflows/deploy.yml   # build + actions/deploy-pages
```
Dependencies: `vite`, `typescript`, `@techstark/opencv-js` (imported inside the worker), `vitest`, `@playwright/test`.

## Milestones
1. **Scaffold.** Vite + TS project, camera shown full screen, file-source mode, Pages deploy workflow.
2. **Worker and OpenCV.** Worker loads OpenCV and returns the preprocessing and edge views to the debug panel.
3. **Lines.** Line extraction and the two vanishing-point families, drawn in the debug view.
4. **First quad.** Rectification, comb fit and verification produce the first quad overlay.
5. **Stability.** Refinement, tracking mode, One-Euro smoothing, hold-then-fade.
6. **Tuning.** Adjust on recorded clips of wooden and vinyl boards; tune thresholds; check performance on a real Android phone.
7. **Auto board profile (lock and reset).**
   - **Lock:** after the first confident full detection (and after it is confirmed over a few frames), infer the board's profile:
     - the best verification channel (L, a\* or b\*) and the contrast level on it;
     - the surround type (dark ring, frame or bare edge), from the margin-ring statistics.
   - **Once locked:**
     - preprocessing computes only the gradient and edges for the channels that profile needs (e.g. L only for printed black/grey boards);
     - verification uses the locked channel, with a threshold scaled to the observed contrast;
     - the margin test expects the observed surround.
   - **Unlock:** the profile is dropped automatically after a long loss of the board (e.g. more than 5 s with no detection), so a different board can be picked up.
   - **UI:** a single "Reset board" button returns to auto mode. There is no manual picker. The debug panel shows the locked profile.
   - **Goal:** performance first. Measure full-detection time before and after on the real photos and the synthetic set. Accuracy must not regress; check it on the real photos (printed B&W is the only real profile available) and on the synthetic vinyl and wood sets.

## Verification
- **Unit tests (Vitest)** for the pure geometry: VP estimation, rectification, comb fitting, corner-order stabilisation. Test inputs are synthetic line sets.
- **Synthetic end-to-end.** A generator draws 8x8 boards in wood and green/cream colours with random homographies, lighting gradients, noise, blur and blobs that stand in for pieces. Run the pipeline on these under Node, using OpenCV.js there too. Assert that the corner error is below 2% of the board diagonal and that the recall target is met.
- **Browser smoke test (Playwright + Chromium at `/opt/pw-browsers`).**
  - Launch Chromium with `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=board.y4m`, where the `.y4m` clip is generated from the synthetic frames.
  - Check that the page starts, the worker loads, and a quad is reported within N seconds.
  - Take a screenshot of the overlay.
- **Manual check on a device.** Open the Pages URL on Android Chrome. Point the camera at a wooden board and a vinyl board, both overhead and oblique, with and without pieces, in daylight and under a lamp. Use the debug panel to watch FPS (target at least 10 detections/s on a mid-range phone) and confidence.
- `npm run build` must pass, and the deploy workflow must publish to `https://mkalam-alami.github.io/chess-ref/` (Pages source set to "GitHub Actions").
