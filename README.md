# chess-ref

Proof of concept: a static web page that opens the rear phone camera fullscreen and detects the 8x8 playing area of a chessboard, drawing the quad over the video. Everything runs in the browser (Vite + TypeScript; detection in a Web Worker with OpenCV.js from `@techstark/opencv-js`). Target: Android Chrome.

Status: camera, overlay, debug panel and the worker/preprocessing stub are in place. Line, vanishing-point, grid, verification and refinement stages (`src/vision/*`) are still to do.

## Run

    npm install
    npm run dev        # camera needs a secure context: localhost or https
    npm test           # vitest unit tests
    npm run typecheck
    npm run build
    npm run e2e        # Playwright smoke test (Chromium, fake camera); builds first

Use the "Load file" input to feed an image or video instead of the camera. Toggle the debug panel with the corner button or a three-finger tap.

## Deploy

Pushes to `main` run `.github/workflows/deploy.yml`, which builds and publishes `dist` to GitHub Pages at https://mkalam-alami.github.io/chess-ref/. In the repo settings, set Pages source to "GitHub Actions".
