import { createRequire } from 'node:module';
import type { CV } from '../../src/vision/preprocess';

let cached: Promise<{ cv: CV }> | null = null;

/**
 * Loads OpenCV.js under Node. The package exports a ready object, a promise, or an object that fires
 * onRuntimeInitialized; it is required (not imported) so vite never tries to unwrap the thenable export.
 * The result is boxed so returning it from an async function does not unwrap it either.
 */
export function loadCv(): Promise<{ cv: CV }> {
  cached ??= (async () => {
    const require = createRequire(import.meta.url);
    let cv = require('@techstark/opencv-js') as Record<string, unknown>;
    if (!cv.Mat && typeof cv.then === 'function') {
      try {
        const resolved = (await (cv as unknown as Promise<Record<string, unknown>>)) as Record<string, unknown>;
        if (resolved?.Mat) cv = resolved;
      } catch {
        /* fall through to the runtime callback */
      }
    }
    if (!cv.Mat) {
      await new Promise<void>((resolve) => {
        cv.onRuntimeInitialized = () => resolve();
      });
    }
    return { cv: cv as unknown as CV };
  })();
  return cached;
}
