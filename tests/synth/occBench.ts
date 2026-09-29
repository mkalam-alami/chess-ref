// Occupancy part of `npm run bench`: synthetic starting positions, then a short game from the same camera.
import { applyH, homographyFrom4, invert3, mul3, type Mat3, type Point } from '../../src/geom/homography';
import type { Detector } from '../../src/vision/detector';
import { OccupancyTracker } from '../../src/vision/occupancy';
import type { CV } from '../../src/vision/preprocess';
import { makeBoardSample } from './generate';

/** Ground truth (generator cell frame) re-indexed into hb's cell frame; null if hb is not a relabelling of it. */
export function alignGt(gt: Uint8Array, gtCorners: readonly Point[], hb: Mat3): Uint8Array | null {
  const hgt = homographyFrom4([[0, 0], [8, 0], [8, 8], [0, 8]], gtCorners);
  const inv = hgt && invert3(hgt);
  if (!inv) return null;
  const m = mul3(inv, hb);
  const out = new Uint8Array(64);
  for (let c = 0; c < 64; c++) {
    const [u, v] = applyH(m, [(c & 7) + 0.5, (c >> 3) + 0.5]);
    const i = Math.floor(u);
    const j = Math.floor(v);
    if (i < 0 || j < 0 || i > 7 || j > 7 || Math.hypot(u - i - 0.5, v - j - 0.5) > 0.25) return null;
    out[c] = gt[j * 8 + i]!;
  }
  return out;
}

const sq = (s: string) => (s.charCodeAt(1) - 49) * 8 + (s.charCodeAt(0) - 97);
/** 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 (a capture) in generator cells (white on ranks j = 0-1). */
export const GAME: ReadonlyArray<readonly [number, number]> = [
  ['e2', 'e4'], ['e7', 'e5'], ['g1', 'f3'], ['b8', 'c6'], ['f1', 'b5'], ['a7', 'a6'], ['b5', 'c6'],
].map(([a, b]) => [sq(a!), sq(b!)] as const);

const FRAMES_PER_POS = 5;

export function occupancyBench(cv: CV, det: Detector, n: number, seed: number): string {
  const lines: string[] = [];
  for (const elev of ['overhead', 'oblique'] as const)
    for (const palette of ['wood', 'vinyl', 'printed'] as const) {
      let tried = 0, used = 0, boots = 0, startCells = 0, startOk = 0, startDrop = 0;
      let seqFrames = 0, seqDrop = 0, seqCells = 0, seqOk = 0, finalOk = 0, finalN = 0;
      for (let k = 0; k < n; k++) {
        const sd = seed + k;
        tried++;
        const s0 = makeBoardSample(cv, sd, { elev, palette, pieces: 'start' });
        const f0 = { data: s0.rgba, width: s0.width, height: s0.height };
        const r = det.detect(f0 as unknown as ImageData, {});
        const gt0 = r.hb && s0.corners ? alignGt(s0.occupancy!, s0.corners, r.hb) : null;
        if (!r.hb || !gt0) continue;
        used++;
        const hb = r.hb;
        const tr = new OccupancyTracker();
        let t = 0;
        let out = tr.update(f0, hb, {}, t);
        for (let i = 1; i < FRAMES_PER_POS; i++) out = tr.update(f0, hb, {}, (t += 100));
        if (out.stats.state === 'calibrated') boots++;
        if (!out.grid) startDrop++;
        else for (let c = 0; c < 64; c++) (startCells++, out.grid[c] === gt0[c] && startOk++);
        if (out.stats.state !== 'calibrated') continue;
        // Same camera (hb reused), pieces moved: the committed grid must follow after the hold frames.
        let last: Uint8Array | null = null;
        let lastGt: Uint8Array | null = null;
        for (let m = 1; m <= GAME.length; m++) {
          const s = makeBoardSample(cv, sd, { elev, palette, pieces: 'start' }, GAME.slice(0, m));
          const f = { data: s.rgba, width: s.width, height: s.height };
          const gt = alignGt(s.occupancy!, s.corners!, hb)!;
          for (let i = 0; i < FRAMES_PER_POS; i++) {
            out = tr.update(f, hb, {}, (t += 100));
            seqFrames++;
            if (!out.grid) seqDrop++;
          }
          // Score the committed grid at the end of each position (after FRAMES_PER_POS frames).
          if (out.grid) {
            last = out.grid;
            for (let c = 0; c < 64; c++) (seqCells++, out.grid[c] === gt[c] && seqOk++);
          }
          lastGt = gt;
        }
        if (last && lastGt) {
          finalN++;
          let ok = 0;
          for (let c = 0; c < 64; c++) if (last[c] === lastGt[c]) ok++;
          if (ok === 64) finalOk++;
        }
      }
      const pct = (a: number, b: number) => (b ? ((a / b) * 100).toFixed(1) + '%' : '-');
      lines.push(
        `${`${elev}-${palette}`.padEnd(18)} det ${used}/${tried} calib ${boots}/${used} start grid ${pct(startOk, startCells)} drop ${startDrop}` +
          ` | game: cells ${pct(seqOk, seqCells)} frame drops ${pct(seqDrop, seqFrames)} final exact ${finalOk}/${finalN}`,
      );
    }
  return lines.join('\n');
}
