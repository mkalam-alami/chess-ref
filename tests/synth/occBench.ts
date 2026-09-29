// Occupancy part of `npm run bench`: synthetic starting positions, then a short game from the same camera.
import { applyH, homographyFrom4, invert3, mul3, type Mat3, type Point } from '../../src/geom/homography';
import type { Detector } from '../../src/vision/detector';
import { BOARD_DIHEDRAL } from '../../src/geom/cornerOrder';
import { boardHandedness, chooseOrientation, OccupancyTracker, ORIENT_MAPS, START_SQUARES, toCellOrder } from '../../src/vision/occupancy';
import type { CV } from '../../src/vision/preprocess';
import type { Params } from '../../src/worker/protocol';
import { makeBoardSample } from './generate';

/** Generator cell of each of hb's cells (hb cell c covers generator cell out[c]); null if hb is not a relabelling of
 *  the generator's board. */
export function cellCorrespondence(gtCorners: readonly Point[], hb: Mat3): Uint8Array | null {
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
    out[c] = j * 8 + i;
  }
  return out;
}

/** Ground truth (generator cell frame) re-indexed into hb's cell frame; null if hb is not a relabelling of it. */
export function alignGt(gt: Uint8Array, gtCorners: readonly Point[], hb: Mat3): Uint8Array | null {
  const corr = cellCorrespondence(gtCorners, hb);
  return corr && corr.map((g) => gt[g]!);
}

/**
 * Expected orientation (square -> hb cell) of a 'start'-mode sample: its white pieces start on generator rows 0-1
 * with generator cell (0, 0) as a1 (the generator's frame is right-handed seen from its camera, above the board),
 * so square sq is generator cell sq. Null for a board whose (0, 0) is light (`meta.flip`), where no orientation
 * puts a1 on a dark square: vision must refuse to orient.
 */
export function expectedOrientation(gtCorners: readonly Point[], hb: Mat3, flip: boolean): Uint8Array | null {
  const corr = cellCorrespondence(gtCorners, hb);
  if (!corr || flip) return null;
  const out = new Uint8Array(64);
  for (let c = 0; c < 64; c++) out[corr[c]!] = c;
  return out;
}

const sq = (s: string) => (s.charCodeAt(1) - 49) * 8 + (s.charCodeAt(0) - 97);
/** 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 (a capture) in generator cells (white on ranks j = 0-1). */
export const GAME: ReadonlyArray<readonly [number, number]> = [
  ['e2', 'e4'], ['e7', 'e5'], ['g1', 'f3'], ['b8', 'c6'], ['f1', 'b5'], ['a7', 'a6'], ['b5', 'c6'],
].map(([a, b]) => [sq(a!), sq(b!)] as const);

/** Tracker params that reduce the whole-board-in-view check (boardFramed) to the board corners at z = 0 with no margin:
 *  the generator only keeps the board itself inside the image, so tall pieces on the far rank may reach past the
 *  frame edge. For tests about something else on such boards. */
export const FRAME_CORNERS_ONLY: Params = { occFrameTopZ: 0, occFrameMargin: 0 };

const FRAMES_PER_POS = 5;

const same = (a: ArrayLike<number>, b: ArrayLike<number>) => a.length === b.length && Array.from(a).every((v, i) => v === b[i]);

export function occupancyBench(cv: CV, det: Detector, n: number, seed: number): string {
  const lines: string[] = [];
  for (const elev of ['overhead', 'oblique'] as const)
    for (const palette of ['wood', 'vinyl', 'printed'] as const) {
      // Boards not wholly in view by boardFramed's default rule (never calibrated on, see FRAME_CORNERS_ONLY).
      let unframed = 0;
      let tried = 0, used = 0, boots = 0, startCells = 0, startOk = 0, startDrop = 0;
      let seqFrames = 0, seqDrop = 0, seqCells = 0, seqOk = 0, finalOk = 0, finalN = 0;
      // Orientation after the start frames: right / refused on a light-a1 board (expected) / missed / WRONG.
      let orOk = 0, orRefused = 0, orFlip = 0, orMissed = 0, orWrong = 0;
      // Re-orientation after a 2 s loss and a quarter-turn relabelling, against the final position as the hint.
      let reN = 0, reOk = 0, reWrong = 0, reFrames = 0;
      // Smallest margin (nats) of the adopted placement over the best different one, over both.
      let minGap = Infinity;
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
        if (!out.observation.framed) unframed++;
        for (let i = 1; i < FRAMES_PER_POS; i++) out = tr.update(f0, hb, {}, (t += 100));
        if (out.stats.state === 'setup' || out.stats.state === 'calibrated') boots++;
        if (!out.grid) startDrop++;
        else for (let c = 0; c < 64; c++) (startCells++, out.grid[c] === gt0[c] && startOk++);
        if (out.stats.state !== 'setup' && out.stats.state !== 'calibrated') continue;
        const flip = s0.meta.flip === 1;
        const want = expectedOrientation(s0.corners!, hb, flip);
        if (flip) orFlip++;
        if (!out.orientation) want ? orMissed++ : orRefused++;
        else if (want && same(out.orientation, want)) {
          orOk++;
          const ch = chooseOrientation(out.logLik, START_SQUARES, boardHandedness(hb), -1, 0, 64);
          if ('gap' in ch) minGap = Math.min(minGap, ch.gap);
        }
        else orWrong++;
        // Same camera (hb reused), pieces moved: the committed grid must follow after the hold frames.
        let last: Uint8Array | null = null;
        let lastGt: Uint8Array | null = null;
        let lastSq: Uint8Array | null = null;
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
          lastSq = s.occupancy!;
        }
        if (want && out.orientation && lastSq) {
          // Board lost for 2 s, re-detected in another labelling (a quarter turn, as TrackingSession may return once
          // its orientation reference expired); the game's position is the hint.
          const hb2 = mul3(hb, BOARD_DIHEDRAL[1 + (k % 3)]!);
          const want2 = expectedOrientation(s0.corners!, hb2, false)!;
          const f = makeBoardSample(cv, sd, { elev, palette, pieces: 'start' }, GAME);
          const fr = { data: f.rgba, width: f.width, height: f.height };
          tr.noBoard((t += 100));
          t += 2000;
          tr.setPosition(lastSq);
          reN++;
          for (let i = 1; i <= 8; i++) {
            const o = tr.update(fr, hb2, {}, (t += 100));
            if (o.orientation) {
              const ch = chooseOrientation(o.logLik, lastSq, boardHandedness(hb2), -1, 0, 64);
              if ('gap' in ch) minGap = Math.min(minGap, ch.gap);
              if (same(o.orientation, want2)) (reOk++, (reFrames += i));
              else reWrong++;
              break;
            }
            if (i === 8 && process.env.BENCH_VERBOSE) {
              const g = ORIENT_MAPS.findIndex((m) => same(m, want2));
              const cells = toCellOrder(lastSq, want2);
              const miss = Array.from(cells).map((v, c) => Math.exp(o.logLik[c * 3 + v]!)).map((p, c) => [c, cells[c], o.raw[c], p.toFixed(3)]).filter((x) => +x[3]! < 0.1);
              const ch = chooseOrientation(o.logLik, lastSq, boardHandedness(hb2), -1, 0, 64);
              console.log(`re-orient failed ${elev}-${palette} seed ${sd}: ${tr.orientReason} dropped ${o.stats.dropped} frozen ${o.stats.frozen} want g ${g} got ${JSON.stringify(ch)} misses ${JSON.stringify(miss)}`);
            }
          }
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
        `${`${elev}-${palette}`.padEnd(18)} det ${used}/${tried} unframed ${unframed} calib ${boots}/${used} start grid ${pct(startOk, startCells)} drop ${startDrop}` +
          ` | game: cells ${pct(seqOk, seqCells)} frame drops ${pct(seqDrop, seqFrames)} final exact ${finalOk}/${finalN}` +
          ` | orient: ok ${orOk}/${boots - orFlip} refused(light a1) ${orRefused}/${orFlip} missed ${orMissed} WRONG ${orWrong}` +
          ` | re-orient: ok ${reOk}/${reN}${reOk ? ` (${(reFrames / reOk).toFixed(1)} fr)` : ''} WRONG ${reWrong} | min gap ${minGap.toFixed(0)}`,
      );
    }
  return lines.join('\n');
}
