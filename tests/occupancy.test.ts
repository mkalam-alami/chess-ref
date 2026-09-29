import { beforeAll, describe, expect, it } from 'vitest';
import { mul3, type Mat3 } from '../src/geom/homography';
import { Detector } from '../src/vision/detector';
import type { CV } from '../src/vision/preprocess';
import { boardFramed, cameraFromHomography, cellFootprints, type FrameRect, OCCUPANCY_PARAMS, OccupancyFilter, OccupancyTracker, startGrid } from '../src/vision/occupancy';
import { OCC_BLACK, OCC_EMPTY, OCC_WHITE, type Params } from '../src/worker/protocol';
import { realCases } from './synth/bench';
import { alignGt, FRAME_CORNERS_ONLY, GAME } from './synth/occBench';
import { loadCv } from './synth/cvNode';
import fs from 'node:fs';
import path from 'node:path';
import { H, makeBoardSample, makeCamera, W } from './synth/generate';

/** The rendered boards keep only the board (z = 0) inside the image: see FRAME_CORNERS_ONLY. */
const P = FRAME_CORNERS_ONLY;

/** Generator camera with the principal point at the image centre; hb in cell units (board 0..8). */
function cam(azDeg: number, elDeg: number, f: number, D: number) {
  const c = makeCamera((azDeg * Math.PI) / 180, (elDeg * Math.PI) / 180, 0.05, f, D, W / 2, H / 2);
  const hb = mul3(c.hBoard, [1, 0, -4, 0, 1, -4, 0, 0, 1]);
  return { c, hb };
}

describe('cameraFromHomography', () => {
  it('recovers the focal length and projects points above the board on oblique views', () => {
    for (const [az, el, f, D] of [[20, 40, 520, 14], [130, 55, 650, 13], [250, 32, 450, 16]] as const) {
      const { c, hb } = cam(az, el, f, D);
      const bc = cameraFromHomography(hb, W, H)!;
      expect(bc.nominal).toBe(false);
      expect(Math.abs(bc.f - f) / f).toBeLessThan(0.02);
      for (const [x, y, z] of [[1, 1, 1], [4, 6, 1.5], [7.5, 0.5, 0.4]] as const) {
        const p = bc.project(x, y, z)!;
        const q = c.project(x - 4, y - 4, z);
        expect(Math.hypot(p[0] - q[0], p[1] - q[1])).toBeLessThan(1);
      }
      expect(bc.centre[2]).toBeGreaterThan(0);
    }
  });

  it('keeps +z towards the camera for mirrored relabellings of hb and falls back near overhead', () => {
    const { c, hb } = cam(60, 45, 560, 14);
    // Mirror the board labels (x <-> y): orientation reverses, the camera must still be on the +z side.
    const mirrored = mul3(hb, [0, 1, 0, 1, 0, 0, 0, 0, 1]);
    const bc = cameraFromHomography(mirrored, W, H)!;
    expect(bc.centre[2]).toBeGreaterThan(0);
    const p = bc.project(2, 3, 1)!; // mirrored (2, 3) = original (3, 2)
    const q = c.project(3 - 4, 2 - 4, 1);
    expect(Math.hypot(p[0] - q[0], p[1] - q[1])).toBeLessThan(1);
    const top = cameraFromHomography(cam(10, 89.5, 560, 14).hb, W, H)!;
    expect(top.nominal).toBe(true);
  });
});

/** Camera turned down by `deg` about its own x axis (K R K^-1): the board moves up the image, the principal point
 *  stays at the centre. */
function pitch(deg: number, f: number): Mat3 {
  const a = (deg * Math.PI) / 180;
  const K: Mat3 = [f, 0, W / 2, 0, f, H / 2, 0, 0, 1];
  const Ki: Mat3 = [1 / f, 0, -W / 2 / f, 0, 1 / f, -H / 2 / f, 0, 0, 1];
  return mul3(K, mul3([1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)], Ki));
}

describe('boardFramed', () => {
  const full = [0, 0, W, H] as const;
  const framedFor = (hb: Mat3, rect: FrameRect = full, params: Params = {}) => boardFramed(cameraFromHomography(hb, W, H)!, rect, params);

  it('accepts a board wholly inside the frame, with the margin as a fraction of the short side', () => {
    const r = framedFor(cam(20, 40, 560, 14).hb);
    expect(r.framed).toBe(true);
    expect(r.worst).toBeGreaterThan(0.02 * H);
    // A margin wider than the room left: refused.
    expect(framedFor(cam(20, 40, 560, 14).hb, full, { occFrameMargin: (r.worst + 1) / H }).framed).toBe(false);
  });

  it('refuses a board with a corner outside the frame', () => {
    const r = framedFor(cam(0, 30, 560, 10).hb);
    expect(r.framed).toBe(false);
    expect(r.worst).toBeLessThan(0);
    expect(framedFor(cam(0, 30, 560, 10).hb, full, { occFrameTopZ: 0, occFrameMargin: 0 }).framed).toBe(false);
  });

  it('refuses an oblique board whose far-rank piece tops leave through the top edge', () => {
    // Camera turned down: the board sits at the top of the image, its corners inside with room to spare.
    const hb = mul3(pitch(12, 560), cam(0, 45, 560, 15).hb);
    const corners = framedFor(hb, full, { occFrameTopZ: 0 });
    expect(corners.framed).toBe(true);
    expect(corners.worst).toBeGreaterThan(0.02 * H);
    const r = framedFor(hb);
    expect(r.framed).toBe(false);
    expect(r.worst).toBeLessThan(0);
  });

  it('checks against the visible rect, not the whole frame', () => {
    const { hb } = cam(20, 40, 560, 14);
    expect(framedFor(hb, full).framed).toBe(true);
    // A portrait screen showing the middle of a landscape frame (a cropped view): the board's sides are cut off.
    expect(framedFor(hb, [W / 2 - 0.3 * H, 0, W / 2 + 0.3 * H, H]).framed).toBe(false);
    expect(framedFor(hb, [0, 100, W, H]).framed).toBe(false);
  });
});

describe('cellFootprints', () => {
  it('marks samples behind a nearer square as occludable, and none in a top view', () => {
    const { hb } = cam(-90, 35, 560, 14); // camera on the -y side: row 0 is nearest
    const bc = cameraFromHomography(hb, W, H)!;
    expect(bc.centre[1]).toBeLessThan(0);
    const fp = cellFootprints(bc, W, H);
    const occluders = (cell: number) => {
      const s = new Set<number>();
      for (let k = cell * fp.n; k < (cell + 1) * fp.n; k++) for (let o = fp.occStart[k]!; o < fp.occStart[k + 1]!; o++) s.add(fp.occCells[o]!);
      return s;
    };
    // Cell (3, 2) can be hidden by (3, 1) in front of it, never by (3, 3) behind it.
    expect(occluders(2 * 8 + 3).has(1 * 8 + 3)).toBe(true);
    expect(occluders(2 * 8 + 3).has(3 * 8 + 3)).toBe(false);
    const topFp = cellFootprints(cameraFromHomography(cam(10, 89.5, 560, 14).hb, W, H)!, W, H);
    // Overhead: only the stem samples can touch direct neighbours; the base centre is never occluded.
    for (let c = 0; c < 64; c++) expect(topFp.occStart[c * topFp.n + 1]! - topFp.occStart[c * topFp.n]!).toBe(0);
  });
});

describe('OccupancyFilter', () => {
  const params = { occHoldFrames: 3, occFreezeCells: 6, occSettleMs: 400, occCellMin: 0.2 };
  const conf = new Float32Array(64).fill(1);
  it('commits a change only after consistent frames and ignores low-confidence cells', () => {
    const f = new OccupancyFilter();
    f.reset(startGrid(0, 0));
    const raw = startGrid(0, 0);
    raw[12] = OCC_EMPTY;
    raw[28] = OCC_WHITE;
    f.update(raw, conf, params, 0);
    f.update(raw, conf, params, 100);
    expect(f.committed![28]).toBe(OCC_EMPTY);
    f.update(raw, conf, params, 200);
    expect(f.committed![28]).toBe(OCC_WHITE);
    expect(f.committed![12]).toBe(OCC_EMPTY);
    const lowConf = new Float32Array(64).fill(1);
    lowConf[40] = 0.05;
    const r2 = new Uint8Array(f.committed!);
    r2[40] = OCC_BLACK;
    for (let t = 300; t < 1000; t += 100) f.update(r2, lowConf, params, t);
    expect(f.committed![40]).toBe(OCC_EMPTY);
  });

  it('freezes on a mass change; with hand remnants (outliers) the settle lasts the full occSettleMs cap', () => {
    const f = new OccupancyFilter();
    f.reset(startGrid(0, 0));
    const hand = startGrid(0, 0);
    for (let c = 16; c < 26; c++) hand[c] = OCC_BLACK;
    f.update(startGrid(0, 0), conf, params, 0);
    expect(f.update(hand, conf, params, 50)).toBe(true); // 10 cells flipped at once
    // No flips any more, but 2 cells match no class model (a hand remnant): never calm.
    expect(f.update(hand, conf, params, 100, true, 2)).toBe(true);
    expect(f.update(hand, conf, params, 150, true, 2)).toBe(true);
    expect(f.update(hand, conf, params, 300, true, 2)).toBe(true);
    expect(f.committed![20]).toBe(OCC_EMPTY); // still frozen (cap: 400 ms from t = 50)
    expect(f.update(hand, conf, params, 460, true, 2)).toBe(false);
    expect(f.committed![20]).toBe(OCC_BLACK);
    f.noBoard(500);
    expect(f.frozen(700, params)).toBe(true);
    expect(f.frozen(900, params)).toBe(false);
  });

  it('adaptive settle: occSettleFrames calm frames end it early, not before occSettleMinMs', () => {
    const f = new OccupancyFilter();
    const start = startGrid(0, 0);
    f.reset(start);
    const hand = startGrid(0, 0);
    for (let c = 16; c < 26; c++) hand[c] = OCC_BLACK;
    const moved = new Uint8Array(start);
    moved[12] = OCC_EMPTY;
    moved[28] = OCC_WHITE;
    f.update(start, conf, params, 0);
    expect(f.update(hand, conf, params, 66)).toBe(true); // hand frames: mass flips, unstable
    expect(f.update(start, conf, params, 133)).toBe(true); // hand gone: 10 cells flip back, still unstable
    expect(f.update(moved, conf, params, 200)).toBe(true); // calm 1 (2 flips: the move)
    expect(f.update(moved, conf, params, 266)).toBe(false); // calm 2, 133 ms after the last unstable frame
    // The floor: two calm frames in quick succession do not end the settle before occSettleMinMs.
    f.update(hand, conf, params, 1000);
    expect(f.update(start, conf, params, 1010)).toBe(true); // unstable (flips back)
    expect(f.update(start, conf, params, 1030)).toBe(true); // calm 1
    expect(f.update(start, conf, params, 1060)).toBe(true); // calm 2, but only 50 ms
    expect(f.frozen(1109, params)).toBe(true);
    expect(f.frozen(1110, params)).toBe(false);
    // Legacy behaviour: occSettleFrames above what the frame rate delivers in occSettleMs = the fixed settle.
    const fixed = { ...params, occSettleFrames: 10 };
    f.update(hand, conf, fixed, 2000);
    for (let t = 2066; t < 2400; t += 66) expect(f.update(hand, conf, fixed, t)).toBe(true);
    expect(f.update(hand, conf, fixed, 2400)).toBe(false);
  });

  it('adaptive settle: continued flipping stays frozen; the cap counts from the last unstable frame', () => {
    const f = new OccupancyFilter();
    const start = startGrid(0, 0);
    f.reset(start);
    const hand = startGrid(0, 0);
    for (let c = 16; c < 26; c++) hand[c] = OCC_BLACK;
    f.update(start, conf, params, 0);
    // A hand moving over the board for 1.5 s: every frame flips 10 cells.
    let t = 0;
    for (let i = 1; i <= 22; i++) expect(f.update(i & 1 ? hand : start, conf, params, (t = i * 66))).toBe(true);
    // Then small flips (1-5 cells per frame: neither unstable nor calm) with outliers: frozen up to the cap.
    const wobble = (k: number) => {
      const g = new Uint8Array(start);
      for (let c = 40; c < 40 + (k % 5) + 1; c++) g[c] = OCC_WHITE;
      return g;
    };
    const lastUnstable = t;
    for (let k = 0; k < 10; k++) {
      t += 66;
      expect(f.update(wobble(k), conf, params, t, true, 1)).toBe(t - lastUnstable < 400);
    }
  });

  it('adaptive settle: the first frame after a board loss is a reference, not a calm frame', () => {
    const f = new OccupancyFilter();
    const start = startGrid(0, 0);
    f.reset(start);
    f.update(start, conf, params, 0);
    f.noBoard(100);
    expect(f.update(start, conf, params, 166)).toBe(true); // reference
    expect(f.update(start, conf, params, 233)).toBe(true); // calm 1
    expect(f.update(start, conf, params, 300)).toBe(false); // calm 2
    // markUnstable (a dropped frame) keeps the reference: two frames after it suffice.
    f.markUnstable(400);
    expect(f.update(start, conf, params, 466)).toBe(true); // calm 1
    expect(f.update(start, conf, params, 533)).toBe(false); // calm 2
    // Steady glare: an outlier cell seen on settled frames is the baseline and does not block the early settle.
    f.update(start, conf, params, 600, true, 1);
    f.markUnstable(700);
    expect(f.update(start, conf, params, 766, true, 1)).toBe(true);
    expect(f.update(start, conf, params, 833, true, 1)).toBe(false);
    // One more outlier than the baseline (occSettleOutliers 0) does.
    f.markUnstable(900);
    expect(f.update(start, conf, params, 966, true, 2)).toBe(true);
    expect(f.update(start, conf, params, 1033, true, 2)).toBe(true);
    expect(f.update(start, conf, params, 1300, true, 2)).toBe(false); // cap
  });
});

interface Eval {
  n: number;
  boots: number;
  rawAcc: number;
  gridAcc: number;
  dropped: number;
}

describe('occupancy on synthetic starting positions', () => {
  let cv: CV;
  let det: Detector;
  beforeAll(async () => {
    ({ cv } = await loadCv());
    det = new Detector(cv);
  }, 60_000);

  it('calibrates on the starting position with the right orientation and classifies the frame', () => {
    const ev: Eval = { n: 0, boots: 0, rawAcc: 0, gridAcc: 0, dropped: 0 };
    const lines: string[] = [];
    let seed = 7000;
    for (const elev of ['overhead', 'oblique'] as const)
      for (const palette of ['wood', 'vinyl', 'printed'] as const)
        for (let k = 0; k < 4; k++) {
          const s = makeBoardSample(cv, seed++, { elev, palette, pieces: 'start' });
          const frame = { data: s.rgba, width: s.width, height: s.height };
          const r = det.detect(frame as unknown as ImageData, {});
          if (!r.hb || !s.corners) continue;
          const gt = alignGt(s.occupancy!, s.corners, r.hb);
          if (!gt) continue;
          const tr = new OccupancyTracker();
          let out = tr.update(frame, r.hb, P, 0);
          for (let t = 1; t < 5; t++) out = tr.update(frame, r.hb, P, t * 100);
          ev.n++;
          if (out.stats.state === 'calibrated') ev.boots++;
          let rawOk = 0;
          for (let c = 0; c < 64; c++) if (out.raw[c] === gt[c]) rawOk++;
          ev.rawAcc += rawOk / 64;
          if (out.grid) {
            let ok = 0;
            for (let c = 0; c < 64; c++) if (out.grid[c] === gt[c]) ok++;
            ev.gridAcc += ok / 64;
          } else ev.dropped++;
          lines.push(`${s.label}-${s.seed}: ${out.stats.state} raw ${rawOk}/64 low ${out.stats.lowCells} ${out.grid ? '' : 'dropped'}`);
        }
    console.log(lines.join('\n'));
    console.log(`synthetic start: n=${ev.n} calibrated ${ev.boots} raw acc ${((ev.rawAcc / ev.n) * 100).toFixed(1)}% grid acc (non-dropped) ${((ev.gridAcc / Math.max(1, ev.n - ev.dropped)) * 100).toFixed(1)}% dropped ${ev.dropped}`);
    expect(ev.n).toBeGreaterThanOrEqual(10);
    expect(ev.boots / ev.n).toBeGreaterThanOrEqual(0.75);
    expect(ev.rawAcc / ev.n).toBeGreaterThanOrEqual(0.85);
  }, 120_000);
});

describe('occupancy class log-likelihoods', () => {
  let cv: CV;
  let det: Detector;
  beforeAll(async () => {
    ({ cv } = await loadCv());
    det = new Detector(cv);
  }, 60_000);

  it('ranks the true class first in most cells and is roughly calibrated', () => {
    let n = 0;
    let top = 0;
    let pSum = 0;
    // Low-visibility cells (vis < 0.3): vis is not exposed, so a twin tracker with a linear visibility weight (same
    // state: logLik does not feed back) recovers it from the ratio of the logit spreads, vis^(1 - gamma).
    const gamma = OCCUPANCY_PARAMS.find((p) => p.name === 'occLikVisGamma')!.default;
    const spread = (l: Float32Array) => Math.max(l[0]!, l[1]!, l[2]!) - Math.min(l[0]!, l[1]!, l[2]!);
    let nLow = 0;
    let topLow = 0;
    let pLow = 0;
    let seed = 7100;
    for (const elev of ['overhead', 'oblique'] as const)
      for (const palette of ['wood', 'vinyl', 'printed'] as const)
        for (let k = 0; k < 2; k++) {
          const sd = seed++;
          const s0 = makeBoardSample(cv, sd, { elev, palette, pieces: 'start' });
          const f0 = { data: s0.rgba, width: s0.width, height: s0.height };
          const r = det.detect(f0 as unknown as ImageData, {});
          if (!r.hb || !s0.corners || !alignGt(s0.occupancy!, s0.corners, r.hb)) continue;
          const tr = new OccupancyTracker();
          const twin = new OccupancyTracker();
          const lin = { ...P, occLikVisGamma: 1 };
          let out = tr.update(f0, r.hb, P, 0);
          let outLin = twin.update(f0, r.hb, lin, 0);
          expect(out.logLik.length).toBe(192);
          let t = 0;
          for (let i = 1; i < 5; i++) {
            out = tr.update(f0, r.hb, P, (t += 100));
            outLin = twin.update(f0, r.hb, lin, t);
          }
          if (out.stats.state !== 'calibrated') continue;
          for (let m = 0; m <= 3; m++) {
            const s = makeBoardSample(cv, sd, { elev, palette, pieces: 'start' }, GAME.slice(0, m));
            const gt = alignGt(s.occupancy!, s.corners!, r.hb)!;
            const fr = { data: s.rgba, width: s.width, height: s.height };
            for (let i = 0; i < 4; i++) {
              out = tr.update(fr, r.hb, P, (t += 100));
              outLin = twin.update(fr, r.hb, lin, t);
            }
            // committedProb: probability of the committed class, present even when this frame's grid is dropped.
            expect(out.committedProb).not.toBeNull();
            if (out.grid)
              for (let c = 0; c < 64; c++)
                expect(out.committedProb![c]).toBeCloseTo(Math.exp(out.logLik[c * 3 + out.grid[c]!]!), 5);
            for (let c = 0; c < 64; c++) {
              const l = out.logLik.subarray(c * 3, c * 3 + 3);
              expect(Math.abs(Math.exp(l[0]!) + Math.exp(l[1]!) + Math.exp(l[2]!) - 1)).toBeLessThan(1e-4);
              let arg = 0;
              for (let q = 1; q < 3; q++) if (l[q]! > l[arg]!) arg = q;
              n++;
              pSum += Math.exp(l[arg]!);
              if (arg === gt[c]) top++;
              const sg = spread(l);
              const vis = sg > 1e-6 ? (spread(outLin.logLik.subarray(c * 3, c * 3 + 3)) / sg) ** (1 / (1 - gamma)) : 0;
              if (vis < 0.3) {
                nLow++;
                pLow += Math.exp(l[arg]!);
                if (arg === gt[c]) topLow++;
              }
            }
          }
        }
    const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
    console.log(
      `logLik: n=${n} top-1 ${pct(top / n)} mean predicted ${pct(pSum / n)}; vis<0.3: n=${nLow} top-1 ${pct(topLow / nLow)} mean predicted ${pct(pLow / nLow)}`,
    );
    expect(n).toBeGreaterThanOrEqual(64 * 4 * 6);
    expect(top / n).toBeGreaterThanOrEqual(0.9);
    expect(Math.abs(pSum / n - top / n)).toBeLessThan(0.05);
    // Partly hidden cells are mostly right: no longer grossly under-confident (was ~48% predicted vs ~83% right).
    expect(nLow).toBeGreaterThanOrEqual(30);
    expect(topLow / nLow - pLow / nLow).toBeLessThan(0.2);
  }, 180_000);
});

describe('occupancy on real starting-position photos', () => {
  let cv: CV;
  let det: Detector;
  beforeAll(async () => {
    ({ cv } = await loadCv());
    det = new Detector(cv);
  }, 60_000);

  /** Starting grid in the annotated-corner frame: edge k (corners[k] -> corners[k + 1]) is y = 0, x = 8, y = 8, x = 0. */
  const gtStart = (whiteEdge: number) => [startGrid(0, 0), startGrid(1, 1), startGrid(0, 1), startGrid(1, 0)][whiteEdge]!;

  it('calibrates on the starting position and classifies the cells, within budget at 640 px', () => {
    const meta = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/real/corners.json'), 'utf8')) as Record<string, { whiteEdge?: number }>;
    const lines: string[] = [];
    let n = 0;
    let boots = 0;
    let rawSum = 0;
    let gridSum = 0;
    let grids = 0;
    const times: number[] = [];
    for (const c of realCases(cv, 640)) {
      if (!c.gt) continue;
      const edge = meta[`${c.name.split('@')[0]}.jpg`]?.whiteEdge;
      if (edge === undefined) continue;
      const frame = { data: c.rgba, width: c.width, height: c.height };
      const r = det.detect(frame as unknown as ImageData, {});
      const gt = r.hb && r.corners ? alignGt(gtStart(edge), c.gt, r.hb) : null;
      if (!r.hb || !gt) {
        lines.push(`${c.name}: no aligned detection`);
        continue;
      }
      n++;
      const tr = new OccupancyTracker();
      let out = tr.update(frame, r.hb, {}, 0);
      for (let t = 1; t < 5; t++) out = tr.update(frame, r.hb, {}, t * 100);
      // Steady-state cost per frame once calibrated (the worker's per-frame budget is < 3 ms).
      for (let t = 5; t < 25; t++) {
        const t0 = performance.now();
        tr.update(frame, r.hb, {}, t * 100);
        times.push(performance.now() - t0);
      }
      if (out.stats.state === 'calibrated') boots++;
      let rawOk = 0;
      for (let k = 0; k < 64; k++) if (out.raw[k] === gt[k]) rawOk++;
      rawSum += rawOk / 64;
      let gridOk = -1;
      if (out.grid) {
        gridOk = 0;
        for (let k = 0; k < 64; k++) if (out.grid[k] === gt[k]) gridOk++;
        gridSum += gridOk / 64;
        grids++;
      }
      lines.push(`${c.name}: ${out.stats.state} raw ${rawOk}/64 grid ${out.grid ? `${gridOk}/64` : 'dropped'} low ${out.stats.lowCells}`);
    }
    times.sort((a, b) => a - b);
    const med = times[times.length >> 1] ?? NaN;
    console.log(lines.join('\n'));
    console.log(`real start: n=${n} calibrated ${boots} raw acc ${((rawSum / Math.max(1, n)) * 100).toFixed(1)}% grid acc ${((gridSum / Math.max(1, grids)) * 100).toFixed(1)}% (${grids} not dropped); update median ${med.toFixed(2)} ms p90 ${times[Math.floor(times.length * 0.9)]?.toFixed(2)} ms max ${times[times.length - 1]?.toFixed(2)} ms`);
    expect(n).toBeGreaterThanOrEqual(4);
    expect(boots).toBeGreaterThanOrEqual(4);
    expect(rawSum / n).toBeGreaterThanOrEqual(0.9);
    expect(med).toBeLessThan(10); // loose: CI machines vary; the target is < 3 ms
  }, 120_000);
});
