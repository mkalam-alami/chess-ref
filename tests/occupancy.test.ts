import { beforeAll, describe, expect, it } from 'vitest';
import { homographyFrom4, invert3, mul3, applyH, type Mat3, type Point } from '../src/geom/homography';
import { Detector } from '../src/vision/detector';
import type { CV } from '../src/vision/preprocess';
import { cameraFromHomography, cellFootprints, OccupancyFilter, OccupancyTracker, startGrid } from '../src/vision/occupancy';
import { OCC_BLACK, OCC_EMPTY, OCC_WHITE } from '../src/worker/protocol';
import { loadCv } from './synth/cvNode';
import { H, makeBoardSample, makeCamera, W } from './synth/generate';

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

  it('freezes on a mass change and commits once the board is stable for the settle time', () => {
    const f = new OccupancyFilter();
    f.reset(startGrid(0, 0));
    const hand = startGrid(0, 0);
    for (let c = 16; c < 26; c++) hand[c] = OCC_BLACK;
    f.update(startGrid(0, 0), conf, params, 0);
    expect(f.update(hand, conf, params, 50)).toBe(true); // 10 cells flipped at once
    f.update(hand, conf, params, 100);
    f.update(hand, conf, params, 150);
    f.update(hand, conf, params, 300);
    expect(f.committed![20]).toBe(OCC_EMPTY); // still frozen (settle 400 ms from t = 50)
    f.update(hand, conf, params, 460);
    expect(f.committed![20]).toBe(OCC_BLACK);
    f.noBoard(500);
    expect(f.frozen(700, params)).toBe(true);
  });
});

interface Eval {
  n: number;
  boots: number;
  rawAcc: number;
  gridAcc: number;
  dropped: number;
}

/** Ground truth re-indexed into the detector's hb cell frame (null if hb is not a relabelling of the GT board). */
function alignGt(gt: Uint8Array, gtCorners: readonly Point[], hb: Mat3): Uint8Array | null {
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
          let out = tr.update(frame, r.hb, {}, 0);
          for (let t = 1; t < 5; t++) out = tr.update(frame, r.hb, {}, t * 100);
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
