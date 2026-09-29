// Unit tests of the fixture annotation tooling (tests/tools/annotate.ts, tests/tools/imageio.ts): corner order /
// whiteEdge remapping, the seeded holdout split, EXIF orientation, PNG decoding, metadata stripping, the lattice fit.
import { describe, expect, it } from 'vitest';
import { applyH, homographyFrom4, type Point } from '../src/geom/homography';
import { chessCorners } from './synth/realOccBench';
import { encodePng } from './synth/png';
import { assignSplit, fitHomography, normaliseQuad, stripJpegMetadata, type SourceEntry } from './tools/annotate';
import { applyOrientation, decodePng, encodeJpeg, jpegOrientation } from './tools/imageio';

const GT01: Point[] = [[592, 251], [1212, 235], [1325, 843], [555, 885]];

describe('normaliseQuad', () => {
  it('keeps a normalised quad', () => {
    expect(normaliseQuad(GT01, 2)).toEqual({ corners: GT01, whiteEdge: 2 });
  });

  it('reorders rotated / reflected quads and keeps the same chess corners', () => {
    const want = chessCorners(GT01, 2);
    for (let r = 0; r < 4; r++)
      for (const rev of [false, true]) {
        // q[i] = GT01[perm[i]]; the white edge GT01[2] -> GT01[3] is the edge between q-indices of 2 and 3.
        const perm = [0, 1, 2, 3].map((i) => (rev ? (r - i + 8) % 4 : (r + i) % 4));
        const q = perm.map((i) => GT01[i]!);
        const a = perm.indexOf(2);
        const b = perm.indexOf(3);
        const we = (a + 1) % 4 === b ? a : b;
        const n = normaliseQuad(q, we)!;
        expect(n.corners).toEqual(GT01);
        expect(n.whiteEdge).toBe(2);
        expect(chessCorners(n.corners, n.whiteEdge!)).toEqual(want);
      }
  });

  it('rejects a self-intersecting quad', () => {
    expect(normaliseQuad([GT01[0]!, GT01[2]!, GT01[1]!, GT01[3]!])).toBeNull();
  });
});

describe('assignSplit', () => {
  const mk = (n: number, view: SourceEntry['view'], material: string | null, split: SourceEntry['split'] = null): SourceEntry[] =>
    Array.from({ length: n }, (_, i) => ({ file: `${view}-${material}-${i}.jpg`, title: null, author: null, sourceUrl: null, license: null, view, material, lighting: null, split, notes: null }));

  it('is deterministic, ~1/3 holdout, spread over groups, keeps existing values', () => {
    const run = () => {
      const e = [...mk(6, 'oblique', 'wood'), ...mk(3, 'topdown', 'vinyl'), ...mk(3, 'oblique', 'vinyl'), ...mk(1, 'topdown', 'wood', 'tune')];
      assignSplit(e, 7);
      return e;
    };
    const a = run();
    expect(a.map((e) => e.split)).toEqual(run().map((e) => e.split));
    expect(a.every((e) => e.split === 'tune' || e.split === 'holdout')).toBe(true);
    expect(a.filter((e) => e.split === 'holdout').length).toBe(4); // round(13 / 3)
    expect(a.filter((e) => e.file.startsWith('oblique-wood') && e.split === 'holdout').length).toBe(2);
    expect(a.filter((e) => e.file.startsWith('topdown-vinyl') && e.split === 'holdout').length).toBe(1);
    expect(a.filter((e) => e.file.startsWith('oblique-vinyl') && e.split === 'holdout').length).toBe(1);
    expect(a.at(-1)!.split).toBe('tune');
  });

  it('counts existing holdouts', () => {
    const e = mk(6, 'oblique', null);
    e[0]!.split = 'holdout';
    e[1]!.split = 'holdout';
    assignSplit(e, 1);
    expect(e.filter((x) => x.split === 'holdout').length).toBe(2);
  });
});

/** A JPEG with an EXIF APP1 segment carrying orientation v (big-endian TIFF), inserted after SOI. */
function withExif(jpeg: Uint8Array, v: number): Uint8Array {
  const tiff = [0x4d, 0x4d, 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, v, 0, 0, 0, 0, 0, 0, 0, 0];
  const body = [...Buffer.from('Exif\0\0', 'binary'), ...tiff];
  const len = body.length + 2;
  return Buffer.concat([jpeg.subarray(0, 2), Buffer.from([0xff, 0xe1, len >> 8, len & 255, ...body]), jpeg.subarray(2)]);
}

describe('imageio', () => {
  const w = 5;
  const h = 3;
  const img = { data: new Uint8Array(w * h * 4), width: w, height: h };
  for (let i = 0; i < w * h; i++) img.data.set([i * 10, 255 - i * 10, (i * 37) % 256, 255], i * 4);

  it('reads and strips the EXIF orientation', () => {
    const j = encodeJpeg(img, 90);
    expect(jpegOrientation(j)).toBe(1);
    const e = withExif(j, 6);
    expect(jpegOrientation(e)).toBe(6);
    const s = stripJpegMetadata(e);
    expect(s.removed).toBe(1);
    expect(Buffer.compare(Buffer.from(s.out), j)).toBe(0);
  });

  it('applies orientations (6 = rotate 90 cw, 8 = 90 ccw, 3 = 180)', () => {
    const px = (m: { data: Uint8Array; width: number }, x: number, y: number) => m.data[(y * m.width + x) * 4]!;
    const r6 = applyOrientation(img, 6);
    expect([r6.width, r6.height]).toEqual([h, w]);
    // Displayed top-right = source top-left for a 90 degree clockwise rotation.
    expect(px(r6, h - 1, 0)).toBe(px(img, 0, 0));
    const r8 = applyOrientation(img, 8);
    expect(px(r8, 0, w - 1)).toBe(px(img, 0, 0));
    const r3 = applyOrientation(img, 3);
    expect(px(r3, w - 1, h - 1)).toBe(px(img, 0, 0));
    for (let v = 2; v <= 8; v++) expect(applyOrientation(applyOrientation(img, v), [1, 2, 3, 4, 5, 8, 7, 6][v - 1]!).data).toEqual(img.data);
  });

  it('decodes PNGs written by the bench encoder', () => {
    const d = decodePng(encodePng(img.data, w, h));
    expect([d.width, d.height]).toEqual([w, h]);
    expect(Array.from(d.data)).toEqual(Array.from(img.data));
  });
});

describe('fitHomography', () => {
  it('recovers a homography from lattice points', () => {
    const H = homographyFrom4([[0, 0], [8, 0], [8, 8], [0, 8]], GT01)!;
    const src: Point[] = [];
    for (let j = 1; j <= 7; j++) for (let i = 1; i <= 7; i++) src.push([i, j]);
    const F = fitHomography(src, src.map((p) => applyH(H, p)))!;
    for (const p of [[0, 0], [8, 8], [0, 8]] as Point[]) {
      const a = applyH(F, p);
      const b = applyH(H, p);
      expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeLessThan(1e-6);
    }
  });
});
