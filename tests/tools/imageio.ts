/*
 * Image I/O for the annotation tools (tests/tools/annotate.ts): JPEG decode / encode (jpeg-js), a minimal PNG decoder
 * (8-bit, non-interlaced: grey, RGB, palette, grey+alpha, RGBA; enough for screenshots / converted photos), the EXIF
 * orientation tag of a JPEG, and the 8 EXIF orientation transforms. No dependencies beyond jpeg-js and node:zlib.
 */
import { createRequire } from 'node:module';
import { inflateSync } from 'node:zlib';

export interface Rgba {
  data: Uint8Array;
  width: number;
  height: number;
}

interface JpegJs {
  decode(b: Buffer | Uint8Array, o: { useTArray: boolean; maxMemoryUsageInMB?: number; formatAsRGBA?: boolean }): Rgba;
  encode(img: { data: Uint8Array; width: number; height: number }, quality: number): { data: Buffer };
}

const jpeg = (): JpegJs => createRequire(import.meta.url)('jpeg-js') as JpegJs;

export const isJpeg = (b: Uint8Array) => b.length > 3 && b[0] === 0xff && b[1] === 0xd8;
export const isPng = (b: Uint8Array) => b.length > 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47;
/** HEIC / HEIF / AVIF (ISO BMFF 'ftyp' box): not decodable here. */
export const isHeif = (b: Uint8Array) => b.length > 12 && String.fromCharCode(...b.subarray(4, 8)) === 'ftyp';

export function decodeJpeg(b: Uint8Array): Rgba {
  return jpeg().decode(b, { useTArray: true, maxMemoryUsageInMB: 4096 });
}

export function encodeJpeg(img: Rgba, quality = 88): Buffer {
  return jpeg().encode(img, quality).data;
}

/**
 * EXIF orientation (1..8) of a JPEG, 1 when absent. Throws nothing: a malformed EXIF block reads as 1.
 */
export function jpegOrientation(b: Uint8Array): number {
  if (!isJpeg(b)) return 1;
  let o = 2;
  while (o + 4 <= b.length) {
    if (b[o] !== 0xff) return 1;
    const marker = b[o + 1]!;
    if (marker === 0xd9 || marker === 0xda) return 1; // EOI / start of scan: no more metadata
    const len = (b[o + 2]! << 8) | b[o + 3]!;
    if (marker === 0xe1 && len > 14 && String.fromCharCode(...b.subarray(o + 4, o + 10)) === 'Exif\0\0') {
      try {
        return tiffOrientation(b.subarray(o + 10, o + 2 + len));
      } catch {
        return 1;
      }
    }
    o += 2 + len;
  }
  return 1;
}

function tiffOrientation(t: Uint8Array): number {
  const le = t[0] === 0x49; // 'II'
  const u16 = (p: number) => (le ? t[p]! | (t[p + 1]! << 8) : (t[p]! << 8) | t[p + 1]!);
  const u32 = (p: number) => (le ? (t[p]! | (t[p + 1]! << 8) | (t[p + 2]! << 16)) + t[p + 3]! * 2 ** 24 : t[p]! * 2 ** 24 + ((t[p + 1]! << 16) | (t[p + 2]! << 8) | t[p + 3]!));
  const ifd = u32(4);
  const n = u16(ifd);
  for (let i = 0; i < n; i++) {
    const e = ifd + 2 + i * 12;
    if (u16(e) === 0x0112) {
      const v = u16(e + 8);
      return v >= 1 && v <= 8 ? v : 1;
    }
  }
  return 1;
}

/** Applies EXIF orientation v: returns the image as it should be displayed (orientation 1). */
export function applyOrientation(img: Rgba, v: number): Rgba {
  if (v <= 1 || v > 8) return img;
  const { width: w, height: h, data } = img;
  const swap = v >= 5;
  const W = swap ? h : w;
  const H = swap ? w : h;
  const out = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      // (sx, sy) = source pixel shown at display (x, y).
      let sx: number;
      let sy: number;
      switch (v) {
        case 2: sx = w - 1 - x; sy = y; break;
        case 3: sx = w - 1 - x; sy = h - 1 - y; break;
        case 4: sx = x; sy = h - 1 - y; break;
        case 5: sx = y; sy = x; break;
        case 6: sx = y; sy = h - 1 - x; break;
        case 7: sx = w - 1 - y; sy = h - 1 - x; break;
        default: sx = w - 1 - y; sy = x; break; // 8
      }
      const s = (sy * w + sx) * 4;
      const d = (y * W + x) * 4;
      out[d] = data[s]!;
      out[d + 1] = data[s + 1]!;
      out[d + 2] = data[s + 2]!;
      out[d + 3] = data[s + 3]!;
    }
  return { data: out, width: W, height: H };
}

/** Minimal PNG decoder: bit depth 8, no interlace; colour types 0, 2, 3, 4, 6. Alpha is dropped (composited on white). */
export function decodePng(b: Uint8Array): Rgba {
  let o = 8;
  let width = 0;
  let height = 0;
  let depth = 0;
  let ctype = 0;
  let interlace = 0;
  let palette: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  const u32 = (p: number) => b[p]! * 2 ** 24 + ((b[p + 1]! << 16) | (b[p + 2]! << 8) | b[p + 3]!);
  while (o + 8 <= b.length) {
    const len = u32(o);
    const type = String.fromCharCode(...b.subarray(o + 4, o + 8));
    const d = b.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') {
      width = u32(o + 8);
      height = u32(o + 12);
      depth = d[8]!;
      ctype = d[9]!;
      interlace = d[12]!;
    } else if (type === 'PLTE') palette = d;
    else if (type === 'IDAT') idat.push(d);
    else if (type === 'IEND') break;
    o += 12 + len;
  }
  if (depth !== 8 || interlace !== 0) throw new Error(`unsupported PNG (bit depth ${depth}, interlace ${interlace}); convert it to JPEG first`);
  const ch = ({ 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 } as Record<number, number>)[ctype];
  if (!ch) throw new Error(`unsupported PNG colour type ${ctype}`);
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * ch;
  const cur = new Uint8Array(stride);
  const prev = new Uint8Array(stride);
  const out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)]!;
    const row = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= ch ? cur[i - ch]! : 0;
      const up = prev[i]!;
      const c = i >= ch ? prev[i - ch]! : 0;
      let v = row[i]!;
      if (f === 1) v += a;
      else if (f === 2) v += up;
      else if (f === 3) v += (a + up) >> 1;
      else if (f === 4) {
        const p = a + up - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - up);
        const pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? up : c;
      }
      cur[i] = v & 255;
    }
    for (let x = 0; x < width; x++) {
      let r: number;
      let g: number;
      let bl: number;
      let al = 255;
      const p = x * ch;
      if (ctype === 0) r = g = bl = cur[p]!;
      else if (ctype === 4) {
        r = g = bl = cur[p]!;
        al = cur[p + 1]!;
      } else if (ctype === 3) {
        const k = cur[p]! * 3;
        r = palette?.[k] ?? 0;
        g = palette?.[k + 1] ?? 0;
        bl = palette?.[k + 2] ?? 0;
      } else {
        r = cur[p]!;
        g = cur[p + 1]!;
        bl = cur[p + 2]!;
        if (ctype === 6) al = cur[p + 3]!;
      }
      const q = (y * width + x) * 4;
      const w = 255 - al;
      out[q] = Math.round((r * al + 255 * w) / 255);
      out[q + 1] = Math.round((g * al + 255 * w) / 255);
      out[q + 2] = Math.round((bl * al + 255 * w) / 255);
      out[q + 3] = 255;
    }
    prev.set(cur);
  }
  return { data: out, width, height };
}

/** Decodes a JPEG or PNG file's bytes (as displayed: EXIF orientation applied) and reports the orientation found. */
export function decodeImage(b: Uint8Array): { img: Rgba; orientation: number; format: 'jpeg' | 'png' } {
  if (isJpeg(b)) {
    const orientation = jpegOrientation(b);
    return { img: applyOrientation(decodeJpeg(b), orientation), orientation, format: 'jpeg' };
  }
  if (isPng(b)) return { img: decodePng(b), orientation: 1, format: 'png' };
  if (isHeif(b)) throw new Error('HEIC/HEIF/AVIF file: convert it to JPEG first');
  throw new Error('unknown image format (not JPEG / PNG)');
}
