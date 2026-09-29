// Entry of `npm run annot` (vitest.annot.config.ts): fixture annotation tools, see tests/tools/annotate.ts and the
// workflow in tests/fixtures/web/ANNOTATION.md. Options by environment (vitest takes no free CLI arguments):
//   ANNOT          ingest | split | propose | crops | at | promote | compare   (required)
//   ANNOT_DIR      fixture dir (default tests/fixtures/web)
//   ANNOT_FILTER   substring of the file name (propose / crops / promote / compare); the file for ANNOT=at
//   ANNOT_OUT      PNG output dir (default tests/tools/annot-out/<dir name>; gitignored)
//   ingest:  ANNOT_MAX (1600) long side, ANNOT_QUALITY (88), ANNOT_RENAME=1 renames to NN-short-name.jpg,
//            ANNOT_SPLIT=1 also runs split
//   split:   ANNOT_SEED (1)
//   propose: ANNOT_SIZES (640,960) detection long sides, ANNOT_FORCE=1 re-proposes unreviewed entries,
//            ANNOT_REFINE=0 disables the lattice refinement, ANNOT_CROPS=0 skips writing crops afterwards,
//            BENCH_PARAMS as for the bench
//   crops:   ANNOT_FROM draft (default) | gt, ANNOT_CROP (200) crop side in image px, ANNOT_ZOOM (3)
//   at:      ANNOT_AT="x,y;x,y;..." image points (4 points also give a quad preview), ANNOT_CROP, ANNOT_ZOOM
//   promote: ANNOT_FORCE=1 overwrites differing corners.json entries
//   compare: ANNOT_A (default <dir>/corners.draft.json) vs ANNOT_B (default <dir>/corners.json)
import path from 'node:path';
import fs from 'node:fs';
import { test } from 'vitest';
import type { Point } from '../../src/geom/homography';
import { Detector } from '../../src/vision/detector';
import { loadCv } from '../synth/cvNode';
import { assignSplit, compare, crops, cropsAt, DEFAULT_OUT, ingest, promote, propose, readSources, REPO, writeSources } from './annotate';

test('annot', async () => {
  const env = process.env;
  const cmd = env.ANNOT;
  const dir = path.resolve(REPO, env.ANNOT_DIR ?? 'tests/fixtures/web');
  const out = path.resolve(REPO, env.ANNOT_OUT ?? path.join(DEFAULT_OUT, path.basename(dir)));
  const filter = env.ANNOT_FILTER || undefined;
  const force = env.ANNOT_FORCE === '1';
  const num = (v: string | undefined) => (v ? Number(v) : undefined);
  const print = (lines: string[]) => console.log(`\n${lines.join('\n')}`);
  const split = () => {
    const { entries, shape } = readSources(dir);
    const log = assignSplit(entries, Number(env.ANNOT_SEED ?? 1));
    writeSources(dir, entries, shape);
    return log;
  };
  const cropJob = () => ({ dir, outDir: out, from: (env.ANNOT_FROM === 'gt' ? 'gt' : 'draft') as 'gt' | 'draft', filter, size: num(env.ANNOT_CROP), zoom: num(env.ANNOT_ZOOM) });

  switch (cmd) {
    case 'ingest': {
      const { cv } = await loadCv();
      const log = ingest(cv, dir, { maxSide: num(env.ANNOT_MAX), quality: num(env.ANNOT_QUALITY), rename: env.ANNOT_RENAME === '1' });
      print(env.ANNOT_SPLIT === '1' ? [...log, ...split()] : log);
      return;
    }
    case 'split':
      print(split());
      return;
    case 'propose': {
      const { cv } = await loadCv();
      const det = new Detector(cv);
      try {
        const log = propose(cv, det, dir, {
          sizes: env.ANNOT_SIZES?.split(',').map(Number),
          params: JSON.parse(env.BENCH_PARAMS ?? '{}') as Record<string, number>,
          force, filter, refine: env.ANNOT_REFINE !== '0',
        });
        print(env.ANNOT_CROPS === '0' ? log : [...log, ...crops(cv, cropJob())]);
      } finally {
        det.dispose();
      }
      return;
    }
    case 'crops': {
      const { cv } = await loadCv();
      print(crops(cv, cropJob()));
      return;
    }
    case 'at': {
      const { cv } = await loadCv();
      if (!filter || !env.ANNOT_AT) throw new Error('ANNOT=at needs ANNOT_FILTER=<file> and ANNOT_AT="x,y;x,y;..."');
      const files = fs.readdirSync(dir).filter((f) => f.includes(filter) && /\.jpe?g$/i.test(f));
      if (files.length !== 1) throw new Error(`ANNOT_FILTER=${filter} matches ${files.length} images: ${files.join(', ')}`);
      const pts = env.ANNOT_AT.split(';').filter(Boolean).map((s) => s.split(',').map(Number) as Point);
      print(cropsAt(cv, dir, files[0]!, pts, out, { size: num(env.ANNOT_CROP), zoom: num(env.ANNOT_ZOOM) }));
      return;
    }
    case 'promote':
      print(promote(dir, { force, filter }));
      return;
    case 'compare': {
      const a = path.resolve(REPO, env.ANNOT_A ?? path.join(dir, 'corners.draft.json'));
      const b = path.resolve(REPO, env.ANNOT_B ?? path.join(dir, 'corners.json'));
      console.log(`\n${compare(a, b, filter).text}`);
      return;
    }
    default:
      throw new Error(`ANNOT must be one of ingest | split | propose | crops | at | promote | compare (got ${cmd ?? 'nothing'}); see tests/fixtures/web/ANNOTATION.md`);
  }
}, 1_800_000);
