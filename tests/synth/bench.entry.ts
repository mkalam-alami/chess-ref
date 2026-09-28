// Entry of `npm run bench` (see vitest.bench.config.ts). Environment: BENCH_N (samples per category), BENCH_SEED,
// BENCH_FILTER (substring of the category label), BENCH_DUMP (max failure PNGs per run).
import path from 'node:path';
import fs from 'node:fs';
import { test } from 'vitest';
import { Detector } from '../../src/vision/detector';
import { encodePng } from './png';
import { loadCv } from './cvNode';
import { dumpFailure, formatTable, negativeCases, realCases, recallWhere, runCases, summarize, synthCases, type Outcome } from './bench';

const OUT = path.resolve(import.meta.dirname, 'out');

test('bench', async () => {
  const { cv } = await loadCv();
  const n = Number(process.env.BENCH_N ?? 12);
  const seed = Number(process.env.BENCH_SEED ?? 1000);
  const maxDump = Number(process.env.BENCH_DUMP ?? 16);
  const filter = process.env.BENCH_FILTER;
  if (process.env.BENCH_PREVIEW) {
    // Dump raw samples (no detection) for eyeballing the generator.
    const dir = path.join(OUT, 'preview');
    fs.mkdirSync(dir, { recursive: true });
    for (const c of [...synthCases(cv, Number(process.env.BENCH_PREVIEW), seed, filter ? (s) => `${s.elev}-${s.palette}-${s.pieces}`.includes(filter) : undefined), ...negativeCases(cv)]) {
      fs.writeFileSync(path.join(dir, `${c.name}.png`), encodePng(c.rgba, c.width, c.height));
    }
    console.log('previews in ' + dir);
    return;
  }
  const det = new Detector(cv);
  const t0 = Date.now();
  const only = process.env.BENCH_ONLY; // 'real' | 'synth'
  const cases = [
    ...(only === 'real' ? [] : synthCases(cv, n, seed, filter ? (s) => `${s.elev}-${s.palette}-${s.pieces}`.includes(filter) : undefined)),
    ...(filter || only === 'real' ? [] : negativeCases(cv)),
    ...(filter || only === 'synth' ? [] : [...realCases(cv, 640), ...realCases(cv, 800)]),
  ];
  const genSec = (Date.now() - t0) / 1000;
  const outs: Outcome[] = runCases(det, cases, {}, true);
  if (process.env.BENCH_VERBOSE) {
    for (const o of outs) {
      const d = o.res.debug!;
      const f1 = d.family.filter((x) => x === 1).length;
      const f2 = d.family.filter((x) => x === 2).length;
      const cands = d.candidates.map((c) => c.verify.score.toFixed(2)).join(',');
      console.log(
        `${o.c.name.padEnd(34)} err=${o.err === null ? '-' : (o.err * 100).toFixed(1)} conf=${o.res.confidence.toFixed(2)} segs=${d.segments.length} fam=${f1}/${f2} hyps=${d.hypsX.length}/${d.hypsY.length} cand=[${cands}] ${d.reason ?? ''}`,
      );
    }
  }
  console.log(`\n${formatTable(summarize(outs))}\n`);
  const acc: Record<string, number> = {};
  for (const o of outs) for (const [k, v] of Object.entries(o.res.timings)) acc[k] = (acc[k] ?? 0) + v / outs.length;
  console.log('mean ms per stage: ' + Object.entries(acc).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(', '));
  const synth = (l: string) => l.startsWith('overhead') || l.startsWith('oblique');
  const clear = recallWhere(outs, (l) => synth(l) && l.endsWith('none'));
  const some = recallWhere(outs, (l) => synth(l) && !l.endsWith('none'));
  const all = recallWhere(outs, synth);
  console.log(`synthetic recall: no pieces ${(clear.recall * 100).toFixed(1)}% (n=${clear.n}), with pieces ${(some.recall * 100).toFixed(1)}% (n=${some.n}), all ${(all.recall * 100).toFixed(1)}%  [generation ${genSec.toFixed(1)}s]`);
  fs.rmSync(OUT, { recursive: true, force: true });
  let dumped = 0;
  const files: string[] = [];
  const failures = outs.filter((o) => (o.c.gt && !o.ok) || o.falsePositive);
  // Spread the dumps over categories.
  const seen = new Map<string, number>();
  for (const o of failures) {
    if (dumped >= maxDump) break;
    if (o.c.label.startsWith('real') === false && only === 'real') continue;
    if ((seen.get(o.c.label) ?? 0) >= 2) continue;
    seen.set(o.c.label, (seen.get(o.c.label) ?? 0) + 1);
    files.push(dumpFailure(o, OUT));
    dumped++;
  }
  console.log(`failure PNGs (${failures.length} failures, ${files.length} dumped) in ${OUT}`);
  for (const f of files) console.log('  ' + path.basename(f));
  det.dispose();
}, 900_000);
