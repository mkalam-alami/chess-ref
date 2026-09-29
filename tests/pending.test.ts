import { describe, expect, it } from 'vitest';
import type { PendingChange, PieceCode, PlyInfo } from '../src/game/types';
import { displayPlies, pendingBoard, pendingKey, pendingProgress } from '../src/ui/pending';

const ply = (san: string, from = 0, to = 0, extra: Partial<PlyInfo> = {}): PlyInfo => ({ san, from, to, tentative: false, ...extra });
const pend = (kind: PendingChange['kind'], fromPly: number, plies: PlyInfo[], progress = 0.5): PendingChange => ({ kind, fromPly, plies, progress });

/** Square index from algebraic notation. */
const sq = (s: string) => s.charCodeAt(0) - 97 + 8 * (Number(s[1]) - 1);

function startPieces(): Array<PieceCode | null> {
  const b: Array<PieceCode | null> = new Array(64).fill(null);
  const back = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'] as const;
  for (let f = 0; f < 8; f++) {
    b[f] = `w${back[f]!}`;
    b[8 + f] = 'wP';
    b[48 + f] = 'bP';
    b[56 + f] = `b${back[f]!}`;
  }
  return b;
}

describe('displayPlies', () => {
  const plies = [ply('e4'), ply('e5'), ply('Nf3', 0, 0, { tentative: true })];

  it('shows committed plies alone without a pending change', () => {
    expect(displayPlies(plies, null).map((d) => d.mark)).toEqual(['committed', 'committed', 'committed']);
  });

  it('appends an advance, replaces a revision and strikes a takeback', () => {
    const adv = displayPlies(plies, pend('advance', 3, [ply('Nc6')]));
    expect(adv.map((d) => [d.ply.san, d.mark])).toEqual([['e4', 'committed'], ['e5', 'committed'], ['Nf3', 'committed'], ['Nc6', 'pending']]);
    const rev = displayPlies(plies, pend('revise', 2, [ply('Nc3')]));
    expect(rev.map((d) => [d.ply.san, d.mark])).toEqual([['e4', 'committed'], ['e5', 'committed'], ['Nc3', 'pending']]);
    const back = displayPlies(plies, pend('takeback', 2, []));
    expect(back.map((d) => [d.ply.san, d.mark])).toEqual([['e4', 'committed'], ['e5', 'committed'], ['Nf3', 'dropped']]);
  });

  it('clamps an out-of-range fromPly', () => {
    expect(displayPlies(plies, pend('advance', 9, [ply('Nc6')])).map((d) => d.ply.san)).toEqual(['e4', 'e5', 'Nf3', 'Nc6']);
  });
});

describe('pendingKey / pendingProgress', () => {
  it('keys on the line, not the progress', () => {
    expect(pendingKey(null)).toBe('');
    const a = pend('advance', 0, [ply('e4')], 0.2);
    expect(pendingKey(a)).toBe(pendingKey({ ...a, progress: 0.9 }));
    expect(pendingKey(a)).not.toBe(pendingKey(pend('advance', 0, [ply('d4')])));
    expect(pendingKey(a)).not.toBe(pendingKey(pend('revise', 0, [ply('e4')])));
  });

  it('clamps progress', () => {
    expect(pendingProgress(null)).toBe(0);
    expect(pendingProgress(pend('advance', 0, [], 1.7))).toBe(1);
    expect(pendingProgress(pend('advance', 0, [], NaN))).toBe(0);
    expect(pendingProgress(pend('advance', 0, [], 0.4))).toBe(0.4);
  });
});

describe('pendingBoard', () => {
  it('is null without a pending change', () => {
    expect(pendingBoard({ pieces: startPieces(), plies: [], pending: null })).toBeNull();
  });

  it('plays the pending plies on the current position', () => {
    const b = pendingBoard({ pieces: startPieces(), plies: [], pending: pend('advance', 0, [ply('e4', sq('e2'), sq('e4'))]) })!;
    expect(b.pieces[sq('e4')]).toBe('wP');
    expect(b.pieces[sq('e2')]).toBeNull();
    expect(b.squares).toEqual([sq('e2'), sq('e4')]);
  });

  it('takes back the replaced plies first (a revision e4 -> d4)', () => {
    const pieces = startPieces();
    pieces[sq('e4')] = 'wP';
    pieces[sq('e2')] = null;
    const plies = [ply('e4', sq('e2'), sq('e4'))];
    const b = pendingBoard({ pieces, plies, pending: pend('revise', 0, [ply('d4', sq('d2'), sq('d4'))]) })!;
    expect(b.pieces[sq('e2')]).toBe('wP');
    expect(b.pieces[sq('e4')]).toBeNull();
    expect(b.pieces[sq('d4')]).toBe('wP');
    expect(b.pieces[sq('d2')]).toBeNull();
    // A takeback just undoes.
    const t = pendingBoard({ pieces, plies, pending: pend('takeback', 0, []) })!;
    expect(t.pieces).toEqual(startPieces());
    expect(t.squares).toEqual([]);
  });

  it('moves the rook when castling and promotes', () => {
    const pieces: Array<PieceCode | null> = new Array(64).fill(null);
    pieces[sq('e1')] = 'wK';
    pieces[sq('h1')] = 'wR';
    pieces[sq('a7')] = 'wP';
    const castle = ply('O-O', sq('e1'), sq('g1'));
    const promo = ply('a8=N', sq('a7'), sq('a8'), { promotion: 'n' });
    const b = pendingBoard({ pieces, plies: [], pending: pend('advance', 0, [castle, promo]) })!;
    expect(b.pieces[sq('g1')]).toBe('wK');
    expect(b.pieces[sq('f1')]).toBe('wR');
    expect(b.pieces[sq('h1')]).toBeNull();
    expect(b.pieces[sq('a8')]).toBe('wN');
    // Undoing the castle and the promotion restores king, rook and pawn.
    const u = pendingBoard({ pieces: b.pieces, plies: [castle, promo], pending: pend('takeback', 0, []) })!;
    expect(u.pieces).toEqual(pieces);
  });
});
