import { describe, expect, it } from 'vitest';
import type { PlyInfo } from '../src/game/types';
import { FLASH_MS, flashAge, movePairs, resultText, statusText } from '../src/ui/moves';
import { ARCHIVE_KEY, ARCHIVE_MAX, archivePgn, GAME_KEY, loadArchive, loadSavedGame, saveGame, unarchiveLast } from '../src/ui/storage';

const ply = (san: string, extra: Partial<PlyInfo> = {}): PlyInfo => ({ san, from: 0, to: 0, tentative: false, ...extra });

describe('movePairs', () => {
  it('numbers pairs and leaves black empty after a white move at the tip', () => {
    expect(movePairs([])).toEqual([]);
    expect(movePairs([ply('e4'), ply('e5'), ply('Nf3')])).toEqual([
      { no: 1, white: 0, black: 1 },
      { no: 2, white: 2 },
    ]);
  });
});

describe('statusText', () => {
  const base = { state: 'playing', turn: 'w', check: false, result: '*' } as const;

  it('covers every state', () => {
    expect(statusText({ ...base, state: 'waiting' })).toBe('Set up the starting position');
    expect(statusText(base)).toBe('White to move');
    expect(statusText({ ...base, turn: 'b' })).toBe('Black to move');
    expect(statusText({ ...base, check: true })).toContain('Check');
    expect(statusText({ ...base, state: 'desync' })).toMatch(/doesn.t match/);
    expect(statusText({ ...base, state: 'over', result: '1-0', check: true })).toBe('Checkmate, White wins');
    expect(statusText({ ...base, state: 'over', result: '1/2-1/2' })).toBe('Draw');
    expect(resultText('0-1', false)).toBe('Black wins');
  });
});

describe('flashAge', () => {
  it('flashes only recently corrected plies', () => {
    expect(flashAge(ply('e4'), 1000)).toBeNull();
    expect(flashAge(ply('d4', { correctedAt: 1000 }), 1200)).toBe(200);
    expect(flashAge(ply('d4', { correctedAt: 1000 }), 1000 + FLASH_MS)).toBeNull();
    expect(flashAge(ply('d4', { correctedAt: 1000 }), 900)).toBeNull();
    expect(flashAge(ply('d4', { correctedAt: 1000 }), -Infinity)).toBeNull();
  });
});

class MemStore {
  map = new Map<string, string>();
  getItem(k: string) {
    return this.map.get(k) ?? null;
  }
  setItem(k: string, v: string) {
    this.map.set(k, v);
  }
  removeItem(k: string) {
    this.map.delete(k);
  }
}

describe('game storage', () => {
  it('saves, clears and survives unavailable or corrupt storage', () => {
    const s = new MemStore();
    saveGame('{"x":1}', s);
    expect(loadSavedGame(s)).toBe('{"x":1}');
    saveGame('', s);
    expect(s.map.has(GAME_KEY)).toBe(false);
    expect(loadSavedGame(null)).toBeNull();
    const fail = () => {
      throw new Error('denied');
    };
    const broken = { getItem: fail, setItem: fail, removeItem: fail };
    expect(loadSavedGame(broken)).toBeNull();
    expect(() => saveGame('x', broken)).not.toThrow();
    expect(() => archivePgn('1. e4 *', 0, broken)).not.toThrow();
    s.setItem(ARCHIVE_KEY, 'not json');
    expect(loadArchive(s)).toEqual([]);
  });

  it('keeps the last games and can drop the latest one again', () => {
    const s = new MemStore();
    archivePgn('   ', 0, s);
    expect(loadArchive(s)).toEqual([]);
    for (let k = 0; k < ARCHIVE_MAX + 3; k++) archivePgn(`game ${k}`, k, s);
    const games = loadArchive(s);
    expect(games).toHaveLength(ARCHIVE_MAX);
    expect(games[games.length - 1]).toEqual({ t: ARCHIVE_MAX + 2, pgn: `game ${ARCHIVE_MAX + 2}` });
    unarchiveLast('something else', s);
    expect(loadArchive(s)).toHaveLength(ARCHIVE_MAX);
    unarchiveLast(`game ${ARCHIVE_MAX + 2}`, s);
    expect(loadArchive(s).at(-1)!.pgn).toBe(`game ${ARCHIVE_MAX + 1}`);
  });
});
