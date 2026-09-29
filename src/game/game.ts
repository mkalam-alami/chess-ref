import type { GameEvent, GameSnapshot, Observation } from './types';

/** Tunables of the game layer (thresholds in summed log-likelihood units, times in ms of observation clock). */
export interface GameParams {
  /** Revision window: number of most recent plies that stay correctable (1 by default, 2 supported). */
  revisionDepth: number;
}

export const DEFAULT_GAME_PARAMS: GameParams = { revisionDepth: 1 };

/**
 * Turns occupancy observations into a legal game from the standard starting position.
 * STUB (milestone 9 contracts): the real implementation replaces this file; the public API below is fixed.
 */
export class GameTracker {
  readonly params: GameParams;

  constructor(params: Partial<GameParams> = {}) {
    this.params = { ...DEFAULT_GAME_PARAMS, ...params };
  }

  /** Feeds one frame of evidence; returns what changed. */
  observe(_obs: Observation): GameEvent[] {
    return [];
  }

  snapshot(): GameSnapshot {
    return { state: 'waiting', fen: '', turn: 'w', check: false, result: '*', plies: [], pieces: new Array(64).fill(null), lastMove: null, grid: null, top: [] };
  }

  /** Manual undo of the last ply (sidebar button). */
  undo(): GameEvent[] {
    return [];
  }

  /** Drops the game and waits for the starting position again. */
  newGame(): void {}

  /** Accepts a desync: resume inference from the current game position. */
  continueAfterDesync(): void {}

  /** Changes the promotion piece of ply `ply` (0-based); false when that ply is not a promotion or the change is illegal. */
  setPromotion(_ply: number, _piece: 'q' | 'r' | 'b' | 'n'): boolean {
    return false;
  }

  /** PGN of the game (Seven Tag Roster; extra headers override). */
  pgn(_headers: Record<string, string> = {}): string {
    return '';
  }

  /** Serialised game for localStorage. */
  save(): string {
    return '';
  }

  /** Restores a game saved by `save()`; false (and state unchanged) when invalid. */
  load(_saved: string): boolean {
    return false;
  }
}
