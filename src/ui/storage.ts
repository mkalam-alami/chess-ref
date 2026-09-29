/** localStorage persistence of the current game and an archive of recent games' PGN (every access is guarded). */

export const GAME_KEY = 'chess-ref.game';
export const ARCHIVE_KEY = 'chess-ref.archive';
export const ARCHIVE_MAX = 10;
export const MUTED_KEY = 'chess-ref.muted';

export interface ArchivedGame {
  /** Time archived (ms since epoch). */
  t: number;
  pgn: string;
}

type Store = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

function store(): Store | null {
  try {
    return localStorage;
  } catch {
    return null;
  }
}

export function loadSavedGame(s: Store | null = store()): string | null {
  try {
    return s?.getItem(GAME_KEY) ?? null;
  } catch {
    return null;
  }
}

export function saveGame(saved: string, s: Store | null = store()): void {
  try {
    if (saved) s?.setItem(GAME_KEY, saved);
    else s?.removeItem(GAME_KEY);
  } catch {
    /* storage unavailable or full */
  }
}

/** Whether the sound was muted (default: sound on). */
export function loadMuted(s: Store | null = store()): boolean {
  try {
    return s?.getItem(MUTED_KEY) === '1';
  } catch {
    return false;
  }
}

export function saveMuted(muted: boolean, s: Store | null = store()): void {
  try {
    s?.setItem(MUTED_KEY, muted ? '1' : '0');
  } catch {
    /* storage unavailable */
  }
}

export function loadArchive(s: Store | null = store()): ArchivedGame[] {
  try {
    const v: unknown = JSON.parse(s?.getItem(ARCHIVE_KEY) ?? '[]');
    return Array.isArray(v) ? v.filter((g): g is ArchivedGame => typeof g?.pgn === 'string' && typeof g?.t === 'number') : [];
  } catch {
    return [];
  }
}

function writeArchive(games: ArchivedGame[], s: Store | null): void {
  try {
    s?.setItem(ARCHIVE_KEY, JSON.stringify(games.slice(-ARCHIVE_MAX)));
  } catch {
    /* storage unavailable or full */
  }
}

/** Appends a finished / replaced game's PGN (empty PGNs are ignored), keeping the last ARCHIVE_MAX. */
export function archivePgn(pgn: string, t = Date.now(), s: Store | null = store()): void {
  if (!pgn.trim()) return;
  writeArchive([...loadArchive(s), { t, pgn }], s);
}

/** Removes the most recent archive entry if it is `pgn` (the new game was undone, so that game is current again). */
export function unarchiveLast(pgn: string, s: Store | null = store()): void {
  const games = loadArchive(s);
  if (games.length && games[games.length - 1]!.pgn === pgn) writeArchive(games.slice(0, -1), s);
}
