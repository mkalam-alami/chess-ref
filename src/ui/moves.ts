import type { GameSnapshot, PlyInfo } from '../game/types';

/** How long a corrected ply flashes (ms, observation clock). */
export const FLASH_MS = 1500;

export interface MovePair {
  /** Move number (1-based). */
  no: number;
  /** Ply indices (0-based) of white's and black's move; black is absent after a white move at the tip. */
  white: number;
  black?: number;
}

/** Groups plies into numbered pairs `1. e4 e5` (games always start from the standard position, white first). */
export function movePairs(plies: readonly PlyInfo[]): MovePair[] {
  const out: MovePair[] = [];
  for (let i = 0; i < plies.length; i += 2) out.push(i + 1 < plies.length ? { no: i / 2 + 1, white: i, black: i + 1 } : { no: i / 2 + 1, white: i });
  return out;
}

/** Human-readable result of a finished game. */
export function resultText(result: string, check: boolean): string {
  if (result === '1-0') return check ? 'Checkmate, White wins' : 'White wins';
  if (result === '0-1') return check ? 'Checkmate, Black wins' : 'Black wins';
  if (result === '1/2-1/2') return 'Draw';
  return 'Game over';
}

/** The sidebar / chip status line for a snapshot. */
export function statusText(s: Pick<GameSnapshot, 'state' | 'turn' | 'check' | 'result'>): string {
  switch (s.state) {
    case 'waiting':
      return 'Set up the starting position';
    case 'desync':
      return 'Board doesn’t match';
    case 'over':
      return resultText(s.result, s.check);
    default: {
      const side = s.turn === 'w' ? 'White to move' : 'Black to move';
      return s.check ? `Check — ${side}` : side;
    }
  }
}

/** Age (ms) of a ply's last correction at observation time `now`, or null when it should not flash. */
export function flashAge(ply: PlyInfo, now: number): number | null {
  if (ply.correctedAt === undefined || !Number.isFinite(now)) return null;
  const age = now - ply.correctedAt;
  return age >= 0 && age < FLASH_MS ? age : null;
}

/** Everything the sidebar shows; re-rendered only when this changes. */
function renderKey(s: GameSnapshot): string {
  return JSON.stringify([s.state, s.turn, s.check, s.result, s.plies.map((p) => [p.san, p.tentative, p.correctedAt, p.promotion])]);
}

export type Promotion = 'q' | 'r' | 'b' | 'n';

export interface MoveListActions {
  copy(): void;
  undo(): void;
  newGame(): void;
  continueAfterDesync(): void;
  promote(ply: number, piece: Promotion): void;
}

const COLLAPSED_KEY = 'chess-ref.movesCollapsed';

function loadCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === '1';
  } catch {
    return false;
  }
}

function saveCollapsed(on: boolean): void {
  try {
    localStorage.setItem(COLLAPSED_KEY, on ? '1' : '0');
  } catch {
    /* storage unavailable */
  }
}

const $ = <T extends HTMLElement>(root: HTMLElement, sel: string) => root.querySelector(sel) as T;

/** The move sidebar (#moves in index.html): status line, move pairs, Copy PGN / Undo / New game. */
export class MoveList {
  private list: HTMLElement;
  private status: HTMLElement;
  private desync: HTMLElement;
  private promo: HTMLElement;
  private pgnBox: HTMLTextAreaElement;
  private copyBtn: HTMLButtonElement;
  private undoBtn: HTMLButtonElement;
  private newBtn: HTMLButtonElement;
  private key = '';
  private promoPly = -1;

  constructor(
    readonly root: HTMLElement,
    actions: MoveListActions,
  ) {
    this.list = $(root, '.mv-list');
    this.status = $(root, '.mv-status');
    this.desync = $(root, '.mv-desync');
    this.promo = $(root, '.mv-promo');
    this.pgnBox = $(root, '.mv-pgn');
    this.copyBtn = $(root, '.mv-copy');
    this.undoBtn = $(root, '.mv-undo');
    this.newBtn = $(root, '.mv-new');
    this.copyBtn.addEventListener('click', () => actions.copy());
    this.undoBtn.addEventListener('click', () => actions.undo());
    this.newBtn.addEventListener('click', () => actions.newGame());
    $(root, '.mv-desync-undo').addEventListener('click', () => actions.undo());
    $(root, '.mv-desync-continue').addEventListener('click', () => actions.continueAfterDesync());
    $(root, '.mv-toggle').addEventListener('click', () => this.setCollapsed(!this.collapsed));
    this.pgnBox.addEventListener('blur', () => (this.pgnBox.hidden = true));
    this.list.addEventListener('click', (ev) => {
      const el = (ev.target as HTMLElement).closest<HTMLElement>('.mv-ply.promo');
      if (el) this.openPromo(Number(el.dataset.ply), el);
    });
    this.promo.addEventListener('click', (ev) => {
      const piece = (ev.target as HTMLElement).closest<HTMLElement>('button')?.dataset.piece as Promotion | undefined;
      if (piece && this.promoPly >= 0) actions.promote(this.promoPly, piece);
      this.closePromo();
    });
    this.setCollapsed(loadCollapsed());
  }

  get collapsed(): boolean {
    return this.root.classList.contains('collapsed');
  }

  setCollapsed(on: boolean): void {
    this.root.classList.toggle('collapsed', on);
    const t = $(this.root, '.mv-toggle');
    t.textContent = on ? '‹' : '›';
    t.setAttribute('aria-label', on ? 'Show moves' : 'Hide moves');
    t.setAttribute('aria-expanded', String(!on));
    saveCollapsed(on);
  }

  /** Updates the sidebar; `now` is the latest observation time (the clock of PlyInfo.correctedAt). */
  render(s: GameSnapshot, now: number): void {
    const key = renderKey(s);
    if (key === this.key) return;
    this.key = key;
    this.status.textContent = statusText(s);
    this.root.dataset.state = s.state;
    this.desync.hidden = s.state !== 'desync';
    this.copyBtn.disabled = s.plies.length === 0;
    this.undoBtn.disabled = s.plies.length === 0;
    this.newBtn.disabled = s.state === 'waiting';
    this.closePromo();

    const last = s.plies.length - 1;
    const plyEl = (i: number) => {
      const p = s.plies[i]!;
      const el = document.createElement('span');
      el.className = 'mv-ply';
      el.dataset.ply = String(i);
      el.textContent = p.san;
      if (i === last) el.classList.add('last');
      if (p.tentative) el.classList.add('tentative');
      if (p.promotion) {
        el.classList.add('promo');
        el.setAttribute('role', 'button');
        el.title = 'Change promotion piece';
      }
      const age = flashAge(p, now);
      if (age !== null) {
        el.classList.add('flash');
        // Continue the animation where it is when the list is rebuilt mid-flash.
        el.style.animationDelay = `-${Math.round(age)}ms`;
      }
      return el;
    };
    const rows = movePairs(s.plies).map((m) => {
      const row = document.createElement('div');
      row.className = 'mv-row';
      const no = document.createElement('span');
      no.className = 'mv-no';
      no.textContent = `${m.no}.`;
      row.append(no, plyEl(m.white));
      if (m.black !== undefined) row.append(plyEl(m.black));
      return row;
    });
    if (s.state === 'over' && s.result !== '*') {
      const res = document.createElement('div');
      res.className = 'mv-row mv-result';
      res.textContent = s.result;
      rows.push(res);
    }
    this.list.replaceChildren(...rows);
    this.list.scrollTop = this.list.scrollHeight;
  }

  /** Shows the PGN selected in a text box (clipboard fallback). */
  showPgn(pgn: string): void {
    if (this.collapsed) this.setCollapsed(false);
    this.pgnBox.value = pgn;
    this.pgnBox.hidden = false;
    this.pgnBox.focus();
    this.pgnBox.select();
  }

  private openPromo(ply: number, anchor: HTMLElement): void {
    this.promoPly = ply;
    this.promo.style.top = `${anchor.offsetTop + anchor.offsetHeight - this.list.scrollTop}px`;
    this.promo.hidden = false;
  }

  private closePromo(): void {
    this.promoPly = -1;
    this.promo.hidden = true;
  }
}
