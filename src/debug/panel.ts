import { DEBUG_VIEWS, type DebugView, type Params } from '../worker/protocol';

export const RESOLUTIONS = [480, 640, 800] as const;

interface ParamEntry {
  name: string;
  min: number;
  max: number;
  step: number;
  value: number;
}

/** Registry of tunable numeric parameters; the current values are sent to the worker every frame. */
export class ParamRegistry {
  private entries = new Map<string, ParamEntry>();
  private listeners: Array<(e: ParamEntry) => void> = [];

  register(name: string, min: number, max: number, step: number, def: number): void {
    if (this.entries.has(name)) return;
    const entry = { name, min, max, step, value: def };
    this.entries.set(name, entry);
    this.listeners.forEach((l) => l(entry));
  }

  set(name: string, value: number): void {
    const e = this.entries.get(name);
    if (e) e.value = value;
  }

  values(): Params {
    const out: Params = {};
    for (const e of this.entries.values()) out[e.name] = e.value;
    return out;
  }

  all(): ParamEntry[] {
    return [...this.entries.values()];
  }

  onRegister(fn: (e: ParamEntry) => void): void {
    this.listeners.push(fn);
  }
}

export const registry = new ParamRegistry();
/** Register a slider: `registerParam(name, min, max, step, default)`. */
export const registerParam = registry.register.bind(registry);
export const getParams = registry.values.bind(registry);

export interface DebugStats {
  fps: number;
  detectionsPerSec: number;
  timings: Record<string, number>;
  confidence: number;
  mode: string;
}

const STYLE = `
.dbg-toggle{position:absolute;top:8px;right:8px;z-index:20;width:36px;height:36px;padding:0;border-radius:50%;background:rgba(0,0,0,.5);color:#fff;font-size:16px;border:1px solid rgba(255,255,255,.4)}
.dbg-panel{position:absolute;top:52px;right:8px;z-index:20;max-height:calc(100% - 64px);width:min(280px,calc(100% - 16px));overflow:auto;box-sizing:border-box;padding:10px;border-radius:10px;background:rgba(0,0,0,.72);color:#eee;font:12px/1.4 ui-monospace,monospace}
.dbg-panel label{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:4px 0}
.dbg-panel select,.dbg-panel input[type=range]{flex:1;min-width:0}
.dbg-panel pre{margin:6px 0;white-space:pre-wrap}
`;

export class DebugPanel {
  readonly toggleButton: HTMLButtonElement;
  readonly panel: HTMLDivElement;
  private stats: HTMLPreElement;
  private sliders: HTMLDivElement;
  private _view: DebugView = 'none';
  private _resolution = 640;
  private resolutionListeners: Array<(r: number) => void> = [];
  private viewListeners: Array<(v: DebugView) => void> = [];

  constructor(parent: HTMLElement, reg: ParamRegistry = registry) {
    const style = document.createElement('style');
    style.textContent = STYLE;
    document.head.append(style);

    this.toggleButton = document.createElement('button');
    this.toggleButton.className = 'dbg-toggle';
    this.toggleButton.type = 'button';
    this.toggleButton.textContent = '⚙';
    this.toggleButton.setAttribute('aria-label', 'Toggle debug panel');
    this.toggleButton.addEventListener('click', () => this.toggle());

    this.panel = document.createElement('div');
    this.panel.className = 'dbg-panel';
    this.panel.hidden = true;

    this.stats = document.createElement('pre');
    this.panel.append(this.stats);

    const viewSel = this.select('View', DEBUG_VIEWS, this._view, (v) => {
      this._view = v as DebugView;
      this.viewListeners.forEach((l) => l(this._view));
    });
    const resSel = this.select('Resolution', RESOLUTIONS.map(String), String(this._resolution), (v) => {
      this._resolution = Number(v);
      this.resolutionListeners.forEach((l) => l(this._resolution));
    });
    this.sliders = document.createElement('div');
    this.panel.append(viewSel, resSel, this.sliders);

    reg.all().forEach((e) => this.addSlider(reg, e));
    reg.onRegister((e) => this.addSlider(reg, e));

    parent.append(this.toggleButton, this.panel);

    window.addEventListener(
      'touchstart',
      (ev) => {
        if (ev.touches.length === 3) this.toggle();
      },
      { passive: true },
    );
  }

  get view(): DebugView {
    return this._view;
  }

  get resolution(): number {
    return this._resolution;
  }

  get visible(): boolean {
    return !this.panel.hidden;
  }

  toggle(): void {
    this.panel.hidden = !this.panel.hidden;
  }

  onResolutionChange(fn: (r: number) => void): void {
    this.resolutionListeners.push(fn);
  }

  onViewChange(fn: (v: DebugView) => void): void {
    this.viewListeners.push(fn);
  }

  update(s: DebugStats): void {
    if (this.panel.hidden) return;
    const timings = Object.entries(s.timings)
      .map(([k, v]) => `  ${k.padEnd(14)}${v.toFixed(1)} ms`)
      .join('\n');
    this.stats.textContent =
      `render  ${s.fps.toFixed(0)} fps\ndetect  ${s.detectionsPerSec.toFixed(1)} /s\n` +
      `mode    ${s.mode}\nconf    ${s.confidence.toFixed(2)}\n${timings}`;
  }

  private select(
    label: string,
    options: readonly string[],
    value: string,
    onChange: (v: string) => void,
  ): HTMLLabelElement {
    const wrap = document.createElement('label');
    wrap.append(label);
    const sel = document.createElement('select');
    for (const o of options) sel.append(new Option(o, o, false, o === value));
    sel.addEventListener('change', () => onChange(sel.value));
    wrap.append(sel);
    return wrap;
  }

  private addSlider(reg: ParamRegistry, e: ParamEntry): void {
    const wrap = document.createElement('label');
    const name = document.createElement('span');
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(e.min);
    input.max = String(e.max);
    input.step = String(e.step);
    input.value = String(e.value);
    const show = () => (name.textContent = `${e.name} ${e.value}`);
    show();
    input.addEventListener('input', () => {
      reg.set(e.name, Number(input.value));
      show();
    });
    wrap.append(name, input);
    this.sliders.append(wrap);
  }
}
