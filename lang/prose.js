// Author-written prose, attached to steps and tensors.
//
//   say`...`       what a step does and why, about this calculation
//   aside`...`     a side note: what real or trained models do, history, how to use the page
//   hint`...`      the step's "Try it" line
//   X.says`...`    one sentence shown when a cube of X is hovered
//
// Prose is a tagged template. The typed text is HTML with _{sub} and ^{sup}
// markup, split into paragraphs at blank lines. Everything that names a tensor
// or quotes a number is an interpolation, so the compiler knows what the prose
// refers to and the numbers are read when the text is shown:
//
//   ${Q}                 a tensor: its symbol
//   ${d}  ${d.size}      an axis: its symbol; its size is an ordinary number
//   ${cell(X, 3, 0)}     the current value of one cell
//   ${pct(cell(...))}    ... as a percentage;  ${par(cell(...))}  in brackets when negative
//   ${label(T, 3)}       a label of an axis (a token, a word)
//   ${live(() => html)}  anything else computed when shown
//   ${picture(X)}        a (3, H, W) tensor as the colour picture it stands for
//   ${row} ${col} ${slab} ${value}   in X.says only: the hovered cell

export const row = { ph: 'row' };
export const col = { ph: 'col' };
export const slab = { ph: 'slab' };
export const value = { ph: 'value' };
export const cell = (t, ...addr) => ({ cell: t, addr });
export const pct = (of) => ({ pct: of });
export const par = (of) => ({ par: of });
export const label = (axis, i) => ({ label: axis, i });
export const live = (fn) => ({ live: fn });

// A (3, H, W) tensor of brightnesses between 0 and 1, shown as the colour
// picture it stands for: one square per pixel, read when the text is shown.
export const picture = (t, { size = 168 } = {}) => {
  if (t.rank !== 3 || t.dims[0] !== 3) throw new Error(`picture: ${t.name} must be (3, height, width)`);
  return {
    live: () => {
      const [, H, W] = t.dims;
      const level = (ch, i) => Math.round(255 * Math.min(1, Math.max(0, t.data[ch * H * W + i])));
      let cells = '';
      for (let i = 0; i < H * W; i++)
        cells += `<rect x="${i % W}" y="${Math.floor(i / W)}" width="1" height="1" fill="rgb(${level(0, i)},${level(1, i)},${level(2, i)})"/>`;
      return `<svg class="picture" viewBox="0 0 ${W} ${H}" width="${size}" height="${Math.round((size * H) / W)}" shape-rendering="crispEdges" role="img" aria-label="the ${W} × ${H} picture" style="display:block;margin:2px 0;border-radius:3px">${cells}</svg>`;
    },
  };
};

const isTensor = (v) => !!v && typeof v === 'object' && 'shape' in v && 'data' in v;
const isAxis = (v) => !!v && typeof v === 'object' && 'sym' in v && 'size' in v && !('shape' in v);

export const markupToHtml = (s) => s.replace(/_\{([^}]*)\}/g, '<sub>$1</sub>').replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
function fmt(v, n = 2) {
  if (!Number.isFinite(v)) return v < 0 ? '−∞' : '∞';
  const s = Math.abs(v).toFixed(n);
  return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
}

export const prose = (kind, strings, values) => ({ kind, strings: [...strings], values });

// Tensors the prose names, directly or through one of their cells.
export function refs(item) {
  const out = [];
  const visit = (v) => {
    if (isTensor(v)) out.push(v);
    else if (v && typeof v === 'object') for (const k of ['cell', 'pct', 'par']) if (k in v) visit(v[k]);
  };
  item.values.forEach(visit);
  return out;
}

// Placeholders the prose uses ('row', 'col', 'slab', 'value').
export function placeholders(item) {
  const out = new Set();
  const visit = (v) => {
    if (!v || typeof v !== 'object') return;
    if (v.ph) out.add(v.ph);
    for (const k of ['pct', 'par']) if (k in v) visit(v[k]);
  };
  item.values.forEach(visit);
  return out;
}

// What should have been a reference, or belongs in an aside. Asides are
// exempt: they are where facts about other models and their sizes go.
const OUTSIDE = /\b(trained|training|real models?|GPT-?\d|BERT|in practice|the original (paper|transformer))\b/i;
export function lint(item) {
  if (item.kind === 'aside' || item.kind === 'hint') return [];
  const out = [];
  const text = item.strings.join(' ').replace(/<[^>]*>/g, ' ').replace(/[_^]\{[^}]*\}/g, ' ');
  // Digits that are part of a name (GPT-2, X2) are not data.
  for (const m of text.matchAll(/(?<![\w-])\d+(?:[.,]\d+)*%?/g)) {
    const n = Number(m[0].replace(/[,%]/g, ''));
    if (m[0].includes('.') || m[0].endsWith('%') || n >= 2)
      out.push(`the typed number "${m[0]}" should be a reference (an axis size, cell(...) or live(...)), so it cannot go stale`);
  }
  const claim = OUTSIDE.exec(text);
  if (claim) out.push(`"${claim[0]}" reads as a statement about real or trained models; put it in an aside`);
  return out;
}

// ctx (X.says only): { t, d, r, c, value }.  env: { slabColors }
export function render(item, env = {}, ctx = null) {
  const show = (v) => {
    if (v == null || v === false) return '';
    if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString('en-US') : fmt(v);
    if (typeof v === 'string') return markupToHtml(esc(v));
    if (isTensor(v)) return `<span class="ref">${markupToHtml(v.symbol)}</span>`;
    if (isAxis(v)) return markupToHtml(v.sym);
    if ('cell' in v) return fmt(number(v));
    if ('pct' in v) return `${Math.round(number(v.pct) * 100)}%`;
    if ('par' in v) return number(v.par) < 0 ? `(${fmt(number(v.par))})` : fmt(number(v.par));
    if ('label' in v) return `<span class="tok">${esc(String(v.label.labels[v.i]))}</span>`;
    if ('live' in v) {
      const r = v.live(ctx || {});
      return typeof r === 'number' ? show(r) : (r ?? '');
    }
    if (v.ph === 'value') return fmt(ctx.value);
    if (v.ph === 'slab') return `<span class="hd" style="color:${env.slabColors[ctx.d]}">${ctx.t.shape[0].unit || ctx.t.shape[0].sym} ${ctx.d}</span>`;
    if (v.ph === 'row' || v.ph === 'col') {
      const ax = ctx.t.shape.at(v.ph === 'row' ? -2 : -1);
      const i = v.ph === 'row' ? ctx.r : ctx.c;
      return ax.labels ? `<span class="tok">${esc(String(ax.labels[i]))}</span>` : String(i);
    }
    return '';
  };
  const number = (v) => {
    if (typeof v === 'number') return v;
    if (v.ph === 'value') return ctx.value;
    const t = v.cell;
    return t.data[v.addr.reduce((f, x, i) => f * t.dims[i] + x, 0)];
  };

  const text = item.strings.map((s, i) => markupToHtml(s) + (i < item.values.length ? show(item.values[i]) : '')).join('').trim();
  if (item.kind === 'hint') return text;
  const cls = item.kind === 'aside' ? ' class="aside"' : '';
  return text
    .split(/\n\s*\n/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => (/^<(ul|ol|table|h\d|p|div|pre)\b/.test(chunk) ? chunk : `<p${cls}>${chunk}</p>`))
    .join('');
}
