// Where every tensor sits in space, and how every cell was computed.
// No rendering here: viz.js draws these specs, the inspector shows explainCell().
//
// The model is laid out as a row of SECTIONS along x. Each section can be open
// (every tensor shown) or closed (a labelled box plus its output tensor).
// Tensor positions are local to their section; viz.js slides sections along x
// as they open and close.

import { TOKENS, VOCAB, IDS, T, V, D, H, DK, DFF, TMAX, NB, EPS, HEAD_INFO, phi } from './model.js';

// Heads are stacked along the depth (z) axis, head 0 nearest the default camera.
export const HEAD_GAP = 16;
export const zOf = (h) => (1 - h) * HEAD_GAP;
const HZ = Array.from({ length: H }, (_, h) => zOf(h));

// Step numbers (see steps.js). Block 1 is walked through step by step;
// block 2 is shown complete, one step per sub-layer.
export const STEP = {
  MAP: 0, EMBED: 1, POS: 2,
  LN1: 3, PROJ: 4, SPLIT: 5, SCORE: 6, SCALE: 7, MASK: 8, SOFTMAX: 9, MIX: 10, CONCAT: 11, WO: 12, ADD1: 13,
  LN2: 14, MLP1: 15, GELU: 16, MLP2: 17, ADD2: 18,
  B2ATT: 19, B2MLP: 20,
  LNF: 21, UNEMBED: 22, PROBS: 23, ALL: 24,
};
export const NSTEPS = 25;
// The step at which each part of block b appears.
const ATT_KEYS = ['LN1', 'PROJ', 'SPLIT', 'SCORE', 'SCALE', 'MASK', 'SOFTMAX', 'MIX', 'CONCAT', 'WO', 'ADD1'];
const MLP_KEYS = ['LN2', 'MLP1', 'GELU', 'MLP2', 'ADD2'];
export const stepsFor = (b) =>
  b === 0
    ? STEP
    : Object.fromEntries([...ATT_KEYS.map((k) => [k, STEP.B2ATT]), ...MLP_KEYS.map((k) => [k, STEP.B2MLP])]);

// Inside an attention section, the x where attention's own input (the LayerNorm output) starts.
export const AXL = 14;

// ---------------------------------------------------------------------------
// Sections. Local x = 0 is the section's left edge; the previous section's
// output tensor always ends just before it, at local x = -12 .. -1.
//   wOpen / wClosed: width in each state      box: x extent of the closed box
//   keep: the tensor that stays visible when closed, at local x = keepX
// ---------------------------------------------------------------------------

export const SECTIONS = [];
SECTIONS.push({
  id: 'emb', kind: 'emb', title: 'EMBEDDING', lines: ['lookup + position', '(T,) → (T, d_{model})'],
  wOpen: 58, wClosed: 33, box: [6, 16], keep: 'X0', keepX: 21, big: 'X_{0}', from: STEP.EMBED, to: STEP.POS,
});
for (let b = 0; b < NB; b++) {
  const S = stepsFor(b);
  SECTIONS.push({
    id: `att${b}`, kind: 'att', block: b, title: 'ATTENTION',
    lines: [`LayerNorm → ${H} heads → add`, '(T, d_{model}) → (T, d_{model})'],
    wOpen: 240, wClosed: 35, box: [3, 13], keep: `b${b}.Xa`, keepX: 23, big: `X_{${2 * b + 1}}`, from: S.LN1, to: S.ADD1,
  });
  SECTIONS.push({
    id: `mlp${b}`, kind: 'mlp', block: b, title: 'MLP',
    lines: ['LayerNorm → MLP → add', '(T, d_{model}) → (T, d_{model})'],
    wOpen: 202, wClosed: 35, box: [3, 13], keep: `b${b}.Xm`, keepX: 23, big: `X_{${2 * b + 2}}`, from: S.LN2, to: S.ADD2,
  });
}
SECTIONS.push({
  id: 'out', kind: 'out', title: 'OUTPUT', lines: ['norm → unembed → softmax', '(T, d_{model}) → (T, V)'],
  wOpen: 98, wClosed: 34, box: [3, 13], keep: 'Pr', keepX: 18, big: 'p', from: STEP.LNF, to: STEP.PROBS,
});
// The section a step belongs to (-1: none, the whole model is collapsed).
export const sectionOfStep = (step) => SECTIONS.findIndex((s) => step >= s.from && step <= s.to);
// Every section that is open at a step: the one the step belongs to, plus the
// one before it during that section's first step. A finished section is left
// open for one extra step so that it folds only after the camera has moved on.
export const openSections = (step) => {
  const cur = sectionOfStep(step);
  if (cur < 0) return [];
  return cur > 0 && step === SECTIONS[cur - 1].to + 1 ? [cur - 1, cur] : [cur];
};

// ---------------------------------------------------------------------------
// Names. The residual stream is numbered along the model: X0 after embedding,
// then one more after every sub-layer. Block b reads X(2b) and leaves X(2b+2).
// ---------------------------------------------------------------------------

const digits = (n) => String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[d]);
const LAST = 2 * NB; // index of the stream after the last block
const inIdOf = (b) => (b === 0 ? 'X0' : `b${b - 1}.Xm`);

const tokenRows = TOKENS;
const sameCell = (id) => (d, r, c) => [id, d, r, c];

// spec.get(model) returns [depth][row][col]. x0 is local to the section.
export const SPECS = [];

// ---- Embedding ----
SPECS.push({
  id: 'WE', sec: 'emb', label: 'W_{E}', sym: '(V, d_{model})', note: 'token embedding table · one row per word',
  kind: 'param', group: 'emb', step: STEP.EMBED, x0: 0, cy: 10, zs: [0],
  rowLabels: VOCAB.map((w, id) => ({ text: `${id}  ${w}`, dim: !IDS.includes(id) })),
  colBrackets: 'dmodel',
  get: (m) => [m.WE],
});
SPECS.push({
  id: 'E', sec: 'emb', label: 'E', sym: '(T, d_{model})', note: 'one row looked up per token',
  kind: 'act', group: 'emb', step: STEP.EMBED, x0: 24, cy: 5, zs: [0],
  rowLabels: TOKENS.map((t, i) => `${t} · id ${IDS[i]}`),
  from: (d, r, c) => ['WE', 0, IDS[r], c],
  get: (m) => [m.E],
});
SPECS.push({
  id: 'WP', sec: 'emb', label: 'W_{P}', sym: '(T_{max}, d_{model})', note: 'position embedding table · one row per position',
  kind: 'param', group: 'pos', step: STEP.POS, x0: 0, cy: -9, zs: [0],
  rowLabels: Array.from({ length: TMAX }, (_, i) => ({ text: `pos ${i}`, dim: i >= T })),
  colBrackets: 'dmodel',
  get: (m) => [m.WP],
});
SPECS.push({
  id: 'P', sec: 'emb', label: 'P', sym: '(T, d_{model})', note: 'the first T rows',
  kind: 'act', group: 'pos', step: STEP.POS, x0: 24, cy: -5, zs: [0],
  rowLabels: TOKENS.map((_, i) => `pos ${i}`), title: 'below',
  from: (d, r, c) => ['WP', 0, r, c],
  get: (m) => [m.P],
});
SPECS.push({
  id: 'X0', sec: 'emb', label: 'X_{0} = E + P', sym: '(T, d_{model})', note: 'the residual stream starts here',
  kind: 'act', group: 'resid', step: STEP.POS, x0: 46, cy: 0, zs: [0],
  rowLabels: tokenRows, colBrackets: 'dmodel',
  get: (m) => [m.X0],
});

// ---- LayerNorm: row statistics, gain, normalised copy ----
// LNS maps each of its four tensor ids to what the explanation needs.
const LNS = new Map();
function layerNormSpecs({ sec, pre, key, inId, inK, outK, step, get }) {
  const ids = { mu: `${pre}mu${key}`, sd: `${pre}sd${key}`, g: `${pre}g${key}`, N: `${pre}N${key}` };
  const meta = { ids, inId, inPlain: `X${digits(inK)}`, outPlain: `N${typeof outK === 'number' ? digits(outK) : outK}`, get };
  for (const [part, id] of Object.entries(ids)) LNS.set(id, { ...meta, part });
  SPECS.push({
    id: ids.mu, sec, label: 'μ', sym: '(T, 1)', note: 'mean of each row',
    kind: 'act', group: 'resid', step, x0: 3, cy: 0, zs: [0],
    get: (m) => [get(m).mu],
  });
  SPECS.push({
    id: ids.sd, sec, label: 'σ', sym: '(T, 1)', note: 'spread of each row',
    kind: 'act', group: 'resid', step, x0: 6, cy: 0, zs: [0], title: 'below',
    get: (m) => [get(m).sd],
  });
  SPECS.push({
    id: ids.g, sec, label: `γ_{${outK}}`, sym: '(d_{model},)', num: `(${D},)`, note: 'gain · one per column',
    kind: 'param', group: 'gain', step, x0: AXL, cy: 4.5, zs: [0],
    get: (m) => [[get(m).g]],
  });
  SPECS.push({
    id: ids.N, sec, label: `N_{${outK}} = LayerNorm(X_{${inK}})`, sym: '(T, d_{model})',
    kind: 'act', group: 'norm', step, x0: AXL, cy: 0, zs: [0],
    rowLabels: tokenRows, colBrackets: 'dmodel', title: 'below',
    get: (m) => [get(m).N],
  });
}

// ---- One block: attention section, then MLP section ----
for (let b = 0; b < NB; b++) {
  const S = stepsFor(b);
  const pre = `b${b}.`;
  const B = (m) => m.b[b];
  const att = `att${b}`;
  const mlp = `mlp${b}`;
  const [kIn, kA, kM] = [2 * b, 2 * b + 1, 2 * b + 2];
  const g = (name) => `${name}@${b}`; // colour scales are per block

  layerNormSpecs({
    sec: att, pre, key: 'A', inId: inIdOf(b), inK: kIn, outK: kA, step: S.LN1,
    get: (m) => ({ X: B(m).in, mu: B(m).muA, sd: B(m).sdA, g: B(m).gA, N: B(m).NA }),
  });

  for (const [n, cy] of [['K', 18], ['Q', 0], ['V', -18]]) {
    SPECS.push({
      id: `${pre}W${n}`, sec: att, label: `W_{${n}}`, sym: '(d_{model}, d_{model})', note: 'weights (random here)',
      kind: 'param', group: g('W'), step: S.PROJ, x0: AXL + 20, cy, zs: [0],
      rowBrackets: 'dmodel', colBrackets: 'heads',
      get: (m) => [B(m)['W' + n]],
    });
    SPECS.push({
      id: pre + n, sec: att, label: `${n} = N_{${kA}} W_{${n}}`, sym: '(T, d_{model})',
      kind: 'act', group: g(n), step: S.PROJ, x0: AXL + 40, cy, zs: [0],
      rowLabels: tokenRows, colBrackets: 'heads',
      get: (m) => [B(m)[n]],
    });
  }

  const split = (n, x0, cy, extra = {}) =>
    SPECS.push({
      id: `${pre}${n}h`, sec: att, label: n, sym: '(h, T, d_{k})', note: 'one slab per head',
      kind: 'act', group: g(n), step: S.SPLIT, x0: AXL + x0, cy, zs: HZ,
      rowLabels: tokenRows, from: (d, r, c) => [pre + n, 0, r, d * DK + c], get: (m) => B(m)[n + 'h'], ...extra,
    });
  split('K', 73, 18);
  split('Q', 60, 0);
  split('V', 132, -8, { title: 'below' });

  SPECS.push({
    id: `${pre}KT`, sec: att, label: 'K^{T}', sym: '(h, d_{k}, T)',
    kind: 'act', group: g('K'), step: S.SCORE, x0: AXL + 72, cy: 7, zs: HZ,
    colLabels: tokenRows, from: (d, r, c) => [`${pre}Kh`, d, c, r],
    get: (m) => B(m).KT,
  });

  const scoreLike = (name, label, step, x0, extra) =>
    SPECS.push({
      id: pre + name, sec: att, label, sym: '(h, T, T)', kind: 'score', group: g('score'), step, x0: AXL + x0, cy: 0, zs: HZ,
      rowLabels: tokenRows, colLabels: tokenRows, title: 'below', get: (m) => B(m)[name], ...extra,
    });
  scoreLike('S', 'S = Q K^{T}', S.SCORE, 72, { note: 'row = query token i · column = key token j' });
  scoreLike('Ss', 'S / √d_{k}', S.SCALE, 86, { from: sameCell(`${pre}S`) });
  SPECS.push({
    id: `${pre}M`, sec: att, label: 'M  (mask)', sym: '(T, T)', note: 'no head axis: broadcast to every head',
    kind: 'mask', step: S.MASK, x0: AXL + 100, cy: 0, zs: [zOf(0) + HEAD_GAP],
    rowLabels: tokenRows, title: 'below',
    get: (m) => [m.M],
  });
  scoreLike('Sm', 'S / √d_{k} + M', S.MASK, 100, { from: sameCell(`${pre}Ss`) });
  scoreLike('A', 'A = softmax(…)', S.SOFTMAX, 114, {
    kind: 'attn', group: undefined, note: 'attention weights · every row sums to 1', from: sameCell(`${pre}Sm`),
  });

  SPECS.push({
    id: `${pre}Z`, sec: att, label: 'Z = A V', sym: '(h, T, d_{k})',
    kind: 'act', group: g('Z'), step: S.MIX, x0: AXL + 132, cy: 0, zs: HZ,
    rowLabels: tokenRows,
    get: (m) => B(m).Z,
  });
  SPECS.push({
    id: `${pre}Zc`, sec: att, label: 'Z  heads concatenated', sym: '(T, h·d_{k})', note: 'h·d_{k} = d_{model}',
    kind: 'act', group: g('Z'), step: S.CONCAT, x0: AXL + 148, cy: 0, zs: [0],
    rowLabels: tokenRows, colBrackets: 'heads',
    from: (d, r, c) => [`${pre}Z`, Math.floor(c / DK), r, c % DK],
    get: (m) => [B(m).Zc],
  });
  SPECS.push({
    id: `${pre}WO`, sec: att, label: 'W_{O}', sym: '(d_{model}, d_{model})', note: 'weights (random here)',
    kind: 'param', step: S.WO, x0: AXL + 170, cy: 0, zs: [0],
    rowBrackets: 'heads', colBrackets: 'dmodel',
    get: (m) => [B(m).WO],
  });
  SPECS.push({
    id: `${pre}O`, sec: att, label: 'O = Z W_{O}', sym: '(T, d_{model})', note: "attention's output",
    kind: 'act', group: 'resid', step: S.WO, x0: AXL + 190, cy: 0, zs: [0],
    rowLabels: tokenRows, colBrackets: 'dmodel',
    get: (m) => [B(m).O],
  });
  SPECS.push({
    id: `${pre}Xa`, sec: att, label: `X_{${kA}} = X_{${kIn}} + O`, sym: '(T, d_{model})', note: 'the stream after attention',
    kind: 'act', group: 'resid', step: S.ADD1, x0: AXL + 214, cy: 0, zs: [0],
    rowLabels: tokenRows, colBrackets: 'dmodel', from: sameCell(`${pre}O`),
    get: (m) => [B(m).Xa],
  });

  layerNormSpecs({
    sec: mlp, pre, key: 'M', inId: `${pre}Xa`, inK: kA, outK: kM, step: S.LN2,
    get: (m) => ({ X: B(m).Xa, mu: B(m).muM, sd: B(m).sdM, g: B(m).gM, N: B(m).NM }),
  });
  SPECS.push({
    id: `${pre}W1`, sec: mlp, label: 'W_{1}', sym: '(d_{model}, 4·d_{model})', note: 'weights (random here)',
    kind: 'param', step: S.MLP1, x0: 34, cy: 0, zs: [0],
    rowBrackets: 'dmodel', colBrackets: 'dff',
    get: (m) => [B(m).W1],
  });
  SPECS.push({
    id: `${pre}H`, sec: mlp, label: `H = N_{${kM}} W_{1}`, sym: '(T, 4·d_{model})', note: 'four times wider',
    kind: 'act', group: g('mlp'), step: S.MLP1, x0: 90, cy: 5, zs: [0],
    rowLabels: tokenRows,
    get: (m) => [B(m).H],
  });
  SPECS.push({
    id: `${pre}G`, sec: mlp, label: 'G = GELU(H)', sym: '(T, 4·d_{model})',
    kind: 'act', group: g('mlp'), step: S.GELU, x0: 90, cy: -5, zs: [0],
    rowLabels: tokenRows, colBrackets: 'dff', title: 'below', from: sameCell(`${pre}H`),
    get: (m) => [B(m).G],
  });
  SPECS.push({
    id: `${pre}W2`, sec: mlp, label: 'W_{2}', sym: '(4·d_{model}, d_{model})', note: 'weights (random here)',
    kind: 'param', step: S.MLP2, x0: 146, cy: 0, zs: [0],
    rowBrackets: 'dff', colBrackets: 'dmodel',
    get: (m) => [B(m).W2],
  });
  SPECS.push({
    id: `${pre}F`, sec: mlp, label: 'F = G W_{2}', sym: '(T, d_{model})', note: "the MLP's output",
    kind: 'act', group: 'resid', step: S.MLP2, x0: 166, cy: 0, zs: [0],
    rowLabels: tokenRows, colBrackets: 'dmodel',
    get: (m) => [B(m).F],
  });
  SPECS.push({
    id: `${pre}Xm`, sec: mlp, label: `X_{${kM}} = X_{${kA}} + F`, sym: '(T, d_{model})',
    note: `the stream after block ${b + 1}`,
    kind: 'act', group: 'resid', step: S.ADD2, x0: 190, cy: 0, zs: [0],
    rowLabels: tokenRows, colBrackets: 'dmodel', from: sameCell(`${pre}F`),
    get: (m) => [B(m).Xm],
  });
}

// ---- Output ----
layerNormSpecs({
  sec: 'out', pre: '', key: 'F', inId: inIdOf(NB), inK: LAST, outK: 'f', step: STEP.LNF,
  get: (m) => ({ X: m.Xlast, mu: m.muF, sd: m.sdF, g: m.gF, N: m.NF }),
});
SPECS.push({
  id: 'WET', sec: 'out', label: 'W_{E}^{T}', sym: '(d_{model}, V)', note: 'the embedding table again, transposed',
  kind: 'param', group: 'emb', step: STEP.UNEMBED, x0: 34, cy: 0, zs: [0],
  colLabels: VOCAB, rowBrackets: 'dmodel',
  from: (d, r, c) => ['WE', 0, c, r],
  get: (m) => [m.WET],
});
SPECS.push({
  id: 'L', sec: 'out', label: 'logits = N_{f} W_{E}^{T}', sym: '(T, V)', note: 'one score per vocabulary word',
  kind: 'act', step: STEP.UNEMBED, x0: 58, cy: 0, zs: [0],
  rowLabels: tokenRows, colLabels: VOCAB, title: 'below',
  get: (m) => [m.L],
});
SPECS.push({
  id: 'Pr', sec: 'out', label: 'p = softmax(logits)', sym: '(T, V)', note: 'next-token probabilities · every row sums to 1',
  kind: 'prob', step: STEP.PROBS, x0: 82, cy: 0, zs: [0],
  rowLabels: tokenRows, colLabels: VOCAB, title: 'below', from: sameCell('L'),
  get: (m) => [m.Pr],
});

export const SPEC_BY_ID = new Map(SPECS.map((s) => [s.id, s]));
// Every tensor id belonging to a section, in order (used to frame a whole section).
export const idsOfSection = (secId) => SPECS.filter((s) => s.sec === secId).map((s) => s.id);

// ---------------------------------------------------------------------------
// Explanations
// ---------------------------------------------------------------------------

export const markupToHtml = (s) =>
  s.replace(/_\{([^}]*)\}/g, '<sub>$1</sub>').replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>');

export function fmt(v, digitsN = 2) {
  if (v === -Infinity) return '−∞';
  if (v === Infinity) return '∞';
  let s = v.toFixed(digitsN);
  if (parseFloat(s) === 0) s = (0).toFixed(digitsN);
  return s.replace('-', '−');
}
// A number in brackets when it is negative, for use after an operator.
export const par = (v) => (fmt(v).startsWith('−') ? `(${fmt(v)})` : fmt(v));

const tk = (i) => `<span class="tok">${TOKENS[i]}</span>`;
const wd = (v) => `<span class="tok">${VOCAB[v]}</span>`;
const hd = (h) => `<span class="hd" style="color:${HEAD_INFO[h].color}">head ${h}</span>`;
const grp = (id, d, r0, r1, c0, c1) => ({ id, d, r0, r1, c0, c1 });
const one = (id, d, r, c) => grp(id, d, r, r, c, c);
const RANDOM = 'Learned in a real model; random here.';

function table(head, rows, foot) {
  const th = head.map((h) => `<th>${h}</th>`).join('');
  const tr = rows
    .map((r) => {
      const cls = [r.faint ? 'faint' : '', r.cur ? 'cur' : ''].join(' ').trim();
      return `<tr${cls ? ` class="${cls}"` : ''}>${r.cells.map((c) => `<td>${c}</td>`).join('')}</tr>`;
    })
    .join('');
  const tf = foot ? `<tfoot><tr>${foot.map((c) => `<td>${c}</td>`).join('')}</tr></tfoot>` : '';
  return `<table class="terms"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody>${tf}</table>`;
}

const faint = (p) => Math.abs(p) < 0.005;

// The terms of a dot product a · b as table rows. With more than `max` terms,
// only the largest are listed and the rest are folded into one row.
function dotRows(a, b, name, max = 12) {
  const terms = a.map((x, k) => ({ k, x, w: b[k], p: x * b[k] }));
  const row = (t) => ({ cells: [name(t.k), fmt(t.x), fmt(t.w), fmt(t.p)], faint: faint(t.p) });
  if (terms.length <= max) return terms.map(row);
  const sorted = [...terms].sort((s, t) => Math.abs(t.p) - Math.abs(s.p));
  const top = sorted.slice(0, max - 2).sort((s, t) => s.k - t.k);
  const rest = sorted.slice(max - 2).reduce((s, t) => s + t.p, 0);
  return [...top.map(row), { cells: [`the other ${terms.length - top.length}`, '', '', fmt(rest)], faint: true }];
}

// Returns { id, d, r, c, value, title, html, groups, context }.
// groups: exactly the cells (in other tensors) this cell was computed from.
// context: regions outlined for orientation only, such as the row a copied
// number belongs to. They are not inputs to this cell.
export function explainCell(m, id, d, r, c) {
  const spec = SPEC_BY_ID.get(id);
  const v = spec.get(m)[d][r][c];
  const out = { id, d, r, c, value: v, title: '', html: '', groups: [], context: [] };

  // ---- LayerNorm tensors share one explanation each ----
  if (LNS.has(id)) {
    const ln = LNS.get(id);
    const { X, mu, sd, g } = ln.get(m);
    const inHtml = `X<sub>${ln.inPlain.slice(1)}</sub>`;
    if (ln.part === 'mu') {
      out.title = `μ[${TOKENS[r]}]`;
      out.html = `<p>The average of the ${D} numbers in row ${tk(r)} of ${ln.inPlain}.</p>
        <p class="eq">(${X[r].map((x, i) => (i ? par(x) : fmt(x))).join(' + ')}) / ${D} = ${fmt(v)}</p>`;
      out.groups = [grp(ln.inId, 0, r, r, 0, D - 1)];
    } else if (ln.part === 'sd') {
      const varr = X[r].reduce((s, x) => s + (x - mu[r][0]) ** 2, 0) / D;
      out.title = `σ[${TOKENS[r]}]`;
      out.html = `<p>How spread out row ${tk(r)} of ${ln.inPlain} is around its mean ${fmt(mu[r][0])}:
        the square root of the average squared distance from the mean.</p>
        <p class="eq">√(${fmt(varr, 3)} + ε) = ${fmt(v, 3)}</p>
        <p>ε = ${EPS} only guards against dividing by zero.</p>`;
      out.groups = [grp(ln.inId, 0, r, r, 0, D - 1), one(ln.ids.mu, 0, r, 0)];
    } else if (ln.part === 'g') {
      out.title = `γ[${c}]`;
      out.html = `<p>Gain for dimension ${c}: after a row is normalised, its number in this column is multiplied by ${fmt(v)}.
        The same ${D} gains are used for every token.</p>
        <p>A real model starts every gain at exactly 1 and training moves them. Here they are random values near 1.</p>`;
      out.context = [grp(ln.ids.g, 0, 0, 0, 0, D - 1)];
    } else {
      out.title = `${ln.outPlain}[${TOKENS[r]}, ${c}]`;
      out.html = `<p>Dimension ${c} of ${tk(r)} in ${inHtml}, normalised: subtract the row's mean, divide by the row's spread, multiply by the column's gain.</p>
        <p class="eq">(x − μ) / σ × γ</p>
        <p class="eq">(${fmt(X[r][c])} − ${par(mu[r][0])}) / ${fmt(sd[r][0])} × ${fmt(g[c])} = ${fmt(v)}</p>
        <p>Only numbers from this token's own row are used: tokens do not interact here.</p>`;
      out.groups = [one(ln.inId, 0, r, c), one(ln.ids.mu, 0, r, 0), one(ln.ids.sd, 0, r, 0), one(ln.ids.g, 0, 0, c)];
    }
    return out;
  }

  // ---- Tensors inside a block ----
  const inBlock = /^b(\d+)\.(.+)$/.exec(id);
  if (inBlock) {
    const b = Number(inBlock[1]);
    const base = inBlock[2];
    const blk = m.b[b];
    const I = (x) => `b${b}.${x}`; // id of a tensor in this block
    const tag = `block ${b + 1} · `;
    const [kIn, kA, kM] = [2 * b, 2 * b + 1, 2 * b + 2];
    const N = (k) => `N<sub>${k}</sub>`;
    const X = (k) => `X<sub>${k}</sub>`;

    switch (base) {
      case 'WQ':
      case 'WK':
      case 'WV': {
        const n = base[1];
        const h = Math.floor(c / DK);
        out.title = `${tag}W${n}[${r}, ${c}]`;
        out.html = `<p>Weight: how much dimension ${r} of a token's vector contributes to column ${c} of ${n}
          — that column becomes ${hd(h)}'s dimension ${c % DK}.</p>
          <p>The outlined column is the vector every token row gets dotted with. ${RANDOM}</p>`;
        out.context = [grp(id, 0, 0, D - 1, c, c)];
        break;
      }
      case 'Q':
      case 'K':
      case 'V': {
        const W = blk['W' + base];
        const h = Math.floor(c / DK);
        out.title = `${tag}${base}[${TOKENS[r]}, ${c}]`;
        const rows = dotRows(blk.NA[r], W.map((w) => w[c]), (k) => `dim ${k}`);
        out.html = `<p>Row ${tk(r)} of ${N(kA)} dotted with column ${c} of W<sub>${base}</sub>.
          Column ${c} will become ${hd(h)}'s dimension ${c % DK}.</p>
          ${table(['input', N(kA), `W<sub>${base}</sub>`, 'product'], rows, ['', '', 'Σ', fmt(v)])}`;
        out.groups = [grp(I('NA'), 0, r, r, 0, D - 1), grp(I('W' + base), 0, 0, D - 1, c, c)];
        break;
      }
      case 'Qh':
      case 'Kh':
      case 'Vh': {
        const n = base[0];
        const flat = d * DK + c;
        out.title = `${tag}${n}[head ${d}][${TOKENS[r]}, ${c}]`;
        out.html = `<p>The same number as flat ${n}[${TOKENS[r]}, ${flat}] — a reshape moves no data.</p>
          <p>Flat column h·d<sub>k</sub> + c = ${d}·${DK} + ${c} = ${flat} is simply re-addressed as (${hd(d)}, dim ${c}).</p>`;
        out.groups = [one(I(n), 0, r, flat)];
        break;
      }
      case 'KT': {
        out.title = `${tag}Kᵀ[head ${d}][${r}, ${TOKENS[c]}]`;
        out.html = `<p>Transpose swaps the last two axes: K<sup>T</sup>[${r}, ${c}] = K[${c}, ${r}] inside ${hd(d)}.</p>
          <p>Keys now run along columns, so the key of ${tk(c)} sits directly above column ${tk(c)} of S.</p>`;
        out.groups = [one(I('Kh'), d, c, r)];
        break;
      }
      case 'S': {
        out.title = `${tag}S[head ${d}][${TOKENS[r]} → ${TOKENS[c]}]`;
        const rows = dotRows(blk.Qh[d][r], blk.Kh[d][c], (k) => `dim ${k}`);
        out.html = `<p>${hd(d)}: the <b>query</b> of ${tk(r)} dotted with the <b>key</b> of ${tk(c)}.</p>
          <p class="eq">S[i, j] = q<sub>i</sub> · k<sub>j</sub></p>
          ${table(['', `q (${TOKENS[r]})`, `k (${TOKENS[c]})`, 'q × k'], rows, ['', '', 'Σ', fmt(v)])}`;
        out.groups = [grp(I('Qh'), d, r, r, 0, DK - 1), grp(I('KT'), d, 0, DK - 1, c, c)];
        break;
      }
      case 'Ss': {
        out.title = `${tag}S/√d_k[head ${d}][${TOKENS[r]} → ${TOKENS[c]}]`;
        out.html = `<p class="eq">${fmt(blk.S[d][r][c])} ÷ √${DK} = ${fmt(blk.S[d][r][c])} ÷ ${fmt(m.scale, 0)} = ${fmt(v)}</p>
          <p>Same shape, every score divided by √d<sub>k</sub>.</p>`;
        out.groups = [one(I('S'), d, r, c)];
        break;
      }
      case 'M': {
        out.title = `M[${TOKENS[r]}, ${TOKENS[c]}]`;
        out.html =
          v === -Infinity
            ? `<p>${tk(c)} (position ${c}) comes <b>after</b> ${tk(r)} (position ${r}), so this pair is blocked.
               Adding −∞ makes its softmax weight exactly 0.</p>`
            : m.causal
              ? `<p>${tk(c)} (position ${c}) is not after ${tk(r)} (position ${r}): allowed. Adding 0 changes nothing.</p>`
              : `<p>Mask is off: every token may look at every token (encoder-style). Adding 0 changes nothing.</p>`;
        out.html += `<p>The mask depends only on positions, so it is the same in every block.</p>`;
        break;
      }
      case 'Sm': {
        out.title = `${tag}masked[head ${d}][${TOKENS[r]} → ${TOKENS[c]}]`;
        out.html = `<p class="eq">${fmt(blk.Ss[d][r][c])} + ${m.M[r][c] === 0 ? '0' : '(−∞)'} = ${fmt(v)}</p>
          <p>The mask has no head axis, so the same (T, T) mask is added to every head.</p>`;
        out.groups = [one(I('Ss'), d, r, c), one(I('M'), 0, r, c)];
        break;
      }
      case 'A': {
        const s = blk.Sm[d][r];
        out.title = `${tag}A[head ${d}][${TOKENS[r]} → ${TOKENS[c]}]`;
        const rows = s.map((x, j) => ({
          cells: [TOKENS[j], fmt(x), fmt(Math.exp(x), 3), fmt(blk.A[d][r][j])],
          faint: blk.A[d][r][j] < 0.005,
          cur: j === c,
        }));
        const sum = s.reduce((a, x) => a + Math.exp(x), 0);
        out.html = `<p>${hd(d)}: of everything ${tk(r)} takes in, <b>${Math.round(v * 100)}%</b> comes from ${tk(c)}.</p>
          <p class="eq">A[i, j] = e<sup>s<sub>ij</sub></sup> / Σ<sub>j′</sub> e<sup>s<sub>ij′</sub></sup></p>
          ${table(['key token j', 'score s', 'e<sup>s</sup>', 'weight'], rows, ['Σ', '', fmt(sum, 3), fmt(1)])}`;
        out.groups = [grp(I('Sm'), d, r, r, 0, T - 1)];
        out.context = [grp(I('A'), d, r, r, 0, T - 1)];
        break;
      }
      case 'Z': {
        out.title = `${tag}Z[head ${d}][${TOKENS[r]}, ${c}]`;
        const rows = dotRows(blk.A[d][r], blk.Vh[d].map((row) => row[c]), (j) => TOKENS[j]);
        out.html = `<p>${hd(d)}: dimension ${c} of the new vector for ${tk(r)} — a weighted average of every token's value, using row ${tk(r)} of A as the weights.</p>
          <p class="eq">z<sub>i</sub> = Σ<sub>j</sub> A[i, j] · v<sub>j</sub></p>
          ${table(['source token j', 'A[i, j]', `V[j, ${c}]`, 'product'], rows, ['', '', 'Σ', fmt(v)])}`;
        out.groups = [grp(I('A'), d, r, r, 0, T - 1), grp(I('Vh'), d, 0, T - 1, c, c)];
        break;
      }
      case 'Zc': {
        const h = Math.floor(c / DK);
        out.title = `${tag}Z[${TOKENS[r]}, ${c}]`;
        out.html = `<p>The same number as Z[${hd(h)}][${TOKENS[r]}, ${c % DK}] — the head slabs are laid side by side again.</p>
          <p>Column h·d<sub>k</sub> + c = ${h}·${DK} + ${c % DK} = ${c}.</p>`;
        out.groups = [one(I('Z'), h, r, c % DK)];
        break;
      }
      case 'WO': {
        const h = Math.floor(r / DK);
        out.title = `${tag}WO[${r}, ${c}]`;
        out.html = `<p>Weight: how strongly ${hd(h)}'s output dimension ${r % DK} writes into dimension ${c} of the residual stream.</p>
          <p>The outlined row shows that this one head dimension writes into all ${D} dimensions. ${RANDOM}</p>`;
        out.context = [grp(id, 0, r, r, 0, D - 1)];
        break;
      }
      case 'O': {
        out.title = `${tag}O[${TOKENS[r]}, ${c}]`;
        const rows = dotRows(blk.Zc[r], blk.WO.map((w) => w[c]), (k) => `head ${Math.floor(k / DK)} · dim ${k % DK}`);
        out.html = `<p>What attention writes into dimension ${c} of ${tk(r)}:
          its concatenated row dotted with column ${c} of W<sub>O</sub>.</p>
          ${table(['from', 'Z', 'W<sub>O</sub>', 'product'], rows, ['', '', 'Σ', fmt(v)])}`;
        out.groups = [grp(I('Zc'), 0, r, r, 0, D - 1), grp(I('WO'), 0, 0, D - 1, c, c)];
        break;
      }
      case 'Xa': {
        out.title = `X${digits(kA)}[${TOKENS[r]}, ${c}]`;
        out.html = `<p>Dimension ${c} of ${tk(r)}: what the stream held before this sub-layer, plus what attention wrote.</p>
          <p class="eq">${fmt(blk.in[r][c])} + ${par(blk.O[r][c])} = ${fmt(v)}</p>
          <p>The first term is ${X(kIn)}, from <i>before</i> LayerNorm. LayerNorm only prepared attention's input; it did not change the stream.</p>`;
        out.groups = [one(inIdOf(b), 0, r, c), one(I('O'), 0, r, c)];
        break;
      }
      case 'W1': {
        out.title = `${tag}W1[${r}, ${c}]`;
        out.html = `<p>Weight: how much dimension ${r} of a token's vector contributes to hidden unit ${c}.</p>
          <p>The outlined column is the vector that hidden unit dots with every token row. ${RANDOM}</p>`;
        out.context = [grp(id, 0, 0, D - 1, c, c)];
        break;
      }
      case 'H': {
        out.title = `${tag}H[${TOKENS[r]}, ${c}]`;
        const rows = dotRows(blk.NM[r], blk.W1.map((w) => w[c]), (k) => `dim ${k}`);
        out.html = `<p>Hidden unit ${c} for ${tk(r)}: its row of ${N(kM)} dotted with column ${c} of W<sub>1</sub>.</p>
          ${table(['input', N(kM), 'W<sub>1</sub>', 'product'], rows, ['', '', 'Σ', fmt(v)])}`;
        out.groups = [grp(I('NM'), 0, r, r, 0, D - 1), grp(I('W1'), 0, 0, D - 1, c, c)];
        break;
      }
      case 'G': {
        const h = blk.H[r][c];
        out.title = `${tag}G[${TOKENS[r]}, ${c}]`;
        out.html = `<p>GELU applied to the one number directly above it. Nothing else is involved.</p>
          <p class="eq">GELU(h) = h × Φ(h)</p>
          <p class="eq">${fmt(h)} × ${fmt(phi(h), 3)} = ${fmt(v)}</p>
          <p>Φ(h) is the share of a standard bell curve lying below h: close to 1 for large positive h (the number passes through),
          close to 0 for large negative h (the number is squashed to about 0).</p>`;
        out.groups = [one(I('H'), 0, r, c)];
        break;
      }
      case 'W2': {
        out.title = `${tag}W2[${r}, ${c}]`;
        out.html = `<p>Weight: how strongly hidden unit ${r} writes into dimension ${c} of the residual stream.</p>
          <p>The outlined row shows that this one hidden unit writes into all ${D} dimensions. ${RANDOM}</p>`;
        out.context = [grp(id, 0, r, r, 0, D - 1)];
        break;
      }
      case 'F': {
        out.title = `${tag}F[${TOKENS[r]}, ${c}]`;
        const rows = dotRows(blk.G[r], blk.W2.map((w) => w[c]), (k) => `unit ${k}`);
        out.html = `<p>What the MLP writes into dimension ${c} of ${tk(r)}: its ${DFF} hidden values dotted with column ${c} of W<sub>2</sub>.
          The largest of the ${DFF} products:</p>
          ${table(['hidden', 'G', 'W<sub>2</sub>', 'product'], rows, ['', '', 'Σ', fmt(v)])}`;
        out.groups = [grp(I('G'), 0, r, r, 0, DFF - 1), grp(I('W2'), 0, 0, DFF - 1, c, c)];
        break;
      }
      case 'Xm': {
        out.title = `X${digits(kM)}[${TOKENS[r]}, ${c}]`;
        out.html = `<p>Dimension ${c} of ${tk(r)}: the stream after attention, plus what the MLP wrote.</p>
          <p class="eq">${fmt(blk.Xa[r][c])} + ${par(blk.F[r][c])} = ${fmt(v)}</p>
          <p>This is the output of block ${b + 1}${b + 1 < NB ? ` and the input of block ${b + 2}` : ', the last block'}.</p>`;
        out.groups = [one(I('Xa'), 0, r, c), one(I('F'), 0, r, c)];
        break;
      }
    }
    return out;
  }

  switch (id) {
    // ---- embedding ----
    case 'WE': {
      const at = IDS.map((x, i) => (x === r ? i : -1)).filter((i) => i >= 0);
      out.title = `WE[${VOCAB[r]}, ${c}]`;
      out.html = `<p>Row ${r} of the embedding table is the vector for the word ${wd(r)} (token id ${r}).
        ${at.length ? `This sentence uses it at position ${at.join(', ')}.` : 'This sentence does not use it.'}</p>
        <p>${RANDOM}</p>`;
      out.context = [grp('WE', 0, r, r, 0, D - 1)];
      break;
    }
    case 'E': {
      out.title = `E[${TOKENS[r]}, ${c}]`;
      out.html = `<p>${tk(r)} has token id ${IDS[r]}, so its vector is row ${IDS[r]} of W<sub>E</sub>, copied as it is.
        This number is column ${c} of that row: W<sub>E</sub>[${IDS[r]}, ${c}].</p>
        <p>A lookup, not arithmetic: nothing is multiplied or added. The thin outlines mark the row being copied;
        only the one bright cube is the source of this number.</p>`;
      out.groups = [one('WE', 0, IDS[r], c)];
      out.context = [grp('WE', 0, IDS[r], IDS[r], 0, D - 1), grp('E', 0, r, r, 0, D - 1)];
      break;
    }
    case 'WP': {
      out.title = `WP[pos ${r}, ${c}]`;
      out.html = `<p>Row ${r} of the position table is the vector for position ${r}, whatever token sits there.
        ${r < T ? `Here that is ${tk(r)}.` : `Unused: this sentence has only ${T} tokens. The table has T<sub>max</sub> = ${TMAX} rows, the longest input the model accepts.`}</p>
        <p>${RANDOM}</p>`;
      out.context = [grp('WP', 0, r, r, 0, D - 1)];
      break;
    }
    case 'P': {
      out.title = `P[pos ${r}, ${c}]`;
      out.html = `<p>Position ${r} takes row ${r} of W<sub>P</sub>, copied as it is.
        This number is column ${c} of that row: W<sub>P</sub>[${r}, ${c}].</p>
        <p>The thin outlines mark the row being copied; only the one bright cube is the source of this number.</p>`;
      out.groups = [one('WP', 0, r, c)];
      out.context = [grp('WP', 0, r, r, 0, D - 1), grp('P', 0, r, r, 0, D - 1)];
      break;
    }
    case 'X0': {
      out.title = `X₀[${TOKENS[r]}, ${c}]`;
      out.html = `<p>Dimension ${c} of the vector for ${tk(r)} at position ${r}: its token embedding plus its position embedding.</p>
        <p class="eq">${fmt(m.E[r][c])} + ${par(m.P[r][c])} = ${fmt(v)}</p>
        <p>The two are added, so token and position share the same ${D} dimensions.</p>`;
      out.groups = [one('E', 0, r, c), one('P', 0, r, c)];
      break;
    }

    // ---- output ----
    case 'WET': {
      out.title = `WEᵀ[${r}, ${VOCAB[c]}]`;
      out.html = `<p>The same number as W<sub>E</sub>[${VOCAB[c]}, ${r}]. The output layer reuses the embedding table, turned on its side,
        so column ${wd(c)} here is the embedding of ${wd(c)}.</p>
        <p>This sharing is called weight tying. GPT-2 does it; some models keep a separate output matrix.</p>`;
      out.groups = [one('WE', 0, c, r)];
      break;
    }
    case 'L': {
      out.title = `logits[${TOKENS[r]}, ${VOCAB[c]}]`;
      const rows = dotRows(m.NF[r], m.WE[c], (k) => `dim ${k}`);
      out.html = `<p>Score for ${wd(c)} as the token that follows ${tk(r)}: the final vector at that position dotted with the embedding of ${wd(c)}.</p>
        ${table(['', 'N<sub>f</sub>', `W<sub>E</sub> (${VOCAB[c]})`, 'product'], rows, ['', '', 'Σ', fmt(v)])}`;
      out.groups = [grp('NF', 0, r, r, 0, D - 1), grp('WET', 0, 0, D - 1, c, c)];
      break;
    }
    case 'Pr': {
      const s = m.L[r];
      out.title = `p[${TOKENS[r]}, ${VOCAB[c]}]`;
      const rows = s.map((x, j) => ({
        cells: [VOCAB[j], fmt(x), fmt(Math.exp(x), 2), fmt(m.Pr[r][j])],
        faint: m.Pr[r][j] < 0.005,
        cur: j === c,
      }));
      const sum = s.reduce((a, x) => a + Math.exp(x), 0);
      out.html = `<p>The model's probability that ${wd(c)} comes after ${tk(r)}: <b>${Math.round(v * 100)}%</b>.
        Random weights, so the number is meaningless; the calculation is the real one.</p>
        ${table(['word', 'logit', 'e<sup>logit</sup>', 'p'], rows, ['Σ', '', fmt(sum, 2), fmt(1)])}`;
      out.groups = [grp('L', 0, r, r, 0, V - 1)];
      out.context = [grp('Pr', 0, r, r, 0, V - 1)];
      break;
    }
  }
  return out;
}
