// compile(program) -> scene: everything the 3D runtime needs, derived from the
// program and its layout. Nothing here knows about any particular model.
//
//   scene.tensors    one spec per drawn tensor: position, labels, brackets, colours
//   scene.sections   widths open and folded, the folded box, arrows and operators
//   scene.steps      the tour: title, formula, shapes, text, code, camera view
//   scene.explain    what to say and light up for one cube
//   scene.set        flip a toggle and recompute

import { ONE } from './core.js';
import { layout, GEO, colsOf, rowsOf, slabsOf, isParamLike, isDeck } from './layout.js';
import { explain as explainCell, plain, fmt } from './explain.js';
import { render } from './expr.js';
import { render as renderProse } from './prose.js';

const SLAB_COLORS = ['#b79cff', '#3fd6c0', '#ff7d9c', '#ffd166', '#6cc5ff', '#9be564'];
const DIM = '#9aa7bf';
const SKIP = 0xffb454;
const SOFT = 0x9aa7bf;

export const markupToHtml = (s) => s.replace(/_\{([^}]*)\}/g, '<sub>$1</sub>').replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>');
const H = (t) => markupToHtml(t.symbol);
const simple = (t) => !t.symbol.includes(' ');
const unitOf = (ax) => ax.unit || ax.sym;
const sameDims = (a, b) => a.rank === b.rank && a.dims.every((d, i) => d === b.dims[i]);

// ---------------------------------------------------------------------------
// What an operation is called, on a title, on an arrow and in code.

// The right-hand side of "Q = ...", when it is short enough to put in a title.
function exprOf(t) {
  const op = t.op;
  if (!op) return null;
  const [a, b] = op.inputs;
  if (op.family === 'contract') return simple(a) && simple(b) ? `${a.symbol} ${op.name === 'conv2d' ? '∗ ' : ''}${b.symbol}` : null;
  if (op.name === 'add') return simple(a) && simple(b) ? `${a.symbol} + ${b.symbol}` : null;
  if (op.name === 'softmax') return `softmax(${simple(a) ? a.symbol : '…'})`;
  if (op.family === 'module') return op.inputs.every(simple) ? `${op.title}(${op.inputs.map((x) => x.symbol).join(', ')})` : null;
  if (op.family === 'map' && op.inputs.length === 1 && op.name !== 'map') return `${op.show || op.name.toUpperCase()}(${simple(a) ? a.symbol : '…'})`;
  return null;
}

// A formula with operand names replaced by tensor symbols and axis names by their symbols.
function formulaOf(t, prog) {
  const { op } = t;
  const axes = Object.fromEntries(prog.axes.map((a) => [a.name, a.sym]));
  return render(op.ast, (name) => {
    const k = op.keys.indexOf(name);
    return k >= 0 ? op.inputs[k].symbol : axes[name];
  });
}

// The text on the arrow leading to t.
function arrowLabel(t, prog, attached) {
  const { op } = t;
  if (op.name === 'softmax') return 'softmax';
  if (op.family !== 'map') return null;
  if (op.name === 'add') return attached.length ? `+ ${attached[0].symbol}` : null;
  if (op.name !== 'map') return op.show || op.name.toUpperCase();
  if (op.inputs.length !== 1) return null;
  const axes = Object.fromEntries(prog.axes.map((a) => [a.name, a.sym]));
  const ast = op.ast;
  if (ast.t === 'bin' && ast.a.t === 'var' && ast.a.name === op.keys[0]) {
    const sign = { '+': '+', '-': '−', '*': '×', '/': '÷', '^': '^' }[ast.op];
    return `${sign} ${render(ast.b, (n) => axes[n])}`;
  }
  return render(ast, (n) => axes[n]);
}

const pyName = (t) => t.id.replace(/^.*\./, '');
const small = (v) => (v !== 0 && Math.abs(v) < 1e-3 ? v.toExponential().replace('e-', 'e-') : String(v));
function pyExpr(ast, sub) {
  const go = (n) => {
    switch (n.t) {
      case 'num': return String(n.v);
      case 'group': return `(${go(n.a)})`;
      case 'var': return sub(n.name) ?? n.name;
      case 'neg': return `-${go(n.a)}`;
      case 'call':
        if (n.name === 'mask') return `torch.where(${go(n.args[0])} > 0, 0.0, float('-inf'))`;
        return `${{ sqrt: 'math.sqrt', gelu: 'F.gelu', relu: 'F.relu', exp: 'torch.exp', tanh: 'torch.tanh', sigmoid: 'torch.sigmoid', max: 'torch.maximum', min: 'torch.minimum' }[n.name] || n.name}(${n.args.map(go).join(', ')})`;
      default: return `${go(n.a)} ${n.op === '^' ? '**' : n.op} ${go(n.b)}`;
    }
  };
  return go(ast);
}
// One line of PyTorch that does what the operation does.
function pyOf(t) {
  const { op } = t;
  const [a, b] = op.inputs.map(pyName);
  const ax = (x) => x.name;
  switch (op.name) {
    case 'matmul': return `${a} @ ${b}`;
    case 'conv2d': return `F.conv2d(${a}, ${b}, stride=${op.attrs.stride}, padding=${op.attrs.pad})`;
    case 'mean': return `${a}.mean(-1, keepdim=True)`;
    case 'sum': return `${a}.sum(${op.attrs.pos - op.inputs[0].rank}, keepdim=True)`;
    case 'std': return `((${a} - ${b}) ** 2).mean(-1, keepdim=True).add(${small(op.attrs.eps)}).sqrt()`;
    case 'softmax': return `${a}.softmax(-1)`;
    case 'lookup': return `${a}[${pyName(op.attrs.index)}]`;
    case 'rows': return `${a}[:${ax(t.shape[0])}]`;
    case 'transpose': return `${a}.transpose(-2, -1)`;
    case 'split': return `${a}.view(${[...op.inputs[0].shape.slice(0, -1), ...op.attrs.into].map(ax).join(', ')}).transpose(0, 1)`;
    case 'merge': return `${a}.transpose(0, 1).reshape(${t.shape.map(ax).join(', ')})`;
    case 'channels': return `${a}[${op.attrs.start}:${op.attrs.start + op.attrs.count}]`;
    case 'columns': return `${a}[..., ${op.attrs.start}:${op.attrs.start + op.attrs.count}]`;
    case 'again': return a;
    case 'reshape': return `${a}.view(${t.shape.map(ax).join(', ')})`;
    case 'flatten': return `${a}.flatten(1).T`;
    case 'unflatten': return `${a}.T.reshape(${t.shape.map(ax).join(', ')})`;
    case 'upsample': return `F.interpolate(${a}[None], scale_factor=${op.attrs.factor}, mode='nearest')[0]`;
    case 'concat': return `torch.cat([${a}, ${b}])`;
    case 'groupMean': return `${a}.view(${op.attrs.groups}, -1).mean(1).repeat_interleave(${op.attrs.per})[:, None, None]`;
    case 'groupStd': return `${a}.view(${op.attrs.groups}, -1).var(1, correction=0).add(${small(op.attrs.eps)}).sqrt().repeat_interleave(${op.attrs.per})[:, None, None]`;
    case 'silu': return `F.silu(${a})`;
    case 'module': return `${op.title.toLowerCase().replace(/\W+/g, '_')}(${op.inputs.map(pyName).join(', ')})`;
    case 'causalMask': return `torch.full((${t.shape.map(ax).join(', ')}), float('-inf')).triu(1)`;
    case 'constant': return op.py || 'torch.tensor(...)';
    default: return op.family === 'map' ? pyExpr(op.ast, (n) => (op.keys.includes(n) ? pyName(op.inputs[op.keys.indexOf(n)]) : undefined)) : `${op.name}(${op.inputs.map(pyName).join(', ')})`;
  }
}

// ---------------------------------------------------------------------------

export function compile(prog) {
  const L = layout(prog);
  const drawn = prog.tensors.filter((t) => t.section); // inputs made outside the steps are not drawn
  const slabAxis = drawn.find((t) => t.rank === 3)?.shape[0] || null;
  const nSlabs = slabAxis ? Math.max(...drawn.map((t) => t.view.D)) : 0;
  const slabs = {
    n: nSlabs,
    unit: slabAxis ? unitOf(slabAxis) : '',
    sym: slabAxis ? slabAxis.sym : '',
    colors: Array.from({ length: Math.max(nSlabs, SLAB_COLORS.length) }, (_, i) => SLAB_COLORS[i % SLAB_COLORS.length]),
    gap: GEO.slabGap,
  };
  const order = new Map(prog.tensors.map((t, i) => [t, i]));

  // ---- colour scales: copies share a scale with their source, sums with their terms
  const parent = new Map(drawn.map((t) => [t, t]));
  const find = (t) => {
    while (parent.get(t) !== t) t = parent.get(t);
    return t;
  };
  const join = (a, b) => parent.has(a) && parent.has(b) && parent.set(find(a), find(b));
  for (const t of drawn) {
    if (t.op?.family === 'rearrange') for (const o of t.op.inputs) join(t, o);
    if (t.op?.name === 'add') for (const o of t.op.inputs) if (sameDims(o, t)) join(t, o);
  }

  // ---- axes seen as groups: d_model as h groups of d_k, wherever that split shows
  const grouped = new Map(); // tensor -> { cols?: [a, b], rows?: [a, b] }
  const mark = (t, side, pair) => grouped.set(t, { ...grouped.get(t), [side]: pair });
  for (const t of drawn) {
    if (t.op?.name === 'split') mark(t.op.inputs[0], 'cols', t.op.attrs.into);
    if (t.op?.name === 'merge') mark(t, 'cols', t.op.attrs.from);
  }
  for (const t of [...drawn].reverse()) // a result's column groups are its weight's column groups
    if (t.op?.family === 'contract' && grouped.get(t)?.cols) mark(t.op.inputs[1], 'cols', grouped.get(t).cols);
  for (const t of drawn) // an operand's column groups are its weight's row groups
    if (t.op?.family === 'contract' && grouped.get(t.op.inputs[0])?.cols) mark(t.op.inputs[1], 'rows', grouped.get(t.op.inputs[0]).cols);

  const bracket = (t, side, ax, n) => {
    const pair = grouped.get(t)?.[side];
    if (pair) {
      const [a, b] = pair;
      return Array.from({ length: a.size }, (_, i) => ({ from: i * b.size, to: (i + 1) * b.size - 1, label: `${unitOf(a)} ${i}`, color: SLAB_COLORS[i % SLAB_COLORS.length] }));
    }
    return [{ from: 0, to: n - 1, label: `${ax.sym} = ${ax.size}`, color: DIM }];
  };

  // Rows of a table that nothing reads are dimmed.
  const usedRows = (t) => {
    const readers = drawn.filter((u) => (u.op?.name === 'lookup' || u.op?.name === 'rows') && u.op.inputs[0] === t);
    if (!readers.length) return null;
    const used = new Set();
    for (const u of readers) for (const f of u.op.at) used.add(Math.floor(f / colsOf(t)));
    return used;
  };

  const nested = (t) => {
    const out = Array.from({ length: slabsOf(t) }, () => Array.from({ length: rowsOf(t) }, () => new Array(colsOf(t))));
    for (let i = 0; i < t.size; i++) {
      const [d, r, c] = t.addr(i);
      out[d][r][c] = t.data[i];
    }
    return out;
  };

  // Where a tensor was made, in the reader's words: "Step 1", or the section's title.
  const whereOf = (t) => (t.section.repeat?.name ? `${t.section.repeat.name} ${t.section.repeat.index + 1}` : t.section.title);
  const noteOf = (t, rec) => {
    if (t.note) return t.note;
    const op = t.op;
    if (t.kind === 'input') return 'given as input';
    if (t.kind === 'param' && t.alias) return `the same weights as in ${whereOf(t.alias)}`;
    if (t.kind === 'param') return t.rank === 1 ? 'one per column (random here)' : 'weights (random here)';
    if (op.family === 'module') return `the same ${op.title}, not drawn again`;
    if (op.name === 'lookup') return 'one row looked up per entry';
    if (op.name === 'rows') return `the first ${t.shape[0].sym} rows`;
    if (op.name === 'split') return `one slab per ${unitOf(op.attrs.into[0])}`;
    if (op.name === 'merge') return `${op.attrs.from.map((a) => a.sym).join('·')} = ${op.attrs.axis.sym}`;
    if (op.name === 'channels') return `channel${op.attrs.count > 1 ? `s ${op.attrs.start} to ${op.attrs.start + op.attrs.count - 1}` : ` ${op.attrs.start}`} of ${op.inputs[0].symbol}`;
    if (op.name === 'columns') return `columns ${op.attrs.start} to ${op.attrs.start + op.attrs.count - 1} of ${op.inputs[0].symbol}`;
    if (op.name === 'again') return `${op.inputs[0].symbol} again, from ${op.inputs[0].section.title}`;
    if (op.name === 'flatten') return 'one row per grid cell';
    if (op.name === 'unflatten') return 'one grid cell per row';
    if (op.name === 'upsample') return `every cell copied ${op.attrs.factor} × ${op.attrs.factor}`;
    if (op.name === 'concat') return `${op.inputs[0].symbol}, then ${op.inputs[1].symbol}`;
    if (op.name === 'reshape') return `${op.inputs[0].symbol}, re-addressed`;
    if (op.name === 'groupMean') return `mean of each group of ${op.attrs.per} ${slabs.unit}s`;
    if (op.name === 'groupStd') return `spread of each group of ${op.attrs.per} ${slabs.unit}s`;
    if (op.family === 'rearrange' && op.inputs[0].section !== t.section) return `${op.inputs[0].symbol} again, ${op.name}d`;
    if (op.name === 'softmax') return 'every row sums to 1';
    if (rec.role === 'front') return `no ${slabs.unit} axis: the same plate for every ${slabs.unit}`;
    if (op.family === 'const') return op.rule;
    if (op.family === 'map' && op.inputs.length > 1 && !exprOf(t)) return formulaOf(t, prog);
    return '';
  };

  // ---- tensor specs
  const specs = drawn.map((t) => {
    const rec = L.at.get(t);
    const [R, C] = [rowsOf(t), colsOf(t)];
    const rowAx = t.rank === 2 || t.rank === 3 ? t.shape.at(-2) : null;
    // A bank of filters: the rows are grouped by filter.
    const filters = t.rank === 4
      ? Array.from({ length: t.dims[0] }, (_, i) => ({ from: i * t.dims[2], to: (i + 1) * t.dims[2] - 1, label: `${unitOf(t.shape[0])} ${i}`, color: SLAB_COLORS[i % SLAB_COLORS.length] }))
      : null;
    const colAx = t.shape.at(-1);
    const used = usedRows(t);
    const expr = simple(t) ? exprOf(t) : null;
    const paramLike = isParamLike(t);
    const hasUnder = L.sections.get(t.section).placed.some((q) => q.stackedUnder === t);
    // Where a cube flies in from: its source if it is a copy, otherwise the
    // same cell of the nearest operand of the same shape.
    const twin = t.op && t.op.family !== 'rearrange' ? t.op.inputs.filter((o) => sameDims(o, t)).sort((a, b) => order.get(b) - order.get(a))[0] : null;
    return {
      id: t.id, sec: t.section.id, t,
      label: expr ? `${t.label} = ${expr}` : t.label,
      sym: t.sym, num: t.num, note: noteOf(t, rec),
      // How the tensor is drawn. hint({ show }): 'image' for a picture, 'prob' for numbers between 0 and 1.
      //   param: flat tiles, gold frame (learned)       const, mask: flat tiles, silver frame (fixed in advance)
      kind: t.hints.show || (paramLike ? 'param' : t.op?.family === 'const' ? (t.op.mask ? 'mask' : 'const') : t.op?.name === 'softmax' ? 'prob' : 'act'),
      group: find(t).id,
      step: t.step.index, x0: rec.x0, cy: rec.cy, zs: rec.zs,
      title: rec.below ? 'below' : undefined,
      rowLabels: rowAx?.labels && C > 1 ? rowAx.labels.map((text, r) => ({ text, dim: used ? !used.has(r) : false })) : null,
      colLabels: colAx.labels && C > 1 ? colAx.labels : null,
      colBrackets: t.rank === 2 && C > 1 && colAx !== ONE && !colAx.labels && !hasUnder ? bracket(t, 'cols', colAx, C) : null,
      rowBrackets: filters || (t.rank === 2 && paramLike && !rowAx.labels ? bracket(t, 'rows', rowAx, R) : null),
      from: t.op?.family === 'rearrange'
        ? (d, r, c) => { const [[src, [f]]] = t.op.sources(t.flat(d, r, c)); return [src.id, ...src.addr(f)]; }
        : twin ? (d, r, c) => [twin.id, d, r, c] : null,
      // A weight of exactly 0 that came from a −∞ score.
      blocked: t.op?.name === 'softmax' ? (d, r, c) => t.op.inputs[0].data[t.flat(d, r, c)] === -Infinity : null,
      get: () => nested(t),
    };
  });
  const specOf = new Map(specs.map((s) => [s.t, s]));

  // ---- sections: widths, folded view, arrows
  const sections = prog.sections.map((sec) => {
    const S = L.sections.get(sec);
    const keep = sec.keep;
    const residual = S.keep.ring && S.stream && keep.op.inputs.includes(sec.stream);
    // The example a section reads directly is named in front of its box, unless the section is nothing but that example.
    const given = !S.stream && sec.input !== keep ? sec.input : null;
    const bx = S.stream ? [3, 13] : [6, 16];
    const keepX = bx[1] + (residual ? 10 : 5);
    const flows = [];
    const F = (step, pts, o = {}) => flows.push({ type: 'flow', step, pts, ...o });
    const T = (step, text, at, o = {}) => flows.push({ type: 'op', step, text, at, ...o });

    // Folded view: stream -> box -> (+) -> output, with the residual arc over the box.
    const c = { closed: true };
    if (S.stream) F(0, [[-0.1, 0, 0], [bx[0] - 0.4, 0, 0]], c);
    else if (given) {
      T(0, given.label, [2.2, 0.25, 0], { size: 0.62, ...c });
      T(0, `${given.sym} = ${given.num}`, [2.2, -0.25, 0], { size: 0.5, color: DIM, anchor: [0.5, 1], ...c });
      F(0, [[4.3, 0, 0], [bx[0] - 0.4, 0, 0]], c);
    }
    const inPad = (t) => (specOf.get(t).rowLabels || specOf.get(t).rowBrackets ? 2.7 : 0.8);
    if (residual) {
      F(0, [[bx[1] + 0.4, 0, 0], [bx[1] + 2.7, 0, 0]], c);
      flows.push({ type: 'plus', step: 0, x: bx[1] + 4, y: 0, color: SKIP, ...c });
      F(0, [[bx[1] + 5.3, 0, 0], [keepX - inPad(keep), 0, 0]], c);
      F(0, [[-1, 3.2, 0], [-1, 9, 0], [bx[1] + 4, 9, 0], [bx[1] + 4, 1.4, 0]], { color: SKIP, ...c });
    } else F(0, [[bx[1] + 0.4, 0, 0], [keepX - inPad(keep), 0, 0]], c);

    // Open view.
    const pos = (t) => (t.section === sec ? L.at.get(t) : t === sec.stream ? S.stream : null);
    const outPort = (p) => p.x0 + p.C - 1 + 0.9;
    const inPort = (p) => p.x0 - (p.stream ? 0.8 : inPad(p.t));
    const top = (p) => p.cy + (p.R - 1) / 2;
    const bot = (p) => p.cy - (p.R - 1) / 2;
    const mid = (p) => p.x0 + (p.C - 1) / 2;
    const arcY = S.top + 8.5;
    // One arrow per slab while there are few of them; one for the whole stack otherwise.
    const eachSlab = (t, fn) => (t.rank === 3 && L.at.get(t).zs.length <= 4 && !isDeck(t) ? L.at.get(t).zs.forEach((z, d) => fn(z, d)) : fn(0, null));
    const tag = (step, x, y, z, d) => d != null && T(step, `${slabs.unit} ${d}`, [x, y - 0.35, z], { size: 0.42, color: SLAB_COLORS[d % SLAB_COLORS.length], anchor: [0.5, 1], slab: d });
    // An arrow from a's right edge to b's left edge, bending if they are on different lanes.
    const link = (step, a, b, { z = 0, slab = null, label = null, size = 0.62 } = {}) => {
      const [x1, x2] = [outPort(a), inPort(b)];
      if (x2 - x1 < 1.5) return;
      // Bend right after the source, unless something on the target's lane is in the way:
      // then run along the source's lane and bend just before the target.
      const inWay = a.cy !== b.cy && S.placed.some((q) => q !== a && q !== b && q.role !== 'front' && q.x0 + q.C > x1 + 1 && q.x0 < x2 - 1 && Math.abs(q.cy - b.cy) < q.R / 2 + 0.5);
      const bend = inWay ? x2 - 1.8 : x1 + 1.8;
      F(step, a.cy === b.cy ? [[x1, a.cy, z], [x2, b.cy, z]] : [[x1, a.cy, z], [bend, a.cy, z], [bend, b.cy, z], [x2, b.cy, z]], { slab });
      const lx = a.cy === b.cy ? (x1 + x2) / 2 : (bend + x2) / 2;
      if (label) T(step, label, [lx, b.cy + 0.3, z], { size, slab });
      tag(step, lx, b.cy, z, slab);
    };

    for (const t of sec.tensors) {
      const p = L.at.get(t);
      const op = t.op;
      if (!p || !op) continue;
      const step = t.step.index;
      const attached = op.inputs.filter((o) => L.roles.get(o)?.of === t && L.roles.get(o).type !== 'weight');
      const flowIns = op.inputs.filter((o) => pos(o) && !attached.includes(o));

      if (p.role === 'front') {
        const of = L.at.get(p.of);
        F(of.t.step.index, [[mid(p), p.cy, p.zs[0] - 0.8], [mid(p), p.cy, of.zs[0] + 0.9]], { color: 0xb86a7c });
        continue;
      }
      if (op.family === 'const') continue;

      if (op.family === 'module') {
        // Every input goes in; the first arrow carries the module's name.
        flowIns.map(pos).forEach((a, n) => link(step, a, p, { label: n ? null : op.title, size: 0.8 }));
        continue;
      }

      if (op.family === 'contract') {
        const a = pos(op.inputs[0]);
        const b = pos(op.inputs[1]);
        if (b?.role === 'weight' && b.of === t) {
          if (a) link(step, a, b, { label: op.symbol });
          link(step, b, p);
        } else {
          eachSlab(t, (z, d) => {
            if (a) link(step, a, p, { z, slab: d, label: op.symbol });
            // The operand beside the result points at it, unless labels sit in between.
            if (b?.role === 'beside' && b.of === t && (b.side === 'below' || !specOf.get(t).colLabels))
              F(step, b.side === 'below' ? [[mid(p), top(b) + 0.55, z], [mid(p), bot(p) - 0.65, z]] : [[mid(p), bot(b) - 0.55, z], [mid(p), top(p) + 0.65, z]], { slab: d });
            // A right operand that is in the flow itself comes in with its own arrow.
            else if (b && b.role === 'flow' && b !== a) link(step, b, p, { z, slab: d });
          });
        }
        continue;
      }

      if (op.family === 'rearrange') {
        const a = pos(op.inputs[0]);
        if (!a) continue; // the source is in another section
        const [x1, x2] = [outPort(a), inPort(p)];
        if (op.name === 'split') {
          // A trunk leaves the source, then one branch per slab peels off into depth.
          const straight = a.cy === p.cy;
          const xt = straight ? x1 + 2.2 : x2 - 1.5;
          F(step, [[x1, a.cy, 0], [xt, a.cy, 0]], { head: false });
          eachSlab(t, (z, d) => {
            const pts = [[xt, a.cy, 0]];
            if (z) pts.push([xt, a.cy, z]);
            if (!straight) pts.push([xt, p.cy, z]);
            pts.push([x2, p.cy, z]);
            F(step, pts, { slab: d });
          });
          T(step, `split ${unitOf(op.attrs.into[0])}s`, [straight ? xt : xt - 5, a.cy + 0.5, 0], { size: 0.5 });
        } else if (op.name === 'merge') {
          const xm = (x1 + x2) / 2 + 0.5;
          eachSlab(op.inputs[0], (z, d) => F(step, z ? [[x1, a.cy, z], [xm, a.cy, z], [xm, a.cy, 0]] : [[x1, a.cy, 0], [x2, p.cy, 0]], { head: !z, slab: d }));
          T(step, 'concat', [xm, p.cy + 0.4, 0], { size: 0.54 });
        } else if (p.role === 'beside') {
          // Comes from its source and drops onto its place above or below the result.
          const yEnd = p.side === 'above' ? top(p) + (specOf.get(t).colLabels ? 2.6 : 0.9) : bot(p) - 0.9;
          eachSlab(t, (z, d) => {
            F(step, [[x1, a.cy, z], [mid(p), a.cy, z], [mid(p), yEnd, z]], { slab: d });
            T(step, op.name, [mid(p) + 0.6, (a.cy + yEnd) / 2, z], { size: 0.46, anchor: [0, 0.5], slab: d });
          });
        } else {
          const label = op.name === 'lookup' ? 'look up' : op.name === 'rows' ? `first ${t.shape[0].sym} rows` : op.name === 'channels' || op.name === 'columns' ? null : op.name === 'upsample' ? `upsample × ${op.attrs.factor}` : op.name;
          eachSlab(t, (z, d) => link(step, a, p, { z, slab: d, label, size: 0.5 }));
          for (const o of op.inputs.slice(1)) if (pos(o)) eachSlab(t, (z, d) => link(step, pos(o), p, { z, slab: d }));
        }
        continue;
      }

      if (p.stackedUnder) {
        // Three short arrows down from the operand, with the operation named between them.
        const a = pos(p.stackedUnder);
        const xs = [0.085, 0.5, 0.915].map((f) => a.x0 + (a.C - 1) * f);
        for (const x of xs) F(step, [[x, bot(a) - 0.75, 0], [x, top(p) + 0.75, 0]]);
        const text = `${arrowLabel(t, prog, attached) || op.name}, one cube at a time`;
        for (const x of [(xs[0] + xs[1]) / 2, (xs[1] + xs[2]) / 2]) T(step, text, [x, (bot(a) + top(p)) / 2, 0], { size: 0.5, anchor: [0.5, 0.5] });
        continue;
      }

      if (p.ring) {
        const { x, y } = p.ring;
        const far = (a) => a.stream || x - (a.x0 + a.C) > 30;
        const recs = flowIns.map(pos);
        const color = recs.some(far) ? SKIP : SOFT;
        for (const a of recs) {
          if (far(a)) {
            // A residual connection: up from the stream, along the top, down into the add node.
            // An arc that starts inside the section runs under the one that spans it.
            const xa = a.x0 + a.C - 1;
            const ya = a.stream ? arcY : arcY - 3;
            F(step, [[xa, top(a) + 0.7, 0], [xa, ya, 0], [x, ya, 0], [x, y + 1.4, 0]], { color: SKIP });
            T(step, `residual connection: ${a.t.symbol} skips ahead unchanged`, [(xa + x) / 2, ya + 0.4, 0], { size: 0.8, color: '#ffb454' });
          } else if (a.cy === y) F(step, [[outPort(a), y, 0], [x - 1.3, y, 0]]);
          else F(step, [[outPort(a), a.cy, 0], [x, a.cy, 0], [x, y + Math.sign(a.cy - y) * 1.4, 0]], { color: SOFT });
        }
        flows.push({ type: 'plus', step, x, y, color });
        F(step, [[x + 1.3, y, 0], [inPort(p), y, 0]]);
        continue;
      }

      // Everything else: an arrow from the nearest operand to its left, if
      // nothing stands in between.
      const [a, ...others] = flowIns.map(pos).sort((u, v) => v.x0 + v.C - (u.x0 + u.C));
      if (!a) continue;
      const clear = (from) => !S.placed.some((q) => q !== from && q !== p && q.role === 'flow' && q.x0 > from.x0 && q.x0 < p.x0 && Math.abs(q.cy - p.cy) < (q.R + p.R) / 2);
      if (!clear(a)) continue;
      const label = arrowLabel(t, prog, attached);
      eachSlab(t, (z, d) => link(step, a, p, { z, slab: d, label, size: label && label.length > 5 ? 0.54 : 0.62 }));
      // When the nearest operand came in from a side lane, the one on the result's own lane gets its arrow too.
      // With hint({ arrows: 'all' }) every other operand gets one: a formula of several tensors that all matter.
      const every = t.hints.arrows === 'all';
      for (const o of others)
        if (every ? o.cy !== p.cy || clear(o) : a.cy !== p.cy && o.cy === p.cy && clear(o)) eachSlab(t, (z, d) => link(step, o, p, { z, slab: d }));
    }

    // One glass pane per slab, and the slab axis itself.
    const deep = S.placed.filter((p) => p.t.rank === 3);
    if (deep.length && deep.every((p) => p.zs.length === deep[0].zs.length && !isDeck(p.t))) {
      const step = Math.min(...deep.map((p) => p.t.step.index));
      const [x0, x1] = [Math.min(...deep.map((p) => p.x0)) - 4.5, Math.max(...deep.map((p) => p.x0 + p.C)) + 3.5];
      const [y0, y1] = [Math.min(...deep.map(bot)) - 7, Math.max(...deep.map(top)) + 8];
      deep[0].zs.forEach((z, d) => flows.push({ type: 'pane', step, slab: d, x0, x1, y0, y1, z: z - 0.9, label: `${slabs.unit.toUpperCase()} ${d}` }));
      const f = deep[0];
      F(step, [[mid(f), bot(f) - 1.9, f.zs[0] + 0.5], [mid(f), bot(f) - 1.9, f.zs.at(-1) - 0.5]], { color: 0xcfd8ea });
      T(step, `${slabs.unit} axis: ${slabs.sym} = ${slabs.n}`, [mid(f), bot(f) - 2.4, f.zs[0]], { size: 0.55, anchor: [0.5, 1] });
    }

    return {
      id: sec.id, title: sec.title.toUpperCase(),
      lines: [sec.summary, `${S.stream ? `${sec.stream.sym} → ` : given ? `${given.sym} → ` : ''}${keep.sym}`],
      wOpen: S.width, wClosed: keepX + colsOf(keep), box: bx, keep: keep.id, keepX, big: keep.symbol,
      from: sec.from, to: sec.to, top: S.top, arcY, flows,
    };
  });

  // A bar over each repeated instance and each named part, spanning its sections.
  const bars = [];
  for (const sec of prog.sections) {
    if (!sec.repeat && !sec.part) continue;
    const key = sec.part || sec.repeat.index; // a named part, or one instance of the repeat
    let bar = bars.find((b) => b.key === key);
    if (!bar) bars.push((bar = { key, label: sec.part ? sec.part.name.toUpperCase() : `${sec.repeat.name.toUpperCase()} ${sec.repeat.index + 1}`, titles: [], first: sec.id, yOpen: 0 }));
    bar.last = sec.id;
    bar.titles.push(sec.title);
    bar.yOpen = Math.max(bar.yOpen, L.sections.get(sec).top + 14.5);
  }
  for (const b of bars) b.sub = b.titles.length > 5 ? `${b.titles.length} boxes, from ${b.titles[0]} to ${b.titles.at(-1)}` : b.titles.join(', then ');

  // ---- the tour
  const numWith = (t, pos) => `(${t.dims.map((d, i) => (i === (pos < 0 ? t.rank + pos : pos) ? `<u>${d}</u>` : d)).join(', ')})`;
  const sentence = (t) => {
    const op = t.op;
    const [a, b] = op.inputs;
    const name = `<b>${H(t)}</b>`;
    const deepNote = t.rank === 3 ? ` This happens separately in each of the ${slabs.n} ${slabs.unit} slabs.` : '';
    switch (op.name) {
      case 'matmul':
        return `${name} = ${H(a)} @ ${H(b)}: every row of ${H(a)} is dotted with every column of ${H(b)}.
          The inner sizes must agree: <code>${numWith(a, -1)} @ ${numWith(b, -2)} → ${t.num}</code>.${deepNote}`;
      case 'conv2d':
        return `${name} = ${H(a)} ∗ ${H(b)}: each of the ${b.dims[0]} filters in ${H(b)} slides over ${H(a)} ${op.attrs.stride > 1 ? `in steps of ${op.attrs.stride}` : 'one cell at a time'}${
          op.attrs.pad ? `, with ${op.attrs.pad} cell${op.attrs.pad > 1 ? 's' : ''} of zero padding around the edge` : ''
        }. At every position the ${b.dims[2]} × ${b.dims[3]} window, across all ${a.dims[0]} input channel${a.dims[0] > 1 ? 's' : ''}, is multiplied cell by cell with the filter and added up:
          <code>${a.num} ∗ ${b.num} → ${t.num}</code>.`;
      case 'add': return `${name} = ${H(a)} + ${H(b)}, cell by cell.${sameDims(a, b) ? '' : ` ${H(sameDims(a, t) ? b : a)} has fewer axes and is repeated along the ones it lacks.`}`;
      case 'mean': return `${name}: the mean of each line of ${H(a)} along ${markupToHtml(a.shape[op.attrs.pos].sym)}, so ${a.num} → ${t.num}.`;
      case 'sum': return `${name}: the total of each line of ${H(a)} along ${markupToHtml(a.shape[op.attrs.pos].sym)}, so ${a.num} → ${t.num}.`;
      case 'std': return `${name}: how spread out each of those lines is around its mean, √(mean((x − μ)²) + ε) with ε = ${small(op.attrs.eps)}.`;
      case 'softmax': return `${name}: softmax along the last axis. Each line of ${H(a)} is exponentiated and divided by its total, so every line sums to 1.`;
      case 'lookup':
        return t.dims[0] === 1
          ? `${name} copies the one row of ${H(a)} that the ${op.attrs.index.label} picks. A lookup: nothing is multiplied or added.`
          : `${name} copies one row of ${H(a)} for each of the ${t.dims[0]} ${op.attrs.index.label}. A lookup: nothing is multiplied or added.`;
      case 'rows': return `${name} is the first ${markupToHtml(t.shape[0].sym)} = ${t.dims[0]} rows of ${H(a)}, copied as they are.`;
      case 'transpose': return `${name} is ${H(a)} with its last two axes swapped: ${a.num} → ${t.num}. No number changes.`;
      case 'split': {
        const [g, k] = op.attrs.into;
        return `${name} ${markupToHtml(t.sym)}: the last axis of ${H(a)}, ${markupToHtml(op.attrs.axis.sym)} = ${op.attrs.axis.size}, is re-addressed as
          ${markupToHtml(g.sym)} × ${markupToHtml(k.sym)} = ${g.size} × ${k.size}, and the ${markupToHtml(g.sym)} axis is brought to the front. No number changes.`;
      }
      case 'merge': return `${name}: the ${op.attrs.from[0].size} slabs of ${H(a)} are laid side by side again, ${a.num} → ${t.num}. No number changes.`;
      case 'channels': return `${name}: ${op.attrs.count > 1 ? `channels ${op.attrs.start} to ${op.attrs.start + op.attrs.count - 1}` : `channel ${op.attrs.start}`} of ${H(a)}, copied as ${op.attrs.count > 1 ? 'they are' : 'it is'}: ${a.num} → ${t.num}.`;
      case 'columns': return `${name}: columns ${op.attrs.start} to ${op.attrs.start + op.attrs.count - 1} of ${H(a)}, copied as they are: ${a.num} → ${t.num}. No number changes.`;
      case 'module': {
        const made = op.hidden.filter((h) => h.op);
        return `${name} = ${op.title}(${op.inputs.map(H).join(', ')}): the whole ${op.title} is run again on these ${op.inputs.length} inputs, with the same weights as where it is drawn in full.
          Its ${made.length} intermediate tensors (${made.reduce((n, h) => n + h.size, 0).toLocaleString('en-US')} numbers) are computed exactly and not drawn.${
            Array.from({ length: t.size }, (_, i) => op.sources(i)).every((s) => s.length === op.inputs.length && s.every(([x, cells]) => cells.length === x.size))
              ? ` Every number of ${H(t)} depends on every number of ${op.inputs.map(H).join(', ')}.`
              : ''
          }`;
      }
      case 'again': return `${name} is ${H(a)} from the ${a.section.title} section, brought in unchanged because it is needed here. No number changes.`;
      case 'reshape': return `${name}: the numbers of ${H(a)} in the same order, re-addressed ${a.num} → ${t.num} so that they line up with the axes of what they are combined with next. No number changes.`;
      case 'flatten': return `${name}: every cell of the ${op.attrs.grid.map((x) => x.size).join(' × ')} grid becomes one row, read row by row, and its ${a.dims[0]} channels become that row's columns: ${a.num} → ${t.num}. No number changes.`;
      case 'unflatten': return `${name}: the reverse of flattening. Each of the ${a.dims[0]} rows goes back to its cell of the ${op.attrs.grid.map((x) => x.size).join(' × ')} grid, and its columns become channels again: ${a.num} → ${t.num}. No number changes.`;
      case 'upsample': return `${name}: every cell of ${H(a)} is copied into a ${op.attrs.factor} × ${op.attrs.factor} block, so ${a.num} → ${t.num}. Nothing new is computed.`;
      case 'concat': return `${name}: the ${op.attrs.counts[0]} channels of ${H(a)} followed by the ${op.attrs.counts[1]} channels of ${H(b)}, ${a.num} and ${b.num} → ${t.num}. No number changes.`;
      case 'groupMean': return `${name}: the ${a.dims[0]} channels of ${H(a)} are taken in ${op.attrs.groups} groups of ${op.attrs.per}. Each cube is the mean of its whole group, all ${op.attrs.n} numbers of it, so the channels of one group hold the same value: ${a.num} → ${t.num}.`;
      case 'groupStd': return `${name}: how spread out each of those groups is around its mean, √(mean((x − μ)²) + ε) with ε = ${small(op.attrs.eps)}.`;
      case 'constant':
      case 'causalMask': return `${name}: fixed in advance, neither learned nor computed: ${op.rule}. Drawn as flat tiles with a silver frame.`;
      default: {
        if (op.family !== 'map') return `${name} = ${op.name}(${op.inputs.map(H).join(', ')}).`;
        const stretched = op.inputs.filter((o) => !sameDims(o, t));
        return `${name} = ${markupToHtml(exprOf(t) || formulaOf(t, prog))}, ${op.inputs.length === 1 ? `applied to every cube of ${H(a)} on its own` : 'cell by cell'}.${
          stretched.length ? ` ${stretched.map((o) => `${H(o)} ${markupToHtml(o.sym)}`).join(' and ')} ${stretched.length > 1 ? 'are' : 'is'} repeated along the axes ${stretched.length > 1 ? 'they lack' : 'it lacks'}.` : ''
        }`;
      }
    }
  };
  const shapeRow = (t, note = '') => [t.symbol, t.sym, t.num, note];
  // The author's prose, then the sentences generated from the operations.
  const env = { slabColors: slabs.colors };
  const authored = (b) => b.prose.map((item) => renderProse(item, env)).join('');
  const withDerived = (b, derived) => (b.prose.length ? `${authored(b)}<div class="derived"><div class="derived-h">Exactly what is computed</div>${derived}</div>` : derived);
  const tryOf = (b, fallback) => (b.hint ? renderProse(b.hint, env) : fallback);
  // In the step bar: "Block 1 · Attention". An instance with many sections names itself on its first one only.
  const chapterOf = (sec) => {
    if (sec.part) return prog.sections.find((x) => x.part === sec.part) === sec ? `${sec.part.name} · ${sec.title}` : sec.title;
    if (!sec.repeat) return sec.title;
    const unit = `${sec.repeat.name} ${sec.repeat.index + 1}`;
    if (sec.collapsed) return unit;
    const mine = prog.sections.filter((x) => x.repeat?.index === sec.repeat.index);
    return mine.length > 3 && mine[0] !== sec ? sec.title : `${unit} · ${sec.title}`;
  };
  const HOVER = 'Hover a cube to see exactly which numbers it was computed from. Click it to pin the explanation.';

  const stepOf = (s) => {
    const sec = s.section;
    const news = s.tensors;
    const made = news.filter((t) => t.op);
    const given = news.filter((t) => t.kind === 'input');
    const params = news.filter((t) => t.kind === 'param');
    const operands = [...new Set(made.flatMap((t) => t.op.inputs))].filter((o) => !news.includes(o) && o.section);
    const deepIn = (ids) => ids.some((t) => t.rank === 3);

    if (sec.collapsed) {
      const firstOf = prog.sections.find((x) => x.repeat?.index === 0 && x.title === sec.title);
      return {
        chapter: chapterOf(sec),
        title: `${sec.repeat.name} ${sec.repeat.index + 1} · ${sec.title}`,
        formula: '',
        shapes: [...(sec.stream ? [shapeRow(sec.stream, 'in')] : []), shapeRow(sec.keep, 'out')],
        body: () => {
          const own = params.filter((t) => !t.alias);
          const list = firstOf ? `<ol>${firstOf.steps.map((x) => `<li>${x.title}</li>`).join('')}</ol>` : '';
          const same = firstOf ? `The same ${firstOf.steps.length} step${firstOf.steps.length > 1 ? 's' : ''} as ${firstOf.repeat.name} 1 · ${firstOf.title}` : '';
          const unit = sec.repeat.name.toLowerCase();
          // Its own weights (another block), the same weights (another call of one model), or no weights at all.
          const recap = !firstOf
            ? ''
            : own.length
              ? `<p>${same}, on the stream that block left,
          with this block's own ${params.length} parameter tensors (${params.reduce((n, t) => n + t.size, 0)} numbers):</p>
          ${list}
          <p>Every shape is the same as before; every number is different.</p>`
              : `<p>${same}, on this ${unit}'s inputs${params.length ? ` and with the same ${params.length} weight tensors` : ''}:</p>${list}
          <p>Every shape is the same as before${params.length ? ', and so is every weight' : ''}; every computed number is different.</p>`;
          // A short section also says exactly what it computes.
          const derived = !own.length && made.length <= 6 ? `<div class="derived"><div class="derived-h">Exactly what is computed</div>${made.map((t) => `<p>${sentence(t)}</p>`).join('')}</div>` : '';
          return `${authored(s)}${recap}${derived}`;
        },
        code: made.map((t) => `${pyName(t)} = ${pyOf(t)}`).join('\n'),
        try: tryOf(s, HOVER),
        view: { ids: sec.tensors.map((t) => t.id), az: deepIn(sec.tensors) ? 16 : 10, el: deepIn(sec.tensors) ? 14 : 8, perHead: false },
      };
    }

    // Frame what is new, plus the operands next to it. Tensors placed for a
    // later step (beside a result that does not exist yet) are left out.
    const here = news.filter((t) => { const of = L.at.get(t).of; return !of || of.step.index <= s.index; });
    const S = L.sections.get(sec);
    const pos = (t) => (t.section === sec ? L.at.get(t) : t === sec.stream ? S.stream : null);
    const left = Math.min(...here.map((t) => L.at.get(t).x0));
    const feeding = new Set(here.filter((t) => t.op).flatMap((t) => t.op.inputs));
    // An operand on a side lane (what came from another section, and what was computed from it) is framed from further away.
    const entered = (t) => !!t.op && (t.op.name === 'again' || (t.op.inputs[0]?.section === sec && entered(t.op.inputs[0])));
    // hint({ frame: 'operands' }) on a result asks for all of its operands, however far back they stand.
    const wanted = new Set(here.filter((t) => t.op && t.hints.frame === 'operands').flatMap((t) => t.op.inputs));
    const near = operands.filter((o) => feeding.has(o) && pos(o) && (wanted.has(o) || left - (pos(o).x0 + pos(o).C) < (entered(o) ? 70 : 40)));
    // ... and whatever is already on show above or below them.
    const core = [...near, ...here];
    const [x0, x1] = [Math.min(...core.map((t) => pos(t).x0)), Math.max(...core.map((t) => pos(t).x0 + pos(t).C))];
    const around = S.placed.filter((q) => !core.includes(q.t) && q.t.step.index <= s.index && (!q.of || q.of.step.index <= s.index) && q.x0 < x1 && x0 < q.x0 + q.C).map((q) => q.t);
    const framed = [...core, ...around];
    const deep = deepIn(framed);
    const eqs = made.filter(simple).map((t) => [t, exprOf(t) || (t.op.family === 'map' ? formulaOf(t, prog) : null)]).filter(([, e]) => e);
    const width = Math.max(...made.map((t) => pyName(t).length + pyOf(t).length + 5), 0);
    return {
      chapter: chapterOf(sec),
      title: s.title,
      formula: eqs.map(([t, e]) => `${H(t)} = ${markupToHtml(e)}`).join(' &nbsp;&nbsp; '),
      shapes: [...operands.map((o) => shapeRow(o)), ...given.map((t) => shapeRow(t, 'input')), ...params.map((t) => shapeRow(t, 'weights')), ...made.map((t) => shapeRow(t))],
      body: () =>
        withDerived(s, `${given.map((t) => `<p><b>${H(t)}</b> ${markupToHtml(t.sym)} = ${t.num}: given as input, not computed.</p>`).join('')}${made.map((t) => `<p>${sentence(t)}</p>`).join('')}${
          params.length
            ? params.every((t) => t.alias)
              ? `<p class="aside">${params.map(H).join(', ')}: the same weights as in ${whereOf(params[0].alias)}, drawn again where they are used.</p>`
              : `<p class="aside">${params.map(H).join(', ')}: ${params.reduce((n, t) => n + t.size, 0)} parameters, learned in a real model and random here.
               Flat gold-framed tiles are parameters; cubes are activations, recomputed for every input.</p>`
            : ''
        }`),
      code: made.map((t) => `${`${pyName(t)} = ${pyOf(t)}`.padEnd(width)}# ${plain(t.sym)}`).join('\n'),
      try: tryOf(s, HOVER),
      focus: made.some((t) => t.rank === 3 && t.op.name === 'matmul') ? 0 : undefined,
      view: { ids: framed.map((t) => t.id), az: deep ? 30 : 12, el: deep ? 32 : 9, perHead: deep },
    };
  };

  const nParams = drawn.filter((t) => t.kind === 'param' && !t.alias).reduce((n, t) => n + t.size, 0); // shared weights count once
  const nCubes = drawn.reduce((n, t) => n + t.size, 0);
  const overview = (chapter, title, lead, b) => ({
    chapter, title, formula: '', shapes: null,
    body: () => `${b.prose.length ? authored(b) : `<p>${lead}</p>`}
      <p>${drawn.length} tensors and ${nCubes.toLocaleString('en-US')} numbers are drawn, one cube each; ${nParams.toLocaleString('en-US')} of them are parameters.</p>
      <ul>${prog.sections.map((x) => `<li><b>${x.part ? `${x.part.name} · ` : x.repeat ? `${x.repeat.name} ${x.repeat.index + 1} · ` : ''}${x.title}</b>: ${x.summary}</li>`).join('')}</ul>
      <p><b>How this differs from a real model</b></p>
      <ul>${prog.departures.map((d) => `<li>${d.text}</li>`).join('')}</ul>`,
    code: '',
    try: tryOf(b, 'Click a box to open that section, or press → to walk through every step.'),
    view: { ids: prog.sections.map((x) => x.keep.id), all: true, az: 10, el: 12 },
  });
  const steps = [
    overview('Map', prog.name, 'Each box is one section of the calculation; the tensor after it is what that section hands on.', prog.intro),
    ...prog.steps.map(stepOf),
    overview('All', 'The whole calculation', 'Every section, folded again. The last tensor is the output.', prog.outro),
  ];

  // ---- one cube
  // Cells as outlined regions: one rectangle per slab where the cells fill
  // one, single cells otherwise.
  const boxes = (src, cells) => {
    const bySlab = new Map();
    for (const f of cells) {
      const [d, r, c] = src.addr(f);
      if (!bySlab.has(d)) bySlab.set(d, []);
      bySlab.get(d).push([r, c]);
    }
    const out = [];
    for (const [d, a] of bySlab) {
      const [r0, r1, c0, c1] = [Math.min(...a.map((x) => x[0])), Math.max(...a.map((x) => x[0])), Math.min(...a.map((x) => x[1])), Math.max(...a.map((x) => x[1]))];
      if ((r1 - r0 + 1) * (c1 - c0 + 1) === a.length) out.push({ id: src.id, d, r0, r1, c0, c1 });
      else for (const [r, c] of a) out.push({ id: src.id, d, r0: r, r1: r, c0: c, c1: c });
    }
    return out;
  };
  const rowOf = (t, d, r) => ({ id: t.id, d, r0: r, r1: r, c0: 0, c1: colsOf(t) - 1 });
  const colOf = (t, d, c) => ({ id: t.id, d, r0: 0, r1: rowsOf(t) - 1, c0: c, c1: c });
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const table = (head, rows, foot) =>
    `<table class="terms"><thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows
      .map((r) => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${r.cells.map((x) => `<td>${x}</td>`).join('')}</tr>`)
      .join('')}</tbody><tfoot><tr>${foot.map((x) => `<td>${x}</td>`).join('')}</tr></tfoot></table>`;

  function explain(id, d, r, c) {
    const t = prog.t(id);
    const e = explainCell(t, t.flat(d, r, c));
    // The author's sentence about this cell, then the generated arithmetic.
    const said = t.hover ? renderProse(t.hover, env, { t, d, r, c, value: e.value }) : '';
    const out = { id, d, r, c, value: e.value, title: e.title, html: `${said}<p${said ? ' class="derived-p"' : ''}>${esc(e.text)}</p>`, groups: [], context: [] };
    for (const line of e.eq) out.html += `<p class="eq">${esc(line)}</p>`;
    if (e.terms && 'product' in e.terms[0]) {
      // With many terms, list the largest and fold the rest into one row.
      const MAX = 12;
      const all = e.terms.map((x, k) => ({ ...x, k }));
      const top = all.length <= MAX ? all : [...all].sort((x, y) => Math.abs(y.product) - Math.abs(x.product)).slice(0, MAX - 2).sort((x, y) => x.k - y.k);
      const rows = top.map((x) => ({ cells: [x.aName, fmt(x.a), fmt(x.b), fmt(x.product)], cls: Math.abs(x.product) < 0.005 ? 'faint' : '' }));
      if (top.length < all.length) {
        const shown = new Set(top.map((x) => x.k));
        rows.push({ cells: [`the other ${all.length - top.length}`, '', '', fmt(all.filter((x) => !shown.has(x.k)).reduce((s, x) => s + x.product, 0))], cls: 'faint' });
      }
      const [A, B] = t.op.inputs;
      out.html += table(['from', H(A), H(B), 'product'], rows, ['', '', 'Σ', fmt(e.value)]);
    } else if (e.terms) {
      out.html += table(['cell', 'x', 'share'], e.terms.map((x) => ({ cells: [x.name.slice(x.name.lastIndexOf('[')), fmt(x.x), fmt(x.weight)], cls: [x.weight < 0.005 ? 'faint' : '', x.current ? 'cur' : ''].join(' ').trim() })), ['Σ', '', fmt(1)]);
    }
    // Every source that is drawn lights up (an input made outside the steps is not drawn).
    for (const [src, cells] of e.sources) if (src.section) out.groups.push(...boxes(src, cells));
    // Parameters, constants and inputs have no sources. Say so, or the missing highlight looks like a fault.
    if (!out.groups.length) out.html += '<p class="derived-p"><b>Nothing else lights up:</b> this number is not computed from other numbers on screen.</p>';

    // Outlines for orientation only: the row a copy belongs to, the vector a weight is part of.
    const op = t.op;
    if (t.kind === 'param') {
      const user = drawn.find((u) => u.op?.inputs.includes(t));
      if (t.rank === 1) out.context.push(rowOf(t, 0, 0));
      else if (user?.op.family === 'contract' && user.op.inputs[1] === t) out.context.push(colOf(t, d, c));
      else out.context.push(rowOf(t, d, r));
    } else if (!op) {
      // an input: nothing to outline
    } else if (op.name === 'lookup' || op.name === 'rows') {
      const [[src, [f]]] = e.sources;
      const [sd, sr] = src.addr(f);
      out.context.push(rowOf(src, sd, sr), rowOf(t, d, r));
    } else if (op.name === 'softmax') out.context.push(rowOf(t, d, r));
    return out;
  }

  return {
    title: prog.name,
    subtitle: `Every cube is one real number.${prog.departures.some((d) => d.key === 'untrained') ? ' Untrained: all weights are random.' : ''}`,
    dims: prog.axes.map((a) => `${markupToHtml(a.sym)} = ${a.size}`).join(' · '),
    slabs, tensors: specs, sections, bars, steps, nSteps: steps.length,
    toggles: [...prog.toggles.values()],
    departures: prog.departures,
    explain,
    set: (name, value) => prog.set(name, value),
    prog, layout: L,
  };
}
