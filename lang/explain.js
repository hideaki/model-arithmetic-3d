// What to say about one cell, derived from the operation that produced it.
// Returns plain data (no HTML): a title, a sentence, equation lines, and for
// sums the list of terms. The sentence is generic arithmetic; anything about
// what the numbers mean has to come from the author.

import { ONE } from './core.js';
import { render } from './expr.js';

export const plain = (label) => label.replace(/[{}]/g, '');

export function fmt(v, n = 2) {
  if (!Number.isFinite(v)) return v < 0 ? '−∞' : '∞';
  const s = Math.abs(v).toFixed(n);
  return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
}
const par = (v, n) => (v < 0 ? `(${fmt(v, n)})` : fmt(v, n));

// "Q[robot, 3]": axis labels where the axis has them, numbers otherwise.
export function cellName(t, flat) {
  const dims = t.dims;
  const idx = new Array(dims.length);
  for (let i = dims.length - 1; i >= 0; i--) {
    idx[i] = flat % dims[i];
    flat = (flat - idx[i]) / dims[i];
  }
  const parts = t.shape.map((a, i) => (a === ONE ? null : a.labels ? a.labels[idx[i]] : idx[i])).filter((p) => p !== null);
  return `${plain(t.label)}[${parts.join(', ')}]`;
}

export function explain(t, i) {
  const v = t.data[i];
  const sources = t.prog.sources(t, i);
  const out = { title: cellName(t, i), value: v, text: '', eq: [], terms: null, sources };
  const { op } = t;
  const valuesOf = ([src, cells]) => cells.map((f) => src.data[f]);

  if (t.kind === 'param') {
    out.text = `Learned in a real model; random here (mean ${t.source.mean}, standard deviation ${t.source.std}).${
      t.alias ? ` The very same weight as ${cellName(t.alias, i)} where ${plain(t.alias.label)} first appears: a model's weights do not change from one call to the next.` : ''
    }`;
  } else if (t.kind === 'input') {
    out.text = 'Given as input.';
  } else if (op.family === 'const') {
    out.text = `Fixed in advance, neither learned nor computed: ${op.rule}.`;
  } else if (op.family === 'module') {
    const n = sources.reduce((s, [, cells]) => s + cells.length, 0);
    const total = op.inputs.reduce((s, x) => s + x.size, 0);
    out.text = `Computed by running the whole ${op.title} again, with the same weights, on ${op.inputs.map((x) => plain(x.label)).join(', ')}. ${
      n === total ? `All ${n} of their numbers have` : `${n} of their ${total} numbers have`
    } a part in this one. The steps in between are the ones drawn in full where the ${op.title} first appears; they are computed here and not drawn.`;
  } else if (op.family === 'rearrange') {
    const [[src, [f]]] = sources;
    out.text = op.name === 'again' ? `The same number as ${cellName(src, f)} in the ${src.section.title} section, brought in unchanged.` : `The same number as ${cellName(src, f)}: ${op.name} copies it and does no arithmetic.`;
  } else if (op.family === 'map') {
    const vals = Object.fromEntries(op.keys.map((k, n) => [k, valuesOf(sources[n])[0]]));
    // A single setting (a schedule value, a scale) is shown with more digits than a cell of a tensor.
    const digits = Object.fromEntries(op.keys.map((k, n) => [k, op.inputs[n].size === 1 ? 4 : 2]));
    const sub = (name) => (name in vals ? par(vals[name], digits[name]) : name in op.consts ? String(op.consts[name]) : undefined);
    out.text = `Computed from the cell at the same position in ${sources.map(([s, [f]]) => cellName(s, f)).join(', ')}.`;
    out.eq = [render(op.ast), `${render(op.ast, sub)} = ${fmt(v)}`];
  } else if (op.name === 'mean') {
    const xs = valuesOf(sources[0]);
    out.text = `The average of the ${xs.length} numbers along ${t.op.inputs[0].shape[op.attrs.pos].sym} in ${plain(op.inputs[0].label)}.`;
    out.eq = [`(${xs.map((x, k) => (k ? par(x) : fmt(x))).join(' + ')}) / ${xs.length} = ${fmt(v)}`];
  } else if (op.name === 'sum') {
    const xs = valuesOf(sources[0]);
    out.text = `The total of the ${xs.length} numbers along ${t.op.inputs[0].shape[op.attrs.pos].sym} in ${plain(op.inputs[0].label)}.`;
    out.eq = [`${xs.map((x, k) => (k ? par(x) : fmt(x))).join(' + ')} = ${fmt(v)}`];
  } else if (op.name === 'std') {
    const xs = valuesOf(sources[0]);
    const [mu] = valuesOf(sources[1]);
    const varr = xs.reduce((s, x) => s + (x - mu) ** 2, 0) / xs.length;
    out.text = `How spread out those ${xs.length} numbers are around their mean ${fmt(mu)}.`;
    out.eq = ['√(mean((x − μ)²) + ε)', `√(${fmt(varr, 3)} + ${op.attrs.eps}) = ${fmt(v, 3)}`];
  } else if (op.name === 'groupMean' || op.name === 'groupStd') {
    const xs = valuesOf(sources[0]);
    const g = Math.floor(sources[0][1][0] / xs.length);
    const where = `channels ${g * op.attrs.per} to ${(g + 1) * op.attrs.per - 1} of ${plain(op.inputs[0].label)}`;
    const mu = xs.reduce((s, x) => s + x, 0) / xs.length;
    if (op.name === 'groupMean') {
      out.text = `The average of all ${xs.length} numbers in ${where}: one group. Every channel of the group holds this same mean.`;
      out.eq = [`(sum of ${xs.length} numbers) / ${xs.length} = ${fmt(xs.reduce((s, x) => s + x, 0), 3)} / ${xs.length} = ${fmt(v, 3)}`];
    } else {
      const varr = xs.reduce((s, x) => s + (x - mu) ** 2, 0) / xs.length;
      out.text = `How spread out the ${xs.length} numbers in ${where} are around their mean ${fmt(mu, 3)}.`;
      out.eq = ['√(mean((x − μ)²) + ε)', `√(${fmt(varr, 3)} + ${op.attrs.eps}) = ${fmt(v, 3)}`];
    }
  } else if (op.name === 'softmax') {
    const [src, cells] = sources[0];
    const xs = cells.map((f) => src.data[f]);
    const mx = Math.max(...xs);
    const ex = xs.map((x) => Math.exp(x - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    out.text = `This cell's share of its line: every number is exponentiated, then divided by the line's total.`;
    out.eq = ['e^x / Σ e^x'];
    out.terms = cells.map((f, k) => ({ name: cellName(src, f), x: xs[k], weight: ex[k] / sum, current: f === i }));
  } else if (op.family === 'contract') {
    const [[A, aCells], [B, bCells]] = sources;
    out.text =
      op.name === 'conv2d'
        ? `A window of ${plain(A.label)}, across its ${A.dims[0]} channel${A.dims[0] > 1 ? 's' : ''}, multiplied cell by cell with one filter of ${plain(B.label)}: ${aCells.length} products, added up.${
            aCells.length < B.size / B.dims[0] ? ' The window hangs over the edge here; the padding counts as 0.' : ''
          }`
        : `A row of ${plain(A.label)} dotted with a column of ${plain(B.label)}: ${aCells.length} products, added up.`;
    out.terms = aCells.map((f, k) => ({ a: A.data[f], b: B.data[bCells[k]], product: A.data[f] * B.data[bCells[k]], aName: cellName(A, f), bName: cellName(B, bCells[k]) }));
    out.eq = [`Σ a × b = ${fmt(v)}`];
  }
  return out;
}
