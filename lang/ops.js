// Operations. Each one belongs to a family, and the family says which input
// cells an output cell is computed from:
//
//   rearrange  exactly one input cell              lookup, rows, transpose, split, merge,
//                                                  channels, columns, again, reshape,
//                                                  flatten, unflatten, upsample, concat
//   map        the same-position cell of each       add, gelu, silu, any formula
//              operand (with broadcasting)
//   reduce     every cell along one axis, or        mean, sum, std, softmax,
//              every cell of a group of channels   groupMean, groupStd
//   contract   a row of one operand and a          matmul
//              column of the other, or a window
//              of one and a filter of the other    conv2d
//   const      nothing: fixed by a rule            causalMask, constant
//   module     every cell of every input: a part   module
//              of the program run again, undrawn
//
// An operation supplies run(out), which fills out.data from its inputs, and
// sources(i), which lists the cells output cell i was computed from as
// [[tensor, [flat index, ...]], ...]. check.js verifies that the two agree.

import { Tensor, ONE, current } from './core.js';
import { parse, compile, names as namesIn } from './expr.js';

const sizes = (shape) => shape.map((a) => a.size);
const product = (dims) => dims.reduce((n, d) => n * d, 1);
const unravel = (flat, dims) => {
  const idx = new Array(dims.length);
  for (let i = dims.length - 1; i >= 0; i--) {
    idx[i] = flat % dims[i];
    flat = (flat - idx[i]) / dims[i];
  }
  return idx;
};
const ravel = (idx, dims) => idx.reduce((f, v, i) => f * dims[i] + v, 0);
const sameShape = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const make = (kind, shape, op) => new Tensor(current(op.name), kind, shape, op);

// ---------------------------------------------------------------------------
// Rearrange: every output cell is a copy of one input cell.

function rearrange(name, inputs, shape, indexOf, attrs = {}) {
  const op = {
    family: 'rearrange', name, inputs, attrs,
    build(out) {
      this.which = new Int32Array(out.size);
      this.at = new Int32Array(out.size);
      for (let i = 0; i < out.size; i++) [this.which[i], this.at[i]] = indexOf(i);
    },
    run(out) {
      this.build(out); // the mapping may depend on an index input
      for (let i = 0; i < out.size; i++) out.data[i] = inputs[this.which[i]].data[this.at[i]];
    },
    sources(i) {
      return [[inputs[this.which[i]], [this.at[i]]]];
    },
  };
  return make('act', shape, op);
}

// One row of `table` per entry of `ids`: (V, d) and (T,) -> (T, d).
export function lookup(table, ids) {
  if (ids.kind !== 'input' || ids.rank !== 1) throw new Error(`lookup: ${ids.name} must be a rank-1 input of row numbers`);
  const [rowAxis, ...rest] = table.shape;
  const w = table.size / rowAxis.size;
  return rearrange('lookup', [table], [ids.shape[0], ...rest], (i) => {
    const id = ids.data[(i - (i % w)) / w];
    if (!Number.isInteger(id) || id < 0 || id >= rowAxis.size) throw new Error(`lookup: ${id} is not a row of ${table.name}`);
    return [0, id * w + (i % w)];
  }, { index: ids });
}

// The first rows of `table`, as many as `ax` is long: (Tmax, d) -> (T, d).
export function rows(table, ax) {
  const [rowAxis, ...rest] = table.shape;
  if (ax.size > rowAxis.size) throw new Error(`rows: ${table.name} has ${rowAxis.size} rows, fewer than ${ax.sym} = ${ax.size}`);
  return rearrange('rows', [table], [ax, ...rest], (i) => [0, i]);
}

// Swap the last two axes.
export function transpose(X) {
  if (X.rank < 2) throw new Error(`transpose: ${X.name} has one axis`);
  const shape = [...X.shape.slice(0, -2), X.shape.at(-1), X.shape.at(-2)];
  const [inDims, outDims] = [X.dims, sizes(shape)];
  return rearrange('transpose', [X], shape, (i) => {
    const idx = unravel(i, outDims);
    const n = idx.length;
    [idx[n - 1], idx[n - 2]] = [idx[n - 2], idx[n - 1]];
    return [0, ravel(idx, inDims)];
  });
}

// Split the last axis into two and bring the first of them to the front:
// (T, d_model) -> (h, T, d_k), where d_model = h * d_k.
export function split(X, ax, [a, b]) {
  if (X.shape.at(-1) !== ax) throw new Error(`split: the last axis of ${X.name} is ${X.shape.at(-1).sym}, not ${ax.sym}`);
  if (ax.size !== a.size * b.size) throw new Error(`split: ${ax.sym} = ${ax.size} is not ${a.sym} × ${b.sym} = ${a.size} × ${b.size}`);
  const lead = X.shape.slice(0, -1);
  const shape = [a, ...lead, b];
  const [inDims, outDims] = [X.dims, sizes(shape)];
  return rearrange('split', [X], shape, (i) => {
    const idx = unravel(i, outDims);
    return [0, ravel([...idx.slice(1, -1), idx[0] * b.size + idx.at(-1)], inDims)];
  }, { axis: ax, into: [a, b] });
}

// The inverse of split: (h, T, d_k) -> (T, d_model).
export function merge(X, [a, b], ax) {
  if (X.shape[0] !== a || X.shape.at(-1) !== b) throw new Error(`merge: ${X.name} is ${X.sym}, not (${a.sym}, …, ${b.sym})`);
  if (ax.size !== a.size * b.size) throw new Error(`merge: ${ax.sym} = ${ax.size} is not ${a.sym} × ${b.sym} = ${a.size} × ${b.size}`);
  const shape = [...X.shape.slice(1, -1), ax];
  const [inDims, outDims] = [X.dims, sizes(shape)];
  return rearrange('merge', [X], shape, (i) => {
    const idx = unravel(i, outDims);
    const c = idx.at(-1);
    return [0, ravel([(c - (c % b.size)) / b.size, ...idx.slice(0, -1), c % b.size], inDims)];
  }, { axis: ax, from: [a, b] });
}

// Channels start .. start + ax.size - 1 of X: (c, H, W) -> (ax, H, W).
export function channels(X, start, ax) {
  const [c, ...rest] = X.shape;
  if (X.rank < 2 || start < 0 || start + ax.size > c.size) throw new Error(`channels: ${X.name} ${X.sym} has no channels ${start} to ${start + ax.size - 1}`);
  const w = X.size / c.size;
  return rearrange('channels', [X], [ax, ...rest], (i) => [0, start * w + i], { start, count: ax.size });
}

// Columns start .. start + ax.size - 1 of X, the same run in every row:
// (n, 4h) -> (n, h). One piece of PyTorch's chunk() along the last axis.
export function columns(X, start, ax) {
  const last = X.shape.at(-1);
  if (!Number.isInteger(start) || start < 0 || start + ax.size > last.size) throw new Error(`columns: ${X.name} ${X.sym} has no columns ${start} to ${start + ax.size - 1}`);
  const n = ax.size;
  return rearrange('columns', [X], [...X.shape.slice(0, -1), ax], (i) => [0, ((i - (i % n)) / n) * last.size + start + (i % n)], { start, count: n });
}

// The same tensor again, unchanged: how a tensor made in an earlier section is
// brought into the one that needs it (a prompt into every attention layer, say).
export function again(X) {
  return rearrange('again', [X], X.shape, (i) => [0, i]);
}

// Add or drop unit axes: (1, c) -> (c, 1, 1). The numbers keep their order.
export function reshape(X, shape) {
  const real = (sh) => sh.filter((a) => a !== ONE);
  if (!sameShape(real(X.shape), real(shape))) throw new Error(`reshape: ${X.name} ${X.sym} can only gain or lose unit axes`);
  return rearrange('reshape', [X], shape, (i) => [0, i]);
}

// A grid of channels as a table with one row per grid cell:
// (c, H, W) -> (N, c), where N = H * W. Row r is the cell at (r div W, r mod W).
export function flatten(X, n) {
  if (X.rank !== 3) throw new Error(`flatten: ${X.name} must be (channels, height, width)`);
  const [c, H, W] = X.shape;
  if (n.size !== H.size * W.size) throw new Error(`flatten: ${n.sym} = ${n.size} is not ${H.sym} × ${W.sym} = ${H.size} × ${W.size}`);
  return rearrange('flatten', [X], [n, c], (i) => [0, (i % c.size) * n.size + (i - (i % c.size)) / c.size], { grid: [H, W] });
}

// The inverse of flatten: (N, c) -> (c, H, W).
export function unflatten(X, [H, W]) {
  if (X.rank !== 2) throw new Error(`unflatten: ${X.name} must be (positions, channels)`);
  const [n, c] = X.shape;
  if (n.size !== H.size * W.size) throw new Error(`unflatten: ${n.sym} = ${n.size} is not ${H.sym} × ${W.sym} = ${H.size} × ${W.size}`);
  return rearrange('unflatten', [X], [c, H, W], (i) => [0, (i % n.size) * c.size + (i - (i % n.size)) / n.size], { grid: [H, W] });
}

// Nearest-neighbour upsampling: every cell is copied into a factor x factor
// block. (c, H, W) -> (c, H', W'), with the new axes named in `to`.
export function upsample(X, { factor = 2, to } = {}) {
  if (X.rank !== 3) throw new Error(`upsample: ${X.name} must be (channels, height, width)`);
  const [c, H, W] = X.shape;
  if (!to || to.length !== 2) throw new Error('upsample: name the output height and width axes with { to: [H, W] }');
  to.forEach((ax, i) => {
    if (ax.size !== [H, W][i].size * factor) throw new Error(`upsample: ${factor} × ${[H, W][i].size} is not ${ax.sym} = ${ax.size}`);
  });
  const [Hi, Wi, Ho, Wo] = [H.size, W.size, to[0].size, to[1].size];
  return rearrange('upsample', [X], [c, to[0], to[1]], (i) => {
    const x = i % Wo;
    const y = ((i - x) / Wo) % Ho;
    const ch = (i - x - y * Wo) / (Wo * Ho);
    return [0, (ch * Hi + Math.floor(y / factor)) * Wi + Math.floor(x / factor)];
  }, { factor });
}

// Channels of A followed by channels of B: (a, H, W) and (b, H, W) -> (ax, H, W).
export function concat(A, B, ax) {
  if (A.rank !== B.rank || !sameShape(A.shape.slice(1), B.shape.slice(1)))
    throw new Error(`concat: ${A.name} ${A.sym} and ${B.name} ${B.sym} must agree on every axis but the first`);
  const [a, b] = [A.shape[0], B.shape[0]];
  if (ax.size !== a.size + b.size) throw new Error(`concat: ${ax.sym} = ${ax.size} is not ${a.sym} + ${b.sym} = ${a.size} + ${b.size}`);
  return rearrange('concat', [A, B], [ax, ...A.shape.slice(1)], (i) => (i < A.size ? [0, i] : [1, i - A.size]), { counts: [a.size, b.size] });
}

// ---------------------------------------------------------------------------
// Map: each output cell is a formula of the cell at the same position in every
// operand. Shapes broadcast as in NumPy, aligned from the right, except that
// axes must be the same axis (not merely the same size) or the unit axis.

function broadcast(ins) {
  const rank = Math.max(...ins.map((t) => t.rank));
  const shape = new Array(rank).fill(ONE);
  const owner = new Array(rank).fill(null);
  for (const t of ins)
    t.shape.forEach((a, j) => {
      const p = rank - t.rank + j;
      if (a === ONE) return;
      if (shape[p] !== ONE && shape[p] !== a)
        throw new Error(
          `cannot combine ${owner[p].name} ${owner[p].sym} with ${t.name} ${t.sym}: axis ${shape[p].sym} meets ${a.sym}` +
            (shape[p].size === a.size ? ' (same size, different axes)' : ''),
        );
      shape[p] = a;
      owner[p] = t;
    });
  return shape;
}

// For each output cell, the flat index of the matching cell in an operand.
function indexMap(inShape, outShape) {
  const [inDims, outDims] = [sizes(inShape), sizes(outShape)];
  const off = outShape.length - inShape.length;
  const map = new Int32Array(product(outDims));
  for (let i = 0; i < map.length; i++) {
    const idx = unravel(i, outDims);
    map[i] = ravel(inDims.map((d, j) => (inShape[j] === ONE ? 0 : idx[off + j])), inDims);
  }
  return map;
}

// map('(x - μ) / σ * γ', { x: X, μ: mu, σ: sd, γ: g })
// Names that are not operands are looked up in opts.consts, then among the
// program's axes (an axis name stands for its size, e.g. d_k).
export function map(formula, operands, opts = {}) {
  const p = current('map()');
  const keys = Object.keys(operands);
  const ins = keys.map((k) => operands[k]);
  const ast = parse(formula);
  const used = namesIn(ast);
  for (const k of keys) if (!used.has(k)) throw new Error(`map "${formula}": operand ${k} is never used`);
  const consts = { ...Object.fromEntries(p.axes.map((a) => [a.name, a.size])), ...opts.consts };
  const f = compile(ast, keys, consts);
  const shape = broadcast(ins);
  const maps = ins.map((t) => indexMap(t.shape, shape));
  const v = new Array(ins.length);
  const op = {
    family: 'map', name: opts.name || 'map', show: opts.show, inputs: ins, formula, ast, keys, consts,
    run(out) {
      for (let i = 0; i < out.size; i++) {
        for (let k = 0; k < ins.length; k++) v[k] = ins[k].data[maps[k][i]];
        out.data[i] = f(v);
      }
    },
    sources(i) {
      return ins.map((t, k) => [t, [maps[k][i]]]);
    },
  };
  return make('act', shape, op);
}

export const add = (a, b) => map('a + b', { a, b }, { name: 'add' });
export const gelu = (x) => map('gelu(x)', { x }, { name: 'gelu', show: 'GELU' });
export const sigmoid = (x) => map('sigmoid(x)', { x }, { name: 'sigmoid', show: 'σ' });
export const silu = (x) => map('x * sigmoid(x)', { x }, { name: 'silu', show: 'SiLU' });

// ---------------------------------------------------------------------------
// Reduce: each output cell is computed from every cell along one axis.

// For each output cell, the flat index of the first cell of its line in X.
function lineStarts(X, pos, keep) {
  const dims = X.dims;
  const stride = product(dims.slice(pos + 1));
  const outDims = keep ? dims : dims.map((d, i) => (i === pos ? 1 : d));
  const base = new Int32Array(product(outDims));
  for (let o = 0; o < base.length; o++) {
    const idx = unravel(o, outDims);
    idx[pos] = 0;
    base[o] = ravel(idx, dims);
  }
  return { base, stride, n: dims[pos] };
}
const lineOf = (L, o) => Array.from({ length: L.n }, (_, j) => L.base[o] + j * L.stride);
const axisPos = (X, axis) => (axis < 0 ? X.rank + axis : axis);
const collapsed = (X, pos) => X.shape.map((a, i) => (i === pos ? ONE : a));

// Mean along an axis; the axis stays as a unit axis: (T, d) -> (T, 1).
export function mean(X, { axis = -1 } = {}) {
  const pos = axisPos(X, axis);
  const L = lineStarts(X, pos, false);
  return make('act', collapsed(X, pos), {
    family: 'reduce', name: 'mean', inputs: [X], attrs: { pos, n: L.n },
    run(out) {
      for (let o = 0; o < out.size; o++) {
        let s = 0;
        for (let j = 0; j < L.n; j++) s += X.data[L.base[o] + j * L.stride];
        out.data[o] = s / L.n;
      }
    },
    sources: (o) => [[X, lineOf(L, o)]],
  });
}

// Sum along an axis; the axis stays as a unit axis: (N, N) -> (N, 1).
export function sum(X, { axis = -1 } = {}) {
  const pos = axisPos(X, axis);
  const L = lineStarts(X, pos, false);
  return make('act', collapsed(X, pos), {
    family: 'reduce', name: 'sum', inputs: [X], attrs: { pos, n: L.n },
    run(out) {
      for (let o = 0; o < out.size; o++) {
        let s = 0;
        for (let j = 0; j < L.n; j++) s += X.data[L.base[o] + j * L.stride];
        out.data[o] = s;
      }
    },
    sources: (o) => [[X, lineOf(L, o)]],
  });
}

// Spread around a mean that was computed first: sqrt(mean((x - mean)^2) + eps).
export function std(X, { mean: mu, eps = 0, axis = -1 } = {}) {
  const pos = axisPos(X, axis);
  const shape = collapsed(X, pos);
  if (!mu || !sameShape(mu.shape, shape)) throw new Error(`std: pass the mean of ${X.name} along the same axis as { mean }`);
  const L = lineStarts(X, pos, false);
  return make('act', shape, {
    family: 'reduce', name: 'std', inputs: [X, mu], attrs: { pos, n: L.n, eps },
    run(out) {
      for (let o = 0; o < out.size; o++) {
        let s = 0;
        for (let j = 0; j < L.n; j++) s += (X.data[L.base[o] + j * L.stride] - mu.data[o]) ** 2;
        out.data[o] = Math.sqrt(s / L.n + eps);
      }
    },
    sources: (o) => [[X, lineOf(L, o)], [mu, [o]]],
  });
}

// Softmax along an axis: same shape, every line sums to 1.
export function softmax(X, { axis = -1 } = {}) {
  const pos = axisPos(X, axis);
  const L = lineStarts(X, pos, true);
  return make('act', X.shape, {
    family: 'reduce', name: 'softmax', inputs: [X], attrs: { pos, n: L.n },
    run(out) {
      for (let o = 0; o < out.size; o++) {
        if (L.base[o] !== o) continue; // once per line, at its first cell
        let mx = -Infinity;
        for (let j = 0; j < L.n; j++) mx = Math.max(mx, X.data[o + j * L.stride]);
        let sum = 0;
        for (let j = 0; j < L.n; j++) sum += out.data[o + j * L.stride] = Math.exp(X.data[o + j * L.stride] - mx);
        for (let j = 0; j < L.n; j++) out.data[o + j * L.stride] /= sum;
      }
    },
    sources: (o) => [[X, lineOf(L, o)]],
  });
}

// Group statistics, as in group normalisation. The channels of (c, H, W) are
// taken in `groups` equal runs. Each channel's one cell holds the mean of its
// whole group: every channel in the group, at every position. Channels in the
// same group therefore hold the same number. (c, H, W) -> (c, 1, 1).
function groupOf(X, groups, what) {
  if (X.rank !== 3) throw new Error(`${what}: ${X.name} must be (channels, height, width)`);
  const c = X.shape[0];
  if (!Number.isInteger(groups) || c.size % groups) throw new Error(`${what}: ${c.sym} = ${c.size} channels do not divide into ${groups} groups`);
  const per = c.size / groups;
  const w = X.size / c.size;
  const start = (ch) => (ch - (ch % per)) * w;
  return { per, n: per * w, start, cells: (ch) => Array.from({ length: per * w }, (_, j) => start(ch) + j) };
}

export function groupMean(X, { groups } = {}) {
  const g = groupOf(X, groups, 'groupMean');
  return make('act', [X.shape[0], ONE, ONE], {
    family: 'reduce', name: 'groupMean', inputs: [X], attrs: { groups, per: g.per, n: g.n },
    run(out) {
      for (let ch = 0; ch < out.size; ch++) {
        let s = 0;
        for (let j = 0; j < g.n; j++) s += X.data[g.start(ch) + j];
        out.data[ch] = s / g.n;
      }
    },
    sources: (ch) => [[X, g.cells(ch)]],
  });
}

// Spread of each group around its mean: sqrt(mean((x - mean)^2) + eps).
export function groupStd(X, { groups, mean: mu, eps = 0 } = {}) {
  const g = groupOf(X, groups, 'groupStd');
  const shape = [X.shape[0], ONE, ONE];
  if (!mu || !sameShape(mu.shape, shape)) throw new Error(`groupStd: pass the group means of ${X.name} as { mean }`);
  return make('act', shape, {
    family: 'reduce', name: 'groupStd', inputs: [X, mu], attrs: { groups, per: g.per, n: g.n, eps },
    run(out) {
      for (let ch = 0; ch < out.size; ch++) {
        let s = 0;
        for (let j = 0; j < g.n; j++) s += (X.data[g.start(ch) + j] - mu.data[ch]) ** 2;
        out.data[ch] = Math.sqrt(s / g.n + eps);
      }
    },
    sources: (ch) => [[X, g.cells(ch)], [mu, [ch]]],
  });
}

// ---------------------------------------------------------------------------
// Contract: each output cell is a row of A dotted with a column of B.

// (..., n, k) @ (..., k, m) -> (..., n, m). B may leave out the leading axes,
// in which case the same matrix is used for every slab of A.
export function matmul(A, B) {
  if (A.rank < 2 || B.rank < 2) throw new Error(`matmul: ${A.name} and ${B.name} both need at least two axes`);
  const k = A.shape.at(-1);
  const kb = B.shape.at(-2);
  if (k !== kb)
    throw new Error(
      `matmul: the columns of ${A.name} ${A.sym} are ${k.sym} but the rows of ${B.name} ${B.sym} are ${kb.sym}` +
        (k.size === kb.size ? ' (same size, different axes)' : ''),
    );
  const lead = A.shape.slice(0, -2);
  const shared = B.rank === 2;
  if (!shared && !sameShape(B.shape.slice(0, -2), lead))
    throw new Error(`matmul: the leading axes of ${A.name} ${A.sym} and ${B.name} ${B.sym} differ`);
  const [n, K, m] = [A.shape.at(-2).size, k.size, B.shape.at(-1).size];
  const slabs = product(sizes(lead));
  const bAt = (s) => (shared ? 0 : s * K * m);
  return make('act', [...lead, A.shape.at(-2), B.shape.at(-1)], {
    family: 'contract', name: 'matmul', symbol: '@', inputs: [A, B], attrs: { n, K, m, shared },
    run(out) {
      for (let s = 0; s < slabs; s++)
        for (let i = 0; i < n; i++)
          for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let x = 0; x < K; x++) sum += A.data[(s * n + i) * K + x] * B.data[bAt(s) + x * m + j];
            out.data[(s * n + i) * m + j] = sum;
          }
    },
    sources(o) {
      const j = o % m;
      const row = (o - j) / m; // slab * n + i
      const s = (row - (row % n)) / n;
      const xs = Array.from({ length: K }, (_, x) => x);
      return [[A, xs.map((x) => row * K + x)], [B, xs.map((x) => bAt(s) + x * m + j)]];
    },
  });
}

// Convolution: X (in, H, W) with a bank of filters K (out, in, kh, kw) gives
// (out, H', W'). Each output cell is one window of X, across every input
// channel, multiplied cell by cell with one filter and added up. The output's
// height and width axes are named by the author in `to` and must have the
// sizes the stride and padding imply. Padding cells are zeros that exist
// nowhere, so they are not sources.
export function conv2d(X, K, { stride = 1, pad = 0, to } = {}) {
  if (X.rank !== 3 || K.rank !== 4) throw new Error(`conv2d: ${X.name} must be (in, H, W) and ${K.name} (out, in, kh, kw)`);
  const [cin, H, W] = X.shape;
  const [cout, kin, kh, kw] = K.shape;
  if (cin !== kin)
    throw new Error(`conv2d: ${X.name} ${X.sym} has channels ${cin.sym} but the filters ${K.name} ${K.sym} expect ${kin.sym}` + (cin.size === kin.size ? ' (same size, different axes)' : ''));
  const want = [H, W].map((ax, i) => Math.floor((ax.size + 2 * pad - [kh, kw][i].size) / stride) + 1);
  if (!to || to.length !== 2) throw new Error('conv2d: name the output height and width axes with { to: [H, W] }');
  to.forEach((ax, i) => {
    if (ax.size !== want[i]) throw new Error(`conv2d: with stride ${stride} and padding ${pad} the output ${i ? 'width' : 'height'} is ${want[i]}, but ${ax.sym} = ${ax.size}`);
  });
  const [Ci, Hi, Wi, Co, Kh, Kw, Ho, Wo] = [cin.size, H.size, W.size, cout.size, kh.size, kw.size, to[0].size, to[1].size];
  // The (X cell, K cell) pairs that make up output cell o, in a fixed order.
  const pairs = (o) => {
    const x = o % Wo;
    const y = ((o - x) / Wo) % Ho;
    const f = (o - x - y * Wo) / (Wo * Ho);
    const out = [];
    for (let c = 0; c < Ci; c++)
      for (let dy = 0; dy < Kh; dy++)
        for (let dx = 0; dx < Kw; dx++) {
          const [iy, ix] = [y * stride + dy - pad, x * stride + dx - pad];
          if (iy >= 0 && iy < Hi && ix >= 0 && ix < Wi) out.push([(c * Hi + iy) * Wi + ix, ((f * Ci + c) * Kh + dy) * Kw + dx]);
        }
    return out;
  };
  return make('act', [cout, to[0], to[1]], {
    family: 'contract', name: 'conv2d', symbol: '∗', inputs: [X, K], attrs: { stride, pad },
    run(out) {
      for (let o = 0; o < out.size; o++) {
        let sum = 0;
        for (const [a, b] of pairs(o)) sum += X.data[a] * K.data[b];
        out.data[o] = sum;
      }
    },
    sources(o) {
      const ps = pairs(o);
      return [[X, ps.map((p) => p[0])], [K, ps.map((p) => p[1])]];
    },
  });
}

// ---------------------------------------------------------------------------
// Module: a part of the program run again on other inputs, without drawing it.
//
// fn builds that part from `inputs` out of ordinary operations and returns its
// last tensor. Everything it makes is computed exactly but belongs to no
// section; only the result becomes a tensor on screen. Its weights must be
// shared ones that are drawn elsewhere. The sources of an output cell are the
// input cells it depends on through the undrawn operations, found by following
// their own sources back; check.js verifies them like any other source.
export function module(title, inputs, fn, { prefix = '' } = {}) {
  const p = current('module()');
  if (p.hide) throw new Error('module() cannot be nested');
  if (!p.step) throw new Error(`module ${title} was called outside a step()`);
  const hidden = [];
  const outer = p.prefix;
  p.hide = hidden;
  p.prefix = outer + prefix;
  let result;
  try {
    result = fn();
  } finally {
    p.hide = null;
    p.prefix = outer;
  }
  if (!(result instanceof Tensor) || !hidden.includes(result)) throw new Error(`module ${title}: fn must return the last tensor it computed`);
  // Whatever it reads must be handed to it: otherwise a dependency would go unseen.
  for (const h of hidden)
    for (const o of h.op?.inputs || [])
      if (!hidden.includes(o) && !inputs.includes(o)) throw new Error(`module ${title} reads ${o.name}, which is not one of its inputs`);
  const refresh = () => {
    for (const h of hidden) if (h.op) h.op.run(h);
  };
  // For every cell computed along the way, the set of input cells it depends
  // on, as one bit per input cell; an operation's cell is the union of its sources'.
  let table = null;
  const depend = () => {
    if (table) return table;
    const starts = [];
    let bits = 0;
    for (const x of inputs) {
      starts.push(bits);
      bits += x.size;
    }
    const n = Math.ceil(bits / 32);
    const sets = new Map(
      inputs.map((x, k) => {
        const s = new Uint32Array(x.size * n);
        for (let c = 0; c < x.size; c++) s[c * n + ((starts[k] + c) >>> 5)] |= 1 << ((starts[k] + c) & 31);
        return [x, s];
      }),
    );
    for (const h of hidden) {
      const s = new Uint32Array(h.size * n);
      if (h.op)
        for (let i = 0; i < h.size; i++)
          for (const [src, cells] of h.op.sources(i)) {
            const from = sets.get(src);
            for (const c of cells) for (let w = 0; w < n; w++) s[i * n + w] |= from[c * n + w];
          }
      sets.set(h, s);
    }
    const r = sets.get(result);
    table = Array.from({ length: result.size }, (_, i) =>
      inputs
        .map((x, k) => [x, Array.from({ length: x.size }, (_, c) => c).filter((c) => r[i * n + ((starts[k] + c) >>> 5)] & (1 << ((starts[k] + c) & 31)))])
        .filter(([, cells]) => cells.length),
    );
    return table;
  };
  return make('act', result.shape, {
    family: 'module', name: 'module', title, inputs, hidden, result, refresh,
    run(out) {
      refresh();
      out.data.set(result.data);
    },
    sources: (i) => depend()[i],
  });
}

// ---------------------------------------------------------------------------
// Const: numbers fixed by a rule on positions, neither learned nor computed
// from other tensors.

// (T, T): -inf where column j is after row i, else 0. All zeros when off.
export function causalMask(ax, on) {
  const n = ax.size;
  return make('const', [ax, ax], {
    family: 'const', name: 'causalMask', inputs: [], controls: [on], mask: true,
    py: `torch.full((${n}, ${n}), float('-inf')).triu(1)`,
    rule: 'M[i, j] = −∞ where j > i, otherwise 0',
    run(out) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out.data[i * n + j] = on.value && j > i ? -Infinity : 0;
    },
    sources: () => [],
  });
}

// Numbers fixed in advance: by a rule on positions (a function of the cell's
// indices) or as a chosen setting (a list). `rule` says which, in words; it
// is what the viewer is told. `py` is the PyTorch that would build it.
export function constant(shape, values, { rule, py } = {}) {
  if (!rule) throw new Error('constant: say where the numbers come from with { rule }');
  const dims = sizes(shape);
  if (typeof values !== 'function' && values.length !== product(dims)) throw new Error(`constant: ${values.length} values for ${product(dims)} cells`);
  return make('const', shape, {
    family: 'const', name: 'constant', inputs: [], rule, py,
    run(out) {
      for (let i = 0; i < out.size; i++) out.data[i] = typeof values === 'function' ? values(...unravel(i, dims)) : values[i];
    },
    sources: () => [],
  });
}
