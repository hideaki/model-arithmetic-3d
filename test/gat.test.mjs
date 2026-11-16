// The graph attention network: its numbers against an independent
// calculation, its sources, its prose, its layout and its scene. Also the
// formula function `mask`, which was added to the language for this program.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build } from '../programs/gat.js';
import { checkProgram } from '../lang/check.js';
import { explain } from '../lang/explain.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

const prog = build();
const data = (id) => Array.from(prog.t(id).data);
const close = (got, want, what) => {
  assert.equal(got.length, want.length, what);
  got.forEach((v, i) => assert.ok(v === want[i] || Math.abs(v - want[i]) < 1e-12, `${what}[${i}]: ${v} is not ${want[i]}`));
};

// Plain loops, written without the language. Tables are flat, row by row;
// a tensor with heads is flat as [head][row][column].
const [N, F, K, D] = [8, 6, 2, 3];
const KD = K * D;
const SLOPE = 0.2;

test('every number equals an independent calculation', () => {
  // Taken from the program: the two inputs and the three weight tensors. Everything else is recomputed.
  const [A, X, W, a1, a2] = ['A', 'X', 'W', 'a1', 'a2'].map(data);

  // The example graph is what the page says it is: 9 links, no direction, no node linked to itself.
  assert.equal(A.reduce((s, v) => s + v, 0), 2 * 9);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) assert.equal(A[i * N + j], i === j ? 0 : A[j * N + i]);
  // Every row of the features adds up to 1.
  for (let i = 0; i < N; i++) assert.ok(Math.abs(X.slice(i * F, i * F + F).reduce((s, v) => s + v, 0) - 1) < 1e-12);

  const At = A.map((v, q) => v + (Math.floor(q / N) === q % N ? 1 : 0));
  close(data('I'), A.map((_, q) => (Math.floor(q / N) === q % N ? 1 : 0)), 'I');
  close(data('At'), At, 'At');
  const M = At.map((v) => (v === 1 ? 0 : -Infinity));
  close(data('M'), M, 'M');
  close(data('M1'), M, 'M1');

  const XW = [];
  for (let i = 0; i < N; i++)
    for (let j = 0; j < KD; j++) {
      let s = 0;
      for (let f = 0; f < F; f++) s += X[i * F + f] * W[f * KD + j];
      XW.push(s);
    }
  close(data('XW'), XW, 'XW');
  const Wh = [];
  for (let k = 0; k < K; k++) for (let i = 0; i < N; i++) for (let f = 0; f < D; f++) Wh.push(XW[i * KD + k * D + f]);
  close(data('Wh'), Wh, 'Wh');
  close(data('Wh2'), Wh, 'Wh2');

  const score = (a) => {
    const out = [];
    for (let k = 0; k < K; k++)
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let f = 0; f < D; f++) s += Wh[(k * N + i) * D + f] * a[k * D + f];
        out.push(s);
      }
    return out;
  };
  const [s, t] = [score(a1), score(a2)];
  close(data('s'), s, 's');
  close(data('t'), t, 't');
  close(data('tT'), t, 'tT');

  // e_ij = LeakyReLU(aᵀ [Wh_i ‖ Wh_j]), written as the paper writes it: one vector a over the two rows end to end.
  const e = [];
  for (let k = 0; k < K; k++)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const a = [...a1.slice(k * D, k * D + D), ...a2.slice(k * D, k * D + D)];
        const both = [...Wh.slice((k * N + i) * D, (k * N + i) * D + D), ...Wh.slice((k * N + j) * D, (k * N + j) * D + D)];
        const x = a.reduce((sum, w, n) => sum + w * both[n], 0);
        e.push(x > 0 ? x : SLOPE * x);
      }
  close(data('e'), e, 'e');
  const em = e.map((v, q) => v + M[q % (N * N)]);
  close(data('em'), em, 'em');

  // Softmax over the neighbours of node i only, written over the list of neighbours, not with the mask.
  const alpha = new Array(K * N * N).fill(0);
  for (let k = 0; k < K; k++)
    for (let i = 0; i < N; i++) {
      const nb = [];
      for (let j = 0; j < N; j++) if (i === j || A[i * N + j]) nb.push(j);
      const ex = nb.map((j) => Math.exp(e[(k * N + i) * N + j]));
      const total = ex.reduce((x, y) => x + y, 0);
      nb.forEach((j, n) => (alpha[(k * N + i) * N + j] = ex[n] / total));
    }
  close(data('alpha'), alpha, 'alpha');

  const H = [];
  for (let k = 0; k < K; k++)
    for (let i = 0; i < N; i++)
      for (let f = 0; f < D; f++) {
        let sum = 0;
        for (let j = 0; j < N; j++) sum += alpha[(k * N + i) * N + j] * Wh[(k * N + j) * D + f];
        H.push(sum);
      }
  close(data('H'), H, 'H');
  const Hc = [];
  for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) for (let f = 0; f < D; f++) Hc.push(H[(k * N + i) * D + f]);
  close(data('Hc'), Hc, 'Hc');
  close(data('Ho'), Hc.map((x) => (x > 0 ? x : Math.exp(x) - 1)), 'Ho');

  // What the steps promise: some numbers are negative before ELU; within a head every row of e ranks the
  // columns in the order of t (static attention).
  assert.ok(Hc.some((x) => x < 0) && Hc.some((x) => x > 0));
  for (let k = 0; k < K; k++)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        for (let jj = 0; jj < N; jj++)
          if (t[k * N + j] > t[k * N + jj]) assert.ok(e[(k * N + i) * N + j] >= e[(k * N + i) * N + jj], `head ${k}, row ${i}: ${j} before ${jj}`);
});

test('declared sources are true for every operation, and the prose has no warnings', () => {
  assert.deepEqual(checkProgram(prog), { checked: 15, problems: [] });
  assert.deepEqual(prog.warnings, []);
  for (const s of prog.steps) assert.ok(s.prose.length, `step "${s.title}" has no prose`);
  for (const t of prog.tensors) assert.ok(t.hover, `${t.id} has no hover sentence`);
  assert.deepEqual(prog.departures.map((d) => d.key), [
    'untrained', 'tiny sizes', 'one layer', 'inference only', 'no bias', 'dense table and mask', 'features prepared beforehand', 'no drawing of the graph',
  ]);
});

test('mask: 0 where there is a link, −∞ where there is none, lit by its one cell', () => {
  const M = prog.t('M');
  assert.equal(M.op.family, 'map');
  // Node 7 is linked to node 6 and to itself only.
  assert.deepEqual(Array.from(M.data.slice(56, 64)), [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity, -Infinity, 0, 0]);
  assert.deepEqual(prog.sources(M, 62).map(([src, cells]) => [src.id, cells]), [['At', [62]]]);
  assert.deepEqual(explain(M, 62).eq, ['mask(a)', 'mask(1.00) = 0.00']);
  assert.deepEqual(explain(M, 60).eq, ['mask(a)', 'mask(0.00) = −∞']);
});

test('layout: nothing overlaps, data flows left to right, the mask and the values come in where they are used', () => {
  const L = layout(prog);
  const top = (p) => p.cy + (p.R - 1) / 2;
  const bot = (p) => p.cy - (p.R - 1) / 2;
  for (const sec of prog.sections) {
    const placed = L.sections.get(sec).placed;
    assert.equal(placed.length, sec.tensors.length, sec.id);
    for (const a of placed)
      for (const b of placed) {
        if (a === b) break;
        const apart =
          a.x0 + a.C <= b.x0 || b.x0 + b.C <= a.x0 || bot(a) > top(b) || bot(b) > top(a) ||
          Math.min(...a.zs) > Math.max(...b.zs) || Math.min(...b.zs) > Math.max(...a.zs);
        assert.ok(apart, `${a.t.id} and ${b.t.id} overlap`);
      }
  }
  for (const t of prog.tensors) {
    const p = L.at.get(t);
    if (!t.op || p.role !== 'flow') continue;
    for (const o of t.op.inputs) {
      const q = o.section === t.section ? L.at.get(o) : null;
      if (q && q.role !== 'beside' && q.role !== 'over') assert.ok(q.x0 + q.C <= p.x0, `${t.id} is not right of ${o.id}`);
    }
  }
  const at = (id) => L.at.get(prog.t(id));
  // The features do not read the mask: their section starts afresh.
  assert.ok(!L.sections.get(prog.sections[1]).stream);
  assert.equal(at('X').x0, 14);
  // The weight table and the listener's attention vector sit inline; the line runs through to α and out.
  for (const [w, r] of [['W', 'XW'], ['a1', 's']]) {
    assert.equal(at(w).role, 'weight');
    assert.ok(at(w).x0 + at(w).C <= at(r).x0, w);
  }
  for (const id of ['XW', 'Wh', 's', 'e', 'em', 'alpha', 'H', 'Hc', 'Ho']) assert.equal(at(id).cy, 0, id);
  // The mask enters the scores section on a side lane below the line.
  assert.ok(top(at('M1')) < bot(at('em')));
});

test('the scene: steps, drawing convention, generated text and hover', () => {
  const scene = compile(prog);
  assert.equal(scene.steps.length, 15);
  assert.equal(scene.tensors.length, 21);
  const byId = Object.fromEntries(scene.tensors.map((t) => [t.id, t]));
  const kinds = Object.fromEntries(scene.tensors.map((t) => [t.id, t.kind]));
  assert.deepEqual(
    ['I', 'W', 'a1', 'a2', 'A', 'X', 'M', 'M1', 'e', 'em', 'alpha', 'Ho'].map((id) => kinds[id]),
    ['const', 'param', 'param', 'param', 'act', 'act', 'act', 'act', 'act', 'act', 'prob', 'act'],
  );
  // The shares of unlinked nodes are drawn blocked, exactly where the graph has no link.
  const A = data('A');
  for (let d = 0; d < K; d++)
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) assert.equal(byId.alpha.blocked(d, r, c), !(r === c || A[r * N + c]), `alpha[${d}, ${r}, ${c}]`);

  // For every cube of every tensor, what lights up is exactly its sources.
  let cubes = 0;
  for (const spec of scene.tensors) {
    const t = prog.t(spec.id);
    for (let i = 0; i < t.size; i++) {
      const [d, r, c] = t.addr(i);
      const lit = new Set();
      for (const g of scene.explain(spec.id, d, r, c).groups) for (let rr = g.r0; rr <= g.r1; rr++) for (let cc = g.c0; cc <= g.c1; cc++) lit.add(`${g.id}:${g.d}:${rr}:${cc}`);
      const want = new Set();
      for (const [src, cells] of prog.sources(t, i)) for (const f of cells) want.add(`${src.id}:${src.addr(f).join(':')}`);
      assert.deepEqual([...lit].sort(), [...want].sort(), `${spec.id}[${d}, ${r}, ${c}]`);
      cubes++;
    }
  }
  assert.equal(cubes, 1136); // 1,024 computed numbers, plus the copies of M and Wh brought in with again

  // Inputs, the constant and the weights have no sources, and the inspector says so.
  for (const [id, text] of [['A', 'Not linked: node 2 and node 5.'], ['X', 'the node does not contain it'], ['I', 'Off the diagonal.'], ['W', 'how much word 2 contributes to column 5']]) {
    const ex = scene.explain(id, 0, 2, id === 'A' || id === 'W' ? 5 : 4);
    assert.deepEqual(ex.groups, [], id);
    assert.ok(ex.html.includes(text) && ex.html.includes('Nothing else lights up:'), `${id}: ${ex.html}`);
  }

  // A pair score lights the listener's score and the speaker's score; a share lights its whole row.
  assert.deepEqual(scene.explain('e', 1, 3, 5).groups.map((g) => [g.id, g.d, g.r0, g.c0]), [['s', 1, 3, 0], ['tT', 1, 0, 5]]);
  assert.deepEqual(scene.explain('alpha', 0, 7, 6).groups.map((g) => [g.id, g.d, g.r0, g.r1, g.c0, g.c1]), [['em', 0, 7, 7, 0, 7]]);
  assert.deepEqual(scene.explain('M1', 0, 3, 4).groups.map((g) => [g.id, g.r0, g.c0]), [['M', 3, 4]]);
  assert.match(scene.explain('M', 0, 0, 7).html, /^<p>Node 0 may not listen to node 7: they are not linked\.<\/p>/);

  // Generated sentences and code for the new function and for the formulas.
  assert.match(scene.steps[3].body(), /M<\/b> = mask\(Ã\), applied to every cube of Ã on its own/);
  assert.match(scene.steps[3].code, /M = torch\.where\(At > 0, 0\.0, float\('-inf'\)\)/);
  assert.match(scene.steps[8].code, /e = torch\.maximum\(s \+ tT, 0\.2 \* \(s \+ tT\)\)/);
  assert.match(scene.steps[13].code, /Ho = F\.relu\(Hc\) \+ torch\.exp\(-F\.relu\(-Hc\)\) - 1/);
});
