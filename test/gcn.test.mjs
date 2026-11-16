// The graph convolutional network: its numbers against an independent
// calculation, its sources, its prose, its layout and its scene. Also the
// `sum` operation, which was added to the language for this program.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build } from '../programs/gcn.js';
import { checkProgram, checkSources } from '../lang/check.js';
import { explain } from '../lang/explain.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

const prog = build();
const data = (id) => Array.from(prog.t(id).data);
const close = (got, want, what) => {
  assert.equal(got.length, want.length, what);
  got.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12, `${what}[${i}]: ${v} is not ${want[i]}`));
};

// Plain loops, written without the language. Tables are flat, row by row.
const [N, F, H, C] = [8, 6, 4, 3];
const times = (A, B, n, k, m) => {
  const out = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let x = 0; x < k; x++) s += A[i * k + x] * B[x * m + j];
      out.push(s);
    }
  return out;
};
const rowTotals = (A, n, m) => {
  const out = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += A[i * m + j];
    out.push(s);
  }
  return out;
};

test('every number equals an independent calculation', () => {
  // Taken from the program: the two inputs and the two weight tables. Everything else is recomputed.
  const [A, X0, W0, W1] = [data('A'), data('X0'), data('W0'), data('W1')];

  // The example graph is what the page says it is: 9 links, no direction, no node linked to itself.
  assert.equal(A.reduce((s, v) => s + v, 0), 2 * 9);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) assert.equal(A[i * N + j], i === j ? 0 : A[j * N + i]);

  const At = A.map((v, q) => v + (Math.floor(q / N) === q % N ? 1 : 0));
  close(data('I'), A.map((_, q) => (Math.floor(q / N) === q % N ? 1 : 0)), 'I');
  close(data('At'), At, 'At');
  const deg = rowTotals(At, N, N);
  close(data('deg'), deg, 'deg');
  assert.deepEqual(deg, [3, 3, 4, 4, 3, 4, 3, 2]);
  const dis = deg.map((d) => 1 / Math.sqrt(d));
  close(data('dis'), dis, 'dis');
  close(data('disT'), dis, 'disT');
  const Ah = At.map((v, q) => v * dis[Math.floor(q / N)] * dis[q % N]);
  close(data('Ah'), Ah, 'Ah');
  close(data('Ah1'), Ah, 'Ah1');
  close(data('Ah2'), Ah, 'Ah2');

  const rs = rowTotals(X0, N, F);
  close(data('rs'), rs, 'rs');
  const X = X0.map((v, q) => v / rs[Math.floor(q / F)]);
  close(data('X'), X, 'X');

  const M1 = times(X, W0, N, F, H);
  close(data('M1'), M1, 'M1');
  const P1 = times(Ah, M1, N, N, H);
  close(data('P1'), P1, 'P1');
  const H1 = P1.map((v) => (v > 0 ? v : 0));
  close(data('H1'), H1, 'H1');
  const M2 = times(H1, W1, N, H, C);
  close(data('M2'), M2, 'M2');
  const P2 = times(Ah, M2, N, N, C);
  close(data('P2'), P2, 'P2');
  const Z = [];
  for (let i = 0; i < N; i++) {
    const e = P2.slice(i * C, i * C + C).map(Math.exp);
    const total = e.reduce((a, b) => a + b, 0);
    Z.push(...e.map((v) => v / total));
  }
  close(data('Z'), Z, 'Z');

  // The paper's equation 9 in one line, multiplied in the other order: softmax(Â ReLU(Â X W0) W1).
  const other = times(times(Ah, times(times(Ah, X, N, N, F), W0, N, F, H).map((v) => (v > 0 ? v : 0)), N, N, H), W1, N, H, C);
  other.forEach((v, i) => assert.ok(Math.abs(v - P2[i]) < 1e-12, `equation 9, cell ${i}`));

  // What the steps promise: Â is symmetric, rows of X and of Z add up to 1, ReLU leaves nothing negative.
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) assert.ok(Math.abs(Ah[i * N + j] - Ah[j * N + i]) < 1e-15);
  for (const [id, m] of [['X', F], ['Z', C]]) for (const total of rowTotals(data(id), N, m)) assert.ok(Math.abs(total - 1) < 1e-12, id);
  assert.ok(data('H1').every((v) => v >= 0) && data('P1').some((v) => v < 0));
});

test('declared sources are true for every operation, and the prose has no warnings', () => {
  assert.deepEqual(checkProgram(prog), { checked: 15, problems: [] });
  // ReLU is flat below zero: a small nudge to a negative number moves nothing. The checker must not
  // take that for a source that is not one (it tries the cell on the other side of zero as well).
  assert.deepEqual(checkSources(prog.t('H1')), []);
  assert.deepEqual(prog.warnings, []);
  for (const s of prog.steps) assert.ok(s.prose.length, `step "${s.title}" has no prose`);
  for (const t of prog.tensors) assert.ok(t.hover, `${t.id} has no hover sentence`);
  assert.deepEqual(prog.departures.map((d) => d.key), ['untrained', 'tiny sizes', 'inference only', 'dense tables', 'preparation shown as steps', 'no drawing of the graph']);
});

test('sum: the total of each line, lit by the whole line', () => {
  const [At, deg, rs] = ['At', 'deg', 'rs'].map((id) => prog.t(id));
  assert.equal(deg.op.family, 'reduce');
  assert.deepEqual(deg.dims, [8, 1]);
  // Node 7 is linked to node 6 and to itself.
  const [[src, cells]] = prog.sources(deg, 7);
  assert.equal(src, At);
  assert.deepEqual(cells, [56, 57, 58, 59, 60, 61, 62, 63]);
  const e = explain(deg, 7);
  assert.equal(e.text, 'The total of the 8 numbers along N in Ã.');
  assert.deepEqual(e.eq, ['0.00 + 0.00 + 0.00 + 0.00 + 0.00 + 0.00 + 1.00 + 1.00 = 2.00']);
  assert.equal(explain(rs, 2).eq[0], '1.00 + 1.00 + 1.00 + 0.00 + 0.00 + 0.00 = 3.00');
});

test('layout: nothing overlaps, data flows left to right, the graph enters each layer from a side lane', () => {
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
  // The identity is placed by the sum that uses it, directly above the result.
  assert.deepEqual([at('I').role, at('I').x0], ['beside', at('At').x0]);
  assert.ok(at('I').cy > at('At').cy);
  // The features do not read the graph: their section starts afresh, clear of Â.
  assert.ok(!L.sections.get(prog.sections[1]).stream);
  assert.equal(at('X0').x0, 14);
  // In both layers: the weight table sits inline, and Â comes in below the flow.
  for (const [w, m, a, p, out] of [['W0', 'M1', 'Ah1', 'P1', 'H1'], ['W1', 'M2', 'Ah2', 'P2', 'Z']]) {
    assert.equal(at(w).role, 'weight');
    assert.ok(at(w).x0 + at(w).C <= at(m).x0, w);
    assert.equal(at(m).cy, 0);
    assert.ok(top(at(a)) < bot(at(m)), `${a} is below the flow`);
    assert.equal(at(p).cy, at(a).cy);
    // What a layer hands on sits on the stream line.
    assert.equal(at(out).cy, 0);
  }
});

test('the scene: steps, drawing convention, generated text and hover', () => {
  const scene = compile(prog);
  assert.equal(scene.steps.length, 15);
  assert.equal(scene.tensors.length, 20);
  const kinds = Object.fromEntries(scene.tensors.map((t) => [t.id, t.kind]));
  // Constants are fixed tiles, weights learned tiles, everything else cubes; the answer is drawn as probabilities.
  assert.deepEqual([kinds.I, kinds.W0, kinds.W1, kinds.A, kinds.Ah, kinds.Ah1, kinds.X, kinds.H1, kinds.Z], ['const', 'param', 'param', 'act', 'act', 'act', 'act', 'act', 'prob']);
  assert.ok(scene.steps.every((s) => s.focus === undefined)); // no slabs anywhere on this page

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
  assert.equal(cubes, 716);

  // The two inputs, the constant and a weight table have no sources, and the inspector says so.
  for (const [id, text] of [['A', 'Not linked: node 2 and node 5.'], ['X0', 'Node 2 contains word 1.'], ['I', 'Off the diagonal.'], ['W0', 'how much word 2 contributes to hidden number 1']]) {
    const ex = scene.explain(id, 0, 2, id === 'A' ? 5 : 1);
    assert.deepEqual(ex.groups, [], id);
    assert.ok(ex.html.includes(text) && ex.html.includes('Nothing else lights up:'), `${id}: ${ex.html}`);
  }

  // A degree lights its whole row of Ã; a scaled link lights one cell of Ã and the two scales.
  assert.deepEqual(scene.explain('deg', 0, 7, 0).groups.map((g) => [g.id, g.r0, g.r1, g.c0, g.c1]), [['At', 7, 7, 0, 7]]);
  assert.deepEqual(scene.explain('Ah', 0, 6, 7).groups.map((g) => [g.id, g.r0, g.c0]), [['At', 6, 7], ['dis', 6, 0], ['disT', 0, 7]]);
  assert.match(scene.explain('Ah', 0, 6, 7).html, /^<p>How much node 6 hears from node 7: <b>0\.41<\/b>\.<\/p>/);
  assert.match(scene.explain('Ah', 0, 0, 7).html, /^<p>Node 0 and node 7 are not linked: nothing passes: <b>0\.00<\/b>\.<\/p>/);
  // The copy of Â in a layer lights the original, cell for cell.
  assert.deepEqual(scene.explain('Ah2', 0, 3, 4).groups.map((g) => [g.id, g.r0, g.c0]), [['Ah', 3, 4]]);
  // The graph product: a row of Â and a column of what the weights made.
  const p1 = scene.explain('P1', 0, 7, 2);
  assert.deepEqual(p1.groups.map((g) => [g.id, g.r0, g.r1, g.c0, g.c1]), [['Ah1', 7, 7, 0, 7], ['M1', 0, 7, 2, 2]]);
  assert.ok(!p1.html.includes('Nothing else lights up'));

  // Generated sentences and code for the new operation and for the two formulas.
  assert.match(scene.steps[3].body(), /the total of each line of Ã along N, so \(8, 8\) → \(8, 1\)/);
  assert.match(scene.steps[3].code, /deg = At\.sum\(-1, keepdim=True\)/);
  assert.match(scene.steps[5].code, /Ah = At \* dis \* disT/);
  assert.match(scene.steps[7].code, /X = X0 \/ rs/);
  assert.match(scene.steps[10].code, /H1 = F\.relu\(P1\)/);
});
