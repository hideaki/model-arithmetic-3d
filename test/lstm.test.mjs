// The LSTM language model: its numbers against an independent calculation,
// its sources, its prose, its layout and what hovering lights up.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build, VOCAB, TOKENS, GATES } from '../programs/lstm.js';
import { checkProgram } from '../lang/check.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

const prog = build();
const data = (id) => Array.from(prog.t(id).data);
const close = (got, want, what) => {
  assert.equal(got.length, want.length, what);
  got.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12, `${what}[${i}]: ${v} is not ${want[i]}`));
};
const [V, DE, DH, T] = [VOCAB.length, 6, 5, TOKENS.length];

// Plain loops, written without the language.
// A row vector x (k numbers) times a matrix W (k x m, flat).
const rowTimes = (x, W, m) => Array.from({ length: m }, (_, j) => x.reduce((s, v, i) => s + v * W[i * m + j], 0));
const sig = (v) => 1 / (1 + Math.exp(-v));

test('every number equals an independent calculation', () => {
  // Only the weights are taken from the program; everything else is recomputed.
  const [WE, Wih, Whh, Wout] = ['t0.WE', 't0.Wih', 't0.Whh', 'Wout'].map(data);
  assert.deepEqual([WE.length, Wih.length, Whh.length, Wout.length], [V * DE, DE * 4 * DH, DH * 4 * DH, DH * V]);

  let h = new Array(DH).fill(0);
  let c = new Array(DH).fill(0);
  close(data('t0.h0'), h, 'h0');
  close(data('t0.c0'), c, 'c0');
  TOKENS.forEach((word, t) => {
    const id = VOCAB.indexOf(word);
    const x = WE.slice(id * DE, (id + 1) * DE);
    const ax = rowTimes(x, Wih, 4 * DH);
    const ah = rowTimes(h, Whh, 4 * DH);
    const z = ax.map((v, j) => v + ah[j]);
    const run = (k) => z.slice(k * DH, (k + 1) * DH);
    const [i, f, g, o] = [run(0).map(sig), run(1).map(sig), run(2).map(Math.tanh), run(3).map(sig)];
    const cNew = c.map((v, j) => f[j] * v + i[j] * g[j]);
    const hNew = cNew.map((v, j) => o[j] * Math.tanh(v));

    const at = (name) => data(`t${t}.${name}`);
    close(at('x'), x, `x ${t}`);
    close(at('a'), ax, `a_x ${t}`);
    close(at('r'), ah, `a_h ${t}`);
    close(at('z'), z, `z ${t}`);
    GATES.forEach((name, k) => close(at(`z${name}`), run(k), `z_${name} ${t}`));
    close(at('i'), i, `i ${t}`);
    close(at('f'), f, `f ${t}`);
    close(at('g'), g, `g ${t}`);
    close(at('o'), o, `o ${t}`);
    if (t > 0) close(at('cin'), c, `c brought in ${t}`);
    close(at('c'), cNew, `c ${t}`);
    close(at('h'), hNew, `h ${t}`);

    // What the arithmetic guarantees: gates between 0 and 1, candidate and hidden state between -1 and 1.
    assert.ok([...i, ...f, ...o].every((v) => v > 0 && v < 1) && [...g, ...hNew].every((v) => v > -1 && v < 1), `ranges ${t}`);
    // The same weights at every time step: the same memory, not a copy.
    for (const w of ['WE', 'Wih', 'Whh']) assert.equal(prog.t(`t${t}.${w}`).data, prog.t(`t0.${w}`).data, `${w} ${t}`);
    [h, c] = [hNew, cNew];
  });
  // With an all-zero hidden state the memory says nothing in the first time step.
  assert.ok(data('t0.r').every((v) => v === 0) && data('t1.r').some((v) => v !== 0));

  const logits = rowTimes(h, Wout, V);
  close(data('logits'), logits, 'logits');
  const ex = logits.map(Math.exp);
  const sum = ex.reduce((a, b) => a + b, 0);
  close(data('p'), ex.map((v) => v / sum), 'p');
  assert.ok(Math.abs(data('p').reduce((a, b) => a + b, 0) - 1) < 1e-12);
  // Weights are counted once however many tokens are read.
  const counted = prog.tensors.filter((t) => t.kind === 'param' && !t.alias).reduce((n, t) => n + t.size, 0);
  assert.equal(counted, V * DE + DE * 4 * DH + DH * 4 * DH + DH * V);
});

test('declared sources are true for every operation, and the prose has no warnings', () => {
  assert.deepEqual(checkProgram(prog), { checked: 61, problems: [] });
  assert.deepEqual(prog.warnings, []);
  for (const s of prog.steps) assert.ok(s.prose.length, `step "${s.title}" has no prose`);
  for (const t of prog.tensors) if (t.kind !== 'input') assert.ok(t.hover, `${t.id} has no hover sentence`);
  assert.deepEqual(prog.departures.map((d) => d.key), ['untrained', 'one layer', 'no biases', 'one prediction', 'token looked up inside its time step', 'tiny sizes']);
});

test('layout: nothing overlaps, data flows left to right, the state is handed on along the stream line', () => {
  const L = layout(prog);
  const top = (p) => p.cy + (p.R - 1) / 2;
  const bot = (p) => p.cy - (p.R - 1) / 2;
  for (const sec of prog.sections) {
    const placed = L.sections.get(sec).placed;
    assert.equal(placed.length, sec.tensors.length, sec.id);
    for (const a of placed)
      for (const b of placed) {
        if (a === b) break;
        const apart = a.x0 + a.C <= b.x0 || b.x0 + b.C <= a.x0 || bot(a) > top(b) || bot(b) > top(a);
        assert.ok(apart, `${a.t.id} and ${b.t.id} overlap`);
      }
  }
  for (const t of prog.tensors) {
    const p = L.at.get(t);
    if (!t.op || !p || p.role !== 'flow') continue;
    for (const o of t.op.inputs) {
      const q = o.section === t.section ? L.at.get(o) : null;
      if (q && q.role !== 'beside' && q.role !== 'over') assert.ok(q.x0 + q.C <= p.x0, `${t.id} is not right of ${o.id}`);
    }
  }
  const at = (id) => L.at.get(prog.t(id));
  for (let t = 0; t < T; t++) {
    const k = (name) => at(`t${t}.${name}`);
    // Each weight matrix sits between what it reads and what it makes.
    for (const [x, w, y] of [[t ? null : 'h0', 'Whh', 'r'], ['x', 'Wih', 'a']]) {
      assert.equal(k(w).role, 'weight');
      assert.ok((!x || k(x).x0 + k(x).C <= k(w).x0) && k(w).x0 + k(w).C <= k(y).x0, `${w} ${t}`);
    }
    // The token comes in on a lane of its own, above the memory's.
    assert.ok(bot(k('WE')) > top(k('Whh')) && k('x').cy > k('r').cy, `token lane ${t}`);
    // The four runs, and the gates made from them, stand in the order i, f, g, o from the top.
    for (const names of [GATES.map((g) => `z${g}`), GATES]) {
      const ys = names.map((n) => k(n).cy);
      assert.deepEqual(ys, [...ys].sort((a, b) => b - a), `gate order ${t}`);
      assert.equal(new Set(names.map((n) => k(n).x0)).size, 1, `gate column ${t}`);
    }
    // The new hidden state is the stream going out: on the line y = 0, last on the right.
    assert.equal(k('h').cy, 0, `h ${t}`);
    assert.equal(L.sections.get(prog.t(`t${t}.h`).section).keep.t.id, `t${t}.h`);
    if (t > 0) assert.equal(prog.t(`t${t}.h`).section.stream.id, `t${t - 1}.h`);
  }
  // Before the first token the old cell state is a constant, placed by the result that uses it.
  assert.deepEqual([at('t0.c0').role, at('t0.c0').x0], ['beside', at('t0.c').x0]);
  // Later it is brought in from the time step before, on a side lane below the gates.
  assert.ok(at('t1.cin').cy < at('t1.o').cy && at('t1.cin').x0 + at('t1.cin').C <= at('t1.c').x0);
  assert.equal(at('p').cy, 0);
});

test('the scene: steps, kinds, generated text and hover', () => {
  const scene = compile(prog);
  assert.equal(scene.steps.length, 16);
  assert.equal(scene.tensors.length, 76); // the four token ids are used, not drawn
  // The first time step is walked in nine steps; each later one is a single step.
  assert.deepEqual(prog.sections.map((s) => s.steps.length), [9, 1, 1, 1, 2]);
  assert.deepEqual(scene.bars.map((b) => [b.label, b.sub]), TOKENS.map((w, t) => [`TIME STEP ${t + 1}`, `Reading “${w}”`]));

  // Zeros set by rule are fixed tiles, weights are learned tiles, gates are drawn on the 0-to-1 scale.
  const kinds = Object.fromEntries(scene.tensors.map((t) => [t.id, t.kind]));
  assert.deepEqual(
    ['t0.h0', 't0.c0', 't0.Whh', 't2.WE', 't0.x', 't0.z', 't0.f', 't0.g', 't1.cin', 'p'].map((id) => kinds[id]),
    ['const', 'const', 'param', 'param', 'act', 'act', 'prob', 'act', 'act', 'prob'],
  );
  assert.ok(scene.steps.every((s) => s.focus === undefined)); // no slabs on this page

  // The cut into four runs is told and coded as a copy of columns.
  assert.match(scene.steps[6].body(), /columns 10 to 14 of/);
  assert.match(scene.steps[6].code, /zg = z\[\.\.\., 10:15\]/);
  assert.match(scene.steps[8].code, /c = f \* c0 \+ i \* g/);
  assert.match(scene.steps[9].code, /h = o \* torch\.tanh\(c\)/);

  // A lookup of a single row is told as one row, not as "each of the 1".
  assert.match(scene.steps[3].body(), /copies the one row of W<sub>E<\/sub> that the token id picks/);

  // Every operand of the two state formulas has an arrow into the result (hint arrows: 'all').
  const into = (k, step, id) => {
    const x = scene.layout.at.get(prog.t(id)).x0;
    return scene.sections[k].flows.filter((f) => f.type === 'flow' && !f.closed && f.step === step && Math.abs(f.pts.at(-1)[0] - (x - 0.8)) < 1e-9).length;
  };
  assert.deepEqual([into(0, 8, 't0.c'), into(0, 9, 't0.h')], [3, 2]); // f, i, g (c_0 stands beside); c and o
  assert.deepEqual([into(1, 10, 't1.c'), into(1, 10, 't1.h')], [4, 2]); // f, the old c, i, g; c and o

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
  assert.equal(cubes, 1617);

  // A new cell state lights one cube in each of the forget gate, the old state, the input gate and the candidate.
  assert.deepEqual(scene.explain('t0.c', 0, 0, 2).groups.map((g) => [g.id, g.r0, g.c0, g.c1]), [['t0.f', 0, 2, 2], ['t0.c0', 0, 2, 2], ['t0.i', 0, 2, 2], ['t0.g', 0, 2, 2]]);
  // From the second time step on, the old state is the one brought in, and that copy lights the original.
  assert.deepEqual(scene.explain('t2.c', 0, 0, 4).groups.map((g) => g.id), ['t2.f', 't2.cin', 't2.i', 't2.g']);
  assert.deepEqual(scene.explain('t2.cin', 0, 0, 4).groups.map((g) => [g.id, g.c0]), [['t1.c', 4]]);
  // A hidden state lights the cell state and the output gate; a run lights one cube of z.
  assert.deepEqual(scene.explain('t1.h', 0, 0, 3).groups.map((g) => [g.id, g.c0]), [['t1.c', 3], ['t1.o', 3]]);
  assert.deepEqual(scene.explain('t0.zo', 0, 0, 1).groups.map((g) => [g.id, g.c0, g.c1]), [['t0.z', 16, 16]]);
  // A raw gate number in a later time step reads the hidden state of the step before, across the boxes.
  assert.deepEqual(scene.explain('t3.r', 0, 0, 7).groups.map((g) => g.id), ['t2.h', 't3.Whh']);
  // The token's row lights one tile of the table, in the row of its word.
  assert.deepEqual(scene.explain('t1.x', 0, 0, 5).groups.map((g) => [g.id, g.r0, g.c0]), [['t1.WE', VOCAB.indexOf('cat'), 5]]);

  // Zeros set by rule and weights have no sources, and the inspector says so.
  for (const [id, text] of [['t0.h0', 'zero, by rule'], ['t0.c0', 'zero, by rule'], ['t0.Wih', 'which belongs to the input gate'], ['Wout', 'adds to the score of']]) {
    const ex = scene.explain(id, 0, 0, 1);
    assert.deepEqual(ex.groups, [], id);
    assert.ok(ex.html.includes(text) && ex.html.includes('Nothing else lights up:'), `${id}: ${ex.html}`);
  }
  // A weight met again says it is the very same weight.
  assert.match(scene.explain('t2.Whh', 0, 2, 7).html, /The very same weight as W_hh\[2, 7\]/);
  assert.ok(!scene.explain('t0.c', 0, 0, 1).html.includes('Nothing else lights up'));
});
