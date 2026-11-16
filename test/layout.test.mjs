// The derived layout and scene: properties that must hold for any program,
// checked on the transformer.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build } from '../programs/transformer.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

const prog = build();
const L = layout(prog);
const scene = compile(prog);
const top = (p) => p.cy + (p.R - 1) / 2;
const bot = (p) => p.cy - (p.R - 1) / 2;

test('every drawn tensor has a place, and no two share space', () => {
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
});

test('data flows left to right: a result is never left of what it is computed from', () => {
  for (const t of prog.tensors) {
    const p = L.at.get(t);
    if (!p || !t.op || p.role !== 'flow') continue;
    for (const o of t.op.inputs) {
      const q = o.section === t.section ? L.at.get(o) : null;
      if (!q || q.role === 'beside' || q.role === 'front' || q.role === 'over' || p.stackedUnder === o) continue;
      assert.ok(q.x0 + q.C <= p.x0, `${t.id} at ${p.x0} is not right of ${o.id}`);
    }
  }
});

test('each section hands on its last tensor on the stream line, at its right edge', () => {
  for (const sec of prog.sections) {
    const S = L.sections.get(sec);
    assert.equal(S.keep.cy, 0, sec.id);
    assert.equal(S.keep.x0 + S.keep.C, Math.max(...S.placed.map((p) => p.x0 + p.C)), sec.id);
  }
});

test('an operand beside a result lines its columns up with the result', () => {
  const beside = [...L.at.values()].filter((p) => p.role === 'beside');
  assert.equal(beside.length, 4); // K^T and V, in each block
  for (const p of beside) {
    const of = L.at.get(p.of);
    assert.deepEqual([p.x0, p.C], [of.x0, of.C], p.t.id);
    assert.ok(p.side === 'above' ? bot(p) > top(of) : top(p) < bot(of), p.t.id);
  }
});

test('a repeated section is laid out the same way every time', () => {
  const shape = (id) => L.sections.get(prog.sections.find((s) => s.id === id)).placed.map((p) => [p.t.id.replace(/^b\d\./, ''), p.x0, p.cy, p.zs, p.role, p.below]);
  assert.deepEqual(shape('att1'), shape('att0'));
  assert.deepEqual(shape('mlp1'), shape('mlp0'));
});

test('the scene is complete and consistent', () => {
  const ids = new Set(scene.tensors.map((t) => t.id));
  assert.equal(scene.tensors.length, 80);
  assert.equal(scene.steps.length, prog.nSteps);
  for (const [i, s] of scene.steps.entries()) {
    assert.ok(s.title && s.chapter && s.view.ids.length, `step ${i}`);
    for (const id of s.view.ids) assert.ok(ids.has(id), `step ${i} frames ${id}`);
    assert.equal(typeof s.body(), 'string');
  }
  for (const sec of scene.sections) {
    assert.ok(sec.wOpen > sec.wClosed && ids.has(sec.keep), sec.id);
    for (const f of sec.flows) {
      assert.ok(f.closed ? f.step === 0 : f.step >= sec.from && f.step <= sec.to, `${sec.id}: a ${f.type} at step ${f.step}`);
      const points = f.pts || (f.type === 'pane' ? [[f.x0, f.y0, f.z], [f.x1, f.y1, f.z]] : [f.at || [f.x, f.y, 0]]);
      for (const pt of points) assert.ok(pt.every(Number.isFinite), `${sec.id}: ${JSON.stringify(f)}`);
    }
  }
  // The causal mask is a constant: drawn as fixed tiles, not as cubes or learned weights.
  const kind = (id) => scene.tensors.find((t) => t.id === id).kind;
  assert.deepEqual([kind('b0.M'), kind('b1.M'), kind('b0.WQ'), kind('WET'), kind('b0.Sm')], ['mask', 'mask', 'param', 'param', 'act']);
  // The two tables are learned weights: tiles. The rows taken out of them for this sentence are cubes.
  // A transposed table is still the whole table: tiles.
  assert.deepEqual([kind('WE'), kind('E'), kind('WP'), kind('P'), kind('X0'), kind('WET')], ['param', 'act', 'param', 'act', 'act', 'param']);
  // A copy is drawn on the same colour scale as its source.
  const group = (id) => scene.tensors.find((t) => t.id === id).group;
  assert.equal(group('E'), group('WE'));
  assert.equal(group('b0.Qh'), group('b0.Q'));
  assert.equal(group('WET'), group('WE'));
});

test('what the scene lights up for a cube is exactly its sources', () => {
  let cubes = 0;
  for (const spec of scene.tensors) {
    const t = prog.t(spec.id);
    for (let i = 0; i < t.size; i += 7) {
      const [d, r, c] = t.addr(i);
      const ex = scene.explain(spec.id, d, r, c);
      const lit = new Set();
      for (const g of ex.groups) for (let rr = g.r0; rr <= g.r1; rr++) for (let cc = g.c0; cc <= g.c1; cc++) lit.add(`${g.id}:${g.d}:${rr}:${cc}`);
      const want = new Set();
      for (const [src, cells] of prog.sources(t, i)) for (const f of cells) want.add(`${src.id}:${src.addr(f).join(':')}`);
      assert.deepEqual([...lit].sort(), [...want].sort(), `${spec.id}[${d}, ${r}, ${c}]`);
      assert.equal(ex.value, t.data[i]);
      cubes++;
    }
  }
  assert.ok(cubes > 1200);
});
