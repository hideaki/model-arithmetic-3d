// The YOLO-style detector: its numbers against an independent calculation,
// its sources, its prose, and the layout properties every program must have.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build } from '../programs/yolo.js';
import { checkProgram } from '../lang/check.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

const prog = build();
const data = (id) => Array.from(prog.t(id).data);
const close = (got, want, what) => {
  assert.equal(got.length, want.length, what);
  got.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12, `${what}[${i}]: ${v} is not ${want[i]}`));
};

// Plain loops, written without the language.
function conv(X, [Ci, Hi, Wi], K, [Co, Kh, Kw], stride, pad) {
  const Ho = Math.floor((Hi + 2 * pad - Kh) / stride) + 1;
  const Wo = Math.floor((Wi + 2 * pad - Kw) / stride) + 1;
  const out = [];
  for (let o = 0; o < Co; o++)
    for (let y = 0; y < Ho; y++)
      for (let x = 0; x < Wo; x++) {
        let s = 0;
        for (let c = 0; c < Ci; c++)
          for (let dy = 0; dy < Kh; dy++)
            for (let dx = 0; dx < Kw; dx++) {
              const [iy, ix] = [y * stride + dy - pad, x * stride + dx - pad];
              if (iy >= 0 && iy < Hi && ix >= 0 && ix < Wi) s += X[(c * Hi + iy) * Wi + ix] * K[((o * Ci + c) * Kh + dy) * Kw + dx];
            }
        out.push(s);
      }
  return out;
}
const leaky = (xs) => xs.map((x) => Math.max(x, 0.1 * x));
const sig = (x) => 1 / (1 + Math.exp(-x));

test('every number equals an independent calculation', () => {
  const F1 = conv(data('img'), [3, 16, 16], data('K1'), [4, 3, 3], 2, 1);
  close(data('F1'), F1, 'F1');
  const A1 = leaky(F1);
  close(data('A1'), A1, 'A1');
  const A2 = leaky(conv(A1, [4, 8, 8], data('K2'), [8, 3, 3], 2, 1));
  close(data('A2'), A2, 'A2');
  const Y = conv(A2, [8, 4, 4], data('K3'), [8, 1, 1], 1, 0);
  close(data('Y'), Y, 'Y');

  const cell = (ch, i, j) => Y[(ch * 4 + i) * 4 + j];
  const grid = (chs, f) => chs.flatMap((ch, k) => Array.from({ length: 16 }, (_, q) => f(cell(ch, Math.floor(q / 4), q % 4), k, Math.floor(q / 4), q % 4)));
  close(data('bxy'), grid([0, 1], (t, k, i, j) => ((sig(t) + (k === 0 ? j : i)) * 16) / 4), 'bxy');
  close(data('bwh'), grid([2, 3], (t, k) => [6, 7][k] * Math.exp(t)), 'bwh');
  const obj = grid([4], sig);
  const cls = grid([5, 6, 7], sig);
  close(data('obj'), obj, 'obj');
  close(data('cls'), cls, 'cls');
  close(data('score'), cls.map((c, q) => obj[q % 16] * c), 'score');

  // What decoding guarantees: a centre inside its own cell, a positive size, scores between 0 and 1.
  const bxy = data('bxy');
  for (let q = 0; q < 16; q++) {
    const [i, j] = [Math.floor(q / 4), q % 4];
    assert.ok(bxy[q] > j * 4 && bxy[q] < (j + 1) * 4 && bxy[16 + q] > i * 4 && bxy[16 + q] < (i + 1) * 4, `cell ${i},${j}`);
  }
  assert.ok(data('bwh').every((v) => v > 0) && data('score').every((v) => v > 0 && v < 1));
});

test('declared sources are true for every operation, and the prose has no warnings', () => {
  assert.deepEqual(checkProgram(prog), { checked: 14, problems: [] });
  assert.deepEqual(prog.warnings, []);
  for (const s of prog.steps) assert.ok(s.prose.length, `step "${s.title}" has no prose`);
  for (const t of prog.tensors) assert.ok(t.hover, `${t.id} has no hover sentence`);
});

test('layout: nothing overlaps, data flows left to right, channel stacks line up', () => {
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
  // Filters sit between what they read and what they make, one slice per channel read, at the same depths.
  for (const [x, k, y] of [['img', 'K1', 'F1'], ['A1', 'K2', 'F2']]) {
    assert.equal(at(k).role, 'weight');
    assert.ok(at(x).x0 + at(x).C <= at(k).x0 && at(k).x0 + at(k).C <= at(y).x0, k);
    assert.deepEqual(at(k).zs, at(x).zs, k);
  }
  assert.deepEqual(at('K3').zs, at('A2').zs);
  // Constants are placed by the result that uses them.
  assert.deepEqual([at('cxy').role, at('cxy').x0], ['beside', at('bxy').x0]);
  assert.deepEqual([at('anchor').role, at('anchor').zs], ['over', at('bwh').zs]);
  // The last tensor of the model sits on the stream line.
  assert.equal(at('score').cy, 0);
});

test('the scene: steps, drawn picture, filter brackets and hover', () => {
  const scene = compile(prog);
  assert.equal(scene.steps.length, 13);
  assert.equal(scene.tensors.length, 20);
  assert.equal(scene.tensors.find((t) => t.id === 'img').kind, 'image');
  assert.equal(scene.tensors.find((t) => t.id === 'score').kind, 'prob');
  // Constants are drawn as fixed tiles, weights as learned tiles, everything computed as cubes.
  const kinds = Object.fromEntries(scene.tensors.map((t) => [t.id, t.kind]));
  assert.deepEqual([kinds.cxy, kinds.anchor, kinds.K1, kinds.K3, kinds.F1, kinds.bxy], ['const', 'const', 'param', 'param', 'act', 'act']);
  assert.deepEqual(scene.tensors.find((t) => t.id === 'K2').rowBrackets.length, 8);
  for (const sec of scene.sections) assert.ok(!sec.flows.some((f) => f.type === 'pane'), 'channel decks get no panes');
  assert.ok(scene.steps.every((s) => s.focus === undefined)); // no step opens on a single channel

  // The 1x1 head: one output cell reads one column through the 8 channels, and one filter.
  const ex = scene.explain('Y', 5, 2, 1);
  assert.equal(ex.groups.filter((g) => g.id === 'A2').length, 8);
  assert.ok(ex.groups.filter((g) => g.id === 'A2').every((g) => g.r0 === 2 && g.r1 === 2 && g.c0 === 1 && g.c1 === 1));
  assert.deepEqual(ex.groups.filter((g) => g.id === 'K3').map((g) => [g.r0, g.r1]), Array(8).fill([5, 5]));
  assert.match(ex.html, /^<p>Raw output channel 5 for grid cell \(row 2, column 1\)\.<\/p>/);

  // The first convolution reads the picture: one window per colour sheet lights up.
  const f1 = scene.explain('F1', 2, 3, 4);
  assert.deepEqual(f1.groups.filter((g) => g.id === 'img').map((g) => [g.d, g.r0, g.r1, g.c0, g.c1]), [[0, 5, 7, 7, 9], [1, 5, 7, 7, 9], [2, 5, 7, 7, 9]]);
  assert.equal(f1.groups.filter((g) => g.id === 'K1').length, 3);

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
  assert.equal(cubes, 2462);

  // A constant, a parameter and the picture have no sources, and the inspector says so.
  for (const [id, text] of [['cxy', 'The column number of this grid cell.'], ['K1', 'A weight of filter 0'], ['img', 'the brightness of this colour']]) {
    const ex = scene.explain(id, 0, 2, 1);
    assert.deepEqual(ex.groups, [], id);
    assert.ok(ex.html.includes(text) && ex.html.includes('Nothing else lights up:'), `${id}: ${ex.html}`);
  }
  assert.ok(!scene.explain('bxy', 0, 2, 1).html.includes('Nothing else lights up'));

  // A decoded centre lights its raw number and its cell's own position.
  assert.deepEqual(scene.explain('bxy', 1, 3, 2).groups.map((g) => [g.id, g.d, g.r0, g.c0]), [['txy', 1, 3, 2], ['cxy', 1, 3, 2]]);
  assert.match(scene.steps[8].body(), /W \/ S = 4/);
  assert.match(scene.steps[8].code, /bxy = \(torch\.sigmoid\(txy\) \+ cxy\) \* W \/ S/);
});
