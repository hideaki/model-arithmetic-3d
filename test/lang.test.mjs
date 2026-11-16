// The language itself: what the compiler rejects, formulas, toggles, the
// source checker, and the derived explanations.

import test from 'node:test';
import assert from 'node:assert/strict';

import { program, axis, input, param, random, toggle, departure, section, step } from '../lang/core.js';
import { lookup, split, map, add, mean, softmax, matmul, causalMask } from '../lang/ops.js';
import { parse, compile, render } from '../lang/expr.js';
import { checkSources } from '../lang/check.js';
import { explain } from '../lang/explain.js';
import { build } from '../programs/transformer.js';

// A one-step program around `fn`, with the departure random weights require.
const tiny = (fn, { untrained = true } = {}) =>
  program('tiny', { seed: 3 }, () => {
    if (untrained) departure('untrained', 'random weights');
    section('s', 'Section', () => step('Step', fn));
  });

test('rejects a matmul across two different axes of the same size', () => {
  assert.throws(
    () => tiny(() => {
      const [n, a, b] = [axis('n', 2), axis('a', 4), axis('b', 4)];
      matmul(param('X', 'X', [n, a], random(1)), param('W', 'W', [b, n], random(1))).as('Y');
    }),
    /columns of X \(n, a\) are a but the rows of W \(b, n\) are b \(same size, different axes\)/,
  );
});

test('rejects a parameter given literal numbers', () => {
  assert.throws(() => tiny(() => param('W', 'W', [axis('n', 2)], [0.5, -0.5])), /literal numbers are not allowed/);
});

test('rejects random weights that are not declared as a departure', () => {
  assert.throws(() => tiny(() => param('W', 'W', [axis('n', 2)], random(1)), { untrained: false }), /declare departure\('untrained'/);
});

test('rejects a step that shows no new tensor', () => {
  assert.throws(() => tiny(() => {}), /step "Step" shows no new tensor/);
});

test('rejects an unnamed tensor and a tensor outside a step', () => {
  assert.throws(() => tiny(() => mean(param('X', 'X', [axis('n', 2), axis('m', 3)], random(1)))), /has no name/);
  assert.throws(() => program('p', {}, () => param('W', 'W', [axis('n', 2)], random(1))), /outside a step/);
});

test('rejects formulas with unknown names or unused operands', () => {
  const X = () => param('X', 'X', [axis('n', 2)], random(1));
  assert.throws(() => tiny(() => map('x * k', { x: X() }).as('Y')), /"k", which is neither an operand nor a known constant/);
  assert.throws(() => tiny(() => { const x = X(); map('x + 1', { x, y: x }).as('Y'); }), /operand y is never used/);
  assert.throws(() => tiny(() => map('x +', { x: X() }).as('Y')), /formula "x \+"/);
});

test('rejects combining different axes, and a split that does not divide', () => {
  assert.throws(
    () => tiny(() => {
      const [n, a, b] = [axis('n', 2), axis('a', 3), axis('b', 3)];
      add(param('X', 'X', [n, a], random(1)), param('Y', 'Y', [n, b], random(1))).as('Z');
    }),
    /cannot combine X \(n, a\) with Y \(n, b\): axis a meets b \(same size, different axes\)/,
  );
  assert.throws(
    () => tiny(() => {
      const [n, d, h, k] = [axis('n', 2), axis('d', 12), axis('h', 3), axis('k', 5)];
      split(param('X', 'X', [n, d], random(1)), d, [h, k]).as('Y');
    }),
    /d = 12 is not h × k = 3 × 5/,
  );
});

test('a lookup refuses a row number outside the table', () => {
  assert.throws(
    () => tiny(() => {
      const [v, n, d] = [axis('v', 3), axis('n', 2), axis('d', 2)];
      lookup(param('W', 'W', [v, d], random(1)), input('ids', 'ids', [n], [0, 3])).as('E');
    }),
    /3 is not a row of W/,
  );
});

test('formulas: precedence, unary minus, powers, constants, rendering', () => {
  const f = (src, vals = {}, consts = {}) => compile(parse(src), Object.keys(vals), consts)(Object.values(vals));
  assert.equal(f('1 + 2 * 3'), 7);
  assert.equal(f('(1 + 2) * 3'), 9);
  assert.equal(f('8 / 4 / 2'), 1);
  assert.equal(f('-2 ^ 2'), -4);
  assert.equal(f('2 ^ -1'), 0.5);
  assert.equal(f('(x - μ) / σ * γ', { x: 5, μ: 1, σ: 2, γ: 3 }), 6);
  assert.equal(f('s / sqrt(d_k)', { s: 3 }, { d_k: 4 }), 1.5);
  assert.equal(f('relu(x)', { x: -2 }), 0);
  assert.equal(render(parse('(x - μ) / σ * γ')), '(x − μ) / σ × γ');
  assert.equal(render(parse('s / sqrt(d_k)'), (n) => ({ s: '0.31', d_k: '4' })[n]), '0.31 / √4');
});

test('a toggle recomputes everything downstream', () => {
  const p = program('mask', {}, () => {
    const n = axis('n', 3);
    const on = toggle('causal', true);
    section('s', 'S', () => step('Mask', () => softmax(causalMask(n, on).as('M')).as('A')));
  });
  assert.deepEqual([...p.t('A').data.slice(0, 3)], [1, 0, 0]);
  p.set('causal', false);
  assert.deepEqual([...p.t('A').data.slice(0, 3)], [1 / 3, 1 / 3, 1 / 3]);
});

test('the source checker catches a missing source and a false one', () => {
  const p = tiny(() => {
    const [n, k, m] = [axis('n', 2), axis('k', 3), axis('m', 2)];
    matmul(param('X', 'X', [n, k], random(1)), param('W', 'W', [k, m], random(1))).as('Y');
  });
  const Y = p.t('Y');
  const real = Y.op.sources;
  assert.deepEqual(checkSources(Y), []);

  Y.op.sources = (o) => real(o).slice(0, 1); // forget the column of W
  assert.match(checkSources(Y)[0], /depends on W\[.*\], which is not listed as a source/);

  Y.op.sources = (o) => [...real(o), [p.t('X'), [o === 0 ? 5 : 0]]]; // claim a cell of another row
  assert.match(checkSources(Y)[0], /lists X\[.*\] as a source but does not depend on it/);
});

test('every cell of the transformer can be explained, and the arithmetic shown adds up', () => {
  const p = build();
  let cells = 0;
  for (const t of p.tensors) {
    for (let i = 0; i < t.size; i++) {
      const e = explain(t, i);
      assert.ok(e.title && e.text, `${t.id}[${i}]`);
      if (t.op?.family === 'contract') assert.equal(e.terms.reduce((s, x) => s + x.product, 0), t.data[i], e.title);
      if (t.op?.name === 'softmax') assert.ok(Math.abs(e.terms.reduce((s, x) => s + x.weight, 0) - 1) < 1e-12, e.title);
      cells++;
    }
  }
  assert.equal(cells, 8784 + 6); // every cube, plus the six token ids

  const at = (id, d, r, c) => explain(p.t(id), p.t(id).flat(d, r, c));
  assert.equal(at('E', 0, 1, 3).text, 'The same number as W_E[robot, 3]: lookup copies it and does no arithmetic.');
  assert.equal(at('b0.Qh', 2, 4, 1).title, 'Q[2, was, 1]');
  assert.match(at('b0.NA', 0, 1, 3).eq[1], /^\(.+ − .+\) \/ .+ × .+ = /);
  assert.equal(at('b0.M', 0, 0, 1).text, 'Fixed in advance, neither learned nor computed: M[i, j] = −∞ where j > i, otherwise 0.');
});
