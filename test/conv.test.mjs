// Convolution and banks of filters: numbers, sources, drawing addresses, and
// that layout and scene accept them.

import test from 'node:test';
import assert from 'node:assert/strict';

import { program, axis, input, param, random, departure, section, step } from '../lang/core.js';
import { conv2d, map } from '../lang/ops.js';
import { checkProgram } from '../lang/check.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

// image (1, 8, 8) -> conv 3x3, stride 2, pad 1 -> (4, 4, 4) -> leaky ReLU -> conv 3x3, stride 1, pad 0 -> (3, 2, 2)
const IMAGE = Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.7) + (i % 8 > 3 ? 0.5 : -0.5));
const build = () =>
  program('small convnet', { seed: 4 }, () => {
    departure('untrained', 'random filters');
    const [c0, H, W] = [axis('c_0', 1), axis('H', 8), axis('W', 8)];
    const [c1, H1, W1] = [axis('c_1', 4, { unit: 'channel' }), axis('H_1', 4), axis('W_1', 4)];
    const [c2, H2, W2] = [axis('c_2', 3, { unit: 'channel' }), axis('H_2', 2), axis('W_2', 2)];
    const k = axis('k', 3);
    section('net', 'Backbone', () => {
      const img = step('The image', () => input('img', 'image', [c0, H, W], IMAGE));
      const F1 = step('First convolution', () => conv2d(img, param('K1', 'K_{1}', [c1, c0, k, k], random(0.5)), { stride: 2, pad: 1, to: [H1, W1] }).as('F1', 'F_{1}'));
      const A1 = step('Leaky ReLU', () => map('max(x, 0.1 * x)', { x: F1 }).as('A1', 'A_{1}'));
      step('Second convolution', () => conv2d(A1, param('K2', 'K_{2}', [c2, c1, k, k], random(0.5)), { to: [H2, W2] }).as('F2', 'F_{2}'));
    });
  });

// The same arithmetic, written out independently.
function reference(X, [Ci, Hi, Wi], K, [Co, Kh, Kw], stride, pad) {
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
              const v = iy < 0 || iy >= Hi || ix < 0 || ix >= Wi ? 0 : X[(c * Hi + iy) * Wi + ix];
              s += v * K[((o * Ci + c) * Kh + dy) * Kw + dx];
            }
        out.push(s);
      }
  return out;
}

test('conv2d computes the convolution, with stride and zero padding', () => {
  const p = build();
  const F1 = reference(IMAGE, [1, 8, 8], p.t('K1').data, [4, 3, 3], 2, 1);
  p.t('F1').data.forEach((v, i) => assert.ok(Math.abs(v - F1[i]) < 1e-12, `F1[${i}]`));
  const A1 = F1.map((x) => Math.max(x, 0.1 * x));
  assert.deepEqual([...p.t('A1').data], A1);
  const F2 = reference(A1, [4, 4, 4], p.t('K2').data, [3, 3, 3], 1, 0);
  p.t('F2').data.forEach((v, i) => assert.ok(Math.abs(v - F2[i]) < 1e-12, `F2[${i}]`));
  assert.deepEqual(p.t('F2').dims, [3, 2, 2]);
});

test('declared sources of a convolution are true, padding included', () => {
  const p = build();
  assert.deepEqual(checkProgram(p), { checked: 3, problems: [] });
  // A corner of F1 hangs over the padding: 4 of the 9 window cells exist.
  const F1 = p.t('F1');
  assert.deepEqual(p.sources(F1, 0).map(([, cells]) => cells.length), [4, 4]);
  assert.deepEqual(p.sources(F1, F1.flat(0, 1, 1)).map(([, cells]) => cells.length), [9, 9]);
  // F2 reads every channel of A1: 4 channels x 9 cells.
  assert.deepEqual(p.sources(p.t('F2'), 0).map(([, cells]) => cells.length), [36, 36]);
});

test('conv2d rejects wrong output sizes and mismatched channels', () => {
  const tiny = (fn) =>
    program('t', {}, () => {
      departure('untrained', 'x');
      section('s', 'S', () => step('Conv', fn));
    });
  assert.throws(
    () => tiny(() => {
      const [c, H, W, o, k, H1] = [axis('c', 1), axis('H', 8), axis('W', 8), axis('o', 2), axis('k', 3), axis('H_1', 5)];
      conv2d(input('x', 'x', [c, H, W], new Array(64).fill(0)), param('K', 'K', [o, c, k, k], random(1)), { stride: 2, pad: 1, to: [H1, H1] }).as('Y');
    }),
    /with stride 2 and padding 1 the output height is 4, but H_\{1\} = 5/,
  );
  assert.throws(
    () => tiny(() => {
      const [c, c2, H, W, o, k, H1] = [axis('c', 2), axis('e', 2), axis('H', 4), axis('W', 4), axis('o', 2), axis('k', 3), axis('H_1', 2)];
      conv2d(input('x', 'x', [c, H, W], new Array(32).fill(0)), param('K', 'K', [o, c2, k, k], random(1)), { to: [H1, H1] }).as('Y');
    }),
    /has channels c but the filters K \(o, e, k, k\) expect e \(same size, different axes\)/,
  );
  assert.throws(() => tiny(() => map('x + 1', { x: param('K', 'K', [axis('a', 2), axis('b', 2), axis('k', 3), axis('k2', 3)], random(1)) }).as('Y')), /only a parameter can have 4 axes/);
});

test('a bank of filters is drawn one slab per input channel, filters stacked down the rows', () => {
  const p = build();
  const K2 = p.t('K2'); // (3 filters, 4 channels, 3, 3)
  assert.deepEqual(K2.view, { D: 4, R: 9, C: 3 });
  const seen = new Set();
  for (let i = 0; i < K2.size; i++) {
    const [d, r, c] = K2.addr(i);
    assert.equal(K2.flat(d, r, c), i);
    seen.add(`${d}:${r}:${c}`);
  }
  assert.equal(seen.size, K2.size);
  // Filter 2, channel 1, row 0, column 2 sits in slab 1, row 2*3 + 0, column 2.
  assert.deepEqual(K2.addr(((2 * 4 + 1) * 3 + 0) * 3 + 2), [1, 6, 2]);
});

test('layout and scene accept a convolutional program', () => {
  const p = build();
  const L = layout(p);
  const at = (id) => L.at.get(p.t(id));
  // The filters sit inline, between what they read and what they produce.
  assert.equal(at('K1').role, 'weight');
  assert.ok(at('img').x0 + at('img').C <= at('K1').x0 && at('K1').x0 < at('F1').x0 && at('F1').x0 < at('A1').x0 && at('A1').x0 + at('A1').C <= at('K2').x0 && at('K2').x0 < at('F2').x0);
  assert.deepEqual(at('K2').zs, at('A1').zs); // one filter slab per channel it reads, at the same depths

  const scene = compile(p);
  assert.equal(scene.slabs.n, 4);
  const K2 = scene.tensors.find((t) => t.id === 'K2');
  assert.deepEqual(K2.rowBrackets.map((b) => [b.from, b.to, b.label]), [[0, 2, 'channel 0'], [3, 5, 'channel 1'], [6, 8, 'channel 2']]);
  assert.deepEqual([K2.get().length, K2.get()[0].length, K2.get()[0][0].length], [4, 9, 3]);

  // Hovering an output cell lights one window per input channel and one filter slice per channel.
  const ex = scene.explain('F2', 1, 0, 1);
  assert.equal(ex.groups.filter((g) => g.id === 'A1').length, 4);
  assert.equal(ex.groups.filter((g) => g.id === 'K2').length, 4);
  for (const g of ex.groups) assert.deepEqual([g.r1 - g.r0, g.c1 - g.c0], [2, 2]);
  assert.deepEqual(ex.groups.filter((g) => g.id === 'K2').map((g) => g.r0), [3, 3, 3, 3]); // filter 1
  assert.match(ex.html, /A window of A_1, across its 4 channels, multiplied cell by cell with one filter of K_2: 36 products/);
  assert.match(scene.steps[2].body(), /each of the 4 filters in K<sub>1<\/sub> slides over image in steps of 2, with 1 cell of zero padding/);
  assert.equal(scene.steps[2].code.split('#')[0].trim(), 'F1 = F.conv2d(img, K1, stride=2, padding=1)');
  assert.equal(scene.tensors[0].id, 'img'); // an input made in a step is drawn
  assert.match(scene.steps[1].body(), /<b>image<\/b> .* = \(1, 8, 8\): given as input/);
});
