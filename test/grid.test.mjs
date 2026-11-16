// The operations a U-Net needs on grids of channels: group statistics,
// upsampling, concatenation, grid <-> sequence, and bringing a tensor in from
// another section. Each is checked for its numbers, its sources and what it rejects.

import test from 'node:test';
import assert from 'node:assert/strict';

import { program, axis, ONE, input, param, random, shared, departure, section, step, repeat, part } from '../lang/core.js';
import { again, columns, reshape, flatten, unflatten, upsample, concat, groupMean, groupStd, silu, matmul, transpose, module, map, constant } from '../lang/ops.js';
import { checkSources, checkProgram } from '../lang/check.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';
import { picture, render, prose } from '../lang/prose.js';

const range = (n) => Array.from({ length: n }, (_, i) => i);

// One section holding a (4, 2, 3) grid numbered 0..23, and whatever fn builds from it.
const grid = (fn) => {
  let out;
  const prog = program('grid', {}, () => {
    const [c, H, W] = [axis('c', 4, { deck: true }), axis('H', 2), axis('W', 3)];
    section('s', 'S', () =>
      step('x', () => {
        const X = input('X', 'X', [c, H, W], range(24));
        out = fn(X, { c, H, W });
      }),
    );
  });
  return { prog, out };
};

test('flatten and unflatten: one row per grid cell, and back', () => {
  const { out } = grid((X, { H, W }) => {
    const N = axis('N', 6);
    const F = flatten(X, N).as('F');
    const U = unflatten(F, [H, W]).as('U');
    return { F, U, X };
  });
  assert.deepEqual(out.F.dims, [6, 4]);
  // Row r is the cell at (r div 3, r mod 3); its columns are the four channels.
  assert.deepEqual(Array.from(out.F.data.slice(0, 4)), [0, 6, 12, 18]);
  assert.deepEqual(Array.from(out.F.data.slice(5 * 4)), [5, 11, 17, 23]);
  assert.deepEqual(Array.from(out.U.data), range(24));
  assert.deepEqual(out.F.op.sources(5), [[out.X, [7]]]); // row 1, column 1 is channel 1 of grid cell (0, 1)
  assert.deepEqual([checkSources(out.F), checkSources(out.U)], [[], []]);
  assert.throws(() => grid((X) => flatten(X, axis('N', 5)).as('F')), /N = 5 is not H × W = 2 × 3/);
});

test('upsample copies every cell into a block', () => {
  const { out } = grid((X) => upsample(X, { to: [axis('H2', 4), axis('W2', 6)] }).as('U'));
  assert.deepEqual(out.dims, [4, 4, 6]);
  assert.deepEqual(Array.from(out.data.slice(0, 12)), [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]);
  assert.deepEqual(Array.from(out.data.slice(12, 24)), [3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5]);
  assert.deepEqual(checkSources(out), []);
  assert.throws(() => grid((X) => upsample(X, { to: [axis('H2', 4), axis('W2', 5)] }).as('U')), /2 × 3 is not W2 = 5/);
});

test('concat stacks channels and keeps both sources', () => {
  const { out } = grid((X, { c }) => {
    const Y = again(X).as('Y');
    const cc = axis('cc', 8, { deck: true });
    return { X, Y, Z: concat(X, Y, cc).as('Z') };
  });
  assert.deepEqual(out.Z.dims, [8, 2, 3]);
  assert.deepEqual(Array.from(out.Z.data), [...range(24), ...range(24)]);
  assert.deepEqual(out.Z.op.sources(3), [[out.X, [3]]]);
  assert.deepEqual(out.Z.op.sources(24 + 3), [[out.Y, [3]]]);
  assert.deepEqual(checkSources(out.Z), []);
  assert.throws(() => grid((X) => concat(X, X, axis('cc', 7)).as('Z')), /cc = 7 is not c \+ c = 4 \+ 4/);
});

test('columns takes the same run of columns out of every row', () => {
  const { prog, out } = grid((X) => ({ X, Y: columns(X, 1, axis('w', 2)).as('Y') }));
  assert.deepEqual(out.Y.dims, [4, 2, 2]);
  // Row r of slab d starts at (d * 2 + r) * 3; columns 1 and 2 are copied.
  assert.deepEqual(Array.from(out.Y.data), range(8).flatMap((row) => [row * 3 + 1, row * 3 + 2]));
  assert.deepEqual(out.Y.op.sources(5), [[out.X, [2 * 3 + 2]]]);
  assert.deepEqual(checkSources(out.Y), []);
  const scene = compile(prog);
  assert.match(scene.steps[1].body(), /columns 1 to 2 of/);
  assert.match(scene.steps[1].code, /Y = X\[\.\.\., 1:3\]/);
  assert.throws(() => grid((X) => columns(X, 2, axis('w', 2)).as('Y')), /has no columns 2 to 3/);
});

test('arrows: one from the nearest operand, or one from every operand when hinted', () => {
  const arrows = (hints) => {
    const prog = program('two lanes', {}, () => {
      const [n, m] = [axis('n', 2), axis('m', 3)];
      section('s', 'S', () => {
        // Two computed tensors on two lanes (a fixed operand of the same shape would stand beside the result).
        const [A0, B0] = step('inputs', () => [input('A0', 'A_{0}', [n, m], range(6)), input('B0', 'B_{0}', [n, m], range(6))]);
        const [A, B] = step('doubled', () => [map('2 * x', { x: A0 }).as('A'), map('2 * x', { x: B0 }).as('B')]);
        step('product', () => map('a * b', { a: A, b: B }).as('Z').hint(hints));
      });
    });
    return compile(prog).sections[0].flows.filter((f) => f.type === 'flow' && !f.closed && f.step === 3);
  };
  assert.equal(arrows({}).length, 1);
  const all = arrows({ arrows: 'all' });
  assert.equal(all.length, 2);
  // Both end at the result; the one from the other lane bends.
  assert.deepEqual(all.map((f) => f.pts.at(-1)), [all[0].pts.at(-1), all[0].pts.at(-1)]);
  assert.deepEqual(all.map((f) => f.pts.length).sort(), [2, 4]);
});

test('group statistics: every channel holds the mean and spread of its whole group', () => {
  const { out } = grid((X) => {
    const mu = groupMean(X, { groups: 2 }).as('mu');
    const sd = groupStd(X, { groups: 2, mean: mu }).as('sd');
    return { X, mu, sd };
  });
  assert.deepEqual(out.mu.dims, [4, 1, 1]);
  // Group 0 is channels 0 and 1: the numbers 0..11. Group 1 is 12..23.
  assert.deepEqual(Array.from(out.mu.data), [5.5, 5.5, 17.5, 17.5]);
  const spread = Math.sqrt(range(12).reduce((s, x) => s + (x - 5.5) ** 2, 0) / 12);
  for (const v of out.sd.data) assert.ok(Math.abs(v - spread) < 1e-12);
  assert.deepEqual(out.mu.op.sources(1), [[out.X, range(12)]]);
  assert.deepEqual(out.sd.op.sources(3), [[out.X, range(12).map((i) => 12 + i)], [out.mu, [3]]]);
  assert.deepEqual([checkSources(out.mu), checkSources(out.sd)], [[], []]);
  assert.throws(() => grid((X) => groupMean(X, { groups: 3 }).as('mu')), /4 channels do not divide into 3 groups/);
});

test('reshape only adds or drops unit axes', () => {
  const { out } = grid((X, { c }) => {
    const mu = groupMean(X, { groups: 4 }).as('mu'); // (c, 1, 1)
    const row = reshape(mu, [ONE, c]).as('row');
    return { mu, row, back: reshape(row, [c, ONE, ONE]).as('back') };
  });
  assert.deepEqual(out.row.dims, [1, 4]);
  assert.deepEqual(Array.from(out.back.data), Array.from(out.mu.data));
  assert.deepEqual(checkSources(out.row), []);
  assert.throws(() => grid((X, { H }) => reshape(X, [H, ONE, ONE]).as('r')), /can only gain or lose unit axes/);
});

test('silu is x times sigmoid(x)', () => {
  const { out } = grid((X) => silu(X).as('A'));
  out.data.forEach((v, i) => assert.ok(Math.abs(v - i / (1 + Math.exp(-i))) < 1e-12));
  assert.deepEqual(checkSources(out), []);
});

// Two inputs: the second section does not read the first one's output directly,
// and brings it in part-way through.
const twoInputs = () =>
  program('two inputs', {}, () => {
    const [T, d, N] = [axis('T', 3), axis('d', 4), axis('N', 5)];
    let C;
    section('text', 'Text', () =>
      step('c', () => {
        C = input('C', 'C', [T, d], range(12));
      }),
    );
    section('img', 'Image', () =>
      step('x', () => {
        const X = input('X', 'X', [N, d], range(20));
        const Cc = again(C).as('Cc', 'C');
        const KT = transpose(Cc).as('KT', 'C^{T}').hint({ place: 'flow' });
        matmul(X, KT).as('S', 'S');
      }),
    );
  });

test('a section that does not read the previous one starts afresh, and brings tensors in on a side lane', () => {
  const prog = twoInputs();
  const [text, img] = prog.sections;
  assert.equal(img.stream, null);
  assert.equal(img.input, prog.t('X'));
  assert.deepEqual(Array.from(prog.t('Cc').data), range(12));
  assert.deepEqual(checkProgram(prog), { checked: 3, problems: [] });

  const L = layout(prog);
  const at = (id) => L.at.get(prog.t(id));
  // The copy enters below the flow, to the right of what is already there; its transpose stays on that lane.
  assert.ok(at('Cc').x0 >= at('X').x0 + at('X').C && at('Cc').cy + at('Cc').R / 2 < at('X').cy - at('X').R / 2);
  assert.deepEqual([at('KT').role, at('KT').cy], ['flow', at('Cc').cy]);
  assert.ok(at('S').x0 >= at('KT').x0 + at('KT').C);
  assert.equal(at('S').cy, 0);

  const scene = compile(prog);
  const sec = scene.sections[1];
  // Folded, the second section names its own input; open, both operands of S have an arrow.
  assert.ok(sec.flows.some((f) => f.closed && f.text === 'X'));
  assert.equal(sec.flows.filter((f) => f.type === 'flow' && !f.closed && f.step === 2).length, 3); // Cc → KT, X → S, KT → S
  assert.match(scene.steps[2].body(), /C<\/b> is C from the Text section, brought in unchanged/);
  assert.equal(scene.tensors.find((t) => t.id === 'Cc').note, 'C again, from Text');
  assert.equal(scene.explain('Cc', 0, 1, 2).groups[0].id, 'C');
  void text;
});

// One small layer, y = silu(x @ W), drawn once and then run again as a module on a second input.
const twice = (body) =>
  program('twice', { seed: 3 }, () => {
    const [n, d] = [axis('n', 2), axis('d', 3)];
    departure('untrained', 'random weights');
    const layer = (x) => silu(matmul(x, param('W', 'W', [d, d], random(1))).as('h', 'h')).as('y', 'y');
    section('s', 'S', () =>
      step('x', () => {
        const a = input('a', 'a', [n, d], [1, 2, 3, 4, 5, 6]);
        const b = input('b', 'b', [n, d], [6, 5, 4, 3, 2, 1]);
        body({ a, b, layer, n, d });
      }),
    );
  });

test('shared weights: declared once, the same memory on every later use', () => {
  // A layer repeated twice inside shared(): the second W is the first one again.
  const prog = program('repeated', { seed: 3 }, () => {
    const [n, d] = [axis('n', 2), axis('d', 3)];
    departure('untrained', 'random weights');
    let x;
    shared(() =>
      repeat(2, { prefix: 'r', name: 'Round' }, () =>
        section('s', 'S', () =>
          step('x', () => {
            x ??= input('a', 'a', [n, d], [1, 2, 3, 4, 5, 6]);
            const W = param('W', 'W', [d, d], random(1));
            x = matmul(again(x).as('in', 'x'), W).as('y', 'y');
            assert.throws(() => param('W', 'W', [d, ONE], random(1)), /shared weights were first declared as \(d, d\)/);
          }),
        ),
      ),
    );
  });
  const [W0, W1] = [prog.t('r0.W'), prog.t('r1.W')];
  assert.deepEqual([W0.alias, W1.alias, W1.section.id], [undefined, W0, 's1']);
  assert.equal(W1.data, W0.data); // not a copy: the same numbers
  const scene = compile(prog);
  assert.equal(scene.tensors.find((t) => t.id === 'r1.W').note, 'the same weights as in Round 1');
  assert.match(scene.steps[0].body(), /9 of them are parameters/); // counted once
  // Outside shared(), the same name twice is still a mistake.
  assert.throws(() => twice(({ a, b, layer }) => [layer(a), layer(b)]), /two tensors are named W/);
});

test('module: a part of the program run again, exactly, without being drawn', () => {
  let y1, y2;
  const prog = twice(({ a, b, layer }) =>
    shared(() => {
      y1 = layer(a);
      y2 = module('Layer', [b], () => layer(b), { prefix: 'm.' }).as('y2', 'y');
    }),
  );
  // The same arithmetic with the same weights, on the other input.
  const W = prog.t('W').data;
  const want = [0, 1].flatMap((i) => [0, 1, 2].map((j) => { const v = [0, 1, 2].reduce((s, k) => s + [6, 5, 4, 3, 2, 1][i * 3 + k] * W[k * 3 + j], 0); return v / (1 + Math.exp(-v)); }));
  Array.from(y2.data).forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-12));
  // What it computed on the way exists under its prefix and belongs to no section; only the answer is drawn.
  assert.deepEqual([prog.t('m.h').section, prog.t('m.W').alias, y2.section.id], [null, prog.t('W'), 's']);
  assert.ok(!prog.tensors.includes(prog.t('m.h')) && prog.tensors.includes(y2));
  assert.deepEqual(compile(prog).tensors.map((t) => t.id), ['a', 'b', 'W', 'h', 'y', 'y2']);
  // Its sources are followed back through what is not drawn: a row of the answer depends on the same row of the input, no more.
  assert.deepEqual([y2.op.sources(0), y2.op.sources(5)], [[[prog.t('b'), [0, 1, 2]]], [[prog.t('b'), [3, 4, 5]]]]);
  assert.deepEqual(checkProgram(prog), { checked: 3, problems: [] });
  assert.deepEqual(Array.from(prog.t('m.y').data), Array.from(y2.data)); // put back after the check
  assert.match(compile(prog).explain('y2', 0, 1, 2).html, /running the whole Layer again, with the same weights, on b\. 3 of their 6 numbers have a part/);

  // It may only read what it is handed, and may not bring weights of its own.
  assert.throws(() => twice(({ a, b, layer }) => shared(() => [layer(a), module('Layer', [], () => layer(b), { prefix: 'm.' }).as('y2')])), /module Layer reads b, which is not one of its inputs/);
  assert.throws(() => twice(({ b, layer }) => shared(() => module('Layer', [b], () => layer(b), { prefix: 'm.' }).as('y2'))), /param W: a module can only reuse weights that are declared, and drawn, somewhere else first/);
});

test('a single fixed number sits over the result it enters, and formulas with square roots can be checked', () => {
  const prog = program('scalar', {}, () => {
    const [c, H, W] = [axis('c', 2, { deck: true }), axis('H', 2), axis('W', 2)];
    section('s', 'S', () =>
      step('x', () => {
        const X = input('X', 'X', [c, H, W], range(8));
        const a = constant([ONE], [0.25], { rule: 'a setting' }).as('a', 'a');
        map('x / sqrt(a)', { x: X, a }).as('Y', 'Y');
      }),
    );
  });
  assert.deepEqual(Array.from(prog.t('Y').data), range(8).map((v) => v * 2));
  const at = (id) => layout(prog).at.get(prog.t(id));
  assert.deepEqual([at('a').role, at('a').of.id], ['over', 'Y']);
  assert.ok(at('a').cy > at('Y').cy + at('Y').R / 2);
  assert.deepEqual(checkProgram(prog), { checked: 1, problems: [] });
});

test('a named part: its sections stand under one bar, and its name is said once', () => {
  const build = (wrap) =>
    program('parts', {}, () => {
      const [T, d] = [axis('T', 3), axis('d', 2)];
      departure('untrained', 'random');
      let X;
      section('a', 'Start', () => step('x', () => (X = input('X', 'X', [T, d], range(6)))));
      wrap(() => {
        section('b', 'First', () => step('y', () => (X = silu(X).as('Y', 'Y'))));
        section('c', 'Second', () => step('z', () => (X = silu(X).as('Z', 'Z'))));
      });
    });
  const prog = build((fn) => part('Decoder', fn));
  assert.deepEqual(prog.sections.map((s) => s.part?.name), [undefined, 'Decoder', 'Decoder'].map((x) => x));
  assert.equal(prog.sections[1].part, prog.sections[2].part);
  const scene = compile(prog);
  assert.deepEqual(scene.bars.map((b) => [b.label, b.first, b.last]), [['DECODER', 'b', 'c']]);
  assert.deepEqual(scene.steps.slice(1, 4).map((s) => s.chapter), ['Start', 'Decoder · First', 'Second']);
  assert.match(scene.steps[0].body(), /<li><b>Decoder · Second<\/b>/);
  // A part holds whole sections and nothing is nested in it twice.
  assert.throws(() => build((fn) => part('A', () => part('B', fn))), /cannot be inside another part/);
  assert.throws(() => build((fn) => repeat(1, { prefix: 'r' }, () => part('A', fn))), /cannot be inside another part, a repeat/);
});

test('a result taller than its weight is kept clear of its neighbours; far operands are framed on request', () => {
  let Z;
  const build = (frame) =>
    program('tall', {}, () => {
      const [n, d] = [axis('n', 16), axis('d', 6)];
      departure('untrained', 'random');
      section('s', 'S', () => {
        const X = step('x', () => input('X', 'X', [n, d], range(96)));
        const [Q, K, V] = step('three results from one operand', () => [
          matmul(X, param('WQ', 'W_{Q}', [d, d], random(0.3))).as('Q', 'Q'),
          matmul(X, param('WK', 'W_{K}', [d, d], random(0.3)).hint({ side: 'above' })).as('K', 'K'),
          matmul(X, param('WV', 'W_{V}', [d, d], random(0.3)).hint({ side: 'below' })).as('V', 'V'),
        ]);
        const A = step('scores', () => {
          const S = matmul(Q, transpose(K).as('KT', 'K^{T}').hint({ place: 'flow' })).as('S', 'S');
          return silu(silu(silu(S).as('S1', 'S_{1}')).as('S2', 'S_{2}')).as('A', 'A');
        });
        step('mix', () => (Z = matmul(A, V).as('Z', 'Z').hint(frame ? { frame: 'operands' } : {})));
      });
    });
  const prog = build(true);
  const L = layout(prog);
  const at = (id) => L.at.get(prog.t(id));
  const [top, bot] = [(p) => p.cy + (p.R - 1) / 2, (p) => p.cy - (p.R - 1) / 2];
  // The weights are 6 rows tall and would fit 12 apart; the 16-row results need more, and get it.
  assert.equal(at('Q').cy, 0);
  assert.ok(bot(at('K')) > top(at('Q')) + 5 && top(at('V')) < bot(at('Q')) - 5);
  assert.deepEqual([at('WK').cy, at('WV').cy], [at('K').cy, at('V').cy]);
  // The values stand more than 40 columns behind Z: framed only when the result asks for its operands.
  assert.ok(at('Z').x0 - (at('V').x0 + at('V').C) > 40);
  const view = (p) => compile(p).steps.find((s) => s.title === 'mix').view.ids;
  assert.ok(view(prog).includes('V') && view(prog).includes('A'));
  assert.ok(!view(build(false)).includes('V') && view(build(false)).includes('A'));
});

test('picture: three sheets of brightnesses shown as one square per pixel', () => {
  const { out } = grid((X, { H, W }) => {
    const rgb = axis('rgb', 3, { deck: true });
    const P = input('P', 'P', [rgb, H, W], [0, 0.5, 1, 2, -1, 0.25, ...range(12).map((i) => i / 12)]);
    return { P, X };
  });
  const html = render(prose('say', ['', ''], [picture(out.P, { size: 60 })]));
  assert.match(html, /<svg class="picture" viewBox="0 0 3 2" width="60" height="40"/);
  const rects = html.match(/<rect [^>]*>/g);
  assert.equal(rects.length, 6);
  // Red, green and blue come from the same position of the three sheets; values outside 0..1 are cut off.
  assert.equal(rects[0], '<rect x="0" y="0" width="1" height="1" fill="rgb(0,0,128)"/>');
  assert.equal(rects[3], '<rect x="0" y="1" width="1" height="1" fill="rgb(255,64,191)"/>');
  assert.equal(rects[4], '<rect x="1" y="1" width="1" height="1" fill="rgb(0,85,213)"/>');
  // The numbers are read when the text is shown, not when it was written.
  out.P.data[0] = 1;
  assert.match(render(prose('say', ['', ''], [picture(out.P)])), /<rect x="0" y="0" width="1" height="1" fill="rgb\(255,0,128\)"\/>/);
  assert.throws(() => picture(out.X), /must be \(3, height, width\)/);
});
