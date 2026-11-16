// Author prose: where it goes, what the compiler rejects or flags, and that
// quoted numbers are read live.

import test from 'node:test';
import assert from 'node:assert/strict';

import { program, axis, param, random, departure, section, step, repeat, say, aside, hint } from '../lang/core.js';
import { add, softmax, split } from '../lang/ops.js';
import { row, col, slab, value, cell, pct, label, render, prose } from '../lang/prose.js';
import { compile } from '../lang/scene.js';
import { build } from '../programs/transformer.js';

// Two steps: A, then B = softmax(A). `first` and `second` run inside them.
const two = (first = () => {}, second = () => {}) =>
  program('two', { seed: 5 }, () => {
    departure('untrained', 'random');
    const [n, m] = [axis('n', 2, { labels: ['cat', 'dog'] }), axis('m', 3)];
    section('s', 'Section', () => {
      const A = step('First', () => {
        const A = param('A', 'A', [n, m], random(1));
        first({ A, n, m });
        return A;
      });
      step('Second', () => {
        const B = softmax(A).as('B');
        second({ A, B, n, m });
      });
    });
  });

test('typed numbers in say are flagged; references and asides are not', () => {
  const warn = (fn) => two(fn).warnings;
  assert.match(warn(() => say`Each row has 12 numbers.`)[0], /step "First": the typed number "12" should be a reference/);
  assert.match(warn(() => say`It went from 0.37 to here.`)[0], /"0\.37"/);
  assert.match(warn(() => say`About 30% comes from it.`)[0], /"30%"/);
  assert.deepEqual(warn(({ A, m }) => say`Each row of ${A} has ${m.size} numbers and sums to 1; X_{12} is fine too.`), []);
  assert.deepEqual(warn(() => aside`GPT-2 small uses 768 numbers per token.`), []);
});

test('statements about real or trained models are flagged outside an aside', () => {
  assert.match(two(() => say`In a trained model this head finds the subject.`).warnings[0], /"trained" reads as a statement about real or trained models/);
  assert.match(two(() => say`GPT-2 does the same.`).warnings[0], /"GPT-2"/);
  assert.deepEqual(two(() => aside`In a trained model this head finds the subject.`).warnings, []);
});

test('prose may only name what is on screen by then', () => {
  assert.throws(() => two(undefined, ({ A, B }) => A.says`This feeds ${B}.`), /A\.says mentions B, which first appears later, in "Second"/);
  assert.doesNotThrow(() => two(undefined, ({ A, B }) => B.says`Computed from ${A}.`));
});

test('cell placeholders are only for X.says, and must fit the tensor', () => {
  assert.throws(() => two(() => say`Row ${row} is special.`), /step "First" uses row, which only X\.says understands/);
  assert.throws(() => two(({ A }) => A.says`${slab}: a number.`), /A\.says uses slab, but A \(n, m\) has no slabs/);
});

test('prose renders paragraphs, lists, markup, labels and live values', () => {
  let A;
  const p = two((x) => {
    A = x.A;
    say`The value at ${label(x.n, 1)} is ${cell(A, 1, 2)}, shown as ${pct(cell(A, 1, 2))} too; d_{k} is markup.

      <ul><li>a list stays a list</li></ul>`;
    aside`A side note.`;
    hint`Hover ${A}.`;
    A.says`Row ${row}, column ${col}: ${value}.`;
  });
  const s = p.steps[0];
  const html = s.prose.map((i) => render(i)).join('');
  const v = A.data[5];
  assert.ok(html.startsWith(`<p>The value at <span class="tok">dog</span> is ${v.toFixed(2).replace('-', '−')}, shown as ${Math.round(v * 100)}% too; d<sub>k</sub> is markup.</p>`), html);
  assert.ok(html.includes('<ul><li>a list stays a list</li></ul><p class="aside">A side note.</p>'), html);
  assert.equal(render(s.hint), 'Hover <span class="ref">A</span>.');
  assert.equal(render(A.hover, {}, { t: A, d: 0, r: 1, c: 2, value: 0.5 }), '<p>Row <span class="tok">dog</span>, column 2: 0.50.</p>');
  assert.equal(render(prose('say', ['a < b &amp; ', ''], ['x < y'])), '<p>a < b &amp; x &lt; y</p>'); // typed text is HTML, values are escaped
});

test('where prose is said decides where it is shown', () => {
  const p = program('places', { seed: 2 }, () => {
    departure('untrained', 'random');
    const [n, h, k, d] = [axis('n', 2), axis('h', 2), axis('k', 3), axis('d', 6)];
    say`Opening.`;
    let X;
    section('a', 'A', () => {
      say`Leads the first step.`;
      step('One', () => { X = param('X', 'X', [n, d], random(1)); say`In step one.`; });
    });
    repeat(2, { prefix: 'r', name: 'Round', detail: 'first' }, (i) => {
      section('b', 'B', () => {
        if (i) say`About the second round.`;
        step('Split', () => { X = add(X, param('W', 'W', [n, d], random(1))).as('Y'); say`Step prose for the round.`; });
        step('Heads', () => { split(X, d, [h, k]).as('Z').says`${slab} of the split.`; });
      });
    });
    say`Closing.`;
  });
  const text = (b) => b.prose.map((i) => render(i)).join('');
  assert.equal(text(p.intro), '<p>Opening.</p>');
  assert.equal(text(p.outro), '<p>Closing.</p>');
  assert.equal(text(p.steps[0]), '<p>Leads the first step.</p><p>In step one.</p>');
  assert.equal(text(p.steps[1]), '<p>Step prose for the round.</p>');
  // The second round is one step: it shows its own lead, not the step prose again.
  assert.equal(p.steps.length, 4);
  assert.equal(text(p.steps[3]), '<p>About the second round.</p>');
  assert.ok(p.t('r1.Z').hover, 'hover sentences still apply in the second round');
});

test('the transformer: every step and every tensor is explained, with no warnings', () => {
  const prog = build();
  assert.deepEqual(prog.warnings, []);
  assert.ok(prog.intro.prose.length && prog.outro.prose.length);
  for (const s of prog.steps) assert.ok(s.prose.length, `step "${s.title}" has no prose`);
  for (const t of prog.tensors) if (t.kind !== 'input') assert.ok(t.hover, `${t.id} has no hover sentence`);

  const scene = compile(prog);
  const fmt = (v) => v.toFixed(2).replace('-', '−');
  // Step 13 quotes three live cells of the residual add.
  const [x0, o, xa] = ['X0', 'b0.O', 'b0.Xa'].map((id) => prog.t(id).data[prog.t(id).flat(0, 3, 0)]);
  assert.ok(scene.steps[13].body().includes(`dimension 0 went from ${fmt(x0)} to ${fmt(xa)}`));
  assert.ok(scene.steps[13].body().includes(`(${fmt(x0)} + ${o < 0 ? `(${fmt(o)})` : fmt(o)})`));
  assert.ok(scene.steps[13].body().includes('Exactly what is computed'));
  // Block 2 is one step: its own text, then the generated list, not block 1's step prose.
  assert.ok(scene.steps[19].body().includes('Why stack?') && !scene.steps[19].body().includes('<b>Query</b>'));
  // Hover: the author's sentence first, then the generated arithmetic.
  const ex = scene.explain('b0.S', 1, 2, 3);
  assert.match(ex.html, /^<p><span class="hd"[^>]*>head 1<\/span>: the <b>query<\/b> of <span class="tok">said<\/span> dotted with the <b>key<\/b> of <span class="tok">it<\/span>\.<\/p><p class="derived-p">A row of Q dotted/);

  // Text that quotes values follows the toggle.
  assert.ok(!scene.steps[8].body().includes('currently off'));
  const before = scene.steps[9].body();
  scene.set('causal', false);
  assert.ok(scene.steps[8].body().includes('The mask is currently off'));
  assert.notEqual(scene.steps[9].body(), before);
  scene.set('causal', true);
  assert.equal(scene.steps[9].body(), before);
});
