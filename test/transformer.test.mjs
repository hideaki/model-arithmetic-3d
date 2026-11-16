// The transformer written in the tensor language must reproduce the
// hand-written app exactly: same numbers, same shapes, same steps, and the
// same hover sources for every cube.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build } from '../programs/transformer.js';
import { checkProgram } from '../lang/check.js';
import { compute } from '../src/model.js';
import { SPECS, SECTIONS, NSTEPS, explainCell } from '../src/tensors.js';

const prog = build();
const cellsOf = (spec, m) => spec.get(m).flat(2);
const key = (id, d, r, c) => `${id}:${d}:${r}:${c}`;

test('the program has the same tensors as the app', () => {
  const mine = prog.tensors.filter((t) => t.kind !== 'input').map((t) => t.id).sort();
  assert.deepEqual(mine, SPECS.map((s) => s.id).sort());
});

for (const causal of [true, false]) {
  test(`every number equals compute() exactly (causal ${causal})`, () => {
    prog.set('causal', causal);
    const m = compute({ causal });
    let cubes = 0;
    for (const spec of SPECS) {
      const want = cellsOf(spec, m);
      const t = prog.t(spec.id);
      assert.equal(t.size, want.length, `${spec.id} size`);
      for (let i = 0; i < want.length; i++)
        if (t.data[i] !== want[i]) assert.fail(`${spec.id}[${t.addr(i)}]: ${t.data[i]} is not ${want[i]}`);
      cubes += want.length;
    }
    assert.equal(cubes, 8784);
  });

  test(`hover sources equal the app's for every cube (causal ${causal})`, () => {
    prog.set('causal', causal);
    const m = compute({ causal });
    let cubes = 0;
    for (const spec of SPECS) {
      const t = prog.t(spec.id);
      for (let i = 0; i < t.size; i++) {
        const [d, r, c] = t.addr(i);
        const want = new Set();
        for (const g of explainCell(m, spec.id, d, r, c).groups)
          for (let rr = g.r0; rr <= g.r1; rr++) for (let cc = g.c0; cc <= g.c1; cc++) want.add(key(g.id, g.d, rr, cc));
        const got = new Set();
        for (const [src, cells] of prog.sources(t, i)) for (const f of cells) got.add(key(src.id, ...src.addr(f)));
        assert.deepEqual([...got].sort(), [...want].sort(), `sources of ${spec.id}[${d}, ${r}, ${c}]`);

        // Where the app flies a cube in from must be one of its sources, and
        // for a pure copy it must be the only one.
        if (spec.from) {
          const from = key(...spec.from(d, r, c));
          assert.ok(got.has(from), `${spec.id}[${d}, ${r}, ${c}] flies in from ${from}`);
          if (t.op.family === 'rearrange') assert.equal(got.size, 1);
        }
        cubes++;
      }
    }
    assert.equal(cubes, 8784);
  });
}

test('shape labels are derived from the axes', () => {
  for (const spec of SPECS) assert.equal(prog.t(spec.id).sym, spec.sym, spec.id);
});

test('steps and sections are derived from the structure', () => {
  assert.equal(prog.nSteps, NSTEPS);
  for (const spec of SPECS) assert.equal(prog.t(spec.id).step.index, spec.step, `step of ${spec.id}`);
  for (const spec of SPECS) assert.equal(prog.t(spec.id).section.id, spec.sec, `section of ${spec.id}`);
  assert.deepEqual(
    prog.sections.map((s) => [s.id, s.from, s.to]),
    SECTIONS.map((s) => [s.id, s.from, s.to]),
  );
});

test('declared sources are true for every operation', () => {
  const { checked, problems } = checkProgram(prog);
  assert.deepEqual(problems, []);
  assert.ok(checked > 50, `${checked} operations checked`);
});
