// Checks that an operation's declared sources are true.
//
// For one operation: fill its inputs with random numbers, then nudge every
// input cell in turn and see which output cells move. The cells that move must
// be exactly the cells that list the nudged cell as a source. A source that is
// listed but has no effect, or an effect that is not listed, is a problem.
// A listed cell that has no effect when nudged is tried once more with its
// sign flipped, so that a function with a flat part (ReLU) is not misjudged.
//
// Random inputs are used instead of the program's own values because sources
// are a property of the operation: real values can hide a dependency (a weight
// of exactly 0, or −∞ swallowing whatever is added to it).

import { mulberry32 } from './core.js';

export function checkSources(t, seed = 7) {
  const { op } = t;
  if (!op || !op.inputs.length) return [];
  const rnd = mulberry32(seed);
  const ins = [...new Set(op.inputs)];
  const saved = [t.data, ...ins.map((x) => x.data)];
  const problems = [];
  try {
    // Numbers between -1 and 1; or, for a formula that only takes some numbers
    // (the square root of an operand, say), between 0.1 and 0.9 with a smaller nudge.
    let nudge = 0.37;
    const fill = (lo, hi) => {
      for (const x of ins) x.data = Float64Array.from({ length: x.size }, () => lo + rnd() * (hi - lo));
      t.data = new Float64Array(t.size);
      op.run(t);
    };
    fill(-1, 1);
    if (!t.data.every(Number.isFinite)) {
      fill(0.1, 0.9);
      nudge = 0.05;
    }
    const base = t.data.slice();

    // listed.get(x)[c] = output cells that name cell c of input x as a source
    const listed = new Map(ins.map((x) => [x, Array.from({ length: x.size }, () => new Set())]));
    for (let o = 0; o < t.size; o++)
      for (const [x, cells] of op.sources(o)) for (const c of cells) listed.get(x)[c].add(o);

    for (const x of ins)
      for (let c = 0; c < x.size && problems.length < 5; c++) {
        const old = x.data[c];
        x.data[c] = old + nudge;
        op.run(t);
        x.data[c] = old;
        const want = listed.get(x)[c];
        let mirrored = null; // the output with this cell's sign flipped, computed only if needed
        for (let o = 0; o < t.size; o++) {
          let moved = t.data[o] !== base[o];
          // A function with a flat part (ReLU below zero) does not move for a small nudge.
          // Before calling a listed source unused, try the same cell on the other side of zero.
          if (!moved && want.has(o)) {
            if (!mirrored) {
              const nudged = t.data;
              x.data[c] = -old;
              t.data = new Float64Array(t.size);
              op.run(t);
              mirrored = t.data;
              t.data = nudged;
              x.data[c] = old;
            }
            moved = mirrored[o] !== base[o];
          }
          if (moved === want.has(o)) continue;
          problems.push(
            moved
              ? `${t.id}[${t.addr(o)}] depends on ${x.id}[${x.addr(c)}], which is not listed as a source`
              : `${t.id}[${t.addr(o)}] lists ${x.id}[${x.addr(c)}] as a source but does not depend on it`,
          );
          break;
        }
      }
  } finally {
    [t.data] = saved;
    ins.forEach((x, k) => (x.data = saved[k + 1]));
    op.refresh?.(); // a module recomputes its undrawn tensors from the real inputs again
  }
  return problems;
}

// Every operation in a program.
export function checkProgram(prog) {
  const problems = [];
  let checked = 0;
  for (const t of prog.tensors) {
    if (!t.op || !t.op.inputs.length) continue;
    checked++;
    problems.push(...checkSources(t));
  }
  return { checked, problems };
}
