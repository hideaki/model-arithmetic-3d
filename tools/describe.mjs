// Print what the compiler derives from a program: its declared departures,
// then every step with the tensors it reveals, their shapes and operations.
//
//   node tools/describe.mjs            the step list
//   node tools/describe.mjs b0.S 1 2 3 also explain one cell (id, depth, row, column)

import { explain, plain, fmt } from '../lang/explain.js';

// PROGRAM=yolo node tools/describe.mjs   describes another program
const { build } = await import(`../programs/${process.env.PROGRAM || 'transformer'}.js`);
const prog = build();
const [id, d, r, c] = process.argv.slice(2);

console.log(`${prog.name}: ${prog.tensors.length} tensors, ${prog.tensors.reduce((n, t) => n + (t.kind === 'input' ? 0 : t.size), 0)} cubes, ${prog.nSteps} steps\n`);
console.log('Departures from a real model:');
for (const dep of prog.departures) console.log(`  ${dep.key}: ${dep.text}`);

console.log('\nstep  section  title');
console.log('   0  —        (folded overview)');
for (const s of prog.steps) {
  console.log(`${String(s.index).padStart(4)}  ${s.section.id.padEnd(7)}  ${s.title}`);
  for (const t of s.tensors) {
    const how = t.op ? `${t.op.name}(${t.op.inputs.map((x) => plain(x.label)).join(', ')})` : t.kind === 'input' ? 'input' : `${t.kind}, random std ${t.source.std}`;
    console.log(`        ${plain(t.label).padEnd(12)} ${plain(t.sym).padEnd(18)} = ${t.num.padEnd(11)} ${how}`);
  }
}
console.log(`${String(prog.nSteps - 1).padStart(4)}  —        (folded overview)`);

if (id) {
  const t = prog.t(id);
  const e = explain(t, t.flat(Number(d) || 0, Number(r) || 0, Number(c) || 0));
  console.log(`\n${e.title} = ${fmt(e.value)}\n  ${e.text}`);
  for (const line of e.eq) console.log(`  ${line}`);
  for (const term of e.terms || [])
    console.log('product' in term ? `    ${term.aName} ${fmt(term.a)} × ${term.bName} ${fmt(term.b)} = ${fmt(term.product)}` : `    ${term.current ? '→' : ' '} ${term.name} ${fmt(term.x)} → ${fmt(term.weight)}`);
}
