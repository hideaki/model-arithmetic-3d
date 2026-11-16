// The list of models (models.html) and the folder assembled for a static host
// (tools/build-site.mjs). The list is written by hand, so it is checked against
// the programs: when a model is added or changes size, this test says what to edit.

import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildSite, withoutAskPanel } from '../tools/build-site.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const page = readFileSync(join(ROOT, 'models.html'), 'utf8');
const programs = readdirSync(join(ROOT, 'programs')).filter((f) => f.endsWith('.js')).map((f) => f.slice(0, -3)).sort();

// One card: <a class="card" href="compiled.html?program=NAME"> ... <span class="size">N cubes · M steps</span>
const cards = [...page.matchAll(/<a class="card" href="compiled\.html\?program=([\w-]+)">([\s\S]*?)<\/a>/g)].map(([, name, body]) => {
  const size = /<span class="size">([\d,]+) cubes · (\d+) steps<\/span>/.exec(body);
  return { name, cubes: size && Number(size[1].replace(/,/g, '')), steps: size && Number(size[2]), img: /<img src="([^"]+)"/.exec(body)?.[1] };
});

test('models.html has one card for every program, and no other', () => {
  assert.deepEqual(cards.map((c) => c.name).sort(), programs, 'add or remove a card in models.html (and a row in README.md) to match programs/');
});

test('every card gives the cube and step counts of its program', async () => {
  for (const card of cards) {
    const prog = (await import(`../programs/${card.name}.js`)).build();
    // Counted as tools/describe.mjs counts them: inputs that are not drawn are left out.
    const cubes = prog.tensors.reduce((n, t) => n + (t.kind === 'input' ? 0 : t.size), 0);
    assert.deepEqual({ cubes: card.cubes, steps: card.steps }, { cubes, steps: prog.nSteps }, `the card of "${card.name}" in models.html (README.md has the same numbers)`);
  }
});

test('every picture and link of models.html exists', () => {
  for (const card of cards) if (card.img) assert.ok(existsSync(join(ROOT, card.img)), `${card.img} is missing`);
  assert.ok(existsSync(join(ROOT, 'compiled.html')));
});

test('the assembled site has what the pages load, and no question panel', () => {
  const out = join(mkdtempSync(join(tmpdir(), 'site-')), 'site');
  try {
    buildSite(out);
    assert.equal(readFileSync(join(out, 'index.html'), 'utf8'), page, 'the front page is models.html');
    const compiled = readFileSync(join(out, 'compiled.html'), 'utf8');
    assert.ok(!compiled.includes('id="ask'), 'no part of the question panel is left');
    assert.ok(compiled.includes('<section id="inspector"') && compiled.includes('runtime/main.js'), 'the rest of the page is left as it was');
    // Everything compiled.html names, and everything the list names.
    for (const [, path] of compiled.matchAll(/(?:href|src)="((?!https?:)[^"]+)"/g)) assert.ok(existsSync(join(out, path)), `${path} is missing from the site`);
    for (const card of cards) {
      assert.ok(existsSync(join(out, 'programs', `${card.name}.js`)), `programs/${card.name}.js is missing from the site`);
      if (card.img) assert.ok(existsSync(join(out, card.img)), `${card.img} is missing from the site`);
    }
    for (const dir of ['lang', 'runtime']) assert.deepEqual(readdirSync(join(out, dir)).sort(), readdirSync(join(ROOT, dir)).sort(), `${dir}/ is copied whole`);
    // A second run replaces the first; a folder it did not make is left alone.
    buildSite(out);
    assert.throws(() => buildSite(join(ROOT, 'programs')), /was not made by this script/);
  } finally {
    rmSync(dirname(out), { recursive: true });
  }
});

test('a page without exactly one question panel is refused, not published half-edited', () => {
  assert.throws(() => withoutAskPanel('<aside id="panel"></aside>'), /expected one/);
});
