// Questions from the viewer: what the page tells the model, what the model may
// do on the page, how answers are rendered, and the server's side of a round
// (with its stand-in model, so no API key is needed).

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import { execFileSync } from 'node:child_process';

import { build as buildSd } from '../programs/sd.js';
import { build as buildTransformer } from '../programs/transformer.js';
import { build as buildYolo } from '../programs/yolo.js';
import { compile } from '../lang/scene.js';
import { toText, num, pageContext, stepText, cubeText, viewContext, mismatch, makeTools } from '../runtime/ask-context.js';
import { markdown } from '../runtime/ask.js';

const root = new URL('..', import.meta.url).pathname;
const definitions = JSON.parse(fs.readFileSync(`${root}runtime/ask-tools.json`, 'utf8'));
const sd = compile(buildSd());

// A page that only records what was done to it.
function pageAt(start) {
  const did = [];
  let step = start;
  return { did, step: () => step, go: (i) => ((step = i), did.push(`go ${i}`)), pin: (...a) => did.push(`pin ${a.join(' ')}`), unpin: () => did.push('unpin'), focus: (h) => did.push(`focus ${h}`) };
}

test('page HTML becomes plain text: subscripts, lists, tables, entities, no markup', () => {
  assert.equal(toText('<p>W<sub>E</sub> has <b>8</b> rows &amp; 6 columns.</p><ul><li>one</li><li>two</li></ul>'), 'W_E has 8 rows & 6 columns.\n\n- one\n- two');
  assert.equal(toText('<table><tr><th>cell</th><th>x</th></tr><tr><td>[0]</td><td>−0.33</td></tr></table>'), 'cell | x\n[0] | −0.33');
  assert.equal(toText('before <svg class="picture"><rect x="0"/></svg> after'), 'before [a picture is shown here] after');
  assert.equal(toText('a&nbsp;&nbsp;b &#39;c&#39; &lt;d&gt;'), "a b 'c' <d>");
  assert.deepEqual([num(0.7480684574792961), num(-3), num(-Infinity), num(1e-7)], ['0.748068', '-3', '-infinity', '1e-7']);
});

test('the page context names every step and every tensor, for each of the three models', () => {
  for (const scene of [sd, compile(buildTransformer()), compile(buildYolo())]) {
    const text = pageContext(scene);
    const lines = text.split('\n');
    assert.equal(lines[0], `Model: ${scene.title}`);
    assert.ok(!/<[a-z]/i.test(text), 'no HTML is left');
    for (const d of scene.departures) assert.ok(lines.includes(`- ${d.key}: ${d.text}`), d.key);
    scene.steps.forEach((s, i) => assert.ok(lines.includes(`${i} | ${toText(s.chapter)} | ${toText(s.title)}`), `step ${i}`));
    for (const spec of scene.tensors) assert.equal(lines.filter((l) => l.startsWith(`${spec.id} | `)).length, 1, spec.id);
    // The same page gives the same text: it is what the API caches.
    assert.equal(pageContext(scene), text);
  }
  const line = (id) => pageContext(sd).split('\n').find((l) => l.startsWith(`${id} | `));
  assert.equal(line('DS'), 'DS | S | (H·W, H·W) = (16, 16) | 1 x 16 x 16 | computed | 50 | matmul(DQ, DKT)');
  assert.equal(line('s0.Kin'), 's0.Kin | K_in | (c, c_z, k, k) = (6, 4, 3, 3) | 4 x 18 x 3 | weights | 6 | declared');
  assert.match(line('s1.Wt1'), /\| weights \| 35 \| the same weights as s0\.Wt1$/);
  assert.match(line('s1.epsc'), /\| computed \| 36 \| U-Net\(s1\.cz, s1\.ct, s1\.cC\)$/);
  assert.match(line('Df'), /\| fixed \| 45 \| constant\(\)$/);
  assert.match(line('z'), /\| input \| 3 \| given$/);
  assert.ok(pageContext(sd).length < 60000); // well inside what the server accepts
});

test('the view context is the step the viewer is on and the cube they pinned', () => {
  const view = viewContext(sd, { step: 50, pinned: { id: 'DA', d: 0, r: 5, c: 11 }, focus: -1 });
  assert.ok(view.startsWith('<view>\nStep 50 of 58: Every grid cell meets every other: scores, then shares (chapter: Self-attention)'));
  assert.ok(view.endsWith('</view>'));
  assert.match(view, /Formula: S = Q K\^T/);
  assert.match(view, /- S \(H·W, H·W\) = \(16, 16\)/);
  assert.match(view, /- K\^T \(c_1, H·W\) = \(6, 16\)/); // label markup is flattened too
  assert.match(view, /There is no mask: every cell may look at every other/); // the author's text
  assert.match(view, /Exactly what is computed\nK\^T is K with its last two axes swapped/); // the generated account
  assert.match(view, /Code:\nDKT = DK\.transpose\(-2, -1\)/);
  assert.match(view, /Tensor ids introduced in this step: DKT, DS, DSs, DA/);
  assert.match(view, /Focus: every channel is shown\./);
  // The pinned cube: its exact value, the page's own explanation, and what lights up.
  const value = sd.prog.t('DA').data[5 * 16 + 11];
  assert.ok(view.includes(`The viewer has pinned this cube:\nA[1,1, 2,3] = ${num(value)}   (tensor DA, slab 0, row 5, column 11)`));
  assert.match(view, /Grid cell 1,1 gives \d+% of its attention to grid cell 2,3\./);
  assert.match(view, /Lit on the page as its sources: 16 cells of DSs\./);

  const plain = viewContext(sd, { step: 3, focus: 2 });
  assert.match(plain, /Focus: only channel 2 is shown\.\nNo cube is pinned\.\n<\/view>$/);
  assert.equal(stepText(sd, 57).includes('<svg'), false);
  assert.match(stepText(sd, 57), /\[a picture is shown here\]/);
  // A weight has no sources, and the text says so.
  assert.match(cubeText(sd, 's0.Kin', 0, 0, 0), /Nothing else lights up: this number is not computed from other numbers on screen\./);
});

test('tool inputs are checked against their schema before anything runs', () => {
  const schema = (n) => definitions.find((d) => d.name === n).input_schema;
  assert.equal(mismatch(schema('go_to_step'), { step: 4 }), null);
  assert.equal(mismatch(schema('go_to_step'), {}), '"step" is missing');
  assert.equal(mismatch(schema('go_to_step'), { step: '4' }), '"step" must be a whole number');
  assert.equal(mismatch(schema('go_to_step'), { step: 4.5 }), '"step" must be a whole number');
  assert.equal(mismatch(schema('go_to_step'), { step: 4, extra: 1 }), '"extra" is not a field of this tool');
  assert.equal(mismatch(schema('show_cube'), { tensor: '', slab: 0, row: 0, col: 0 }), '"tensor" must be text');
  assert.equal(mismatch(schema('show_cube'), null), 'the input is not an object');
  assert.equal(mismatch(schema('read_cells'), { tensor: 'DS', slab: 0, row_from: 1 }), null);
  for (const d of definitions) {
    assert.ok(d.description.length > 60 && d.input_schema.type === 'object' && d.input_schema.additionalProperties === false, d.name);
    for (const key of d.input_schema.required) assert.ok(d.input_schema.properties[key], `${d.name}.${key}`);
  }
});

test('the tools: read without touching the page, or move it and say so', () => {
  const page = pageAt(3);
  const tools = makeTools(sd, page, definitions);
  assert.deepEqual(tools.names, definitions.map((d) => d.name));

  // Reading changes nothing on the page.
  const read = tools.run('read_step', { step: 49 });
  assert.equal(read.is_error, false);
  assert.ok(read.content.startsWith('Step 49 of 58: Queries, keys and values: all from the same rows'));
  const cells = tools.run('read_cells', { tensor: 'DS', slab: 0, row_from: 5, row_to: 5, col_from: 10, col_to: 12 });
  const S = sd.prog.t('DS');
  assert.equal(cells.content, `S (tensor DS), slab 0, rows 5 to 5, columns 10 to 12:\nrow 5: ${[10, 11, 12].map((c) => num(S.data[5 * 16 + c])).join(', ')}`);
  assert.equal(cells.note, 'Read 3 numbers of S');
  assert.equal(tools.run('read_cells', { tensor: 'img', slab: 1 }).content.split('\n').length, 17); // a whole 16 x 16 sheet
  assert.deepEqual(page.did, []);

  // Showing a cube that is not on screen moves to the step that introduces it, focuses its slab and pins it.
  const shown = tools.run('show_cube', { tensor: 'img', slab: 2, row: 5, col: 9 });
  assert.deepEqual(page.did, ['go 57', 'focus 2', 'pin img 2 5 9']);
  assert.ok(shown.content.startsWith('The page moved to step 57, which introduces this tensor. The cube is pinned and its sources are lit.'));
  assert.ok(shown.content.includes(`image[2, 5, 9] = ${num(sd.prog.t('img').data[2 * 256 + 5 * 16 + 9])}`));
  assert.equal(shown.note, 'Pinned image[2, 5, 9] (moved to step 57)');
  // One that is on screen is pinned where the viewer is; a tensor of one slab leaves the focus alone.
  page.did.length = 0;
  page.go(51);
  assert.equal(tools.run('show_cube', { tensor: 'DA', slab: 0, row: 0, col: 0 }).note, 'Pinned A[0,0, 0,0]');
  assert.deepEqual(page.did, ['go 51', 'pin DA 0 0 0']);
  // Once its section has folded, the page goes back to it.
  page.go(55);
  tools.run('show_cube', { tensor: 'DA', slab: 0, row: 0, col: 0 });
  assert.deepEqual(page.did.slice(-2), ['go 50', 'pin DA 0 0 0']);

  // Moving to a step releases the cube pinned before: it belongs to what was on screen then.
  const moved = tools.run('go_to_step', { step: 17 });
  assert.deepEqual(page.did.slice(-2), ['unpin', 'go 17']);
  assert.equal(page.step(), 17);
  assert.equal(moved.note, 'Moved to step 17: Every grid cell meets every token');
  assert.ok(moved.content.startsWith('The page is now on this step.\n\nStep 17 of 58'));
});

test('a tool call that cannot be carried out is reported back, and the page is left alone', () => {
  const page = pageAt(10);
  const tools = makeTools(sd, page, definitions);
  const fail = (n, input) => {
    const out = tools.run(n, input);
    assert.equal(out.is_error, true, n);
    return out.content;
  };
  assert.equal(fail('go_to_step', { step: 99 }), 'Error: there is no step 99: steps run from 0 to 58');
  assert.equal(fail('read_cells', { tensor: 'S', slab: 0 }), 'Error: there is no tensor with the id "S". Did you mean: s0.S, DS?');
  assert.equal(fail('read_cells', { tensor: 'nothing', slab: 0 }), 'Error: there is no tensor with the id "nothing". Use an id from the tensor list.');
  assert.equal(fail('show_cube', { tensor: 'DS', slab: 0, row: 16, col: 0 }), 'Error: row 16 is out of range: this tensor has rows 0 to 15');
  assert.equal(fail('show_cube', { tensor: 'DS', slab: 1, row: 0, col: 0 }), 'Error: slab 1 is out of range: this tensor has slabs 0 to 0');
  assert.equal(fail('read_cells', { tensor: 'DS', slab: 0, row_from: 5, row_to: 2 }), 'Error: a range ends before it starts');
  assert.equal(fail('fly', {}), 'There is no tool called "fly".');
  // A malformed input goes back as it was received, so the model can write it again.
  assert.equal(fail('show_cube', { tensor: 'img', slab: 2, row: 5 }), 'Invalid input: "col" is missing. {"INVALID_JSON":"{\\"tensor\\":\\"img\\",\\"slab\\":2,\\"row\\":5}"}');
  assert.deepEqual(page.did, []);
});

test('answers are rendered from a small part of Markdown, and never as HTML', () => {
  assert.equal(markdown('The table is **16 × 16**.\n\nSee `DS`.'), '<p>The table is <b>16 × 16</b>.</p><p>See <code>DS</code>.</p>');
  assert.equal(markdown('Two things:\n- rows ask\n- columns *are* asked\n\nDone.'), '<p>Two things:</p><ul><li>rows ask</li><li>columns <i>are</i> asked</li></ul><p>Done.</p>');
  assert.equal(markdown('1. first\n2. second'), '<ol><li>first</li><li>second</li></ol>');
  assert.equal(markdown('<img src=x onerror=alert(1)> & <b>bold</b>'), '<p>&lt;img src=x onerror=alert(1)&gt; &amp; &lt;b&gt;bold&lt;/b&gt;</p>');
  assert.equal(markdown('a * b * c and 2*3'), '<p>a * b * c and 2*3</p>');
});

// The server's side, run with whatever python3 is installed: these paths need neither the SDK nor a key.
const py = (code, env = {}) => JSON.parse(execFileSync('python3', ['-c', `import json, ask\n${code}`], { cwd: root, env: { ...process.env, ANTHROPIC_API_KEY: '', ASK_ENV_FILE: '/nonexistent/settings', ...env }, encoding: 'utf8' }));
const ask = (text) => `{"page": "Model: test", "messages": [{"role": "user", "content": "<view>step 1</view>\\n<question>\\n${text}\\n</question>"}]}`;

test('the server refuses requests that are not a conversation about a page', () => {
  const codes = py(`
bad = [None, {}, {"page": "", "messages": [{"role": "user", "content": "x"}]}, {"page": "p", "messages": []},
       {"page": "p", "messages": [{"role": "system", "content": "x"}]}, {"page": "p", "messages": [{"role": "user", "content": ""}]},
       {"page": "p", "messages": [{"role": "assistant", "content": "x"}]}, {"page": "p" * 200001, "messages": [{"role": "user", "content": "x"}]},
       {"page": "p", "messages": [{"role": "user", "content": [{"text": "no type"}]}]}]
print(json.dumps([[e["code"] for e in ask.events(b)] for b in bad]))`);
  assert.deepEqual(codes, Array.from({ length: 9 }, () => ['bad_request']));
  // The rules come first and the page's description second, marked for caching; the tools are the page's four.
  const sys = py('print(json.dumps([[b["type"], "cache_control" in b, b["text"][:40]] for b in ask.system_for("PAGE")] + [[t["name"] for t in ask.TOOLS]]))');
  assert.deepEqual(sys.slice(0, 2).map((b) => b.slice(0, 2)), [['text', false], ['text', true]]);
  assert.equal(sys[0][2], 'You answer questions from someone who is');
  assert.equal(sys[1][2], '<page>\nPAGE\n</page>');
  assert.deepEqual(sys[2], definitions.map((d) => d.name));
});

test('after a fallback, thinking and tool calls from before it are not sent back', () => {
  const kept = py(`
T = lambda *types: [{"type": t} for t in types]
print(json.dumps([[b["type"] for b in ask.echoable(c)] for c in [
  T("thinking", "text", "tool_use"),
  T("thinking", "text", "tool_use", "fallback", "thinking", "text", "tool_use"),
  T("fallback", "thinking", "text")]]))`);
  assert.deepEqual(kept, [['thinking', 'text', 'tool_use'], ['text', 'fallback', 'thinking', 'text', 'tool_use'], ['fallback', 'thinking', 'text']]);
});

test('the stand-in model: words arrive one by one, a directive becomes a tool call, results are reported', () => {
  const run = (body) => py(`print(json.dumps(list(ask.events(${body}))))`, { ASK_FAKE: '1' });
  const plain = run(ask('why?'));
  const message = plain.at(-1);
  assert.equal(message.type, 'message');
  assert.equal(message.stop_reason, 'end_turn');
  assert.equal(plain.slice(0, -1).map((e) => e.delta).join(''), message.content[0].text);
  assert.match(message.content[0].text, /I received a view block and the question: “why\?”/);

  const call = run(ask('/tool show_cube {\\"tensor\\": \\"img\\", \\"slab\\": 2, \\"row\\": 5, \\"col\\": 9}')).at(-1);
  assert.equal(call.stop_reason, 'tool_use');
  assert.deepEqual(call.content[1], { type: 'tool_use', id: 'toolu_standin_1', name: 'show_cube', input: { tensor: 'img', slab: 2, row: 5, col: 9 } });

  const after = run(`{"page": "p", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_step", "input": {"step": 1}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Step 1 text"}, {"type": "tool_result", "tool_use_id": "t2", "content": "no such step", "is_error": True}]}]}`).at(-1);
  assert.match(after.content[0].text, /returned 2 result\(s\): Step 1 text; ERROR no such step/);
});

test('settings come from a file outside the project; the environment wins, and unknown names are ignored', () => {
  const dir = fs.mkdtempSync(`${os.tmpdir()}/ask-settings-`);
  const file = `${dir}/.env`;
  fs.writeFileSync(file, ['# a comment', 'ASK_MODEL = claude-haiku-5-5', 'export ASK_EFFORT="low"', 'ANTHROPIC_API_KEY=from-the-file', 'PATH=/nowhere', 'ASK_FAKE=1', ''].join('\n'), { mode: 0o600 });
  const read = (env) => py('import os\nprint(json.dumps([ask.MODEL, ask.EFFORT, os.environ.get("ANTHROPIC_API_KEY"), ask.FAKE, os.environ["PATH"] != "/nowhere"]))', { ASK_ENV_FILE: file, ...env });
  assert.deepEqual(read({}), ['claude-haiku-5-5', 'low', 'from-the-file', false, true]);
  assert.deepEqual(read({ ASK_MODEL: 'claude-sonnet-5-5', ANTHROPIC_API_KEY: 'from-the-shell' }), ['claude-sonnet-5-5', 'low', 'from-the-shell', false, true]);
  // No file: the defaults, and no key.
  assert.deepEqual(py('import os\nprint(json.dumps([ask.MODEL, ask.EFFORT, os.environ.get("ANTHROPIC_API_KEY")]))'), ['claude-opus-5-5', 'medium', '']);
  fs.rmSync(dir, { recursive: true });
});

test('without a key the server says so; it never invents an answer', () => {
  const out = py(`print(json.dumps(list(ask.events(${ask('hello')}))))`);
  assert.equal(out.length, 1);
  assert.equal(out[0].type, 'error');
  // Either the SDK is not installed for this python, or it is and there is no key.
  assert.ok(['no_sdk', 'no_key'].includes(out[0].code), out[0].code);
});
