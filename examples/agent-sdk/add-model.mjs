#!/usr/bin/env node
// Example: the add-model skill, run from a terminal through the Claude Agent SDK.
//
//   node add-model.mjs "graph attention network, one layer"   start the skill for a model
//   node add-model.mjs --say "the arrows on step 6 are wrong" send any first message instead
//   node add-model.mjs --check                                one cheap call: is everything in place?
//
// The program adds nothing to the compiler or to the skill. It gives the agent
// the project folder, the skill in .claude/skills/add-model, a headless browser,
// and a person to answer. What the agent may do without asking is the short
// list in ALLOWED below; everything else is shown to the person first.
//
// Settings, from the environment:
//   ANTHROPIC_API_KEY   the key; if unset, it is taken from the settings file
//   ASK_ENV_FILE        the settings file, default ~/.config/model-arithmetic-3d/.env
//   ADD_MODEL_MODEL     default claude-opus-5-5
//   ADD_MODEL_PORT      the page server's port, default 5174

import { query } from '@anthropic-ai/claude-agent-sdk';
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const MODEL = process.env.ADD_MODEL_MODEL || 'claude-opus-5-5';
const PORT = Number(process.env.ADD_MODEL_PORT || 5174);
const PAGE = `http://localhost:${PORT}`;
const ENV_FILE = (process.env.ASK_ENV_FILE || '~/.config/model-arithmetic-3d/.env').replace(/^~/, homedir());

// What the agent may do without asking. A path that starts with one slash is
// relative to the project folder. Reading inside the project needs no rule.
const ALLOWED = [
  'Skill',
  'WebSearch',
  'WebFetch',
  'Edit(/programs/**)',
  'Edit(/test/**)',
  'Edit(/HANDOFF.md)',
  'Bash(node tools/describe.mjs*)',
  'Bash(PROGRAM=* node tools/describe.mjs*)',
  'Bash(npm test*)',
  'Bash(node --test*)',
  'mcp__playwright__*',
];
// Never, whatever the person answers.
const DENIED = ['Read(~/.config/model-arithmetic-3d/**)', 'Edit(~/.config/model-arithmetic-3d/**)'];

// What this program is, said to the agent. The skill was written for the Claude
// desktop app; these lines cover the places where a terminal differs.
const SETTING = `
You are running inside a small terminal program built on the Claude Agent SDK, not inside the Claude desktop app. The person at the terminal is the owner that the add-model skill speaks of.

- When the skill tells you to stop and wait for the owner (after the proposal, or with a question about something the compiler cannot draw), end your turn with the proposal or the question as your reply. The owner's answer arrives as the next message.
- There is no preview pane and no launch entry to start. The page server for this folder is already running at ${PAGE}, and this program keeps it up. Do not start, stop or kill any server.
- To look at a page, use the Playwright browser tools: browser_navigate to open it, browser_evaluate to run the clock-stepping lines from the skill, browser_take_screenshot to see the result, browser_console_messages for errors. The browser is headless and its viewport is already 1280 x 800. The page server has no tab icon, so every page logs one 404 for /favicon.ico; that is not an error of the page.
- A command outside a short allowed list is shown to the owner for approval, one at a time. Prefer the commands the skill names, and the Read, Glob and Grep tools over shell commands for looking at files.
- A change under lang/ or runtime/ is always shown to the owner before it is made. Ask in words first, as the skill says, so that the approval prompt is not the first the owner hears of it.
`.trim();

// ---------- terminal ----------

const dim = (s) => (process.stdout.isTTY ? `\x1b[2m${s}\x1b[0m` : s);
const bold = (s) => (process.stdout.isTTY ? `\x1b[1m${s}\x1b[0m` : s);

// One reader for the whole run. Lines typed (or piped in) are handed out in
// order; null means there is no one left to answer.
const lines = [];
const waiting = [];
let closed = false;
const rl = createInterface({ input: process.stdin });
rl.on('line', (line) => (waiting.length ? waiting.shift()(line) : lines.push(line)));
rl.on('close', () => {
  closed = true;
  while (waiting.length) waiting.shift()(null);
});
function ask(text) {
  process.stdout.write(text);
  if (lines.length) return Promise.resolve(lines.shift());
  if (closed) return Promise.resolve(null);
  return new Promise((done) => waiting.push(done));
}

// ---------- setup ----------

// The key is read into this process's environment and never printed.
function loadKey() {
  if (process.env.ANTHROPIC_API_KEY || !existsSync(ENV_FILE)) return;
  for (const line of readFileSync(ENV_FILE, 'utf8').split('\n')) {
    const m = line.match(/^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.+?)\s*$/);
    if (m && !process.env.ANTHROPIC_API_KEY) process.env.ANTHROPIC_API_KEY = m[1].replace(/^["']|["']$/g, '');
  }
}

const answers = async (url) => {
  try {
    return (await fetch(url, { signal: AbortSignal.timeout(1500) })).ok;
  } catch {
    return false;
  }
};

// Uses the page server that is already there, or starts one and stops it at the end.
async function pageServer() {
  if (await answers(`${PAGE}/compiled.html`)) return { started: false };
  const python = existsSync(join(ROOT, '.venv/bin/python')) ? join(ROOT, '.venv/bin/python') : 'python3';
  const child = spawn(python, ['serve.py', String(PORT)], { cwd: ROOT, stdio: 'ignore' });
  process.on('exit', () => child.kill());
  for (let i = 0; i < 20; i++) {
    if (await answers(`${PAGE}/compiled.html`)) return { started: true };
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`The page server did not come up on port ${PORT}.`);
}

// The environment handed to the agent. A terminal opened inside Claude Code or the
// Claude desktop app carries that session's own variables. One of them does harm:
// CLAUDE_CODE_ENTRYPOINT (there, "claude-desktop") is sent along with every request
// as the place the request comes from, and with that value the API refused requests
// made with an API key: 400, "Your credit balance is too low", whatever the balance.
// Left out, the SDK names itself ("sdk-ts") and the same key is accepted. The rest
// belong to that other session (its ids, its message socket), so they go too.
function agentEnv() {
  const env = { ...process.env };
  for (const name of Object.keys(env)) if (name.startsWith('CLAUDE')) delete env[name];
  env.CLAUDE_AGENT_SDK_CLIENT_APP = 'add-model-example/0.1';
  return env;
}

// ---------- what the agent asks the person ----------

// Refused before anything else is considered: a call that names the key or its
// file, or a shell command that prints the whole environment. This runs as a hook,
// because commands the agent may run without asking never reach the person.
const touchesKey = (input) =>
  /\.config\/model-arithmetic-3d|ANTHROPIC_API_KEY|ASK_ENV_FILE/.test(JSON.stringify(input)) ||
  /(^|[;&|(]\s*)(printenv|env|set|export\s+-p)\s*($|[;&|)>])/.test(String(input?.command ?? ''));
const KEY_REFUSAL = 'This program does not let the agent near the API key or its file.';
const keyGuard = async (hookInput) =>
  touchesKey(hookInput.tool_input)
    ? { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'deny', permissionDecisionReason: KEY_REFUSAL } }
    : {};

function describe(toolName, input) {
  const rel = (p) => (typeof p === 'string' && p.startsWith(ROOT) ? relative(ROOT, p) : p);
  if (toolName === 'Bash') return `run: ${input.command}`;
  if (toolName === 'Edit' || toolName === 'Write' || toolName === 'NotebookEdit') return `change the file ${rel(input.file_path)}`;
  if (toolName === 'Read') return `read ${rel(input.file_path)}`;
  if (toolName === 'Skill') return `skill ${input.skill}${input.args ? `: ${input.args}` : ''}`;
  if (toolName === 'WebFetch') return `fetch ${input.url}`;
  if (toolName === 'WebSearch') return `search the web: ${input.query}`;
  if (toolName === 'Glob' || toolName === 'Grep') return `${toolName.toLowerCase()} ${input.pattern}`;
  if (toolName.startsWith('mcp__playwright__')) {
    const what = toolName.slice('mcp__playwright__'.length);
    const detail = input.url || (typeof input.function === 'string' ? input.function.replace(/\s+/g, ' ').slice(0, 80) : '');
    return `browser ${what.replace(/^browser_/, '')}${detail ? `: ${detail}` : ''}`;
  }
  return toolName;
}

// The agent's own multiple-choice questions.
async function answerQuestions(input) {
  const out = {};
  for (const q of input.questions || []) {
    console.log(`\n${bold(q.question)}`);
    q.options.forEach((o, i) => console.log(`  ${i + 1}. ${o.label}: ${o.description}`));
    const typed = ((await ask(q.multiSelect ? 'numbers separated by commas, or your own words> ' : 'a number, or your own words> ')) ?? '').trim();
    const picked = typed
      .split(',')
      .map((s) => q.options[Number(s.trim()) - 1])
      .filter(Boolean)
      .map((o) => o.label);
    out[q.question] = picked.length ? picked.join(', ') : typed;
  }
  return { behavior: 'allow', updatedInput: { questions: input.questions, answers: out } };
}

async function canUseTool(toolName, input, { suggestions = [] }) {
  if (toolName === 'AskUserQuestion') return answerQuestions(input);
  if (touchesKey(input)) return { behavior: 'deny', message: KEY_REFUSAL };

  console.log(`\n${bold('The agent asks to')} ${describe(toolName, input)}`);
  if ((toolName === 'Edit' || toolName === 'Write') && typeof (input.new_string ?? input.content) === 'string') {
    const text = input.new_string ?? input.content;
    console.log(dim(text.split('\n').slice(0, 12).join('\n') + (text.split('\n').length > 12 ? '\n…' : '')));
  }
  const typed = await ask('y = yes, a = yes and do not ask again for this, n = no, or say what to do instead> ');
  if (typed === null) return { behavior: 'deny', message: 'No one is at the terminal to approve this.' };
  const reply = typed.trim();
  if (/^y(es)?$/i.test(reply)) return { behavior: 'allow', updatedInput: input };
  if (/^a$/i.test(reply)) {
    // "Do not ask again" lasts for this run only; nothing is written to a settings file.
    const forThisRun = suggestions.map((s) => ({ ...s, destination: 'session' }));
    return { behavior: 'allow', updatedInput: input, updatedPermissions: forThisRun };
  }
  if (!reply || /^n(o)?$/i.test(reply)) return { behavior: 'deny', message: 'The owner said no.' };
  return { behavior: 'deny', message: `The owner said no, and added: ${reply}` };
}

// ---------- the conversation ----------

const userMessage = (text) => ({ type: 'user', message: { role: 'user', content: text }, parent_tool_use_id: null });

async function main() {
  const args = process.argv.slice(2);
  const check = args[0] === '--check';
  const say = args[0] === '--say';
  const text = (say ? args.slice(1) : args).join(' ').trim();
  if (!check && !text) {
    console.log('Usage:\n  node add-model.mjs "<model, paper and variant>"\n  node add-model.mjs --say "<any first message>"\n  node add-model.mjs --check');
    process.exit(2);
  }

  loadKey();
  const server = await pageServer();
  console.log(dim(`project ${ROOT}\npages   ${PAGE} (${server.started ? 'started by this program' : 'already running'})`));

  // The agent's turn ends with a result message; only then is the person asked for the next line.
  let turnEnded;
  const nextTurn = () => new Promise((done) => (turnEnded = done));
  let waitForTurn = nextTurn();
  async function* conversation() {
    yield userMessage(check ? 'Reply with the one word: ready' : say ? text : `/add-model ${text}`);
    while (!check) {
      await waitForTurn;
      waitForTurn = nextTurn();
      let line = '';
      while (line !== null && !line.trim()) line = await ask(`\n${bold('you>')} `);
      if (line === null || /^\/?(quit|exit)$/i.test(line.trim())) return;
      yield userMessage(line);
    }
    await waitForTurn; // --check: keep the input open until the one answer is in
  }

  const run = query({
    prompt: conversation(),
    options: {
      cwd: ROOT,
      model: MODEL,
      systemPrompt: { type: 'preset', preset: 'claude_code', append: SETTING },
      settingSources: ['project'], // the project's .claude folder: this is where the skill is
      skills: ['add-model'],
      allowedTools: ALLOWED,
      disallowedTools: DENIED,
      permissionMode: 'default',
      canUseTool,
      hooks: { PreToolUse: [{ hooks: [keyGuard], onFailure: 'block' }] },
      mcpServers: {
        playwright: {
          command: join(HERE, 'node_modules/.bin/playwright-mcp'),
          args: ['--headless', '--isolated', '--browser', 'chrome', '--viewport-size', '1280x800', '--output-dir', join(tmpdir(), 'add-model-screenshots')],
        },
      },
      env: agentEnv(),
    },
  });

  let cost = 0;
  let greeted = false;
  for await (const m of run) {
    if (m.type === 'system' && m.subtype === 'init') {
      if (greeted) continue; // the SDK repeats this at the start of every turn
      greeted = true;
      const browser = m.mcp_servers.find((s) => s.name === 'playwright');
      console.log(
        dim(
          `model   ${m.model}\nkey     ${m.apiKeySource}\nskill   ${m.skills.includes('add-model') ? 'add-model found' : 'add-model NOT found'}\nbrowser ${browser ? browser.status : 'not configured'}\n`,
        ),
      );
    } else if (m.type === 'assistant') {
      const indent = m.parent_tool_use_id ? '    ' : '';
      for (const block of m.message.content) {
        if (block.type === 'text' && block.text.trim()) console.log(`\n${indent}${block.text.trim()}`);
        if (block.type === 'tool_use' && block.name !== 'AskUserQuestion') console.log(dim(`${indent}· ${describe(block.name, block.input)}`));
      }
    } else if (m.type === 'result') {
      cost = m.total_cost_usd;
      if (m.subtype !== 'success') console.log(`\nThe run stopped: ${m.subtype}`);
      console.log(dim(`\n(${m.num_turns} steps this turn; about $${cost.toFixed(2)} so far, an estimate)`));
      turnEnded();
    }
  }
  rl.close();
}

main().then(() => process.exit(0)).catch((e) => {
  console.error(`\n${e.message}`);
  if (/credit balance/i.test(e.message)) console.error('The account behind the API key has no credit left. Add credit in the Anthropic Console, then run this again.');
  process.exit(1);
});
