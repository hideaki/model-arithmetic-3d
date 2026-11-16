// The question panel: a viewer asks about what they are looking at, and the
// answer comes from a model that is given the page's own text and numbers and
// may move the page (go to a step, pin a cube).
//
// The conversation lives here, in the page. Each round is one POST to /ask on
// the project's server (serve.py), which holds the API key, the rules and the
// tool list. When a round ends by calling tools, they run here, on the page,
// and their results go back in the next round.

import { pageContext, viewContext, makeTools } from './ask-context.js';

const $ = (s) => document.querySelector(s);
const MAX_ROUNDS = 6; // look-ups per question, so one question cannot run away
const MAX_HISTORY = 60; // messages, tool rounds included; beyond it a new conversation starts

const esc = (s) => s.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);

// The small part of Markdown that answers use. The text is escaped first: nothing the model writes becomes HTML.
export function markdown(text) {
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/(^|[\s(])\*([^*\s][^*]*)\*(?=[\s).,;:!?]|$)/g, '$1<i>$2</i>');
  const out = [];
  let list = null;
  const close = () => {
    if (list) out.push(`<${list.tag}>${list.items.map((x) => `<li>${x}</li>`).join('')}</${list.tag}>`);
    list = null;
  };
  for (const block of text.trim().split(/\n{2,}/)) {
    const para = [];
    const flush = () => {
      if (para.length) out.push(`<p>${para.join('<br />')}</p>`);
      para.length = 0;
    };
    for (const line of block.split('\n')) {
      const item = /^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/.exec(line);
      if (item) {
        flush();
        const tag = item[1] ? 'ol' : 'ul';
        if (list?.tag !== tag) close();
        list ??= { tag, items: [] };
        list.items.push(inline(item[2]));
      } else {
        close();
        para.push(inline(line.replace(/^#+\s*/, '')));
      }
    }
    flush();
  }
  close();
  return out.join('');
}

// What to tell the viewer for each way a round can fail. Server messages are shown as they are.
const SAY = {
  no_server:
    'This server cannot answer questions. They are optional and need the project’s own server, <code>serve.py</code>, with the <code>anthropic</code> package and an API key: see “Asking about a page” in README.md. The rest of the page works without them.',
  cut_off: 'The answer was cut off before it finished. Ask again.',
  cancelled: '',
};

// One round: send the conversation, pass the words on as they arrive, return the finished message.
async function round(page, messages, onText, signal) {
  let res;
  try {
    res = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ page, messages }), signal });
  } catch (e) {
    throw { code: signal.aborted ? 'cancelled' : 'no_server' };
  }
  // A plain file server answers a POST with 404, 405 or 501.
  if (!res.ok || !res.body) throw { code: 'no_server' };
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const lines = (pending + value).split('\n');
      pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'text') onText(event.delta);
        else if (event.type === 'error') throw { code: event.code, message: event.message };
        else if (event.type === 'message') return event;
      }
    }
  } catch (e) {
    if (e.code) throw e;
    throw { code: signal.aborted ? 'cancelled' : 'cut_off' };
  }
  throw { code: signal.aborted ? 'cancelled' : 'cut_off' };
}

// page: { scene, step(), pinned(), focus(), go(i), pin(id, d, r, c), unpin(), setFocus(slab) }
export async function initAsk(page) {
  const [toggle, body, log, form, input, send] = ['#ask-toggle', '#ask-body', '#ask-log', '#ask-form', '#ask-input', '#ask-send'].map($);
  if (!toggle) return;
  const definitions = await fetch('runtime/ask-tools.json').then((r) => r.json());
  const tools = makeTools(page.scene, { step: page.step, go: page.go, pin: page.pin, unpin: page.unpin, focus: page.setFocus }, definitions);
  const about = pageContext(page.scene);
  let history = [];
  let running = null; // the AbortController of the question being answered

  const open = (on) => {
    body.hidden = !on;
    toggle.setAttribute('aria-expanded', String(on));
    if (on) input.focus();
  };
  toggle.addEventListener('click', () => open(body.hidden));

  const add = (cls, html = '') => {
    const el = document.createElement('div');
    el.className = cls;
    el.innerHTML = html;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  };
  const busy = (on) => {
    send.textContent = on ? 'Stop' : 'Ask';
    send.classList.toggle('stop', on);
    input.disabled = on;
  };

  async function ask(question) {
    if (history.length > MAX_HISTORY) {
      history = [];
      add('ask-note', 'This conversation was getting long, so a new one starts here.');
    }
    const mark = history.length; // if the answer fails, the conversation goes back to here
    const pinned = page.pinned();
    const view = viewContext(page.scene, { step: page.step(), pinned: pinned && { id: pinned.id, d: pinned.d, r: pinned.r, c: pinned.c }, focus: page.focus() });
    history.push({ role: 'user', content: `${view}\n\n<question>\n${question}\n</question>` });

    add('ask-q', esc(question));
    const answer = add('ask-a', '<p class="ask-wait">Thinking…</p>');
    const ctl = (running = new AbortController());
    busy(true);
    let shown = ''; // every round's words, a blank line between rounds
    const show = () => {
      answer.innerHTML = markdown(shown);
      log.scrollTop = log.scrollHeight;
    };
    try {
      for (let n = 0; ; n++) {
        const message = await round(about, history, (delta) => ((shown += delta), show()), ctl.signal);
        if (message.stop_reason === 'refusal') throw { code: 'refused', message: 'Claude declined to answer this question.' };
        const calls = message.content.filter((b) => b.type === 'tool_use');
        // An answer or a tool call that ran out of room is incomplete: a cut-off tool call must not be run.
        if (message.stop_reason === 'max_tokens') throw { code: 'cut_off', message: 'The answer ran out of room before it finished. Ask for less at a time.' };
        history.push({ role: 'assistant', content: message.content });
        if (message.stop_reason !== 'tool_use' || !calls.length) break;
        // Every call of a round is answered together, in one message.
        const results = calls.map((call) => {
          const out = tools.run(call.name, call.input);
          add(`ask-did${out.is_error ? ' failed' : ''}`, esc(out.note));
          return { type: 'tool_result', tool_use_id: call.id, content: out.content, ...(out.is_error ? { is_error: true } : {}) };
        });
        log.appendChild(answer); // the answer stays below what was done on the way
        history.push({ role: 'user', content: results });
        if (n + 1 >= MAX_ROUNDS) {
          add('ask-note', `Stopped after ${MAX_ROUNDS} look-ups. Ask again to continue.`);
          break;
        }
        if (shown) shown += '\n\n';
      }
      if (!shown.trim()) answer.remove();
    } catch (e) {
      history.length = mark;
      const said = SAY[e.code] ?? esc(e.message || 'Something went wrong. Ask again.');
      if (e.code === 'refused' || !shown.trim()) answer.remove();
      else answer.classList.add('partial');
      if (e.code === 'cancelled') add('ask-note', shown.trim() ? 'Stopped.' : 'Stopped before an answer.');
      else add('ask-note failed', said);
    } finally {
      running = null;
      busy(false);
      input.focus();
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (running) return running.abort();
    const question = input.value.trim();
    if (!question) return;
    input.value = '';
    ask(question);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  window.askPage = { ask, history: () => history }; // handy in the console
}
