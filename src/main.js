import { Viz } from './viz.js';
import { STEPS } from './steps.js';
import { markupToHtml, fmt, SECTIONS } from './tensors.js';

const $ = (s) => document.querySelector(s);

const state = {
  step: 0,
  hoverEx: null,
  pinnedEx: null,
};

const viz = new Viz($('#scene'), {
  onHover(ex) {
    state.hoverEx = ex;
    renderTooltip();
    renderInspector();
  },
  onPin(ex) {
    state.pinnedEx = ex;
    renderInspector();
  },
  // Clicking the box of a closed section jumps to that section's first step.
  onSection(i) {
    go(SECTIONS[i].from);
  },
});
window.viz = viz; // handy in the console

// ---- side panel -------------------------------------------------------------

function shapesHtml(rows) {
  if (!rows) return '';
  return `<div class="shapes">${rows
    .map(
      ([name, sym, num, note]) => `<div class="sh">
        <span class="n">${markupToHtml(name)}</span>
        <code>${markupToHtml(sym)}</code>
        <span class="num">${num}</span>
        ${note ? `<span class="note">${note}</span>` : ''}
      </div>`,
    )
    .join('')}</div>`;
}

function renderPanel() {
  const s = STEPS[state.step];
  $('#panel-body').innerHTML = `
    <div class="kicker">Step ${state.step} of ${STEPS.length - 1}</div>
    <h2>${s.title}</h2>
    ${s.formula ? `<div class="formula">${s.formula}</div>` : ''}
    ${shapesHtml(s.shapes)}
    ${s.body(viz.model)}
    ${s.code ? `<pre class="code"><code>${s.code}</code></pre>` : ''}
    ${s.try ? `<div class="try"><b>Try it</b>${s.try}</div>` : ''}`;
}

// ---- step navigation ----------------------------------------------------------

// One button per step, grouped under its chapter name.
const chips = $('#chips');
const chipButtons = [];
let group = null;
STEPS.forEach((s, i) => {
  if (!group || STEPS[i - 1].chapter !== s.chapter) {
    group = document.createElement('div');
    group.className = 'grp';
    group.innerHTML = `<span class="gl">${s.chapter}</span><div class="row"></div>`;
    chips.appendChild(group);
  }
  const b = document.createElement('button');
  b.textContent = i;
  b.title = s.title;
  b.addEventListener('click', () => go(i));
  group.querySelector('.row').appendChild(b);
  chipButtons.push(b);
});

function setFocus(h) {
  viz.setFocusHead(h);
  [...$('#headseg').querySelectorAll('button')].forEach((b) => b.classList.toggle('on', parseInt(b.dataset.h, 10) === h));
}

function go(i, animate = true) {
  state.step = Math.max(0, Math.min(STEPS.length - 1, i));
  // Steps that are easiest to read one head at a time open focused on a head
  // (keeping the one you already chose); the rest show every head.
  const want = STEPS[state.step].focus;
  setFocus(want == null ? -1 : viz.focusHead >= 0 ? viz.focusHead : want);
  viz.setStep(state.step, STEPS[state.step].view, animate);
  renderPanel();
  $('#panel-body').scrollTop = 0;
  chipButtons.forEach((b, k) => {
    b.classList.toggle('on', k === state.step);
    b.classList.toggle('done', k < state.step);
  });
  [...chips.children].forEach((g) => g.classList.toggle('on', !!g.querySelector('button.on')));
  $('#prev').disabled = state.step === 0;
  $('#next').disabled = state.step === STEPS.length - 1;
  $('#step-title').textContent = STEPS[state.step].title;
  history.replaceState(null, '', `#${state.step}`);
}

$('#prev').addEventListener('click', () => go(state.step - 1));
$('#next').addEventListener('click', () => go(state.step + 1));
$('#reset').addEventListener('click', () => viz.frame(STEPS[state.step].view));
window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea')) return;
  if (e.key === 'ArrowRight') go(state.step + 1);
  else if (e.key === 'ArrowLeft') go(state.step - 1);
  else if (e.key === 'Escape') viz.pin(-1);
});

// ---- controls -------------------------------------------------------------------

$('#causal').addEventListener('change', (e) => {
  viz.setCausal(e.target.checked);
  renderPanel();
});

$('#headseg').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  setFocus(parseInt(b.dataset.h, 10));
  const view = STEPS[state.step].view;
  if (view.perHead) viz.frame(view);
});

// ---- tooltip + inspector ----------------------------------------------------------

const tip = $('#tooltip');
const placeTip = (e) => {
  const r = $('#stage').getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  tip.style.left = `${Math.min(x + 16, r.width - 240)}px`;
  tip.style.top = `${y + 18}px`;
};
$('#stage').addEventListener('pointermove', placeTip);
$('#stage').addEventListener('pointerdown', placeTip);

function renderTooltip() {
  const ex = state.hoverEx;
  tip.hidden = !ex;
  if (ex) tip.innerHTML = `<span class="t">${ex.title}</span><span class="v">${fmt(ex.value, 3)}</span>`;
}

function renderInspector() {
  const ex = state.pinnedEx || state.hoverEx;
  const box = $('#inspector');
  box.hidden = !ex;
  if (!ex) return;
  box.innerHTML = `
    <div class="ins-head">
      <span class="t">${ex.title}</span>
      <span class="v">${fmt(ex.value, 3)}</span>
    </div>
    ${ex.html}
    <div class="ins-foot">${
      state.pinnedEx ? 'Pinned — click the cube again or press Esc to release' : 'Click the cube to pin this'
    }<br />Numbers are shown rounded, so a sum can look off by 0.01.</div>`;
}

// ---- start --------------------------------------------------------------------------

const hashStep = () => {
  const n = parseInt(location.hash.slice(1), 10);
  return Number.isFinite(n) ? n : 0;
};
window.addEventListener('hashchange', () => {
  if (hashStep() !== state.step) go(hashStep());
});
go(hashStep(), false);
