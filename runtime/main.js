// The page around the 3D runtime: step navigation, side panel, controls,
// tooltip and inspector. All content comes from the compiled scene.
//
//   compiled.html?program=transformer    loads programs/transformer.js

import { Viz } from './viz.js';
import { compile, markupToHtml } from '../lang/scene.js';
import { fmt } from '../lang/explain.js';
import { initAsk } from './ask.js';

const $ = (s) => document.querySelector(s);

const name = new URLSearchParams(location.search).get('program') || 'transformer';
if (!/^[\w-]+$/.test(name)) throw new Error(`bad program name: ${name}`);
const { build } = await import(`../programs/${name}.js`);
const scene = compile(build());
const STEPS = scene.steps;

document.title = scene.title;
$('#title').textContent = scene.title;
$('#subtitle').textContent = scene.subtitle;
$('#dims').innerHTML = scene.dims;

const state = { step: 0, hoverEx: null, pinnedEx: null };

const viz = new Viz($('#scene'), scene, {
  onHover(ex) {
    state.hoverEx = ex;
    renderTooltip();
    renderInspector();
  },
  onPin(ex) {
    state.pinnedEx = ex;
    renderInspector();
  },
  // How much of the stage the step bar covers, so the camera keeps content above it.
  bottomInset: () => ($('#stepnav')?.offsetHeight ?? 0) + 20,
  // Clicking the box of a folded section jumps to that section's first step.
  onSection(i) {
    go(scene.sections[i].from);
  },
});
window.viz = viz; // handy in the console
window.scene = scene;

// ---- side panel -------------------------------------------------------------

function shapesHtml(rows) {
  if (!rows) return '';
  return `<div class="shapes">${rows
    .map(
      ([n, sym, num, note]) => `<div class="sh">
        <span class="n">${markupToHtml(n)}</span>
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
    ${s.body()}
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
// A long tour keeps its buttons in one row that scrolls, so the bar does not grow over the stage.
const LONG = STEPS.length > 30;
$('#stepnav').classList.toggle('long', LONG);
const showChip = (smooth) => {
  if (!LONG) return;
  const [box, chip] = [chips.getBoundingClientRect(), chipButtons[state.step].closest('.grp').getBoundingClientRect()];
  const left = chips.scrollLeft + chip.left - box.left - (box.width - chip.width) / 2;
  chips.scrollTo({ left, behavior: smooth ? 'smooth' : 'auto' });
};

// ---- controls -------------------------------------------------------------------

const controls = $('#controls');
for (const t of scene.toggles) {
  const label = document.createElement('label');
  label.className = 'switch';
  label.innerHTML = `<input type="checkbox" ${t.value ? 'checked' : ''} /><span class="track"></span><span>${t.label}</span>`;
  label.querySelector('input').addEventListener('change', (e) => {
    viz.setToggle(t.name, e.target.checked);
    renderPanel();
  });
  controls.appendChild(label);
}

// Rank-3 tensors are drawn as slabs; one slab can be singled out.
let seg = null;
if (scene.slabs.n) {
  seg = document.createElement('div');
  seg.className = 'seg';
  seg.innerHTML = `<span class="seg-label">Focus ${scene.slabs.unit}</span><button data-h="-1" class="on">All</button>${Array.from(
    { length: scene.slabs.n },
    (_, h) => `<button data-h="${h}" style="--on: ${scene.slabs.colors[h]}">${h}</button>`,
  ).join('')}`;
  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    setFocus(parseInt(b.dataset.h, 10));
    const view = STEPS[state.step].view;
    if (view.perHead) viz.frame(view);
  });
  controls.appendChild(seg);
}

function setFocus(h) {
  viz.setFocusHead(h);
  if (!seg) return;
  for (const b of seg.querySelectorAll('button')) {
    const on = parseInt(b.dataset.h, 10) === h;
    b.classList.toggle('on', on);
    b.style.color = on && h >= 0 ? scene.slabs.colors[h] : '';
    b.style.borderColor = on && h >= 0 ? scene.slabs.colors[h] : '';
  }
}

function go(i, animate = true) {
  state.step = Math.max(0, Math.min(STEPS.length - 1, i));
  // Steps that are easiest to read one slab at a time open focused on one
  // (keeping the one already chosen); the rest show every slab.
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
  showChip(animate);
  $('#prev').disabled = state.step === 0;
  $('#next').disabled = state.step === STEPS.length - 1;
  $('#step-title').textContent = STEPS[state.step].title;
  history.replaceState(null, '', `${location.search}#${state.step}`);
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

// ---- tooltip + inspector ----------------------------------------------------------

const tip = $('#tooltip');
const placeTip = (e) => {
  const r = $('#stage').getBoundingClientRect();
  tip.style.left = `${Math.min(e.clientX - r.left + 16, r.width - 240)}px`;
  tip.style.top = `${e.clientY - r.top + 18}px`;
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

// Questions about what is on screen: the model is told the step and the pinned cube, and may move the page.
initAsk({
  scene,
  step: () => state.step,
  pinned: () => state.pinnedEx,
  focus: () => viz.focusHead,
  go,
  setFocus,
  pin: (id, d, r, c) => viz.pin(viz.byId.get(id).index(d, r, c)),
  unpin: () => viz.pin(-1),
});
