// The tensor language: axes, sources and structure. Operations live in ops.js.
//
// A program is ordinary JavaScript run inside program(). Running it builds a
// graph of tensors; every tensor knows its shape as a list of named axes, how
// its numbers are computed, and which input cells each of its cells came from.
// Everything a presentation needs (shape labels, hover sources, steps,
// sections) is read off that graph.

import { prose, refs, placeholders, lint } from './prose.js';

export function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let cur = null;
export const current = (what) => {
  if (!cur) throw new Error(`${what} must be used inside program()`);
  return cur;
};

// ---------------------------------------------------------------------------
// Axes: a named dimension. Shapes are lists of axes, and two axes are the same
// only if they are the same object, whatever their sizes.

const autoSym = (name) => name.replace(/_([A-Za-z0-9]+)/g, '_{$1}');

export class Axis {
  // unit: what one step along the axis is called ("head"), for labels.
  // deck: when this axis is drawn as slabs, stand them close together like a
  //   deck of sheets (channels) instead of far apart (heads).
  constructor(name, size, { sym = autoSym(name), labels = null, unit = null, deck = false } = {}) {
    if (!Number.isInteger(size) || size < 1) throw new Error(`axis ${name}: size must be a positive integer`);
    if (labels && labels.length !== size) throw new Error(`axis ${name}: ${labels.length} labels for size ${size}`);
    Object.assign(this, { name, size, sym, labels, unit, deck });
  }
}

// The unit axis a reduction leaves behind, e.g. the 1 in (T, 1). It is the
// only axis that broadcasts.
export const ONE = new Axis('1', 1);

export function axis(name, size, opts) {
  const p = current('axis()');
  if (p.axes.some((a) => a.name === name)) throw new Error(`axis ${name} is declared twice`);
  const a = new Axis(name, size, opts);
  p.axes.push(a);
  return a;
}

export const shapeSym = (shape) => `(${shape.map((a) => a.sym).join(', ')}${shape.length === 1 ? ',' : ''})`;
export const shapeNum = (shape) => `(${shape.map((a) => a.size).join(', ')}${shape.length === 1 ? ',' : ''})`;

// ---------------------------------------------------------------------------
// Tensors

export class Tensor {
  constructor(prog, kind, shape, op = null) {
    if (shape.some((a) => !(a instanceof Axis))) throw new Error('a shape is a list of axes made with axis()');
    if (shape.length < 1 || shape.length > 4) throw new Error(`rank ${shape.length} tensors cannot be drawn yet (1 to 4 axes)`);
    if (shape.length === 4 && kind !== 'param') throw new Error('only a parameter can have 4 axes: a bank of filters (out, in, height, width)');
    // Inside module() tensors are computed but not drawn: they belong to no section or step.
    const hidden = prog.hide;
    if (kind !== 'input' && !prog.step && !hidden) throw new Error(`a ${op ? op.name : kind} tensor was created outside a step()`);
    this.prog = prog;
    this.kind = kind; // 'input' | 'param' | 'const' | 'act'
    this.shape = shape;
    this.op = op;
    this.size = shape.reduce((n, a) => n * a.size, 1);
    this.data = new Float64Array(this.size);
    this.id = null;
    this.label = null;
    this.note = '';
    this.shapeOverride = null;
    this.hints = {};
    this.hover = null;
    this.prefix = prog.prefix;
    this.section = hidden ? null : prog.section;
    this.step = hidden ? null : prog.step;
    if (hidden) hidden.push(this);
    else {
      prog.tensors.push(this);
      if (this.step) this.step.tensors.push(this);
    }
  }

  // Name the tensor: id for the program, label (with _{sub} markup) for people.
  as(id, label = id, { note = '', shape = null } = {}) {
    const full = this.prefix + id;
    if (this.prog.byId.has(full)) throw new Error(`two tensors are named ${full}`);
    Object.assign(this, { id: full, label, note, shapeOverride: shape });
    this.prog.byId.set(full, this);
    return this;
  }

  // Layout hints, for the few places the derived layout needs a nudge:
  //   side: 'above' | 'below'   which way to move if the natural spot is taken
  //   arrows: 'all'             an arrow from every operand, not only the nearest
  hint(hints) {
    Object.assign(this.hints, hints);
    return this;
  }

  // One sentence for the inspector when a cube of this tensor is hovered.
  // ${row}, ${col}, ${slab} and ${value} stand for the hovered cell.
  says(strings, ...values) {
    this.hover = prose('hover', strings, values);
    return this;
  }

  // The symbol alone, when the label also spells out a definition ("X_{0} = E + P").
  get symbol() { return this.label.split(' = ')[0]; }
  get rank() { return this.shape.length; }
  get dims() { return this.shape.map((a) => a.size); }
  get sym() { return this.shapeOverride || shapeSym(this.shape); }
  get num() { return shapeNum(this.shape); }
  get name() { return this.label || this.id || `an unnamed ${this.op ? this.op.name : this.kind}`; }

  // How the tensor is drawn: D slabs of R rows by C columns. The last axis
  // runs along columns, the one before it along rows, the one before that
  // along depth. A bank of filters (out, in, height, width) is drawn with one
  // slab per input channel and the filters stacked down the rows.
  get view() {
    const d = this.dims;
    if (d.length === 4) return { D: d[1], R: d[0] * d[2], C: d[3] };
    return { D: d.length > 2 ? d[0] : 1, R: d.length > 1 ? d.at(-2) : 1, C: d.at(-1) };
  }

  // Drawing address [depth, row, column] of a cell, and back.
  addr(flat) {
    const d = this.dims;
    if (d.length === 4) {
      const x = flat % d[3];
      const y = ((flat - x) / d[3]) % d[2];
      const c = ((flat - x - y * d[3]) / (d[3] * d[2])) % d[1];
      const o = (flat - x - y * d[3] - c * d[3] * d[2]) / (d[3] * d[2] * d[1]);
      return [c, o * d[2] + y, x];
    }
    const cols = d[d.length - 1];
    const rows = d.length > 1 ? d[d.length - 2] : 1;
    const c = flat % cols;
    const r = ((flat - c) / cols) % rows;
    return [(flat - c - r * cols) / (cols * rows), r, c];
  }
  flat(depth, row, col) {
    const d = this.dims;
    if (d.length === 4) {
      const y = row % d[2];
      return ((((row - y) / d[2]) * d[1] + depth) * d[2] + y) * d[3] + col;
    }
    const cols = d[d.length - 1];
    const rows = d.length > 1 ? d[d.length - 2] : 1;
    return (depth * rows + row) * cols + col;
  }
}

// ---------------------------------------------------------------------------
// Program

export class Program {
  constructor(name, { seed = 1 } = {}) {
    Object.assign(this, {
      name, seed, axes: [], tensors: [], byId: new Map(), sections: [], steps: [],
      toggles: new Map(), departures: [], warnings: [], prefix: '', inner: false,
      intro: { prose: [], hint: null }, outro: { prose: [], hint: null }, instance: null, collapse: false, repeating: null, section: null, step: null,
      shared: null, hide: null, part: null,
    });
    this.rnd = mulberry32(seed);
  }

  gauss(std) {
    let u = 0;
    while (u === 0) u = this.rnd();
    return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.rnd());
  }

  t(id) {
    const t = this.byId.get(id);
    if (!t) throw new Error(`no tensor named ${id}`);
    return t;
  }

  // Recompute every tensor, in the order the program created them.
  run() {
    for (const t of this.tensors) if (t.op) t.op.run(t);
    return this;
  }

  set(name, value) {
    const g = this.toggles.get(name);
    if (!g) throw new Error(`no toggle named ${name}`);
    g.value = value;
    return this.run();
  }

  // The exact cells a cell was computed from: [[tensor, [flat index, ...]], ...]
  sources(t, flat) {
    return t.op ? t.op.sources(flat) : [];
  }

  finish() {
    const errors = [];
    this.steps.forEach((s, i) => (s.index = i + 1)); // step 0 is the folded overview
    this.nSteps = this.steps.length + 2; // ... and so is the last one
    for (const sec of this.sections) {
      sec.from = sec.steps[0]?.index;
      sec.to = sec.steps.at(-1)?.index;
      if (!sec.steps.length) errors.push(`section ${sec.id} has no steps`);
      sec.tensors = this.tensors.filter((t) => t.section === sec);
    }
    // A section hands its last tensor on to the next one: the stream. A
    // section that does not read it starts afresh (a second input, say).
    this.sections.forEach((sec, i) => {
      sec.keep = sec.tensors.at(-1);
      const prev = i ? this.sections[i - 1].keep : null;
      sec.stream = prev && sec.tensors.some((t) => t.op && t.op.name !== 'again' && t.op.inputs.includes(prev)) ? prev : null;
      // The example this section reads, when it reads one directly.
      const inputs = this.tensors.filter((t) => t.kind === 'input');
      sec.input = inputs.find((t) => t.section === sec) || inputs.find((t) => !t.section && sec.tensors.some((u) => u.op?.inputs.includes(t) || u.op?.attrs?.index === t)) || null;
    });
    for (const s of this.steps) if (!s.tensors.length) errors.push(`step "${s.title}" shows no new tensor`);

    // Prose said in a section before its first step opens that step.
    for (const sec of this.sections) {
      const first = sec.steps[0];
      if (!first) continue;
      first.prose.unshift(...sec.lead.prose);
      first.hint ??= sec.lead.hint;
    }
    // Prose may only name what is on screen by then, and typed numbers and
    // statements about other models are flagged.
    const check = (where, item, at) => {
      for (const r of refs(item))
        if (r.step && r.step.index > at) errors.push(`${where} mentions ${r.name}, which first appears later, in "${r.step.title}"`);
      for (const w of lint(item)) this.warnings.push(`${where}: ${w}`);
    };
    for (const b of [this.intro, this.outro, ...this.steps]) {
      const where = b.title ? `step "${b.title}"` : b === this.intro ? 'the opening text' : 'the closing text';
      const at = b.index ?? (b === this.intro ? 0 : Infinity);
      for (const item of [...b.prose, ...(b.hint ? [b.hint] : [])]) {
        check(where, item, at);
        if (placeholders(item).size) errors.push(`${where} uses ${[...placeholders(item)].join(', ')}, which only X.says understands`);
      }
    }
    for (const t of this.tensors) {
      if (!t.hover) continue;
      check(`${t.name}.says`, t.hover, t.step ? t.step.index : 0);
      const ph = placeholders(t.hover);
      if (ph.has('slab') && t.rank < 3) errors.push(`${t.name}.says uses slab, but ${t.name} ${t.sym} has no slabs`);
      if (ph.has('row') && t.rank < 2) errors.push(`${t.name}.says uses row, but ${t.name} ${t.sym} has one axis`);
    }
    for (const t of this.tensors) if (!t.id) errors.push(`${t.name} in step "${t.step?.title}" has no name; add .as(id)`);
    const random = this.tensors.filter((t) => t.kind === 'param' && t.source.type === 'random');
    if (random.length && !this.departures.some((d) => d.key === 'untrained'))
      errors.push(`${random.length} parameters are random; declare departure('untrained', ...) so the presentation says so`);
    if (errors.length) throw new Error(`program "${this.name}" was rejected:\n  - ${errors.join('\n  - ')}`);
    return this.run();
  }
}

export function program(name, opts, fn) {
  if (cur) throw new Error('program() cannot be nested');
  cur = new Program(name, opts);
  try {
    fn();
    return cur.finish();
  } finally {
    cur = null;
  }
}

// ---------------------------------------------------------------------------
// Sources: where numbers that are not computed come from.

export const random = (std, { mean = 0 } = {}) => ({ type: 'random', std, mean });

// A parameter. Its numbers come only from a declared source, never from the
// program text: that is how the "no hand-set weights" policy is enforced.
export function param(id, label, shape, source) {
  const p = current('param()');
  if (!source || source.type !== 'random')
    throw new Error(`param ${id}: give its source as random(std); literal numbers are not allowed`);
  // Inside shared(), a parameter declared again under the same name is the
  // same weights: the same numbers in the same memory, drawn where they are used.
  const first = p.shared?.get(id);
  if (first) {
    if (first.shape.length !== shape.length || first.shape.some((a, i) => a !== shape[i])) throw new Error(`param ${id}: shared weights were first declared as ${first.sym}`);
    const t = new Tensor(p, 'param', shape);
    Object.assign(t, { source: first.source, alias: first, data: first.data });
    return t.as(id, label);
  }
  if (p.hide) throw new Error(`param ${id}: a module can only reuse weights that are declared, and drawn, somewhere else first (inside shared())`);
  const t = new Tensor(p, 'param', shape);
  t.source = source;
  for (let i = 0; i < t.size; i++) {
    const g = p.gauss(source.std);
    t.data[i] = source.mean ? source.mean + g : g;
  }
  if (p.shared) p.shared.set(id, t);
  return t.as(id, label);
}

// Everything declared inside fn shares its parameters by name: a model that
// is run several times (a denoiser called at every sampling step) declares its
// weights once and meets the same ones again on every later call.
export function shared(fn) {
  const p = current('shared()');
  if (p.shared) throw new Error('shared() cannot be nested');
  p.shared = new Map();
  try {
    return fn();
  } finally {
    p.shared = null;
  }
}

// The example being processed (token ids, an image, ...). An input created
// inside a step is drawn there, like any tensor; one created outside the
// steps (token ids, say) is used but not drawn.
export function input(id, label, shape, values) {
  const t = new Tensor(current('input()'), 'input', shape);
  const flat = values.flat(Infinity);
  if (flat.length !== t.size) throw new Error(`input ${id}: ${flat.length} values for shape ${t.num}`);
  t.data.set(flat);
  return t.as(id, label);
}

// A switch the viewer can flip; operations that read it recompute.
export function toggle(name, value, { label = name } = {}) {
  const g = { name, value, label };
  current('toggle()').toggles.set(name, g);
  return g;
}

// A way this presentation departs from a real model; shown to the viewer.
export function departure(key, text) {
  current('departure()').departures.push({ key, text });
}

// ---------------------------------------------------------------------------
// Prose (see prose.js). Where it is said decides where it is shown:
//   inside a step                      that step
//   in a section, before its steps     the section's first step (its only
//                                      step, when a repeat is shown briefly)
//   before the first section           the opening overview
//   after the last section             the closing overview

function addProse(kind, strings, values) {
  const p = current(`${kind}\`...\``);
  if (p.hide || (p.collapse && p.inner)) return;
  const bucket = !p.section ? (p.sections.length ? p.outro : p.intro) : p.step && !p.collapse ? p.step : p.section.lead;
  const item = prose(kind, strings, values);
  if (kind === 'hint') bucket.hint = item;
  else bucket.prose.push(item);
}
export const say = (strings, ...values) => addProse('say', strings, values);
export const aside = (strings, ...values) => addProse('aside', strings, values);
export const hint = (strings, ...values) => addProse('hint', strings, values);

// ---------------------------------------------------------------------------
// Structure

// opts.summary: one line for the folded view of the section.
export function section(id, title, opts, fn) {
  if (typeof opts === 'function') [opts, fn] = [{}, opts];
  const p = current('section()');
  if (p.hide) return fn(); // inside module(): computed, not drawn
  if (p.section) throw new Error('section() cannot be nested');
  const sec = { id: id + (p.instance ?? ''), title, summary: opts.summary || '', steps: [], repeat: p.repeating, part: p.part, collapsed: p.collapse, lead: { prose: [], hint: null } };
  p.sections.push(sec);
  p.section = sec;
  // A repeated instance shown briefly gets one step for the whole section.
  if (p.collapse) p.step = newStep(p, title);
  try {
    return fn();
  } finally {
    p.section = null;
    p.step = null;
  }
}

function newStep(p, title) {
  const s = { title, section: p.section, tensors: [], prose: [], hint: null };
  p.steps.push(s);
  p.section.steps.push(s);
  return s;
}

export function step(title, fn) {
  const p = current('step()');
  if (p.hide) return fn(); // inside module(): computed, not drawn
  if (!p.section) throw new Error(`step "${title}" is outside a section()`);
  if (p.collapse) {
    // A repeat shown briefly: its steps run, but their prose is not shown again.
    p.inner = true;
    try {
      return fn();
    } finally {
      p.inner = false;
    }
  }
  if (p.step) throw new Error('step() cannot be nested');
  p.step = newStep(p, title);
  try {
    return fn();
  } finally {
    p.step = null;
  }
}

// Run fn n times. Tensor ids get the prefix and instance number ("b0.Q"),
// section ids the instance number ("att0"). With detail: 'first', only the
// first instance is walked step by step. name ("Block") titles each instance.
// A named part of the program that is not repeated (a decoder after a loop,
// say): its sections stand under one bar, like one instance of a repeat.
export function part(name, fn) {
  const p = current('part()');
  if (p.hide) return fn();
  if (p.part || p.repeating || p.section) throw new Error(`part "${name}" cannot be inside another part, a repeat or a section`);
  p.part = { name };
  try {
    return fn();
  } finally {
    p.part = null;
  }
}

export function repeat(n, { prefix = '', detail = 'all', name = '' } = {}, fn) {
  const p = current('repeat()');
  try {
    for (let i = 0; i < n; i++) {
      p.prefix = `${prefix}${i}.`;
      p.instance = i;
      p.collapse = detail === 'first' && i > 0;
      p.repeating = { name, index: i, count: n };
      fn(i);
    }
  } finally {
    p.prefix = '';
    p.instance = null;
    p.collapse = false;
    p.repeating = null;
  }
}
