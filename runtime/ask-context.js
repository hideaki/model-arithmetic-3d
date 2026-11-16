// What a page tells the model about itself, and what the model may do on it.
//
// Everything here is a plain function of the compiled scene, so it runs in
// tests without a browser. The page side (ask.js) sends the text and runs the
// tools; the server side (ask.py) holds the rules and the tool definitions.
//
//   pageContext(scene)             the page as a whole: sent once, cached by the API
//   viewContext(scene, view)       what the viewer is looking at: sent with every question
//   makeTools(scene, page)         the four tools of runtime/ask-tools.json, acting on the page

const ENTITIES = { amp: '&', lt: '<', gt: '>', nbsp: ' ', quot: '"' };

// The page's HTML as plain text: subscripts as _x, lists as "- ", tables as "a | b".
export function toText(html) {
  return String(html ?? '')
    .replace(/<svg[\s\S]*?<\/svg>/g, '[a picture is shown here]')
    .replace(/<sub>(.*?)<\/sub>/g, '_$1')
    .replace(/<sup>(.*?)<\/sup>/g, '^$1')
    .replace(/([_^])\{([^}]*)\}/g, '$1$2') // the same, in the page's own label markup
    .replace(/<\/t[dh]>\s*<t[dh][^>]*>/g, ' | ')
    .replace(/<li[^>]*>/g, '\n- ')
    .replace(/<br\s*\/?>/g, '\n')
    .replace(/<\/(p|div|h\d|ul|ol|tr|table|pre)>/g, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/&(#?\w+);/g, (m, k) => ENTITIES[k] ?? (k[0] === '#' ? String.fromCodePoint(Number(k.slice(1))) : m))
    .replace(/[ \t]+/g, ' ')
    .replace(/ *\n */g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

const name = (t) => t.label.replace(/[{}]/g, '');
// A number as the model should quote it: enough digits to be exact for the page's purposes.
export const num = (v) => (Number.isFinite(v) ? String(Number(v.toPrecision(6))) : v < 0 ? '-infinity' : 'infinity');

const KIND = { param: 'weights', const: 'fixed', mask: 'fixed' };
const kindOf = (spec) => (spec.t.kind === 'input' ? 'input' : KIND[spec.kind] || 'computed');
function madeBy(t) {
  if (t.alias) return `the same weights as ${t.alias.id}`;
  if (!t.op) return t.kind === 'input' ? 'given' : 'declared';
  return `${t.op.family === 'module' ? t.op.title : t.op.name}(${t.op.inputs.map((x) => x.id).join(', ')})`;
}

// The page as a whole. Stable for a given page, so the API can cache it.
export function pageContext(scene) {
  const last = scene.steps.length - 1;
  return [
    `Model: ${scene.title}`,
    toText(scene.subtitle),
    `Axis sizes: ${toText(scene.dims)}`,
    '',
    'What this page leaves out or changes, compared with a real model (declared on the page):',
    ...scene.departures.map((d) => `- ${d.key}: ${d.text}`),
    '',
    `Steps of the tour, 0 to ${last} (number, chapter, title). Steps 0 and ${last} are the folded overview of everything:`,
    ...scene.steps.map((s, i) => `${i} | ${toText(s.chapter)} | ${toText(s.title)}`),
    '',
    'Tensors (id | name on the page | shape = sizes | drawn as slabs x rows x columns | kind | step that introduces it | made by):',
    ...scene.tensors.map((spec) => {
      const { D, R, C } = spec.t.view;
      return `${spec.id} | ${name(spec.t)} | ${toText(spec.sym)} = ${spec.num} | ${D} x ${R} x ${C} | ${kindOf(spec)} | ${spec.step} | ${madeBy(spec.t)}`;
    }),
    '',
    'Kinds: "weights" are learned in a real model (flat gold tiles); "fixed" numbers are set by a rule or by hand (flat silver tiles); "computed" numbers depend on the input (cubes); "input" is given.',
  ].join('\n');
}

// One step, in full: what the side panel shows for it.
export function stepText(scene, i) {
  const s = scene.steps[i];
  const made = scene.tensors.filter((spec) => spec.step === i).map((spec) => spec.id);
  return [
    `Step ${i} of ${scene.steps.length - 1}: ${toText(s.title)} (chapter: ${toText(s.chapter)})`,
    s.formula ? `Formula: ${toText(s.formula)}` : '',
    s.shapes?.length ? `Shapes:\n${s.shapes.map(([n, sym, numbers, note]) => `- ${toText(n)} ${toText(sym)} = ${numbers}${note ? ` (${toText(note)})` : ''}`).join('\n')}` : '',
    toText(s.body()),
    s.code ? `Code:\n${toText(s.code)}` : '',
    s.try ? `Try it: ${toText(s.try)}` : '',
    made.length ? `Tensor ids introduced in this step: ${made.join(', ')}` : '',
  ]
    .filter(Boolean)
    .join('\n\n');
}

// One cube: its value, the page's explanation of it, and what lights up.
export function cubeText(scene, id, d, r, c) {
  const ex = scene.explain(id, d, r, c);
  const lit = new Map();
  for (const g of ex.groups) lit.set(g.id, (lit.get(g.id) || 0) + (g.r1 - g.r0 + 1) * (g.c1 - g.c0 + 1));
  return [
    `${toText(ex.title)} = ${num(ex.value)}   (tensor ${id}, slab ${d}, row ${r}, column ${c})`,
    toText(ex.html),
    lit.size ? `Lit on the page as its sources: ${[...lit].map(([src, n]) => `${n} cell${n > 1 ? 's' : ''} of ${src}`).join(', ')}.` : '',
  ]
    .filter(Boolean)
    .join('\n');
}

// What the viewer is looking at now. view: { step, pinned: {id, d, r, c} | null, focus: slab number or -1 }
export function viewContext(scene, { step, pinned = null, focus = -1 }) {
  return [
    '<view>',
    stepText(scene, step),
    '',
    scene.slabs.n ? `Focus: ${focus >= 0 ? `only ${scene.slabs.unit} ${focus} is shown` : `every ${scene.slabs.unit} is shown`}.` : '',
    pinned ? `The viewer has pinned this cube:\n${cubeText(scene, pinned.id, pinned.d, pinned.r, pinned.c)}` : 'No cube is pinned.',
    '</view>',
  ]
    .filter((line) => line !== '')
    .join('\n');
}

// The reason an input does not fit a tool's schema, or null. The API does not check inputs that are streamed.
export function mismatch(schema, input) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) return 'the input is not an object';
  for (const key of schema.required || []) if (!(key in input)) return `"${key}" is missing`;
  for (const [key, value] of Object.entries(input)) {
    const want = schema.properties?.[key];
    if (!want) return `"${key}" is not a field of this tool`;
    if (want.type === 'integer' && !Number.isInteger(value)) return `"${key}" must be a whole number`;
    if (want.type === 'string' && (typeof value !== 'string' || !value)) return `"${key}" must be text`;
  }
  return null;
}

const MAX_CELLS = 400;

// The tools, acting on the page.
//   page: { step(), go(i), pin(id, d, r, c), unpin(), focus(slab) }   all but the first change what the viewer sees
// run(name, input) resolves to { content, is_error, note }: content goes back to the model, note is shown to the viewer.
export function makeTools(scene, page, definitions) {
  const schemas = new Map(definitions.map((def) => [def.name, def.input_schema]));
  const specs = new Map(scene.tensors.map((spec) => [spec.id, spec]));
  const last = scene.steps.length - 1;

  const stepOf = ({ step }) => {
    if (step < 0 || step > last) throw new Error(`there is no step ${step}: steps run from 0 to ${last}`);
    return step;
  };
  const tensorOf = ({ tensor }) => {
    const spec = specs.get(tensor);
    if (spec) return spec;
    const near = scene.tensors.filter((x) => name(x.t) === tensor || x.id.toLowerCase() === tensor.toLowerCase() || x.id.endsWith(`.${tensor}`)).map((x) => x.id);
    throw new Error(`there is no tensor with the id "${tensor}".${near.length ? ` Did you mean: ${near.slice(0, 8).join(', ')}?` : ' Use an id from the tensor list.'}`);
  };
  const within = (what, v, n) => {
    if (v < 0 || v >= n) throw new Error(`${what} ${v} is out of range: this tensor has ${what}s 0 to ${n - 1}`);
    return v;
  };
  // A tensor is on show from the step that introduces it until its section folds.
  const visible = (spec) => {
    const sec = scene.sections.find((s) => s.id === spec.sec);
    return page.step() >= spec.step && page.step() <= sec.to;
  };

  const tools = {
    go_to_step(input) {
      const step = stepOf(input);
      page.unpin(); // a cube pinned earlier belongs to what the viewer was looking at before
      page.go(step);
      return { content: `The page is now on this step.\n\n${stepText(scene, step)}`, note: `Moved to step ${step}: ${toText(scene.steps[step].title)}` };
    },
    read_step(input) {
      const step = stepOf(input);
      return { content: stepText(scene, step), note: `Read step ${step}` };
    },
    show_cube(input) {
      const spec = tensorOf(input);
      const { D, R, C } = spec.t.view;
      const [d, r, c] = [within('slab', input.slab, D), within('row', input.row, R), within('column', input.col, C)];
      const moved = !visible(spec);
      if (moved) page.go(spec.step);
      if (D > 1) page.focus(d);
      page.pin(spec.id, d, r, c);
      const ex = scene.explain(spec.id, d, r, c);
      return {
        content: `${moved ? `The page moved to step ${spec.step}, which introduces this tensor. ` : ''}The cube is pinned and its sources are lit.\n\n${cubeText(scene, spec.id, d, r, c)}`,
        note: `Pinned ${toText(ex.title)}${moved ? ` (moved to step ${spec.step})` : ''}`,
      };
    },
    read_cells(input) {
      const spec = tensorOf(input);
      const { D, R, C } = spec.t.view;
      const d = within('slab', input.slab, D);
      const [r0, r1] = [within('row', input.row_from ?? 0, R), within('row', input.row_to ?? R - 1, R)];
      const [c0, c1] = [within('column', input.col_from ?? 0, C), within('column', input.col_to ?? C - 1, C)];
      if (r1 < r0 || c1 < c0) throw new Error('a range ends before it starts');
      const count = (r1 - r0 + 1) * (c1 - c0 + 1);
      if (count > MAX_CELLS) throw new Error(`that is ${count} numbers; ask for at most ${MAX_CELLS} at a time (a narrower range of rows or columns)`);
      const rows = [];
      for (let r = r0; r <= r1; r++) {
        const values = [];
        for (let c = c0; c <= c1; c++) values.push(num(spec.t.data[spec.t.flat(d, r, c)]));
        rows.push(`row ${r}: ${values.join(', ')}`);
      }
      return {
        content: `${name(spec.t)} (tensor ${spec.id}), slab ${d}, rows ${r0} to ${r1}, columns ${c0} to ${c1}:\n${rows.join('\n')}`,
        note: `Read ${count} number${count > 1 ? 's' : ''} of ${name(spec.t)}`,
      };
    },
  };

  return {
    names: Object.keys(tools),
    run(toolName, input) {
      const schema = schemas.get(toolName);
      if (!schema || !tools[toolName]) return { content: `There is no tool called "${toolName}".`, is_error: true, note: `Unknown tool ${toolName}` };
      const wrong = mismatch(schema, input);
      // An input that does not fit is sent back as it was received, so the model can write it again.
      if (wrong) return { content: `Invalid input: ${wrong}. ${JSON.stringify({ INVALID_JSON: JSON.stringify(input) })}`, is_error: true, note: `A ${toolName} call was malformed` };
      try {
        return { is_error: false, ...tools[toolName](input) };
      } catch (e) {
        return { content: `Error: ${e.message}`, is_error: true, note: `${toolName} failed: ${e.message}` };
      }
    },
  };
}
