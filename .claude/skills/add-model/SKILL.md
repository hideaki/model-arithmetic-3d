---
name: add-model
description: Add a model to this project as a compiled 3D presentation (one cube per number). Use when asked to "show how X works", to add or present a new model, or to write a new program in programs/. Carries the owner's rules, the propose-then-wait step, the checks, and what to do when the compiler cannot draw something.
---

# Add a model

Used once, end to end, for the LSTM page (2026-11-14) and revised from what that showed.

In this project a model is written once as a short program in `programs/`, and the compiler in `lang/` derives the 3D page from it. You are the front end: you research the model, propose a plan, write the program and its prose, and fix what fails. You never write scene code, coordinates or runtime changes for one model. If the page cannot be made right from the program, that is a gap in the compiler: see "When the compiler cannot draw something".

## Read first

- `HANDOFF.md` sections 2 (rules), 5 (the language), 6 (layout), 8 (known limits) and 9 (working notes).
- `programs/yolo.js` and `test/yolo.test.mjs`, in full. They are the pattern to follow.
- `programs/lstm.js`, if the model runs the same weights several times (`shared`, `repeat`, `again`); `programs/sd.js` for `module` and `part`.
- To judge what the compiler can draw: `lang/ops.js` (is there an operation?), `lang/layout.js` (where will a tensor be put?) and the places in `lang/scene.js` that name operations (arrows, notes, sentences, PyTorch code).

## The owner's rules

These are requirements. Each came from a decision or a correction.

- **Accurate over simple.** Exact numbers, exact shapes, the real arithmetic of the model. Shrink sizes; do not simplify the calculation without declaring it.
- **Realistic but opaque.** Weights are seeded random and the page says so. No invented features, reserved dimensions or roles ("the subject-finder head"). What a trained model does is told in prose only, inside `aside`.
- **Hover is exact.** A cube lights exactly the cells it was computed from. The language guarantees this for its own operations; do not work around it.
- **A missing highlight is explained.** Weights, constants and inputs have no sources, and their hover sentence says what they are.
- **Say what is left out.** Every departure from the real model is declared with `departure(key, text)`.
- **Drawing convention.** Computed values are cubes; learned weights are gold tiles; numbers fixed by a rule or by hand are silver tiles (`constant`, with a `rule`). Rows taken from a table for this input are cubes.
- **Prose is authored, arithmetic is generated.** You write why a step exists. The compiler writes what is computed. Numbers in your prose are interpolations (`${c1.size}`, `${live(...)}`), never typed.
- **Slow, sequenced motion** is the runtime's job. Do not change it for one model.

## The work, in order

### 1. Research

Find out exactly what the model computes. Use the links you were given; otherwise the original paper and one reference implementation, and name both. If a documentation page will not load, read the docstring in the library's source file instead. Write the forward pass as equations with a shape on every line before you design anything. Say in the proposal what you read and what is from memory.

### 2. Proposal

Write it for the owner, who decides. It has these parts, in this order:

1. **Variant**: which published model and version, and why that one.
2. **Example input**: the one sentence, picture or other example the page will process.
3. **Toy sizes**: every axis with its name and size, next to the real model's size, and an estimate of the cube count (the existing pages have about 1,600, 2,500, 8,800 and 24,400).
4. **Step outline**: sections and steps, one line each, with the tensors each step reveals and their shapes. Say that the order of steps may change, because it also decides the layout.
5. **Departures**: everything left out or changed, as it will be declared.
6. **Compiler gaps**: every piece sorted as in the section below, with the choices for each.
7. **What the viewer will not see**: anything true of the model that the page cannot show.

### 3. Stop

Give the proposal to the owner in your reply and wait for the answer. If another session started you, also leave the proposal where that session can read it. Write no program before the answer. If the owner changes the plan, the changed plan is the plan.

### 4. Program

Write `programs/<name>.js`, exporting `build`. Work one section at a time and run `PROGRAM=<name> node tools/describe.mjs` after each: the compiler rejects a program with a list of reasons, and the list is easier to clear when short.

What is easy to miss:

- **Axes are compared by identity.** Two axes of the same size are different axes. `ONE` is the only axis that broadcasts, so tensors of one row that must combine across repeated steps use `ONE`, not an axis of size 1 per step.
- **Every drawn tensor needs `.as(id, label)` and a hover sentence with `.says`.** Every step needs `say`. Statements about trained or real models go in `aside`. Random weights need `departure('untrained', ...)`.
- **The order in which tensors are created is the order of the tour and also decides the layout.** Create first what continues the line coming in from the previous section. A weight table or constant created first lands at the left end of that line and looks as if it were fed by it.
- **Only one tensor is handed from a section to the next**: the last one created. Anything else needed later is brought in with `again(X)`, where it is used.
- **An input that is used but not drawn** (token ids) is created before the first section.
- **In a `repeat` with `detail: 'first'`**, prose inside the steps of later instances is dropped. Say their text in the section, after its steps. Hover sentences are still needed in every instance.
- **A formula of several tensors gets an arrow from only one of them**, unless the result carries `.hint({ arrows: 'all' })`.
- **Section titles are quoted inside generated sentences** ("in the … section"). Write titles that read well there.
- An opening `say` before the first section and a closing one after the last are the two folded overviews.

### 5. Checks

Write `test/<name>.test.mjs` on the pattern of `test/yolo.test.mjs`. It has four parts:

1. **An independent recalculation.** Plain loops that import nothing from `lang/`. Take only the inputs and the parameters from the program (`prog.t(id).data`), recompute everything else, and compare every computed tensor on the page. Break the expected arithmetic once on purpose to see the test fail.
2. **Sources and prose.** `checkProgram(prog)` reports no problems, `prog.warnings` is empty, every step has prose, every drawn tensor has a hover sentence.
3. **Layout.** Nothing overlaps, and data flows left to right. Printing `layout(prog).sections` shows where everything was put before you open a browser.
4. **The scene.** Step and tensor counts, and for every cube of every tensor: what lights up is exactly its sources. Tensors without sources say "Nothing else lights up".

Then run:

```bash
PROGRAM=<name> node tools/describe.mjs
npm test
```

All tests must pass, the old ones included.

### 6. Look at the page

Tests do not see the page. Look at it.

- Start the `compiled-dev` preview entry (port 5174). If the port is taken, another session's copy is already serving this folder: open the page on it. Never start, stop or kill anything on port 5173, and never kill a server you did not start.
- Open `http://localhost:5174/compiled.html?program=<name>&r=1`. After every edit, change `r`: a reload without a new query string runs the old files.
- Set the viewport to 1280 x 800 first, and back to the desktop preset when done.
- The pane throttles animation, so step the clock yourself. Go to a step, tick, wait a second, tick a little more, wait, then take the screenshot:
  ```js
  settle = (n) => { viz.clock.getDelta = () => 0.05; for (let i = 0; i < n; i++) viz.tick(); };
  location.hash = '#5';   // then, after a moment:
  settle(900);            // wait 1 s
  settle(30);             // wait 1 s, then screenshot
  ```
  With fewer ticks the picture shows the step before. Check that the step's newest tensor is in the picture. Large pages need more (the Stable Diffusion page about 1,800).
- Pin a cube without a mouse: `viz.pin(viz.byId.get('F1').index(d, r, c))`, then read `#inspector`.
- Read the generated text too: `scene.steps[n].body()` and `scene.steps[n].code`.

Check: no console errors; every tensor of the step is in frame; titles can be read; no arrow runs through a tensor; every operand that matters has an arrow; a pinned cube lights what its sentence says; the generated sentences read correctly. View every step of a model with 20 steps or fewer. For a larger one, view each kind of step at least once and list the steps you did not view.

Fix what is wrong from the program first: the order of operations, how steps and sections are divided, titles, and the hints in `HANDOFF.md` section 6. If that is not enough, it is a compiler gap.

### 7. Report and record

Tell the owner plainly: what was built (tensors, cubes, steps), the declared departures, where you left the approved outline and why, each compiler change you made, each gap you met and what was done about it, which steps you did not view, and anything that still looks wrong.

Update `HANDOFF.md`: the test count and page list (section 3), the code map (section 4), the operation table if you added one (section 5), the model's state (section 7), what is not done (section 8) and the next steps (section 10).

## When the compiler cannot draw something

Sort every piece of the model into one of three kinds, in the proposal where you can:

- **Expressible now.** Go ahead.
- **A small addition**: a new operation in an existing family in `lang/ops.js` (with `run`, exact `sources`, its lines in `lang/scene.js` and a test), or a new opt-in hint. Only with the owner's approval. Add what this model strictly needs and no more, leave the other pages as they were, and name it in the report as a compiler change.
- **Not drawable**: a loop that depends on the data; a display that is not a tensor (boxes on a picture, a graph's dots and lines); a computed tensor with more than three axes; a second kind of slab on one page; a layout the rules in section 6 do not give. `HANDOFF.md` section 8 lists the limits met so far.

For anything but the first kind, tell the owner: "The compiler cannot draw this yet", what it is, and why. Offer the choices: leave it out as a declared departure, show a static variant, a small addition, or compiler work as a separate task. Record the gap in `HANDOFF.md` section 8.

A gap found after the proposal was approved is handled the same way, but do not stop at the first one: finish everything that does not depend on it (the tests, the rest of the browser look), collect what you found, and ask once.

Do not fake it: no numbers typed into the program, no hidden shortcut that changes the arithmetic, no scene code.

## Finished means

- `programs/<name>.js`: prose on every step, a hover sentence on every drawn tensor, departures declared, zero prose warnings.
- `test/<name>.test.mjs` with the four parts above, and `npm test` passing.
- The page loads with no console errors and has been looked at as described.
- `HANDOFF.md` is updated.
- The report is written.

## Do not

- Read `~/.config/transformer-3d/.env`. It holds the owner's API key.
- Edit `src/`, except `src/style.css`. It is the frozen reference.
- Reorder parameter declarations in `programs/transformer.js`: its test depends on the order.
- Polish the compiler or the other pages while adding a model. List what you noticed in the report.
