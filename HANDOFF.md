# Handoff: 3D presentations of AI/ML arithmetic

Written 2026-11-14, at the end of a long session, and updated later the same day when the third model was started (section 7) when the authoring skill and a fourth model were added (sections 7 and 7b), and when a fifth model was added as the skill's second test, and on 2026-11-15 when a sixth (a graph attention network) was added through the skill run from a terminal, for whoever picks this up next (a new Claude session or a person). It covers what the project is, the rules the owner has set, how the code is organised, what is finished, what is not, and how to check work.

## 1. What this is

An interactive 3D page that shows **exactly how a model computes**: every number is one cube, every tensor carries its shape, and hovering a cube lights up the numbers it was computed from.

It started as one hand-built page for a transformer. It is now a **compiler**: a model is written once as a short program in a small tensor language, and the positions, arrows, camera views, step text, hover highlights and checks are derived from it. Six models exist: a two-block GPT-style transformer, a YOLOv3-style detector, a Stable Diffusion v1-style text-to-image model, from prompt to picture (in progress: its weights are still random, so the picture shows nothing), a one-layer LSTM language model, a two-layer graph convolutional network, and one layer of a graph attention network.

The owner's longer-term aim is a system driven in natural language ("show me how X works", optionally with links to papers). An agent would research, propose a plan, write the program and its prose, and fix failures; the compiler stays the deterministic back end. The first piece of that agent layer exists: a repository skill, `.claude/skills/add-model/SKILL.md`, which carries the workflow and was used twice, to add the LSTM and the graph convolutional network (section 7b). Before that, Claude played the role by hand.

## 2. Rules the owner has set

These came out of explicit decisions and corrections. Treat them as requirements.

- **Accurate over simple.** The point is "the understanding of how exactly the calculation is done". Exact numbers, exact shapes (symbolic and numeric), the real arithmetic.
- **Realistic but opaque.** No invented feature columns, reserved dimensions or pre-assigned roles (for example no "subject-finder head"). Weights are seeded random; the page says so. What trained models do is described in prose only, clearly marked.
- **Hover must be exact.** A cube lights exactly the cells it was computed from, never a whole row "for context". Orientation outlines are a separate, dimmer thing.
- **A missing highlight must be explained.** Cubes with no sources (weights, constants, inputs) say so in the inspector.
- **Slow, sequenced motion.** Reveals take about 3 seconds, left to right. A finished section stays open one extra step and folds only after the camera has moved on, so nothing folds in view.
- **Say what is left out.** Every program declares its departures from a real model, and they are shown on the page.
- **Drawing convention** (approved 2026-11-13):
  - Full cube, grey frame: computed from the input.
  - Flat tile, gold frame: fixed and learned (weights).
  - Flat tile, silver frame: fixed and set by a rule or by hand (constants, the causal mask).
  - Rows taken out of a table for this input are **cubes**: E (the token rows) and P (the position rows). The owner questioned drawing them as learned weights on 2026-11-14. The tables themselves, W_E and W_P, stay gold tiles: both are learned in GPT-2 and in CLIP. A transposed table is still the whole table and stays tiles.
- **Prose is authored, arithmetic is generated.** Author text explains purpose; generated text states what is computed and cannot disagree with the numbers. Both are shown.

## 3. How to run and check

```bash
npm test                    # 91 tests, about 6 seconds
npm run describe            # prints the transformer's derived step list
PROGRAM=sd npm run describe # ... or another program's (yolo, sd, lstm, gcn, gat)
python3 serve.py 5173       # no-cache static server
```

Pages:

- `compiled.html` : the transformer, from the compiler
- `compiled.html?program=yolo` : the YOLO detector, from the compiler
- `compiled.html?program=sd` : the Stable Diffusion-style denoising loop, from the compiler
- `compiled.html?program=lstm` : the LSTM language model, from the compiler
- `compiled.html?program=gcn` : the graph convolutional network, from the compiler
- `compiled.html?program=gat` : the graph attention network, from the compiler
- `index.html` : the original hand-written transformer, kept as the reference

There is no build step and no dependencies to install. Three.js r0.170 loads from a CDN through an import map. The project has been a git repository since 2026-11-15 (branch `main`, no remote); everything before that date is one first commit. `.venv/`, `node_modules/` and settings files are ignored.

## 4. Code map

| Path | Lines | What it is |
|---|---|---|
| `lang/core.js` | 449 | Axes, tensors, program, sources (`param`, `input`, `toggle`, `departure`), structure (`section`, `step`, `repeat`, `part`), prose entry points, compile-time checks |
| `lang/ops.js` | 606 | Operations, grouped by family (see section 5) |
| `lang/expr.js` | 148 | Formula parser used by `map`; drives both arithmetic and displayed formulas |
| `lang/prose.js` | 143 | Author prose: tagged templates, references, lint, rendering, `picture` |
| `lang/check.js` | 95 | Verifies declared sources by perturbing every input cell |
| `lang/explain.js` | 109 | Generated explanation of one cell (plain data) |
| `lang/layout.js` | 245 | Derived positions for every tensor |
| `lang/scene.js` | 732 | `compile(program)`: everything the runtime needs |
| `runtime/viz.js` | 1175 | Model-agnostic 3D runtime (Three.js) |
| `runtime/main.js` | 232 | Page shell: step bar, panel, controls, inspector |
| `runtime/ask.js` | 199 | The question panel: the conversation, one round per POST to `/ask`, running the tools, rendering answers |
| `runtime/ask-context.js` | 217 | What the page tells the model (page text, view text) and the four tools acting on the page; no browser needed |
| `runtime/ask-system.md` | 28 | The standing rules for the model. Authored prose: review it like the pages' text |
| `runtime/ask-tools.json` | 58 | The tool definitions, read by both the server and the page |
| `ask.py` | 208 | Server side of a question: checks the request, calls the Claude API with streaming, relays events; has a stand-in model for tests |
| `serve.py` | 79 | Static files with caching off, plus `POST /ask`; serves nothing whose name starts with a dot |
| `programs/transformer.js` | 562 | The transformer program, mostly prose |
| `programs/yolo.js` | 309 | The YOLO program |
| `programs/sd.js` | 1274 | The Stable Diffusion-style program: text, noise, three sampling steps and the decoder |
| `programs/lstm.js` | 345 | The LSTM program: four time steps with shared weights, then the prediction |
| `programs/gcn.js` | 294 | The graph convolutional network: the graph scaled once, the features, two layers |
| `programs/gat.js` | 322 | The graph attention network: the graph and its mask, the features per head, attention scores, the weighted sum |
| `test/*.test.mjs` | 2516 | Twelve test files |
| `.claude/skills/add-model/SKILL.md` | 143 | The authoring skill: how to add a model (section 7b) |
| `tools/describe.mjs` | 36 | Prints steps, shapes and one cell's explanation; `PROGRAM=name` picks the program |
| `examples/agent-sdk/` | 293 | An example only: the skill run from a terminal through the Claude Agent SDK (section 7b). Has its own `package.json` and `node_modules`; nothing else uses it |
| `src/` | 3065 | The hand-written app. **Frozen reference; do not edit** except `src/style.css`, which both pages share |

`runtime/viz.js` began as a copy of `src/viz.js` with every transformer-specific part removed. The two will stay duplicated until the hand-written app is retired.

## 5. The tensor language in brief

A program is ordinary JavaScript run inside `program(...)`. Read `programs/yolo.js` for a compact, complete example.

**Axes are named and compared by identity, not size.** `axis('d_model', 12)`. A matmul across two different axes of the same size is rejected. `ONE` is the unit axis and the only one that broadcasts. Options: `labels`, `unit` ("head", "channel"), `sym`, `deck` (see section 6).

**Every operation belongs to a family that fixes its sources**, which is what hover lights:

| Family | An output cell comes from | Operations |
|---|---|---|
| rearrange | exactly one input cell | `lookup`, `rows`, `transpose`, `split`, `merge`, `channels`, `columns`, `again`, `reshape`, `flatten`, `unflatten`, `upsample`, `concat` |
| map | the same-position cell of each operand, with broadcasting | `map(formula, operands)`, `add`, `gelu`, `sigmoid`, `silu`; formula functions include `mask(a)` (0 where a > 0, −∞ elsewhere) |
| reduce | every cell along one axis, or every cell of a group of channels | `mean`, `sum`, `std`, `softmax`, `groupMean`, `groupStd` |
| contract | a row and a column, or a window and a filter | `matmul`, `conv2d` |
| const | nothing | `causalMask`, `constant` |
| module | the input cells it depends on, traced through operations that are not drawn | `module` |

**Tensors have 1 to 3 axes**; a parameter may have 4 (a bank of filters), drawn with one slab per input channel and the filters stacked down the rows.

**Sections need not form one line.** A section's stream is the previous section's last tensor only if the section reads it; otherwise the section starts afresh (text, timestep and noise in the Stable Diffusion program). A tensor that is needed in a later section and is not the stream is brought in with `again(X)`: an explicit, drawn copy whose hover lights the original.

**One model, called several times.** `shared(fn)` makes parameters declared again under the same name the same weights (the same memory); each use is still a drawn tensor, noted "the same weights as in Step 1", and counted once. `module(title, inputs, fn)` runs a part of the program again without drawing it: everything `fn` builds is computed exactly but belongs to no section, and only the result is a tensor on screen. Its sources are found by following the undrawn operations' own sources back to the inputs, so hover stays exact and `check.js` verifies it. A module may only read the inputs it is handed and may only use shared weights that are drawn somewhere else.

**Group statistics** (`groupMean`, `groupStd`) give `(c, 1, 1)`: one cube per channel, the channels of a group holding the same number. This keeps group norm an ordinary broadcast.

**Structure drives the tour.** `section` gives a foldable box, `step` one tour step, `repeat(n, { detail: 'first' })` walks the first instance in detail and shows later ones as a single step each. The compiler adds a folded overview as the first and last step. `part(name, fn)` groups sections that are not repeated (the decoder) under one bar; its name is said once, in the chapter of its first box.

**What the compiler rejects** (each has a test): mismatched axes; a parameter given literal numbers; random weights without `departure('untrained', ...)`; a step that shows no new tensor; an unnamed tensor; a formula with an unknown name or unused operand; prose naming a tensor not yet on screen; cell placeholders used in the wrong place.

**Prose** (`say`, `aside`, `hint`, `X.says`) is a tagged template. Tensors, axis sizes and values are interpolations, so they stay live. The lint warns (does not fail) on typed numbers that look like data and on words such as "trained" or "GPT-2" outside an `aside`. All three programs currently have zero warnings, and tests assert that. `${picture(X)}` in prose shows a `(3, H, W)` tensor of brightnesses as the colour picture it stands for, one square per pixel, read when the text is shown.

## 6. Layout rules

Coordinates are local to a section, one unit per cube, data flowing left to right along y = 0. Rules by role:

- **flow**: right of its operands, on the lane of the first; moves up or down a lane if the spot is taken.
- **weight**: a parameter multiplied on the right sits inline (A, W, result). When the lane is taken, the pair moves up or down until the result is clear too, not only the weight (a 16-row result next to a 6-row weight).
- **beside**: an activation multiplied on the right, or a same-shape constant, sits directly above or below the result.
- **over**: a per-column gain or a per-slab constant sits just above the result.
- **front**: an operand with no depth axis is a plate in front of the slabs.
- **stack**: a cell-by-cell function of a wide tensor goes underneath it.
- **ring**: adding two streams leaves room for a "+" node; a far operand becomes a residual arc. An arc that starts inside the section runs under the one that spans it.
- **over** also holds a single fixed number (a schedule value, a setting): one silver tile above the result it enters.
- **apart**: a section that does not read the one before it starts 14 units in, clear of that section's last tensor.
- **lane**: a tensor brought in with `again` part-way through a section enters on a side lane below the flow, level with where the flow has got to; what is computed from it stays on that lane. Arrows from a side lane bend just before their target when something is in the way.
- An activation multiplied on the right is placed **beside** the result only if it is a rearranged view (K transposed, V split); anything computed in the flow stays in the flow and gets its own arrow.

Rank-3 tensors are drawn as slabs. **Heads stand far apart; channels are declared `deck: true` and stand close together.** This distinction was the main transformer-specific assumption YOLO exposed.

Manual hints: `.hint({ side: 'below' })` on a tensor, `.hint({ place: 'flow' })` to keep a transposed operand in the flow instead of beside the result, `.hint({ show: 'image' | 'prob' })` for colouring, `.hint({ frame: 'operands' })` on a result so that the step's camera view takes in all of its operands, however far back they stand (otherwise only those within 40 columns are framed), and `.hint({ arrows: 'all' })` on a result of a cell-by-cell formula so that every operand gets an arrow into it (otherwise only the nearest operand does, plus the one on the result's own lane when the nearest came in from a side lane).

## 7. State of the six models

**Transformer** (`compiled.html`): 80 tensors, 8,784 cubes, 25 steps. Tests prove the compiled version matches the hand-written app cube for cube: numbers bit for bit (causal mask on and off), shapes, step and section of every tensor, and hover sources. All steps and tensors have authored prose, ported and adapted from the hand-written app.

**YOLO** (`compiled.html?program=yolo`): 20 tensors, 2,462 cubes, 13 steps. A 16 x 16 colour picture, two stride-2 convolutions with leaky ReLU, a 1x1 head giving 8 numbers per grid cell, then decoding into box centres, sizes, objectness, class probabilities and scores. Tests recompute every number with independently written loops and check the highlight for every cube.

Declared departures in YOLO: untrained; one scale and one anchor; no batch norm or bias; no suppression step; tiny sizes.

**Stable Diffusion-style text-to-image model** (`compiled.html?program=sd`): 257 drawn tensors, 24,414 cubes, 59 steps. Started 2026-11-14; the owner chose the v1-style U-Net denoiser over a transformer denoiser and approved the toy outline. Built so far:

- **Text**: token rows plus position rows give the context C (4 × 6). **Noise**: a seeded 4 × 4 × 4 latent, z_667.
- **Three sampling steps** (t = 667, 334, 1), under bars STEP 1 to 3. Each step: the timestep as a vector; the U-Net with the prompt; the U-Net with an empty prompt; guidance (ε = ε_∅ + 7.5 · (ε_c − ε_∅)); the DDIM update (guess the clean latent, then mix the noise back in for the next timestep). The schedule is Stable Diffusion's scaled-linear one; ᾱ values and the guidance scale are single silver tiles.
- **The U-Net is drawn in full once**, for the first call (8 boxes, 23 tour steps): convolution in, a ResBlock walked in detail, cross-attention walked in detail, stride-2 conv down, a ResBlock in one step, upsample and conv, concatenate the saved skip tensor, a ResBlock from 12 channels to 6 in one step, norm, SiLU and conv out.
- **Its five other calls are modules**: the three inputs are brought in, one arrow labelled U-Net leads to the answer, and hovering the answer lights all three inputs completely (true, and checked). Their intermediate tensors exist under a prefix (`s1.c.h2`, `s2.u.out`) for tests.
- Steps 2 and 3 are shown briefly, one tour step per box. The owner accepted showing the five later U-Net calls by their result only (2026-11-14); it is a choice made in this program, not a rule of the compiler.
- **The decoder** (built 2026-11-14), under a bar DECODER: 7 boxes, 13 tour steps, its own 20 weight tensors, run once on the finished latent z_0. Divide by the scaling factor 0.18215 (a silver tile); a 1 × 1 convolution; convolution in (4 → 6 channels); a ResBlock with no time vector, in one step; self-attention over the 16 grid cells, single head, in four steps (queries on the line, keys above, values below); upsample and convolve to 8 × 8; a ResBlock that narrows 6 → 4 channels with a 1 × 1 shortcut, in one step; upsample and convolve to 16 × 16; group norm, SiLU, convolution out to 3 channels; then `min(max(o / 2 + 0.5, 0), 1)`. The result `img` is drawn as three tinted sheets, and the last step and the closing overview show it as a colour picture in the text panel. With random weights it is coloured noise.

Tests recompute the whole loop and the whole decoder with independently written loops and check the highlight of every cube.

Agreed plan for the rest: a tiny trained checkpoint, because a generator with random weights only turns noise into noise. PyTorch is not installed on this machine; NumPy is.

Declared departures: untrained; U-Net drawn once; decoder trimmed (two doublings and two ResBlocks, where the real one has three doublings and 14 ResBlocks); three sampling steps; text encoder cut to embedding plus position; U-Net trimmed; attention block trimmed to single-head cross-attention; no biases; tiny sizes.

**LSTM language model** (`compiled.html?program=lstm`): 76 drawn tensors, 1,617 cubes, 16 steps. Added 2026-11-14 as the test of the authoring skill (section 7b); the owner chose it over a graph convolutional network and a Vision Transformer, and approved the outline.

- **The example**: "The cat sat on", 4 tokens from a vocabulary of 8; 6 numbers per token (d_e), 5 per state (d_h). The arithmetic is that of `torch.nn.LSTM` (gate order i, f, g, o) followed by the decoder of PyTorch's `word_language_model` example.
- **One box per token**, under bars TIME STEP 1 to 4, all with the same weights (`shared`). The first is walked in nine steps: the hidden state h_0 starts as zeros (silver tiles); h · W_hh; the token's row is looked up in W_E; x · W_ih; the sum z (1 × 20); z is cut into four runs of 5 (`columns`); sigmoid for the gates i, f, o and tanh for the candidate g; c = f · c + i · g; h = o · tanh(c). The three later time steps are one tour step each.
- **The state**: h is handed from box to box as the stream. c is brought into the next box with `again`, on a side lane, where it is used.
- **The prediction**: h_4 · W_out gives 8 logits, softmax gives 8 probabilities.

Tests recompute every number with independently written loops (taking only the weights from the program) and check the highlight of every cube.

Declared departures: untrained; one layer (the example stacks 2); no biases; one prediction, after the last token only (real code predicts at every position and returns log-probabilities); the token is looked up inside its time step (real code looks all tokens up first); tiny sizes.

What it added to the compiler, each approved by the owner: the operation `columns(X, start, ax)` (a run of columns copied out of every row, PyTorch's `chunk` along the last axis); the opt-in hint `arrows: 'all'` (section 6), used on the new cell state and the new hidden state so that every gate has an arrow into them; and a sentence for a lookup of a single row ("copies the one row of W_E that the token id picks"). The other pages draw and read as before. Nothing in `runtime/` was changed.

**Graph convolutional network** (`compiled.html?program=gcn`): 20 drawn tensors, 716 cubes, 15 steps. Added 2026-11-14 as the second test of the authoring skill (section 7b), by a session that started from the skill alone; the owner approved the outline. The model is the two-layer network of Kipf and Welling (2017), equation 9, in the order of the paper's own code (`tkipf/gcn`): weights first, then the graph; no bias; softmax over the classes.

- **The example**: 8 papers with 9 links between them, and for each paper which of 6 words it contains. Both are made up for the page. 4 hidden numbers, 3 classes.
- **Graph** (5 steps): the table of links A; A + I, with I as silver tiles; each row added up to the degree d (`sum`); s = 1 / √d and its transpose; Â = Ã × s × sᵀ, cell by cell.
- **Features** (2 steps): the 0/1 word table X₀; each row divided by its total.
- **Layer 1** and **Layer 2** (3 steps each): X · W, then Â · (X W), then ReLU in the first layer and softmax in the second. Â is brought into each layer with `again` and enters on a side lane below the flow; the line from box to box is X → H₁ → Z.

Tests recompute every number with independently written loops (taking only the two inputs and the two weight tables from the program), check the same result with the paper's order of multiplication, and check the highlight of every cube.

Declared departures: untrained; tiny sizes; inference only (no dropout, no loss, no weight decay); dense tables where real code is sparse; the preparation of Â and of the features shown as steps, where real code does it once beforehand; no drawing of the graph.

What it added to the compiler, each approved by the owner: the operation `sum(X, { axis })` in the reduce family (the total of each line; the line lights up), with its line of PyTorch and its sentence in `lang/scene.js` and its per-cell explanation in `lang/explain.js`; and a change to the source checker. `lang/check.js` nudges every input cell by a fixed amount and expects the output to move, and ReLU of a number well below zero does not move, so it reported the true sources of a ReLU as unused. This was the first page with plain ReLU (the others use GELU, leaky ReLU, SiLU, sigmoid, tanh). Before calling a listed source unused, the checker now tries the same cell with its sign flipped. A source that is listed and truly unused is still caught (`test/lang.test.mjs`). The other pages draw and read as before. Nothing in `runtime/` was changed.

**Graph attention network** (`compiled.html?program=gat`): 21 drawn tensors, 1,136 cubes (1,024 computed numbers plus two `again` copies), 15 steps. Added 2026-11-15 through the skill, run from a terminal with the Claude Agent SDK (section 7b); the owner chose the paper's hidden layer over its output layer and approved the outline. The model is one layer of Veličković et al. (2018), equations 1 to 5: K heads joined side by side, then ELU; the heads' weight tables held as one table split into heads and the attention vector a held as its halves a₁ and a₂, as PyG's `GATConv` does. The research was from memory (paper, the authors' TensorFlow code, PyG), not read in that session.

- **The example**: the graph and words of the GCN page (8 papers, 9 links, 6 words), so the two pages can be compared; the features enter already divided by their row totals. 2 heads of 3 numbers.
- **Graph** (3 steps): A; Ã = A + I; the mask M = mask(Ã), 0 where linked and −∞ elsewhere (cubes: the mask comes from the input).
- **Transform** (3 steps): X; X · W (6 × 6); split into two heads Wh (2, 8, 3).
- **Attention scores** (4 steps): s = Wh a₁ and t = Wh a₂ (2, 8, 1), tᵀ; e = max(s + tᵀ, 0.2 · (s + tᵀ)), arrows from both; M brought in with `again` on a side lane, ẽ = e + M; α = softmax(ẽ), unlinked cells drawn blocked.
- **Weighted sum** (3 steps): Wh brought in again (beside H), H = α Wh; heads merged to (8, 6); ELU, written as `relu(x) + exp(−relu(−x)) − 1` so that its generated PyTorch line is valid.

Tests recompute every number with independently written loops (e from the paper's form aᵀ[Wh_i ‖ Wh_j], softmax over each node's list of neighbours rather than with the mask), check the static-attention ordering the prose states, and check the highlight of every cube and that α is blocked exactly where there is no link.

Declared departures: untrained; tiny sizes; one layer; inference only (no dropout); no bias; dense table with a −∞ mask (PyG scores only along links; the authors' code uses −10⁹); features prepared beforehand; no drawing of the graph.

What it added to the compiler, approved by the owner after asking why: the formula function `mask(a)` in `lang/expr.js` (0 where a > 0, −∞ elsewhere) and its PyTorch line `torch.where(a > 0, 0.0, float('-inf'))` in `lang/scene.js`. The alternatives were `log(Ã)` (exact, but its generated code line is not PyTorch and it is a trick) and `−10⁹ · (1 − Ã)` (the authors' code, but it washes out the colour scale and is not drawn as blocked). The other pages draw and read as before. Nothing in `runtime/` was changed.

## 7a. Questions from the viewer (prototype, 2026-11-14)

Every compiled page has an **Ask about this page** panel under the step text. The owner chose the route "API key and a small server" over publishing as a Claude artifact or running Claude Code locally (see the comparison in project memory).

**How a question travels.** The page keeps the conversation. For each question it sends `POST /ask` with two things: a description of the whole page (model name, declared departures, every step, every tensor with id, shape and how it is made; about 27,000 characters for the Stable Diffusion page) and the messages so far. The newest message carries a `<view>` block: the step the viewer is on, in full, and the cube they pinned with the page's own explanation of it. `ask.py` adds the rules (`runtime/ask-system.md`) and the tools, calls the API with streaming, and relays one JSON object per line: words as they arrive, then the finished message. If the model called tools, the page runs them, appends the results and posts again, at most 6 rounds per question.

**The four tools** run on the page: `go_to_step` and `show_cube` move the page (the viewer sees it; `show_cube` goes to the tensor's step if it is not on show, focuses its slab and pins it), `read_step` and `read_cells` only read. Inputs are checked against the schema on the page before anything runs, because the API does not check streamed tool inputs.

**The rules the model is held to** (in `ask-system.md`): numbers come from the page, never from memory; what this page does is kept apart from what real models do; no invented meanings while the weights are random; show on the page where that helps; be brief.

**Server settings** (environment): `ANTHROPIC_API_KEY`; `ASK_MODEL` (default `claude-opus-5-5`); `ASK_EFFORT` (default `medium`); `ASK_FAKE=1` replaces the model with a stand-in that turns `/tool <name> <json>` into that tool call and otherwise echoes, so the page and the round trip can be tested without a key. The description of the page is marked for prompt caching. If the model's safety checks decline a request, the API retries it on the model Anthropic recommends (`fallbacks: "default"`).

**Running it.** The server needs the `anthropic` package, installed in `.venv/` in the project (not in the system Python):

```bash
.venv/bin/python serve.py 5173
```

The key is read from a settings file **outside the project**, `~/.config/transformer-3d/.env` (a line `ANTHROPIC_API_KEY=...`; `ASK_MODEL` and `ASK_EFFORT` may go there too; `ASK_ENV_FILE` names another file). It is kept out of the project folder because a plain `python3 -m http.server` started there would serve a `.env` to the whole local network, and out of the shell's environment because Claude Code bills a key it finds in `ANTHROPIC_API_KEY` instead of using the Claude plan. A value set in the environment still wins. The tests point `ASK_ENV_FILE` at a file that does not exist, so they never touch the real key. With the system `python3`, or with no key, the pages still work and the panel says what is missing. The launch entries `compiled-dev` (5174) and `ask-stand-in` (5175, stand-in model) both use `.venv`.

**Guards.** The server listens on 127.0.0.1 only. `/ask` accepts only JSON, and only from its own pages (a JSON content type cannot be sent from another site without a preflight, and the `Origin` must match), so another web page cannot spend the key. Request size, message count and page-text size are capped. Answers are escaped before a small Markdown renderer runs, so nothing the model writes becomes HTML. None of this is enough for a public server: that needs sign-in or per-visitor limits.

**State.** Works against the real API (first run 2026-11-14, `claude-opus-5-5` at medium effort, no changes needed). Three real questions on the Stable Diffusion page, each answered in about 7 seconds:

- With a cube of the decoder's attention pinned, "why so much attention to that one, and which cell gets the most?": it quoted the pinned value exactly, found the row's largest share, pinned that cube with `show_cube`, and said the pattern means nothing because the weights are random. The two values it quoted were checked against the page's data.
- "Where does the text prompt get into the U-Net? Show me.": it moved the page to step 15 with `go_to_step`, explained keys and values, and correctly named the three steps of the promptless calls. It remembered the previous question.
- "What does this cube tell me?" with a cross-attention weight pinned: read the cube from the view, no tool call.

Also tested with the stand-in model in the browser (each tool, a failing tool, Stop, missing key) and in `test/ask.test.mjs`. Not tested: a refusal, an answer cut off by the token limit, a conversation long enough to restart, the transformer and YOLO pages with the real model, and what a question costs (estimated at a few cents on this model; the page description is about 10,000 tokens and is cached for five minutes).

**The owner's key** is in `~/.config/transformer-3d/.env` (created by the owner 2026-11-14; never read it, check only its shape). If questions start failing with "the API key was rejected", the key has expired or was revoked, and a new one is needed.

## 7b. The authoring skill (2026-11-14)

`.claude/skills/add-model/SKILL.md` is a Claude Code project skill: any session opened in this folder can use it (`/add-model <model, links>`, or it is picked up when someone asks to show how a model works). It is the workflow that was done by hand for YOLO and Stable Diffusion, written down:

1. Research the model's exact arithmetic.
2. Propose: variant, example input, toy sizes, step outline, departures, compiler gaps, what the viewer will not see.
3. Stop for the owner's decision.
4. Write the program, 5. the tests, 6. look at every step in a browser, 7. report and update this document.

It carries the rules of section 2, and a rule for what the compiler cannot draw: sort every piece as expressible, a small addition (owner's approval needed), or not drawable; say so plainly; never fake it.

**The one test so far** was the LSTM. Followed as written, it produced a page that compiled on the first run and needed one approved compiler addition. Following it also showed where it was unclear, and it was revised: what to read to judge the compiler's limits; that creation order decides the layout; how prose works in a repeat shown briefly; how many clock ticks a screenshot needs; to collect gaps found late and ask once. The gaps that belong to the compiler, not to the skill, are in section 8.

**The third run** (2026-11-15) added the graph attention network from a terminal through the Agent SDK program, with the owner at the terminal: proposal, the owner's choices, a program that compiled on its first run with no prose warnings, one approved compiler addition (section 7), tests and a look at all 15 steps. This was the first run of the skill itself through `examples/agent-sdk/`.

**The second test** was the graph convolutional network, by a fresh session that was given the skill and nothing else, and told not to edit it. The skill got that session through research, a proposal with both expected gaps named before any code (no sum along an axis; a graph's dots and lines), a program that compiled on its first run with no prose warnings, the tests and a look at all 15 steps. It did not prepare the session for three things found after the proposal: the source checker rejecting plain ReLU (fixed with the owner's approval, section 7), operands whose arrow is dropped, and the overview hiding Â (both left as recorded limits, section 8). The owner was asked twice: the proposal, and those three together. The session's list of where the skill was unclear or silent is in the notes of that group of sessions (`skill-test-2-findings.md`); the skill itself has not been revised from it yet.

**Outside the desktop app (2026-11-15).** `examples/agent-sdk/add-model.mjs` runs the same skill from a terminal through the Claude Agent SDK, to show that the authoring loop does not depend on the app. The skill is loaded unchanged; the program adds a headless browser for step 6, a page server if none is running, and an approval prompt for everything outside a short allowed list (so a change under `lang/` cannot be made without the owner's `y`). Its `README.md` has the commands and the limits. Checked with three small runs against the real API: start-up, the approval prompt and a typed refusal, the refusal of anything that names the key, and the headless browser opening a page of this project. A run of the skill itself was done on 2026-11-15: the graph attention network (section 7). One trap: run from a terminal inside Claude Code or the desktop app, an Agent SDK program inherits `CLAUDE_CODE_ENTRYPOINT=claude-desktop`, and the API then refuses every request made with an API key (400, "Your credit balance is too low", whatever the balance). The program leaves the session's `CLAUDE…` variables out.

**What the skill cannot do.** It does not make a model drawable that the compiler cannot draw: it only says so. The owner still has to answer the proposal, and nothing checks the prose for truth.

## 8. Not done, and known problems

- **One arrow per formula is still the default.** A cell-by-cell formula with operands on several lanes gets an arrow from the nearest operand only, unless the result carries `.hint({ arrows: 'all' })`. The LSTM uses the hint; YOLO's scores and Stable Diffusion's guidance and sampler formulas do not, and were not looked at again.
- **LSTM: the folded overview shows only h between the boxes.** One tensor is handed from a box to the next (the stream); the cell state's hand-over is visible only inside a box. The same limit would hide any second carried state.
- **LSTM: the three later time steps show a whole box (18 tensors) in one view**, so the cubes are small. The same rough spot as the one-step ResBlocks of the Stable Diffusion page.
- **LSTM: looked at, lightly.** All 16 steps were viewed once at 1280 x 800 with the clock stepped by hand; five cubes were pinned without a mouse. Not viewed in real time, not at phone width.

- **The source checker can still miss a flat function.** It tries each cell at two values (nudged, and with its sign flipped; section 7, graph network). A formula that is flat on both sides, such as a clamp between two limits fed with numbers outside them, would still be reported as not depending on its source.
- **An operand on the result's own lane gets no arrow when another tensor stands between.** With `arrows: 'all'`, an operand on a different lane always gets its arrow, but one on the same lane is skipped if anything is in between (`clear` in `lang/scene.js`), since the arrow would run through it. On the graph network page Â = Ã × s × sᵀ has an arrow from sᵀ only, though Ã is the operand that matters most, and X = X₀ / r has one from r only. Nothing in the program can move d, s and sᵀ off the lane. Hover and the generated sentence are right.
- **GCN: the folded overview does not show Â entering the layers.** The same limit as the LSTM's cell state below: one tensor is handed from box to box. Here it hides the model's main structure, the same graph used in both layers; the overview reads as one line Graph → Features → Layer 1 → Layer 2.
- **GCN: the graph is not drawn as dots and lines.** A display that is not a tensor, like YOLO's boxes. The owner chose to leave it out, declared as a departure (2026-11-14).
- **GAT: looked at once.** All 15 steps viewed at 1280 x 800 with the clock stepped by hand; three cubes pinned without a mouse. Rough spots: the steps with both heads (7 to 12) are wide views with small cubes; in step 6 and step 12 a folded neighbouring box stands in front of a slab (head 1's Wh, α); the merge label reads "K·F′ = K·F′" because the joined axis's symbol is already the product; "head axis: K = 2" touches H's shape line in steps 11 and 12. The folded overview does not show M entering the scores box (the one-stream limit above).
- **GCN: looked at once.** All 15 steps were viewed at 1280 x 800 with the clock stepped by hand; eight cubes were pinned without a mouse and one highlight was looked at. Not viewed in real time, not at phone width.

- **Viewer questions: three real answers seen, all good, but that is a small sample** (section 7a). The rules in `runtime/ask-system.md` are a first draft the owner has not reviewed. A conversation is lost on reload. The model cannot yet change the Focus control on its own or rerun the model with a changed number.
- **Stable Diffusion: no trained weights yet** (section 7). The picture at the end is coloured noise.
- **Stable Diffusion: the folded overview is unreadable.** It is one row of 32 small boxes, and it is the first and the last screen of the tour. Grouped folding (one box per sampling step and one for the decoder, six boxes in all) was planned and is not built.
- **Stable Diffusion is heavier to draw.** About 7.5 ms per animation tick in the Browser pane with 24,414 cubes, against about 2.2 ms before the decoder (and 2.8 ms for the transformer page). Not measured on the owner's own browser.
- **Stable Diffusion: looked at, not polished.** About 20 of the loop's 46 steps and all 13 of the decoder's were viewed on the page (camera settled by stepping the clock; hover was checked by tests, not by mouse). Known rough spots: the one-step ResBlocks show about 20 tensors at once; the Focus control lists 12 channels because the widest stack has 12; in the decoder's attention the 16 column labels of the score tables are small. A tour with more than 30 steps now keeps its step buttons in one row that scrolls sideways (`#stepnav.long` in `src/style.css`), so the bar no longer grows over the stage; the transformer and YOLO pages are below that and unchanged.
- **The camera now keeps content above the step bar**, whose height the page reports (`bottomInset` in `runtime/main.js`). This also moved the framing on the transformer and YOLO pages slightly; one step of each was looked at afterwards.
- **One slab axis per page.** The Focus control and slab colours assume a single kind of slab (heads, or channels). A program mixing heads and channels would label both the same way; the Stable Diffusion program avoids this by using one attention head.

- **Boxes are not drawn on the picture.** YOLO's 16 predicted boxes exist only as numbers. This needs a display that is not a tensor. The owner was asked whether to do this next and has not answered.
- **Non-maximum suppression is not shown.** It is a data-dependent loop, which the language cannot express. A static variant ("fast NMS") would be expressible.
- **Crowded labels in YOLO.** In the decoding steps, 4 x 4 tensors are small next to their titles and some overlap. In the folded overview, a neighbouring title partly covers a section box label.
- **Back sheets of a deck** can only be hovered after choosing that channel with Focus channel.
- **Constant titles** are pale steel blue, close to normal white; the frame and flatness are the real cues.
- **Never watched in real time.** All browser checks stepped the animation clock by hand (see section 9). Fold timing was measured on the hand-written app only.
- **Not every step viewed.** Roughly 15 of 25 transformer steps and 8 of 13 YOLO steps have been looked at on the compiled pages.
- **Phone width** has not been checked since the transformer grew a second block.
- **Narrow stages cut titles.** At a stage about 880 px wide, the top title in the transformer's Q, K, V step is cut off.
- **No checkpoint loading.** Only random sources exist; `checkpoint(...)` was designed but not built.
- **No PyTorch front end and no natural-language agent.** Both were planned as later phases.
- **Prose truth is unchecked.** The lint is keyword-based, and anything inside `live(...)` is opaque to it.
- **Tensor names in prose are not interactive.** They render as `<span class="ref">` but hovering them does nothing yet.

## 9. Working notes that save time

- **Port 5173 may already be in use by the owner's own server** (`python3 -m http.server 5173`). Do not kill it. That server caches, so tell the owner to hard-reload (Cmd+Shift+R). For your own checks use the `compiled-dev` preview entry in `.claude/launch.json`, which runs `serve.py` on port 5174 with caching off. If 5174 is taken too, another session's `compiled-dev` is serving this same folder: open your page on it and leave it running.
- **A hash-only URL change does not reload modules.** After editing code, navigate to a URL with a changed query string (for example `?r=7`), or the browser keeps running the old files.
- **The Browser pane throttles animation when hidden.** Do not wait on animation frames. Step the clock instead:
  ```js
  viz.clock.getDelta = () => 0.05; for (let i = 0; i < 320; i++) viz.tick();
  ```
  Then wait about a second and tick a few more times before taking a screenshot, or the image lags one state behind. In practice 320 ticks was too few even for the small LSTM page (the picture showed the step before); 900 ticks, a second's wait, 30 more and another wait worked. The Stable Diffusion page needs about 1,800 ticks for a step to settle and about 3,300 for the overview to fold.
- **Set the viewport to 1280 x 800 before judging layout**, and reset it to the desktop preset afterwards. The default pane is narrow enough to trigger the phone layout.
- **Text on a folded box faces the camera, and is slid towards it each frame** (`clearBoxLabel` in `runtime/viz.js`) until it is nearer than the box's nearest corner. Without that, seen from the side, half of the text sank into the box (reported by the owner 2026-11-14). Other objects that really are in front still hide it.
- `window.viz` and `window.scene` are exposed on the compiled pages. `viz.pin(viz.byId.get('F1').index(d, r, c))` pins a cube without a mouse.
- **Hover bugs hide where tests are thin.** Two were reported by the owner: the picture not lighting under F1 (inputs were skipped), and c_xy lighting nothing (correct, but unexplained). The fix both times included a test over every cube. When adding an operation or a new kind of tensor, add the all-cubes highlight test for its program.
- **The transformer test is a regression lock.** `test/transformer.test.mjs` compares against `src/model.js` and `src/tensors.js`. Random numbers match because parameters are drawn in declaration order from one seeded stream; reordering parameter declarations in `programs/transformer.js` will break it.

## 10. Sensible next steps

On 2026-11-14 the owner said the compiler is good enough for now and turned to the system around it. Confirm with the owner before starting any of these.

**The system around the compiler**

1. **The authoring agent, after its second test.** The skill has now been followed by a fresh session for a model the compiler fits less well, the graph convolutional network (section 7b). Next: revise the skill from that session's findings; then let the skill start from a paper link, and decide whether the proposal step should become a form the owner fills in.
2. **Viewer questions, second pass** (section 7a). Have the owner review `runtime/ask-system.md`; try the transformer and YOLO pages; compare cheaper models and lower effort; measure what a question costs; keep the conversation across a reload; let the model set the Focus control; a "what if this number were different" tool that reruns the program.
3. **A stronger number check.** Today the same author writes a program and its independent recalculation. For models that exist in a standard library, compare against that library with the same toy sizes and weights.
4. **A front page** listing the models, and a decision on where pages are published.

**The pages themselves**

5. **Grouped folding of the Stable Diffusion overview** (section 8): six boxes in place of 32. It is the first screen of the tour and cannot be read.
6. **A tiny trained checkpoint for Stable Diffusion**, so the picture shows the prompt. Needs checkpoint loading in the language and a way to train (PyTorch is not installed; NumPy is).
7. Draw YOLO's predicted boxes on the picture, and the graph convolutional network's graph as dots and lines: both need a display that is not a tensor. Fix label crowding in YOLO's decoding steps.
8. Make tensor names in the prose interactive (hover a name, outline the tensor).
9. Graph attention network: check its arithmetic against the paper and PyG's `GATConv` source (the research for it was from memory), ideally with the number check of item 3.

When adding a model, use the skill (`.claude/skills/add-model/SKILL.md`): propose the variant, toy sizes, step outline and departures first, get a decision, then write the program.

## 11. Memory

Claude's project memory for this folder holds the same goals and decisions in shorter form (two files: the project goal, and the compiler direction). A new session in this folder loads them automatically; this document is the fuller version.
