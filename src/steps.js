// The guided tour: what each step does, with exact shapes.
// `shapes` rows are [name, symbolic shape, numeric shape, note?] using _{sub} markup.
// `view` tells the camera which tensors to frame and from where.
// `chapter` groups the step buttons. The order here must match STEP in tensors.js.
//
// Block 1 is walked through one operation at a time. Block 2 repeats it with
// new weights, so it gets one step per sub-layer.

import { TOKENS, VOCAB, IDS, T, V, D, H, DK, DFF, TMAX, NB } from './model.js';
import { SECTIONS, fmt, par, idsOfSection } from './tensors.js';

const pct = (v) => `${Math.round(v * 100)}%`;
const tok = (i) => `<span class="tok">${TOKENS[i]}</span>`;
const word = (v) => `<span class="tok">${VOCAB[v]}</span>`;

const TD = `(${T}, ${D})`;
const HTD = `(${H}, ${T}, ${DK})`;
const HTT = `(${H}, ${T}, ${T})`;
const LAST = 2 * NB; // the stream after the last block is X_LAST
const PER_BLOCK = 2 * D + 4 * D * D + 2 * D * DFF;
const PARAMS_TOY = V * D + TMAX * D + NB * PER_BLOCK + D;

// Tensor ids inside block 1 and block 2.
const B1 = (...names) => names.map((n) => `b0.${n}`);
// Every section's output tensor: the whole model in its collapsed form.
const KEPT = SECTIONS.map((s) => s.keep);

// A row of A to quote in the text: the first one whose rounded percentages add up to exactly 100.
function exampleRow(A, M) {
  for (let h = 0; h < H; h++)
    for (let i = 3; i < T; i++)
      if (A[h][i].reduce((s, w) => s + Math.round(w * 100), 0) === 100) return { h, i };
  return { h: 0, i: 3 };
}
const rowText = (A, M, h, i) =>
  A[h][i]
    .map((w, j) => (M[i][j] === 0 ? `${pct(w)} on ${tok(j)}` : null))
    .filter(Boolean)
    .join(', ');
const argmax = (row) => row.indexOf(Math.max(...row));

const lnBody = (which, inName, gain, extra) => `
  <p>${which} Every row of ${inName} is normalised <b>on its own</b>:</p>
  <ul>
    <li><b>μ</b> — the mean of the row's ${D} numbers. One per token: shape (T, 1).</li>
    <li><b>σ</b> — how spread out the row is around that mean. Also (T, 1).</li>
    <li>Subtract μ and divide by σ. Every row now has mean 0 and variance 1.</li>
    <li>Multiply each column by its gain <b>${gain}</b>, a vector of ${D} numbers shared by all tokens.</li>
  </ul>
  <p><b>Broadcasting:</b> μ and σ have one number per row and are stretched across the ${D} columns;
  ${gain} has one number per column and is stretched down the ${T} rows.</p>
  ${extra}`;

const sizeTable = () => `
  <table class="sizes">
    <thead><tr><th></th><th>this toy</th><th>GPT-2 small</th></tr></thead>
    <tbody>
      <tr><td>V (vocabulary)</td><td>${V}</td><td>50,257</td></tr>
      <tr><td>T<sub>max</sub> (context)</td><td>${TMAX}</td><td>1024</td></tr>
      <tr><td>d<sub>model</sub></td><td>${D}</td><td>768</td></tr>
      <tr><td>h (heads)</td><td>${H}</td><td>12</td></tr>
      <tr><td>d<sub>k</sub> = d<sub>model</sub> / h</td><td>${DK}</td><td>64</td></tr>
      <tr><td>MLP width</td><td>${DFF}</td><td>3072</td></tr>
      <tr><td>blocks</td><td>${NB}</td><td>12</td></tr>
      <tr><td>weights</td><td>${PARAMS_TOY.toLocaleString('en-US')}</td><td>≈ 124 million</td></tr>
    </tbody>
  </table>`;

export const STEPS = [
  // ------------------------------------------------------------------ map
  {
    chapter: 'Map',
    title: 'The whole model on one line',
    formula: 'ids → embedding → [ attention → MLP ] × ' + NB + ' → output',
    shapes: [
      ['token ids', '(T,)', `(${T},)`],
      [`X_{0} … X_{${LAST}}`, '(T, d_{model})', TD, 'the residual stream, the same shape all the way'],
      ['p', '(T, V)', `(${T}, ${V})`, 'next-token probabilities'],
    ],
    body: () => `
      <p>This is a complete GPT-style transformer with ${NB} blocks, folded up. Read it left to right.</p>
      <ul>
        <li><b>The sheets of cubes</b> are the <b>residual stream</b>: one row per token, d<sub>model</sub> = ${D}
        numbers per row. It is written X<sub>0</sub> after the embedding and gets a new number after every
        sub-layer, up to X<sub>${LAST}</sub>.</li>
        <li><b>Each box</b> is a sub-layer. It reads the stream, computes something, and <b>adds</b> its result back.
        The orange arcs are the stream being carried around each box unchanged.</li>
        <li><b>A block</b> is one attention sub-layer followed by one MLP sub-layer. The two blocks have identical
        structure and different weights.</li>
      </ul>
      <p>Stepping forward opens one box at a time into every number it contains, while the rest stay folded.
      You can also <b>click any box</b> to open it.</p>
      <h3>These numbers are random</h3>
      <p>This model is <b>untrained</b>. Every table and weight matrix holds random numbers, so nothing it computes
      means anything. A trained model holds different numbers in the same places and does exactly the same
      arithmetic — and the arithmetic is what this page shows.</p>`,
    try: 'Press → to start with the embedding, or click a box. Drag to orbit, scroll to zoom.',
    view: { ids: KEPT, all: true, az: 10, el: 12 },
  },

  // ------------------------------------------------------------------ embedding
  {
    chapter: 'Embed',
    title: 'Tokens become rows of a table',
    formula: 'E = W<sub>E</sub>[ids]',
    shapes: [
      ['ids', '(T,)', `(${T},)`, 'integers, not shown as cubes'],
      ['W_{E}', '(V, d_{model})', `(${V}, ${D})`, 'embedding table'],
      ['E', '(T, d_{model})', TD],
    ],
    body: () => `
      <p>The model never sees text. A tokenizer has already turned the sentence into <b>token ids</b>, integers
      that index a fixed vocabulary: ${TOKENS.map((t, i) => `${tok(i)} → ${IDS[i]}`).join(', ')}.</p>
      <p>Each id selects one row of the embedding table W<sub>E</sub>. This is a <b>lookup, not arithmetic</b>:
      the six rows are copied out, in sentence order.</p>
      <ul>
        <li><b>V = ${V}</b> — vocabulary size, one table row per word. GPT-2 has 50,257 sub-word pieces.</li>
        <li><b>T = ${T}</b> — sequence length. (Toy tokenizer: one word = one token.)</li>
        <li><b>d<sub>model</sub> = ${D}</b> — how many numbers describe each token. GPT-2 small uses 768.</li>
        <li>Real code carries a leading batch axis, <code>(B, T, d_model)</code>. Every step treats each sequence
        in the batch independently, so B is left out.</li>
      </ul>`,
    code: `tok_emb = W_E[idx]   # idx: (B, T) integers  ->  (B, T, d_model)`,
    try: 'Hover a cube in E: the one cube it was copied from lights up in the table. Click to pin, then drag to look around.',
    view: { ids: ['WE', 'E'], az: 14, el: 8 },
  },
  {
    chapter: 'Embed',
    title: 'Add position',
    formula: 'X<sub>0</sub> = E + P',
    shapes: [
      ['W_{P}', '(T_{max}, d_{model})', `(${TMAX}, ${D})`, 'position table'],
      ['P', '(T, d_{model})', TD, 'its first T rows'],
      ['X_{0}', '(T, d_{model})', TD],
    ],
    body: () => `
      <p>Nothing in E says where a token sits: a word gets the same row wherever it appears. So a second table,
      W<sub>P</sub>, holds one row per <b>position</b>. Its first T rows are taken and <b>added</b> to E.</p>
      <p>Added, not placed side by side: token and position share all ${D} dimensions, and the shape stays
      (T, d<sub>model</sub>).</p>
      <ul>
        <li><b>T<sub>max</sub> = ${TMAX}</b> — the number of rows in W<sub>P</sub>, and so the longest input the
        model can take. GPT-2: 1024. The dimmed rows are unused by this ${T}-token sentence.</li>
      </ul>
      <p>X<sub>0</sub> is the start of the <b>residual stream</b>: every later step reads from it and adds its
      result back into it. No column has a name; a single number means nothing by itself.</p>
      <p class="aside">GPT-2 learns this table. The original Transformer used fixed sine and cosine patterns instead,
      and many recent models leave position out here and inject it inside attention.</p>`,
    code: `pos_emb = W_P[:T]          # (T, d_model)
x = tok_emb + pos_emb      # (B, T, d_model)`,
    try: 'Hover a cube in X<sub>0</sub> to see the two numbers that were added.',
    view: { ids: ['WE', 'WP', 'E', 'P', 'X0'], az: 12, el: 8 },
  },

  // ------------------------------------------------------------------ block 1: attention
  {
    chapter: 'Block 1 · Attention',
    title: 'LayerNorm before attention',
    formula: 'N<sub>1</sub> = (X<sub>0</sub> − μ) / σ × γ<sub>1</sub>',
    shapes: [
      ['X_{0}', '(T, d_{model})', TD],
      ['μ, σ', '(T, 1)', `(${T}, 1)`, 'one per row'],
      ['γ_{1}', '(d_{model},)', `(${D},)`, 'one per column · learned'],
      ['N_{1}', '(T, d_{model})', TD],
    ],
    body: () =>
      lnBody('The first block starts here.', 'X<sub>0</sub>', 'γ<sub>1</sub>', `
      <p>This keeps the scale of what attention reads steady, however large the numbers in the stream have grown.
      Tokens do not interact: each row uses only its own numbers.</p>
      <p><b>The stream itself is not changed.</b> N<sub>1</sub> is a normalised copy handed to attention;
      X<sub>0</sub> carries on untouched (you will see it again at the residual add).</p>
      <p class="aside">This is the “pre-norm” layout of GPT-2 and most later models. The 2017 Transformer normalised
      after each sub-layer instead. LayerNorm usually also adds a learned offset β per column; it is left out here.</p>`),
    code: `n1 = ln_1(x)   # (B, T, d_model)`,
    try: 'Hover a cube in N<sub>1</sub>: the four numbers it is built from light up.',
    view: { ids: ['X0', ...B1('muA', 'sdA', 'gA', 'NA')], az: 14, el: 10 },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Three linear projections: Q, K, V',
    formula: 'Q = N<sub>1</sub> W<sub>Q</sub> &nbsp;&nbsp; K = N<sub>1</sub> W<sub>K</sub> &nbsp;&nbsp; V = N<sub>1</sub> W<sub>V</sub>',
    shapes: [
      ['N_{1}', '(T, d_{model})', TD],
      ['W_{Q}, W_{K}, W_{V}', '(d_{model}, d_{model})', `(${D}, ${D})`, 'weights'],
      ['Q, K, V', '(T, d_{model})', TD, 'result ×3'],
    ],
    body: () => `
      <p>The same input is multiplied by three different weight matrices, giving every token three
      new vectors. The names describe how each one is used later:</p>
      <ul>
        <li><b>Query</b> — compared against every token's key.</li>
        <li><b>Key</b> — what queries are compared against.</li>
        <li><b>Value</b> — what gets passed along once the comparison has set the weights.</li>
      </ul>
      <p>Matrix multiplication needs the inner sizes to agree:
      <code>(${T}, <u>${D}</u>) @ (<u>${D}</u>, ${D}) → (${T}, ${D})</code>. Each row is multiplied by the
      <i>same</i> matrix, so tokens still know nothing about each other.</p>
      <p>The flat gold-framed tiles are <b>parameters</b> (fixed after training); cubes are <b>activations</b>
      (recomputed for every input). Each W holds d<sub>model</sub>² = ${D * D} weights here, 589,824 in GPT-2 small.</p>
      <p class="aside">The brackets under each W show which output columns belong to which head. The original paper
      writes a separate W<sub>i</sub><sup>Q</sup> ∈ ℝ<sup>d<sub>model</sub> × d<sub>k</sub></sup> per head; implementations
      store them side by side as one d<sub>model</sub> × d<sub>model</sub> matrix.</p>`,
    code: `q = n1 @ W_Q    # (B, T, d_model)
k = n1 @ W_K    # (B, T, d_model)
v = n1 @ W_V    # (B, T, d_model)`,
    try: 'Hover a cube in Q: it is one row of N<sub>1</sub> dotted with one column of W<sub>Q</sub>. The table shows all 12 products.',
    view: { ids: B1('NA', 'WK', 'WQ', 'WV', 'K', 'Q', 'V'), az: 12, el: 8 },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Split into heads',
    formula: '(T, d<sub>model</sub>) → (T, h, d<sub>k</sub>) → (h, T, d<sub>k</sub>)',
    shapes: [
      ['Q', '(T, d_{model})', TD],
      ['reshape', '(T, h, d_{k})', `(${T}, ${H}, ${DK})`],
      ['transpose', '(h, T, d_{k})', HTD, 'same for K and V'],
    ],
    body: () => `
      <p><b>No arithmetic happens here.</b> Each token's ${D} numbers are cut into h = ${H} chunks of
      d<sub>k</sub> = d<sub>model</sub> / h = ${DK}. Columns 0–3 go to head 0, 4–7 to head 1, 8–11 to head 2.
      Then the head axis is moved to the front, so each head owns its own (T, d<sub>k</sub>) matrix.</p>
      <p>That new axis is the <b>depth</b> of this model: one glass pane per head. Until the heads are concatenated
      again, every head runs the identical recipe on its own pane, in parallel and without talking to the others —
      to the code, “head” is just another batch axis.</p>
      <p class="aside">The slabs are drawn far apart so you can see between them. <b>Focus head</b>
      (top left) moves the camera to one head and shrinks the others to dots; <b>All</b> brings them back.</p>`,
    code: `q = q.view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
k = k.view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)
v = v.view(B, T, h, d_k).transpose(1, 2)   # (B, h, T, d_k)`,
    try: 'Hover a cube in a split slab: the outlined cube in flat Q is the very same number.',
    view: { ids: B1('Q', 'K', 'Qh', 'Kh'), az: 32, el: 30, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Every query meets every key',
    formula: 'S = Q K<sup>T</sup> &nbsp;&nbsp;→&nbsp;&nbsp; S[h, i, j] = q<sub>i</sub> · k<sub>j</sub>',
    shapes: [
      ['Q', '(h, T, d_{k})', HTD],
      ['K^{T}', '(h, d_{k}, T)', `(${H}, ${DK}, ${T})`, 'last two axes swapped'],
      ['S', '(h, T, T)', HTT, 'raw scores'],
    ],
    body: () => `
      <p>This is where the famous <b>T × T</b> shape is born. Inside each head, the query of token i is dotted
      with the key of token j, for every pair (i, j). Row = the token that is asking; column = the token being
      looked at.</p>
      <p>The layout makes the multiplication visible: each score cube sits <b>to the right of its query row</b>
      and <b>below its key column</b>. That is why K is transposed — its T axis has to run sideways to line up
      with the columns of S. In each head, read top to bottom: K, its transpose K<sup>T</sup>, then S.</p>
      <p><code>(${H}, ${T}, <u>${DK}</u>) @ (${H}, <u>${DK}</u>, ${T}) → (${H}, ${T}, ${T})</code>. The d<sub>k</sub> axis is summed
      away; a second T axis appears. This is the O(T²) cost of attention.</p>
      <p>A large positive score means the query and key vectors point the same way; a large negative one means
      they point in opposite directions.</p>
      <p class="aside">This step opens on one head; the dotted slabs behind it are the other heads doing exactly
      the same thing with their own four columns of Q and K.</p>`,
    code: `att = q @ k.transpose(-2, -1)   # (B, h, T, T)`,
    try: 'Hover a score cube: its query row and key column light up, with the four products that add up to it.',
    focus: 0,
    view: { ids: B1('Qh', 'Kh', 'KT', 'S'), az: 30, el: 32, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Scale by 1/√d_k',
    formula: 'S / √d<sub>k</sub>',
    shapes: [['S / √d_{k}', '(h, T, T)', HTT, 'shape unchanged']],
    body: () => `
      <p>Every score is divided by √d<sub>k</sub> = √${DK} = ${fmt(Math.sqrt(DK), 0)}.</p>
      <p><b>Why?</b> A dot product adds up d<sub>k</sub> terms. If each term has variance around 1, the sum has
      variance around d<sub>k</sub>, so scores grow like √d<sub>k</sub> as heads get wider. Large scores push the
      softmax (two steps ahead) towards a hard one-hot choice, where gradients vanish. Dividing by √d<sub>k</sub>
      keeps the spread of scores the same no matter how wide the head is.</p>
      <p class="aside">Both score matrices share one colour scale, so you can see every cube fade to half strength.</p>`,
    code: `att = att / math.sqrt(d_k)   # (B, h, T, T)`,
    view: { ids: B1('S', 'Ss'), az: 30, el: 32, zoom: 1.1, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Causal mask',
    formula: 'S / √d<sub>k</sub> + M, &nbsp; M[i, j] = 0 if j ≤ i, else −∞',
    shapes: [
      ['M', '(T, T)', `(${T}, ${T})`, 'broadcast over h'],
      ['masked scores', '(h, T, T)', HTT],
    ],
    body: (m) => `
      <p>A GPT-style model predicts the next token at <i>every</i> position at once during training.
      For that to be honest, token i must not see tokens after it. So the upper triangle (j &gt; i) is
      set to −∞ — the small dark-red cubes.</p>
      <p><b>Broadcasting:</b> M has shape (T, T) with no head axis. The same mask is added to each of the
      ${H} head slabs; it is drawn as a single plate in front of the slabs, pushed back through all of them.</p>
      <p>Encoders such as BERT skip this step and let every token see the whole sequence. Use the
      <b>Causal mask</b> switch (top left) to compare.
      ${m.causal ? '' : '<b>The mask is currently off, so M is all zeros.</b>'}</p>`,
    code: `att = att.masked_fill(mask[:T, :T] == 0, float('-inf'))`,
    view: { ids: B1('Ss', 'M', 'Sm'), az: 30, el: 32, zoom: 1.2, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Softmax: scores become weights',
    formula: 'A = softmax(S / √d<sub>k</sub> + M) &nbsp; along the last axis',
    shapes: [['A', '(h, T, T)', HTT, 'each row sums to 1']],
    body: (m) => {
      const A = m.b[0].A;
      const { h, i } = exampleRow(A, m.M);
      return `
      <p>Each <b>row</b> is exponentiated and divided by its row total. Every row of A is now a set of weights
      that sum to 1: how token i splits its attention over the tokens it may look at. e<sup>−∞</sup> = 0,
      so masked positions get exactly zero.</p>
      <p>Softmax runs over the last axis (keys), so rows sum to 1 — columns do not have to.</p>
      <h3>Reading a row</h3>
      <p>Take row ${tok(i)} in head ${h}. Its weights are ${rowText(A, m.M, h, i)}. In the next step, that is the
      mix of value vectors ${tok(i)} receives from this head.</p>
      <p>The three heads give three different tables from the same input, only because their columns of
      W<sub>Q</sub> and W<sub>K</sub> differ.</p>
      <h3>Do the patterns mean anything?</h3>
      <p>Here, no: the weights are random, so where each head looks is arbitrary. In a trained model,
      recognisable patterns appear — a head that puts nearly all its weight on the previous token, say, or one
      that links a pronoun to the noun it refers to. Nobody assigns those jobs. They come out of training and are
      named afterwards by people studying the model, and many heads have no tidy description at all.</p>
      ${m.causal ? `<p class="aside">Row ${tok(0)} can only see itself, so its weight is 1.00 in every head.</p>` : ''}`;
    },
    code: `att = F.softmax(att, dim=-1)   # (B, h, T, T)`,
    try: 'Hover a cube in A to see the whole row go through the softmax, number by number.',
    view: { ids: B1('Sm', 'A'), az: 30, el: 32, zoom: 1.1, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Mix the values',
    formula: 'Z = A V &nbsp;&nbsp;→&nbsp;&nbsp; z<sub>i</sub> = Σ<sub>j</sub> A[i, j] · v<sub>j</sub>',
    shapes: [
      ['A', '(h, T, T)', HTT],
      ['V', '(h, T, d_{k})', HTD],
      ['Z', '(h, T, d_{k})', HTD],
    ],
    body: () => `
      <p>Each token's new vector is a <b>weighted average of all the value vectors</b>, using its row of A
      as the weights. Attention is the only kind of step in the whole model where information moves from one token
      to another.</p>
      <p><code>(${H}, ${T}, <u>${T}</u>) @ (${H}, <u>${T}</u>, ${DK}) → (${H}, ${T}, ${DK})</code>. The T × T table is used up:
      we are back to one row per token, d<sub>k</sub> numbers wide, per head.</p>
      <p>Notice the division of labour: Q and K decided <i>how much</i> each token counts; V is <i>what</i> gets carried over.
      V sits under Z so that each column of V lines up with the column of Z it feeds.</p>`,
    code: `z = att @ v   # (B, h, T, d_k)`,
    try: 'Hover a cube in Z: one row of A times one column of V.',
    focus: 0,
    view: { ids: B1('A', 'Z', 'Vh'), az: 30, el: 32, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Concatenate the heads',
    formula: '(h, T, d<sub>k</sub>) → (T, h, d<sub>k</sub>) → (T, h · d<sub>k</sub>)',
    shapes: [
      ['Z', '(h, T, d_{k})', HTD],
      ['transpose', '(T, h, d_{k})', `(${T}, ${H}, ${DK})`],
      ['reshape', '(T, h·d_{k}) = (T, d_{model})', TD],
    ],
    body: () => `
      <p>The exact inverse of the split, and again <b>no arithmetic</b>: the head slabs slide out of depth and sit
      side by side, so each token has one row of h · d<sub>k</sub> = ${H} · ${DK} = ${D} = d<sub>model</sub> numbers again.</p>
      <p>This is why d<sub>k</sub> is chosen as d<sub>model</sub> / h: the heads' outputs tile the model width exactly.</p>`,
    code: `z = z.transpose(1, 2).contiguous().view(B, T, d_model)   # (B, T, d_model)`,
    view: { ids: B1('Z', 'Zc'), az: 30, el: 32, zoom: 1.05, perHead: true },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Output projection',
    formula: 'O = Z W<sub>O</sub>',
    shapes: [
      ['Z', '(T, d_{model})', TD],
      ['W_{O}', '(d_{model}, d_{model})', `(${D}, ${D})`, 'weights'],
      ['O', '(T, d_{model})', TD],
    ],
    body: () => `
      <p>Until now the heads have worked in separate d<sub>k</sub>-wide compartments. W<sub>O</sub> is one more
      weight matrix that maps the concatenated head outputs back into the ${D} dimensions of the residual stream.</p>
      <p>Rows of W<sub>O</sub> are grouped by head (brackets on its left); columns are residual-stream dimensions.
      Every head dimension can write into every column, so the heads' contributions are <b>summed in the same
      space</b>. No head has dimensions of its own in the output.</p>
      <p class="aside">W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub> are all the weights attention has:
      4 · d<sub>model</sub>² = ${4 * D * D} here, about 2.36 million per block in GPT-2 small.</p>`,
    code: `o = z @ W_O   # (B, T, d_model)`,
    view: { ids: B1('Zc', 'WO', 'O'), az: 12, el: 8 },
  },
  {
    chapter: 'Block 1 · Attention',
    title: 'Add back to the residual stream',
    formula: 'X<sub>1</sub> = X<sub>0</sub> + O',
    shapes: [
      ['X_{0}', '(T, d_{model})', TD, 'from before LayerNorm'],
      ['O', '(T, d_{model})', TD],
      ['X_{1}', '(T, d_{model})', TD],
    ],
    body: (m) => {
      const b = m.b[0];
      return `
      <p>Attention does not replace the token vectors — it <b>adds to them</b>. The orange line is the residual
      (skip) connection. Follow it back: it starts at X<sub>0</sub>, <i>before</i> LayerNorm, so the un-normalised
      stream is what gets carried forward.</p>
      <h3>Reading the result</h3>
      <p>Every number has moved. For ${tok(3)}, dimension 0 went from ${fmt(m.X0[3][0])} to ${fmt(b.Xa[3][0])}
      (${fmt(m.X0[3][0])} + ${par(b.O[3][0])}). Token identity, position and now attention's output all live in the
      same ${D} dimensions, added on top of one another; nothing is kept in a separate compartment.</p>
      <p>With random weights the change is arbitrary. In a trained model, this addition is how a token's vector
      picks up information from the tokens it attended to.</p>`;
    },
    code: `x = x + o   # (B, T, d_model)`,
    try: 'Hover a cube in X<sub>1</sub> to see the before + written = after sum.',
    view: { ids: B1('O', 'Xa'), az: 14, el: 10 },
  },

  // ------------------------------------------------------------------ block 1: MLP
  {
    chapter: 'Block 1 · MLP',
    title: 'LayerNorm before the MLP',
    formula: 'N<sub>2</sub> = (X<sub>1</sub> − μ) / σ × γ<sub>2</sub>',
    shapes: [
      ['X_{1}', '(T, d_{model})', TD],
      ['μ, σ', '(T, 1)', `(${T}, 1)`, 'one per row'],
      ['γ_{2}', '(d_{model},)', `(${D},)`, 'its own learned gain'],
      ['N_{2}', '(T, d_{model})', TD],
    ],
    body: () =>
      lnBody('The second half of the block begins the same way as the first.', 'X<sub>1</sub>', 'γ<sub>2</sub>', `
      <p>Same operation as before, on the stream as it is now, with a separate gain vector. Again the result is a
      normalised copy: X<sub>1</sub> itself carries on to the next residual add.</p>`),
    code: `n2 = ln_2(x)   # (B, T, d_model)`,
    view: { ids: B1('Xa', 'muM', 'sdM', 'gM', 'NM'), az: 14, el: 10 },
  },
  {
    chapter: 'Block 1 · MLP',
    title: 'MLP: expand to four times the width',
    formula: 'H = N<sub>2</sub> W<sub>1</sub>',
    shapes: [
      ['N_{2}', '(T, d_{model})', TD],
      ['W_{1}', '(d_{model}, 4·d_{model})', `(${D}, ${DFF})`, 'weights'],
      ['H', '(T, 4·d_{model})', `(${T}, ${DFF})`],
    ],
    body: () => `
      <p>The MLP (also called the feed-forward layer) works on <b>each token separately</b>. Unlike attention,
      no row ever looks at another row here.</p>
      <p>First every token's ${D} numbers are turned into ${DFF} “hidden” numbers. Each hidden number is the token's
      row dotted with one column of W<sub>1</sub>:
      <code>(${T}, <u>${D}</u>) @ (<u>${D}</u>, ${DFF}) → (${T}, ${DFF})</code>.</p>
      <p>The factor 4 is a convention from the original paper (512 → 2048); GPT-2 small goes 768 → 3072.</p>
      <p class="aside">W<sub>1</sub> alone holds ${D * DFF} weights, as many as all four attention matrices together.</p>`,
    code: `h = n2 @ W_1   # (B, T, 4 * d_model)`,
    try: 'Hover a cube in H: one row of N<sub>2</sub> dotted with one column of W<sub>1</sub>.',
    view: { ids: B1('NM', 'W1', 'H'), az: 8, el: 8 },
  },
  {
    chapter: 'Block 1 · MLP',
    title: 'GELU: bend every number',
    formula: 'G = GELU(H), &nbsp; GELU(h) = h · Φ(h)',
    shapes: [['G', '(T, 4·d_{model})', `(${T}, ${DFF})`, 'shape unchanged']],
    body: (m) => {
      const neg = m.b[0].H.flat().filter((x) => x < 0).length;
      return `
      <p>Each of the ${T * DFF} numbers is passed through the same curve, <b>one at a time</b>. No cube depends on any
      other; G sits directly under H so every cube lands below the one it came from.</p>
      <p>Φ(h) is the fraction of a standard bell curve lying below h. So a large positive number passes through almost
      unchanged, a large negative one is squashed to about 0, and numbers near 0 are bent smoothly between the two.
      Here ${neg} of the ${T * DFF} numbers in H are negative: watch the blue cubes fade.</p>
      <p><b>Why it is needed:</b> without a non-linear step, multiplying by W<sub>1</sub> and then W<sub>2</sub> would
      be the same as multiplying by one matrix (their product), and the extra width would buy nothing.</p>
      <p class="aside">GELU is what GPT-2 and BERT use. The original Transformer used ReLU, which sets negatives to exactly 0.</p>`;
    },
    code: `g = F.gelu(h)   # (B, T, 4 * d_model)`,
    try: 'Hover a cube in G for the exact h × Φ(h).',
    view: { ids: B1('H', 'G'), az: 8, el: 10 },
  },
  {
    chapter: 'Block 1 · MLP',
    title: 'MLP: project back down',
    formula: 'F = G W<sub>2</sub>',
    shapes: [
      ['G', '(T, 4·d_{model})', `(${T}, ${DFF})`],
      ['W_{2}', '(4·d_{model}, d_{model})', `(${DFF}, ${D})`, 'weights'],
      ['F', '(T, d_{model})', TD],
    ],
    body: () => `
      <p>The ${DFF} hidden numbers of each token are combined back into ${D}, ready to be written into the stream:
      <code>(${T}, <u>${DFF}</u>) @ (<u>${DFF}</u>, ${D}) → (${T}, ${D})</code>.</p>
      <p>W<sub>2</sub> is drawn tall because it has one <b>row per hidden unit</b>. A useful way to read it: each
      hidden unit owns one row of W<sub>2</sub>, and adds that row to the output, scaled by how active the unit is.</p>
      <h3>Where the weights are</h3>
      <p>W<sub>1</sub> and W<sub>2</sub> hold 2 · ${D} · ${DFF} = ${2 * D * DFF} weights; attention holds ${4 * D * D}.
      Two thirds of a block's weights sit in the MLP, and that ratio is the same at full size
      (about 4.7 million against 2.4 million in GPT-2 small).</p>`,
    code: `f = g @ W_2   # (B, T, d_model)`,
    try: `Hover a cube in F: ${DFF} products are added; the table lists the largest.`,
    view: { ids: B1('G', 'W2', 'F'), az: 10, el: 6 },
  },
  {
    chapter: 'Block 1 · MLP',
    title: 'Add back to the stream: block 1 is done',
    formula: 'X<sub>2</sub> = X<sub>1</sub> + F',
    shapes: [
      ['X_{1}', '(T, d_{model})', TD, 'from before LayerNorm'],
      ['F', '(T, d_{model})', TD],
      ['X_{2}', '(T, d_{model})', TD, 'same shape as X₀'],
    ],
    body: (m) => {
      const b = m.b[0];
      return `
      <p>The MLP's output is added to the stream through a second residual connection, again starting from
      before the LayerNorm. For ${tok(3)}, dimension 0: ${fmt(b.Xa[3][0])} + ${par(b.F[3][0])} = ${fmt(b.Xm[3][0])}.</p>
      <p>That completes one <b>transformer block</b>: normalise, attend, add; normalise, MLP, add.</p>
      <p><b>The output has exactly the shape of the input</b>, (T, d<sub>model</sub>). That is what lets blocks be
      stacked: X<sub>2</sub> is now handed to block 2 exactly as X<sub>0</sub> was handed to block 1.</p>`;
    },
    code: `x = x + f   # (B, T, d_model)`,
    view: { ids: B1('F', 'Xm'), az: 14, el: 10 },
  },

  // ------------------------------------------------------------------ block 2
  {
    chapter: 'Block 2',
    title: 'Block 2: attention again, with its own weights',
    formula: 'X<sub>3</sub> = X<sub>2</sub> + Attention(LayerNorm(X<sub>2</sub>))',
    shapes: [
      ['X_{2}', '(T, d_{model})', TD, 'what block 1 left in the stream'],
      ['inside', '(h, T, d_{k}) and (h, T, T)', `${HTD}, ${HTT}`, 'identical to block 1'],
      ['X_{3}', '(T, d_{model})', TD],
    ],
    body: (m) => {
      const [A1, A2] = [m.b[0].A, m.b[1].A];
      const { h, i } = exampleRow(A2, m.M);
      return `
      <p>Everything from the last eleven steps happens again, in the same order and with the same shapes:
      LayerNorm, three projections, split into heads, scores, scale, mask, softmax, mix the values, concatenate,
      project, add. The whole sub-layer is shown at once this time.</p>
      <p>Only two things differ from block 1:</p>
      <ul>
        <li><b>Its input is X<sub>2</sub></b>, which already contains what block 1's attention and MLP added.</li>
        <li><b>Its weights are its own</b>: γ<sub>3</sub>, W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub>
        here are different numbers from block 1's.</li>
      </ul>
      <p>So the attention tables differ. Row ${tok(i)}, head ${h}: in block 1 it was ${rowText(A1, m.M, h, i)};
      here it is ${rowText(A2, m.M, h, i)}.</p>
      <h3>Why stack?</h3>
      <p>A head in block 2 reads a stream that earlier heads have already written into, so it can react to what
      they found. In trained models this is how multi-step behaviour is built: later layers work on the results of
      earlier ones.</p>`;
    },
    code: `x = x + attn_2(ln_3(x))   # (B, T, d_model)`,
    try: 'Everything here can be hovered, just as in block 1. Use Focus head to isolate one head.',
    view: { ids: idsOfSection('att1'), az: 16, el: 14, perHead: false },
  },
  {
    chapter: 'Block 2',
    title: 'Block 2: MLP again, with its own weights',
    formula: `X<sub>4</sub> = X<sub>3</sub> + MLP(LayerNorm(X<sub>3</sub>))`,
    shapes: [
      ['X_{3}', '(T, d_{model})', TD],
      ['inside', '(T, 4·d_{model})', `(${T}, ${DFF})`, 'identical to block 1'],
      ['X_{4}', '(T, d_{model})', TD, 'the stream after the last block'],
    ],
    body: (m) => `
      <p>The same four operations as in block 1 — LayerNorm, widen with W<sub>1</sub>, GELU, narrow with
      W<sub>2</sub> — then the residual add. Again the shapes are identical and the weights
      (γ<sub>4</sub>, W<sub>1</sub>, W<sub>2</sub>) are block 2's own.</p>
      <p>For ${tok(3)}, dimension 0 has now been through four additions:
      ${fmt(m.X0[3][0])} → ${fmt(m.b[0].Xa[3][0])} → ${fmt(m.b[0].Xm[3][0])} → ${fmt(m.b[1].Xa[3][0])} → ${fmt(m.b[1].Xm[3][0])}.</p>
      <p>X<sub>4</sub> is the end of the stack. GPT-2 small repeats this block 12 times and the largest GPT-2 48 times;
      nothing changes except the number of repetitions and the size of each matrix.</p>`,
    code: `x = x + mlp_2(ln_4(x))   # (B, T, d_model)`,
    view: { ids: idsOfSection('mlp1'), az: 10, el: 8 },
  },

  // ------------------------------------------------------------------ output
  {
    chapter: 'Output',
    title: 'Final LayerNorm',
    formula: `N<sub>f</sub> = (X<sub>${LAST}</sub> − μ) / σ × γ<sub>f</sub>`,
    shapes: [
      [`X_{${LAST}}`, '(T, d_{model})', TD],
      ['μ, σ', '(T, 1)', `(${T}, 1)`, 'one per row'],
      ['γ_{f}', '(d_{model},)', `(${D},)`, 'its own learned gain'],
      ['N_{f}', '(T, d_{model})', TD],
    ],
    body: () =>
      lnBody('After the last block, the stream is normalised once more.', `X<sub>${LAST}</sub>`, 'γ<sub>f</sub>', `
      <p>Nothing is added back this time. N<sub>f</sub> is what the output layer reads.</p>`),
    code: `nf = ln_f(x)   # (B, T, d_model)`,
    view: { ids: [`b${NB - 1}.Xm`, 'muF', 'sdF', 'gF', 'NF'], az: 14, el: 10 },
  },
  {
    chapter: 'Output',
    title: 'Unembedding: a score for every word',
    formula: 'logits = N<sub>f</sub> W<sub>E</sub><sup>T</sup>',
    shapes: [
      ['N_{f}', '(T, d_{model})', TD],
      ['W_{E}^{T}', '(d_{model}, V)', `(${D}, ${V})`, 'the embedding table, transposed'],
      ['logits', '(T, V)', `(${T}, ${V})`],
    ],
    body: () => `
      <p>Each position's final vector is dotted with the embedding of <b>every word in the vocabulary</b>, giving
      V = ${V} scores per position: <code>(${T}, <u>${D}</u>) @ (<u>${D}</u>, ${V}) → (${T}, ${V})</code>.
      A word scores high when the final vector points the same way as that word's embedding.</p>
      <p>The d<sub>model</sub> axis disappears and a vocabulary axis takes its place: the model is back in the world
      of words.</p>
      <h3>Tied weights</h3>
      <p>The matrix used here is the embedding table from the start, turned on its side — the cubes fly in from
      the embedding. The same ${V * D} numbers turn ids into vectors at the start and vectors into word scores at the end.</p>
      <p class="aside">GPT-2 ties these weights. Some models keep a separate output matrix of the same shape.</p>`,
    code: `logits = nf @ W_E.T   # (B, T, V)`,
    try: 'Hover a logit: the row of N<sub>f</sub> and the word’s embedding column light up.',
    view: { ids: ['NF', 'WET', 'L'], az: 12, el: 10 },
  },
  {
    chapter: 'Output',
    title: 'Softmax: next-token probabilities',
    formula: 'p = softmax(logits) &nbsp; along the last axis',
    shapes: [['p', '(T, V)', `(${T}, ${V})`, 'each row sums to 1']],
    body: (m) => {
      const last = T - 1;
      const best = argmax(m.Pr[last]);
      return `
      <p>The same softmax as in attention, now over the vocabulary axis: each row becomes ${V} probabilities that sum to 1.</p>
      <p><b>Row i is the model's guess for the token that comes after position i.</b> There is one guess per
      position, not one for the whole sentence.</p>
      <ul>
        <li><b>Generating text</b> uses only the last row. Here, after ${tok(last)}, the most likely next token is
        ${word(best)} at ${pct(m.Pr[last][best])}. One token is picked from that row, appended to the input, and the
        whole model runs again.</li>
        <li><b>Training</b> uses all T rows at once: row i is compared with the token that really is at position i + 1.
        The causal mask is what makes that fair, because row i was computed without seeing later tokens.
        ${m.causal ? '' : '<b>The mask is off right now, so every row has seen the whole sentence.</b>'}</li>
      </ul>
      <p>With random weights these probabilities are arbitrary; a trained model would put its weight on words that
      actually tend to follow.</p>`;
    },
    code: `probs = F.softmax(logits, dim=-1)                  # (B, T, V)
next_id = torch.multinomial(probs[:, -1], 1)       # sample from the last row`,
    try: 'Hover a cube in p to see its whole row go through the softmax.',
    view: { ids: ['L', 'Pr'], az: 12, el: 10 },
  },
  {
    chapter: 'All',
    title: 'The whole model, again',
    formula: `X<sub>${LAST}</sub> = X<sub>0</sub> + (${LAST} additions) &nbsp;→&nbsp; p = softmax(LayerNorm(X<sub>${LAST}</sub>) W<sub>E</sub><sup>T</sup>)`,
    shapes: [
      ['ids', '(T,)', `(${T},)`],
      ['residual stream', '(T, d_{model})', TD, `X₀ … X${'₀₁₂₃₄₅₆₇₈₉'[LAST]} and every LayerNorm output`],
      ['inside attention', '(h, T, d_{k}) and (h, T, T)', `${HTD}, ${HTT}`],
      ['inside the MLP', '(T, 4·d_{model})', `(${T}, ${DFF})`],
      ['logits, p', '(T, V)', `(${T}, ${V})`],
    ],
    body: () => `
      <p>Folded up again, every box now stands for something you have seen in full. The stream keeps the shape
      (T, d<sub>model</sub>) from X<sub>0</sub> to X<sub>${LAST}</sub>; each sub-layer branches off, computes in a
      different shape, and adds a (T, d<sub>model</sub>) result back.</p>
      <p>Click any box to reopen it.</p>
      <h3>The same thing at real size</h3>
      ${sizeTable()}
      <p class="aside">Left out of this model: the batch axis B, biases (including LayerNorm's offset β) and dropout.
      Everything is random, scaled so the cubes show contrast; a freshly initialised real model uses much smaller
      weights, which would make every attention row nearly uniform.</p>`,
    try: 'Click a box to open it. Drag to orbit, scroll to zoom, right-drag to pan.',
    view: { ids: KEPT, all: true, az: 10, el: 12 },
  },
];
