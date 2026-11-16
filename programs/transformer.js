// A GPT-2-style transformer with two blocks, written in the tensor language.
//
//   tokens -> embedding lookup + position
//          -> 2 x { [ LayerNorm -> attention -> add ] -> [ LayerNorm -> MLP -> add ] }
//          -> LayerNorm -> unembedding -> softmax
//
// Row-vector convention: a token is a row, and a linear map is X @ W with W of
// shape (in, out).
//
// The operations decide every number, shape, position and arrow. The prose
// (say, aside, hint, .says) is the author's: it explains what each step is for.

import { program, axis, input, param, random, toggle, departure, section, step, repeat, say, aside, hint } from '../lang/core.js';
import { lookup, rows, transpose, split, merge, map, add, gelu, mean, std, softmax, matmul, causalMask } from '../lang/ops.js';
import { row, col, slab, value, cell, pct, par, label, live } from '../lang/prose.js';

export const VOCAB = ['.', 'The', 'a', 'and', 'cat', 'dog', 'happy', 'it', 'mat', 'on', 'ran', 'robot', 'said', 'sat', 'tired', 'was'];
export const TOKENS = ['The', 'robot', 'said', 'it', 'was', 'tired'];
const BLOCKS = 2;
const tok = (text) => `<span class="tok">${text}</span>`;

export const build = () =>
  program('GPT-style transformer, two blocks', { seed: 11 }, () => {
    const V = axis('V', VOCAB.length, { labels: VOCAB });
    const T = axis('T', TOKENS.length, { labels: TOKENS });
    const Tmax = axis('T_max', 8, { labels: Array.from({ length: 8 }, (_, i) => `pos ${i}`) });
    const d = axis('d_model', 12);
    const h = axis('h', 3, { unit: 'head' });
    const dk = axis('d_k', 4);
    const dff = axis('d_ff', 48, { sym: '4·d_{model}' });

    const causal = toggle('causal', true, { label: 'Causal mask' });

    departure('untrained', 'Every table and weight matrix is seeded random numbers; a trained model holds different numbers in the same places.');
    departure('weight scale', 'The random weights are larger than a real initialisation (about 0.02) so the cubes show contrast.');
    departure('no biases', 'Biases are switched off, as in nanoGPT with bias=False. They add a vector after each matrix multiply and change no shapes.');
    departure('tiny sizes', 'V = 16, d_model = 12, 3 heads and 2 blocks, where GPT-2 small has 50,257, 768, 12 and 12.');

    const ids = input('ids', 'token ids', [T], TOKENS.map((t) => VOCAB.indexOf(t)));

    say`This is a complete GPT-style transformer with ${BLOCKS} blocks, folded up. Read it left to right.

      <ul>
        <li><b>The sheets of cubes</b> are the <b>residual stream</b>: one row per token, ${d} = ${d.size} numbers per row.
        It gets a new name after every sub-layer.</li>
        <li><b>Each box</b> is a sub-layer. It reads the stream, computes something, and <b>adds</b> its result back.
        The orange arcs are the stream being carried around each box unchanged.</li>
        <li><b>A block</b> is one attention sub-layer followed by one MLP sub-layer. The blocks have identical
        structure and different weights.</li>
      </ul>

      Stepping forward opens one box at a time into every number it contains, while the rest stay folded.
      You can also <b>click any box</b> to open it.`;
    hint`Press → to start with the embedding, or click a box. Drag to orbit, Shift-drag to move sideways, scroll to zoom.`;

    // LayerNorm over the last axis, written from the primitives.
    const layerNorm = (X, key, k) => {
      const mu = mean(X).as(`mu${key}`, 'μ', { note: 'mean of each row' });
      const sd = std(X, { mean: mu, eps: 1e-5 }).as(`sd${key}`, 'σ', { note: 'spread of each row' });
      const g = param(`g${key}`, `γ_{${k}}`, [d], random(0.2, { mean: 1 }));
      const N = map('(x - μ) / σ * γ', { x: X, μ: mu, σ: sd, γ: g }).as(`N${key}`, `N_{${k}}`);

      say`Every row of ${X} is normalised <b>on its own</b>:

        <ul>
          <li>${mu}: the mean of the row's ${d.size} numbers. One per token.</li>
          <li>${sd}: how spread out the row is around that mean.</li>
          <li>Subtract ${mu} and divide by ${sd}. Every row now has mean 0 and variance 1.</li>
          <li>Multiply each column by its gain ${g}, a vector of ${d.size} numbers shared by all tokens.</li>
        </ul>

        <b>Broadcasting:</b> ${mu} and ${sd} have one number per row and are stretched across the columns;
        ${g} has one number per column and is stretched down the rows.`;

      mu.says`The average of the numbers in row ${row} of ${X}.`;
      sd.says`How spread out row ${row} of ${X} is around its mean: the square root of the average squared distance
        from the mean. The small ε only guards against dividing by zero.`;
      g.says`Gain for dimension ${col}: after a row is normalised, its number in this column is multiplied by ${value}.
        The same gains are used for every token.`;
      N.says`Dimension ${col} of ${row}, normalised: subtract the row's mean, divide by the row's spread, multiply by
        the column's gain. Only numbers from this token's own row are used: tokens do not interact here.`;
      return N;
    };

    let WE, E, X;
    section('emb', 'Embedding', { summary: 'lookup + position' }, () => {
      step('Tokens become rows of a table', () => {
        WE = param('WE', 'W_{E}', [V, d], random(0.8));
        E = lookup(WE, ids).as('E', 'E');

        say`The model never sees text. A tokenizer has already turned the sentence into <b>token ids</b>, integers
          that index a fixed vocabulary: ${live(() => TOKENS.map((t, i) => `${tok(t)} → ${ids.data[i]}`).join(', '))}.

          Each id selects one row of the embedding table ${WE}. This is a <b>lookup, not arithmetic</b>:
          the rows are copied out, in sentence order.

          <ul>
            <li><b>${V} = ${V.size}</b>: vocabulary size, one table row per word.</li>
            <li><b>${T} = ${T.size}</b>: sequence length. This toy tokenizer makes one word one token.</li>
            <li><b>${d} = ${d.size}</b>: how many numbers describe each token.</li>
          </ul>`;
        aside`GPT-2 has 50,257 sub-word pieces in its vocabulary and describes each token with 768 numbers.
          Real code also carries a leading batch axis, (B, T, d_{model}); every step treats each sequence in the
          batch independently, so B is left out here.`;
        hint`Hover a cube in ${E}: the one cube it was copied from lights up in the table. Click to pin, then drag to look around.`;

        WE.says`Row ${row} of the embedding table is the vector for that word.`;
        E.says`The vector for ${row}, copied from the embedding table as it is. A lookup, not arithmetic: the thin
          outlines mark the row being copied; only the one bright cube is the source of this number.`;
      });

      step('Add position', () => {
        const WP = param('WP', 'W_{P}', [Tmax, d], random(0.5)).hint({ side: 'below' });
        const P = rows(WP, T).as('P', 'P');
        X = add(E, P).as('X0', 'X_{0}');

        say`Nothing in ${E} says where a token sits: a word gets the same row wherever it appears. So a second table,
          ${WP}, holds one row per <b>position</b>. Its first ${T} rows are taken and <b>added</b> to ${E}.

          Added, not placed side by side: token and position share all ${d.size} dimensions, and the shape stays ${X.sym}.

          <ul>
            <li><b>${Tmax} = ${Tmax.size}</b>: the number of rows in ${WP}, and so the longest input the model can take.
            The dimmed rows are unused by this sentence.</li>
          </ul>

          ${X} is the start of the <b>residual stream</b>: every later step reads from it and adds its result back
          into it. No column has a name; a single number means nothing by itself.`;
        aside`GPT-2 learns this table and allows 1024 positions. The original Transformer used fixed sine and cosine
          patterns instead, and many recent models leave position out here and inject it inside attention.`;
        hint`Hover a cube in ${X} to see the two numbers that were added.`;

        WP.says`Row ${row} of the position table: the vector for that position, whatever token sits there.`;
        P.says`The position vector for ${row}, copied from the position table as it is. It does not depend on which
          tokens the sentence holds, only on how many there are.`;
        X.says`Dimension ${col} of ${row}: its token embedding plus its position embedding. The two are added,
          so token and position share the same dimensions.`;
      });
    });

    repeat(BLOCKS, { prefix: 'b', name: 'Block', detail: 'first' }, (b) => {
      const [kA, kM] = [2 * b + 1, 2 * b + 2];

      section('att', 'Attention', { summary: 'LayerNorm → 3 heads → add' }, () => {
        const Xin = X;
        if (b > 0) {
          say`Everything from block 1's attention happens again, in the same order and with the same shapes.
            The whole sub-layer is shown at once this time.

            Only two things differ:

            <ul>
              <li><b>Its input is ${Xin}</b>, which already contains what the earlier block's attention and MLP added.</li>
              <li><b>Its weights are its own</b>: different numbers from the earlier block's.</li>
            </ul>

            <h3>Why stack?</h3>

            A head in this block reads a stream that earlier heads have already written into, so it can react to
            what they found.`;
          aside`In trained models this is how multi-step behaviour is built: later layers work on the results of earlier ones.`;
          hint`Everything here can be hovered, just as in block 1. Use Focus head to isolate one head.`;
        }

        const N = step('LayerNorm before attention', () => {
          const N = layerNorm(Xin, 'A', kA);
          say`This keeps the scale of what attention reads steady, however large the numbers in the stream have grown.

            <b>The stream itself is not changed.</b> ${N} is a normalised copy handed to attention; ${Xin} carries
            on untouched (you will see it again at the residual add).`;
          aside`This is the “pre-norm” layout of GPT-2 and most later models. The 2017 Transformer normalised after
            each sub-layer instead. LayerNorm usually also adds a learned offset β per column; it is left out here.
            A real model starts every gain at exactly 1 and training moves them.`;
          hint`Hover a cube in ${N}: the four numbers it is built from light up.`;
          return N;
        });

        const W = {};
        const [Q, K, Vv] = step('Three linear projections: Q, K, V', () => {
          const out = ['Q', 'K', 'V'].map((n) => {
            W[n] = param(`W${n}`, `W_{${n}}`, [d, d], random(n === 'V' ? 0.35 : 0.34));
            W[n].says`Weight: how much dimension ${row} of a token's vector contributes to column ${col} of ${n}.
              The outlined column is the vector every token row is dotted with.`;
            return matmul(N, W[n]).as(n, n).says`Row ${row} of ${N} dotted with column ${col} of ${W[n]}.`;
          });

          say`The same input is multiplied by three different weight matrices, giving every token three new vectors.
            The names describe how each one is used later:

            <ul>
              <li><b>Query</b>: compared against every token's key.</li>
              <li><b>Key</b>: what queries are compared against.</li>
              <li><b>Value</b>: what gets passed along once the comparison has set the weights.</li>
            </ul>

            Each row is multiplied by the <i>same</i> matrix, so tokens still know nothing about each other.

            The brackets under each matrix show which output columns belong to which ${h.unit}.`;
          aside`The original paper writes a separate matrix per head; implementations store them side by side as one
            matrix. Each of these holds 589,824 weights in GPT-2 small.`;
          hint`Hover a cube in ${out[0]}: it is one row of ${N} dotted with one column of ${W.Q}.`;
          return out;
        });

        const [Qh, Kh, Vh] = step('Split into heads', () => {
          const out = [Q, K, Vv].map((x, i) =>
            split(x, d, [h, dk]).as(`${'QKV'[i]}h`, 'QKV'[i]).says`The same number as in the flat matrix: a reshape moves
              no data. Its column is simply re-addressed as (${slab}, dimension ${col}).`);

          say`<b>No arithmetic happens here.</b> Each token's ${d.size} numbers are cut into ${h} = ${h.size} chunks of
            ${dk} = ${d} / ${h} = ${dk.size}. Then the ${h.unit} axis is moved to the front, so each ${h.unit} owns its
            own matrix.

            That new axis is the <b>depth</b> of this model: one glass pane per ${h.unit}. Until the heads are
            concatenated again, every head runs the identical recipe on its own pane, in parallel and without
            talking to the others. To the code, a head is just another batch axis.`;
          aside`The slabs are drawn far apart so you can see between them. <b>Focus head</b> (top left) moves the
            camera to one head and shrinks the others to dots; <b>All</b> brings them back.`;
          hint`Hover a cube in a split slab: the outlined cube in the flat matrix is the very same number.`;
          return out;
        });

        const S = step('Every query meets every key', () => {
          const KT = transpose(Kh).as('KT', 'K^{T}');
          const S = matmul(Qh, KT).as('S', 'S');

          say`This is where the famous <b>${T} × ${T}</b> shape is born. Inside each head, the query of token i is
            dotted with the key of token j, for every pair (i, j). Row = the token that is asking; column = the token
            being looked at.

            The layout makes the multiplication visible: each score cube sits <b>to the right of its query row</b>
            and <b>below its key column</b>. That is why ${Kh} is transposed: its ${T} axis has to run sideways to line
            up with the columns of ${S}.

            The ${dk} axis is summed away; a second ${T} axis appears. This is the O(T²) cost of attention.

            A large positive score means the query and key vectors point the same way; a large negative one means
            they point in opposite directions.`;
          aside`This step opens on one head; the dotted slabs behind it are the other heads doing exactly the same
            thing with their own columns.`;
          hint`Hover a score cube: its query row and key column light up, with the products that add up to it.`;

          KT.says`Transpose swaps the last two axes, so keys now run along columns: the key of ${col} sits directly
            above column ${col} of the scores.`;
          S.says`${slab}: the <b>query</b> of ${row} dotted with the <b>key</b> of ${col}.`;
          return S;
        });

        const Ss = step('Scale by 1/√d_k', () => {
          const Ss = map('s / sqrt(d_k)', { s: S }).as('Ss', 'S / √d_{k}');

          say`Every score is divided by √${dk} = √${dk.size}.

            <b>Why?</b> A dot product adds up ${dk} terms. If each term has variance around 1, the sum has variance
            around ${dk}, so scores grow like √${dk} as heads get wider. Large scores push the softmax (two steps
            ahead) towards a hard one-hot choice. Dividing by √${dk} keeps the spread of scores the same no matter
            how wide the head is.`;
          aside`A near one-hot softmax is where gradients vanish during training, which is the reason the original
            paper gives for this step.`;

          Ss.says`${slab}: the score of ${row} for ${col}, divided by √${dk}.`;
          return Ss;
        });

        const Sm = step('Causal mask', () => {
          const M = causalMask(T, causal).as('M', 'M');
          const Sm = add(Ss, M).as('Sm', 'S / √d_{k} + M');

          say`Token i must not see tokens after it. So the upper triangle (j &gt; i) is set to −∞: the small
            dark-red cells.

            ${M} is drawn as flat tiles with a silver frame. Like a weight it does not depend on the sentence, but
            it is fixed by a rule instead of being learned.

            <b>Broadcasting:</b> ${M} has shape ${M.sym} with no ${h.unit} axis. The same mask is added to each of the
            ${h.size} slabs; it is drawn as a single plate in front of the slabs, pushed back through all of them.

            Use the <b>Causal mask</b> switch (top left) to compare.
            ${live(() => (causal.value ? '' : '<b>The mask is currently off, so it is all zeros.</b>'))}`;
          aside`A GPT-style model predicts the next token at every position at once during training; hiding later
            tokens is what makes that honest. Encoders such as BERT skip this step and let every token see the
            whole sequence.`;

          M.says`${col} as seen from ${row}: ${live(({ value: v }) =>
            v === -Infinity
              ? 'blocked, because it comes later. Adding −∞ makes its softmax weight exactly 0.'
              : 'allowed. Adding 0 changes nothing.')}`;
          Sm.says`${slab}: the scaled score plus the mask. The mask has no ${h.unit} axis, so the same plate is
            added to every ${h.unit}.`;
          return Sm;
        });

        const A = step('Softmax: scores become weights', () => {
          const A = softmax(Sm).as('A', 'A');
          const at = 3; // the row quoted in the text
          const rowText = (hh) =>
            Array.from({ length: T.size }, (_, j) => j)
              .filter((j) => Sm.data[Sm.flat(hh, at, j)] !== -Infinity)
              .map((j) => `${Math.round(A.data[A.flat(hh, at, j)] * 100)}% on ${tok(TOKENS[j])}`)
              .join(', ');

          say`Each <b>row</b> is exponentiated and divided by its row total. Every row of ${A} is now a set of weights
            that sum to 1: how token i splits its attention over the tokens it may look at. Masked positions get
            exactly zero.

            Softmax runs over the last axis (keys), so rows sum to 1; columns do not have to.

            <h3>Reading a row</h3>

            Take row ${label(T, at)} in ${h.unit} 0. Its weights are ${live(() => rowText(0))}. In the next step, that is
            the mix of value vectors ${label(T, at)} receives from this ${h.unit}.

            The heads give different tables from the same input, only because their columns of ${W.Q} and ${W.K} differ.

            <h3>Do the patterns mean anything?</h3>

            Here, no: the weights are random, so where each head looks is arbitrary.`;
          aside`In a trained model, recognisable patterns appear: a head that puts nearly all its weight on the previous
            token, say, or one that links a pronoun to the noun it refers to. Nobody assigns those jobs. They come
            out of training and are named afterwards by people studying the model, and many heads have no tidy
            description at all.`;
          hint`Hover a cube in ${A} to see the whole row go through the softmax, number by number.`;

          A.says`${slab}: of everything ${row} takes in, <b>${pct(value)}</b> comes from ${col}.`;
          return A;
        });

        const Z = step('Mix the values', () => {
          const Z = matmul(A, Vh).as('Z', 'Z');

          say`Each token's new vector is a <b>weighted average of all the value vectors</b>, using its row of ${A} as
            the weights. Attention is the only kind of step in the whole model where information moves from one
            token to another.

            The ${T} × ${T} table is used up: we are back to one row per token, ${dk} numbers wide, per ${h.unit}.

            Notice the division of labour: the queries and keys decided <i>how much</i> each token counts; the values
            are <i>what</i> gets carried over. ${Vh} sits under ${Z} so that each of its columns lines up with the
            column of ${Z} it feeds.`;
          hint`Hover a cube in ${Z}: one row of ${A} times one column of ${Vh}.`;

          Z.says`${slab}: dimension ${col} of the new vector for ${row}, a weighted average of every token's value
            using row ${row} of the attention weights.`;
          return Z;
        });

        const Zc = step('Concatenate the heads', () => {
          const Zc = merge(Z, [h, dk], d).as('Zc', 'Z', { shape: '(T, h·d_{k})' });

          say`The exact inverse of the split, and again <b>no arithmetic</b>: the slabs slide out of depth and sit side
            by side, so each token has one row of ${h} · ${dk} = ${h.size} · ${dk.size} = ${d.size} = ${d} numbers again.

            This is why ${dk} is chosen as ${d} / ${h}: the heads' outputs tile the model width exactly.`;

          Zc.says`The same number as in the slabs, which are laid side by side again.`;
          return Zc;
        });

        const O = step('Output projection', () => {
          const WO = param('WO', 'W_{O}', [d, d], random(0.3));
          const O = matmul(Zc, WO).as('O', 'O');

          say`Until now the heads have worked in separate compartments, ${dk} wide. ${WO} is one more weight matrix
            that maps the concatenated head outputs back into the ${d.size} dimensions of the residual stream.

            Rows of ${WO} are grouped by ${h.unit} (brackets on its left); columns are residual-stream dimensions.
            Every head dimension can write into every column, so the heads' contributions are <b>summed in the same
            space</b>. No head has dimensions of its own in the output.`;
          aside`The three projections and this matrix are all the weights attention has: about 2.36 million per
            block in GPT-2 small.`;

          WO.says`Weight: how strongly head dimension ${row} writes into dimension ${col} of the residual stream.`;
          O.says`What attention writes into dimension ${col} of ${row}.`;
          return O;
        });

        X = step('Add back to the residual stream', () => {
          const Xa = add(Xin, O).as('Xa', `X_{${kA}}`);

          say`Attention does not replace the token vectors: it <b>adds to them</b>. The orange line is the residual
            (skip) connection. Follow it back: it starts at ${Xin}, <i>before</i> LayerNorm, so the un-normalised
            stream is what gets carried forward.

            <h3>Reading the result</h3>

            Every number has moved. For ${label(T, 3)}, dimension 0 went from ${cell(Xin, 3, 0)} to ${cell(Xa, 3, 0)}
            (${cell(Xin, 3, 0)} + ${par(cell(O, 3, 0))}). Token identity, position and now attention's output all live
            in the same ${d.size} dimensions, added on top of one another; nothing is kept in a separate compartment.`;
          aside`With random weights the change is arbitrary. In a trained model, this addition is how a token's
            vector picks up information from the tokens it attended to.`;
          hint`Hover a cube in ${Xa} to see the before + written = after sum.`;

          Xa.says`Dimension ${col} of ${row}: what the stream held before this sub-layer, plus what attention wrote.`;
          return Xa;
        });
      });

      section('mlp', 'MLP', { summary: 'LayerNorm → MLP → add' }, () => {
        const Xin = X;
        if (b > 0) {
          say`The same operations as in the earlier block: LayerNorm, widen, GELU, narrow, then the residual add.
            Again the shapes are identical and the weights are this block's own.`;
          aside`GPT-2 small repeats this block 12 times and the largest GPT-2 48 times; nothing changes except the
            number of repetitions and the size of each matrix.`;
        }

        const N = step('LayerNorm before the MLP', () => {
          const N = layerNorm(Xin, 'M', kM);
          say`Same operation as before, on the stream as it is now, with a separate gain vector. Again the result is
            a normalised copy: ${Xin} itself carries on to the next residual add.`;
          return N;
        });

        let W1;
        const Hh = step('MLP: expand to four times the width', () => {
          W1 = param('W1', 'W_{1}', [d, dff], random(0.3));
          const Hh = matmul(N, W1).as('H', 'H');

          say`The MLP (also called the feed-forward layer) works on <b>each token separately</b>. Unlike attention,
            no row ever looks at another row here.

            First every token's ${d.size} numbers are turned into ${dff.size} “hidden” numbers. Each hidden number is
            the token's row dotted with one column of ${W1}.`;
          aside`The factor of four is a convention from the original paper (512 → 2048); GPT-2 small goes 768 → 3072.`;
          hint`Hover a cube in ${Hh}: one row of ${N} dotted with one column of ${W1}.`;

          W1.says`Weight: how much dimension ${row} of a token's vector contributes to hidden unit ${col}.`;
          Hh.says`Hidden unit ${col} for ${row}.`;
          return Hh;
        });

        const G = step('GELU: bend every number', () => {
          const G = gelu(Hh).as('G', 'G');

          say`Each of the ${Hh.size} numbers is passed through the same curve, <b>one at a time</b>. No cube depends on
            any other; ${G} sits directly under ${Hh} so every cube lands below the one it came from.

            GELU(h) = h · Φ(h), where Φ(h) is the fraction of a standard bell curve lying below h. So a large positive
            number passes through almost unchanged, a large negative one is squashed to about 0, and numbers near 0
            are bent smoothly between the two. Here ${live(() => Hh.data.filter((x) => x < 0).length)} of the
            ${Hh.size} numbers in ${Hh} are negative: watch the blue cubes fade.

            <b>Why it is needed:</b> without a non-linear step, multiplying by ${W1} and then by the next matrix would
            be the same as multiplying by one matrix (their product), and the extra width would buy nothing.`;
          aside`GELU is what GPT-2 and BERT use. The original Transformer used ReLU, which sets negatives to exactly 0.`;
          hint`Hover a cube in ${G} for the exact formula with its numbers.`;

          G.says`GELU applied to the one number directly above it. Nothing else is involved.`;
          return G;
        });

        const F = step('MLP: project back down', () => {
          const W2 = param('W2', 'W_{2}', [dff, d], random(0.2));
          const F = matmul(G, W2).as('F', 'F');

          say`The ${dff.size} hidden numbers of each token are combined back into ${d.size}, ready to be written into
            the stream.

            ${W2} is drawn tall because it has one <b>row per hidden unit</b>. A useful way to read it: each hidden
            unit owns one row of ${W2}, and adds that row to the output, scaled by how active the unit is.

            <h3>Where the weights are</h3>

            ${W1} and ${W2} hold ${W1.size + W2.size} weights; attention's four matrices hold ${4 * d.size * d.size}.
            Two thirds of a block's weights sit in the MLP.`;
          aside`That ratio is the same at full size: about 4.7 million against 2.4 million in GPT-2 small.`;
          hint`Hover a cube in ${F}: ${dff.size} products are added; the table lists the largest.`;

          W2.says`Weight: how strongly hidden unit ${row} writes into dimension ${col} of the residual stream.`;
          F.says`What the MLP writes into dimension ${col} of ${row}.`;
          return F;
        });

        X = step('Add back to the stream: the block is done', () => {
          const Xm = add(Xin, F).as('Xm', `X_{${kM}}`);

          say`The MLP's output is added to the stream through a second residual connection, again starting from
            before the LayerNorm. For ${label(T, 3)}, dimension 0: ${cell(Xin, 3, 0)} + ${par(cell(F, 3, 0))} = ${cell(Xm, 3, 0)}.

            That completes one <b>transformer block</b>: normalise, attend, add; normalise, MLP, add.

            <b>The output has exactly the shape of the input</b>, ${Xm.sym}. That is what lets blocks be stacked:
            ${Xm} is handed to the next block exactly as this block's own input was.`;

          Xm.says`Dimension ${col} of ${row}: the stream after attention, plus what the MLP wrote.`;
          return Xm;
        });
      });
    });

    section('out', 'Output', { summary: 'norm → unembed → softmax' }, () => {
      const N = step('Final LayerNorm', () => {
        const N = layerNorm(X, 'F', 'f');
        say`After the last block, the stream is normalised once more. Nothing is added back this time: ${N} is what
          the output layer reads.`;
        return N;
      });

      const L = step('Unembedding: a score for every word', () => {
        // Tied weights: the output layer reuses the embedding table, transposed.
        const WET = transpose(WE).as('WET', 'W_{E}^{T}');
        const L = matmul(N, WET).as('L', 'logits');

        say`Each position's final vector is dotted with the embedding of <b>every word in the vocabulary</b>, giving
          ${V} = ${V.size} scores per position. A word scores high when the final vector points the same way as that
          word's embedding.

          The ${d} axis disappears and a vocabulary axis takes its place: the model is back in the world of words.

          <h3>Tied weights</h3>

          The matrix used here is the embedding table from the start, turned on its side: the cubes fly in from the
          embedding. The same ${WE.size} numbers turn ids into vectors at the start and vectors into word scores at
          the end.`;
        aside`GPT-2 ties these weights. Some models keep a separate output matrix of the same shape.`;
        hint`Hover a logit: the row of ${N} and the word's embedding column light up.`;

        WET.says`The same number as in the embedding table, turned on its side: column ${col} here is the embedding
          of ${col}.`;
        L.says`Score for ${col} as the token that follows ${row}: the final vector at that position dotted with the
          embedding of ${col}.`;
        return L;
      });

      step('Softmax: next-token probabilities', () => {
        const Pr = softmax(L).as('Pr', 'p');
        const last = T.size - 1;
        const best = () => {
          const p = Array.from(Pr.data.subarray(last * V.size, (last + 1) * V.size));
          const k = p.indexOf(Math.max(...p));
          return `${tok(VOCAB[k])} at ${Math.round(p[k] * 100)}%`;
        };

        say`The same softmax as in attention, now over the vocabulary axis: each row becomes ${V.size} probabilities
          that sum to 1.

          <b>Row i is the model's guess for the token that comes after position i.</b> There is one guess per
          position, not one for the whole sentence.

          <ul>
            <li><b>Generating text</b> uses only the last row. Here, after ${label(T, last)}, the most likely next token
            is ${live(best)}. One token is picked from that row, appended to the input, and the whole model runs again.</li>
            <li><b>Checking every position at once</b> uses all the rows: row i is compared with the token that really
            is at position i + 1. The causal mask is what makes that fair, because row i was computed without seeing
            later tokens. ${live(() => (causal.value ? '' : '<b>The mask is off right now, so every row has seen the whole sentence.</b>'))}</li>
          </ul>`;
        aside`That comparison at every position is how the model is trained. With random weights these probabilities
          are arbitrary; a trained model would put its weight on words that actually tend to follow.`;
        hint`Hover a cube in ${Pr} to see its whole row go through the softmax.`;

        Pr.says`The model's probability that ${col} comes after ${row}: <b>${pct(value)}</b>. Random weights, so the
          number is meaningless; the calculation is the real one.`;
      });
    });

    say`Folded up again, every box now stands for something you have seen in full. The stream keeps the shape
      ${X.sym} from start to end; each sub-layer branches off, computes in a different shape, and adds a result of
      that same shape back.

      Click any box to reopen it.`;
    aside`Also left out of this model: the batch axis B, LayerNorm's offset β and dropout.`;
    hint`Click a box to open it. Drag to orbit, Shift-drag (or right-drag) to move sideways, scroll to zoom.`;
  });
