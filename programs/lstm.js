// An LSTM language model, written in the tensor language.
//
//   for each token, in order, with the same weights every time:
//     h_{t-1} @ W_hh                       what the memory says to the gates
//     token -> row of W_E -> x_t @ W_ih    what the token says to the gates
//     z = the sum, cut into four runs:     i, f, o through a sigmoid, g through tanh
//     c_t = f * c_{t-1} + i * g            the cell state: forget some, write some
//     h_t = o * tanh(c_t)                  the hidden state: what is let out
//   h_T @ W_out -> softmax                 the next word
//
// The arithmetic is that of torch.nn.LSTM (one layer, no biases) followed by
// the decoder of PyTorch's word_language_model example. The first time step
// is walked in detail; the later ones are one tour step each.

import { program, axis, ONE, input, param, random, shared, departure, section, step, repeat, say, aside, hint } from '../lang/core.js';
import { lookup, matmul, add, sigmoid, map, columns, constant, again, softmax } from '../lang/ops.js';
import { row, col, value, pct, live } from '../lang/prose.js';

export const VOCAB = ['.', 'The', 'cat', 'dog', 'mat', 'on', 'sat', 'the'];
export const TOKENS = ['The', 'cat', 'sat', 'on'];
export const GATES = ['i', 'f', 'g', 'o']; // the order PyTorch stacks them in
const PARTS = ['the input gate', 'the forget gate', 'the candidate', 'the output gate'];
const tok = (text) => `<span class="tok">${text}</span>`;

export const build = () =>
  program('LSTM language model', { seed: 11 }, () => {
    const V = axis('V', VOCAB.length, { labels: VOCAB });
    const de = axis('d_e', 6);
    const dh = axis('d_h', 5);
    const G = axis('d_g', GATES.length * dh.size, { sym: '4·d_{h}' });
    const T = TOKENS.length;
    // One token id per time step. They are used, not drawn.
    const ids = TOKENS.map((t, i) => input(`id${i}`, 'token id', [ONE], [VOCAB.indexOf(t)]));
    const sentence = () => TOKENS.map(tok).join(' ');
    const part = ({ c }) => PARTS[Math.floor(c / dh.size)];

    departure('untrained', 'Every weight is a seeded random number, so the gates open and close arbitrarily and the predicted word means nothing; a trained model holds different numbers in the same places.');
    departure('one layer', 'The PyTorch example stacks 2 LSTM layers: the second reads the hidden states of the first where this one reads token rows.');
    departure('no biases', 'A real LSTM adds two learned bias rows to the raw gate numbers, and the output layer adds one more. They change no shapes.');
    departure('one prediction', 'Real code predicts the next word after every token and returns the logarithm of the probabilities. Here only the prediction after the last token is computed.');
    departure('token looked up inside its time step', 'Real code looks up the rows of all tokens first and then walks through them. Here each time step looks up its own token; the numbers are the same.');
    departure('tiny sizes', 'A vocabulary of 8 words, 4 tokens, 6 numbers per token and 5 per state, where the PyTorch example defaults to 200 numbers for each and reads 35 tokens at a time.');

    say`An LSTM reads a sentence <b>one token at a time</b> and keeps a memory of what it has read so far. The
      memory is two rows of numbers that are rewritten after every token: the <b>cell state</b> c, the memory
      itself, and the <b>hidden state</b> h, the part of the memory that is let out.

      <ul>
        <li><b>One box per token.</b> The sentence here is ${live(sentence)}, so there are ${T} boxes, read
        left to right. Each box is called a time step.</li>
        <li><b>The same weights in every box.</b> Only the token and the two rows of memory change from one
        box to the next.</li>
        <li><b>Gates.</b> Inside a box, rows of numbers between 0 and 1 decide what to forget, what to write
        and what to let out.</li>
        <li><b>A prediction at the end.</b> The last hidden state is turned into a probability for every word
        of the vocabulary: the model's guess for the next word.</li>
      </ul>`;
    hint`Press → to start with the first token, or click a box. Drag to orbit, Shift-drag to move sideways, scroll to zoom.`;

    let h = null; // the hidden state, handed from each time step to the next
    let c = null; // the cell state, likewise
    shared(() => {
      repeat(T, { prefix: 't', name: 'Time step', detail: 'first' }, (i) => {
        const word = TOKENS[i];
        section('cell', `Reading “${word}”`, { summary: 'one LSTM cell' }, () => {
          const hPrev =
            i > 0
              ? h
              : step('The memory starts empty', () => {
                  const h0 = constant([ONE, dh], () => 0, { rule: 'all zeros, because nothing has been read yet', py: 'torch.zeros(1, d_h)' }).as('h0', 'h_{0}');

                  say`Before the first token the model has read nothing, so its hidden state ${h0} is a row of
                    zeros.

                    The hidden state is the model's running summary of the sentence so far: ${dh} = ${dh.size}
                    numbers, however long the sentence gets. Every time step reads the one before it and writes
                    the next.

                    These zeros are drawn as flat tiles with a silver frame: they are fixed by a rule, neither
                    learned nor computed.`;
                  aside`Real code may be handed a starting state, for instance the one left by the previous
                    stretch of text. Without one, PyTorch starts from zeros. The PyTorch example uses 200 numbers
                    for the state.`;
                  hint`Hover a tile of ${h0}: nothing else lights up, and the inspector says why.`;

                  h0.says`Position ${col} of the hidden state before anything has been read: zero, by rule.`;
                  return h0;
                });

          const [Whh, r] = step('What the memory tells the gates', () => {
            const Whh = param('Whh', 'W_{hh}', [dh, G], random(0.6));
            const r = matmul(hPrev, Whh).as('r', 'a_{h}');

            say`${hPrev} is multiplied by the weight matrix ${Whh}. The result ${r} has ${G.size} numbers:
              ${dh.size} for each of ${GATES.length} blocks, side by side. They are the memory's say in what the
              gates will do.

              <b>In this first time step the hidden state is all zeros</b>, so every product is zero and ${r} is
              all zeros too: the memory has nothing to say yet. From the next time step on, this multiplication
              is where everything read earlier comes back in.

              <b>Why ${G} columns?</b> A cell has three gates and one candidate for new content, and each needs
              ${dh} raw numbers. Their weights are kept in one matrix, ${GATES.length} blocks of ${dh.size}
              columns, so that one multiplication computes all of them.`;
            aside`PyTorch calls this matrix weight_hh and stores the blocks in the order input gate, forget
              gate, candidate (it says “cell gate”), output gate. This page keeps that order.`;
            hint`Hover a cube in ${r}: the row ${hPrev} and one column of ${Whh} light up.`;

            Whh.says`Weight: how much position ${row} of the hidden state adds to raw number ${col}, which belongs to ${live(part)}.`;
            r.says`What the memory adds to raw number ${col}, which belongs to ${live(part)}: the hidden state
              dotted with one column of ${Whh}.`;
            return [Whh, r];
          });

          const x = step('The token becomes a row of numbers', () => {
            const WE = param('WE', 'W_{E}', [V, de], random(0.8));
            const x = lookup(WE, ids[i]).as('x', `x_{${i + 1}}`);

            say`Now the token. The model never sees text: a tokenizer has already turned ${live(() => tok(word))}
              into the number ${live(() => ids[i].data[0])}, and that number picks one row of the embedding table
              ${WE}. The row is copied out as ${x}. <b>A lookup, not arithmetic.</b>

              <ul>
                <li><b>${V} = ${V.size}</b>: vocabulary size, one table row per word.</li>
                <li><b>${de} = ${de.size}</b>: how many numbers describe a token.</li>
              </ul>

              Only this one token enters this time step. The tokens before it can reach the calculation only
              through the memory, and the tokens after it have not been read yet.`;
            aside`An LSTM adds no position numbers to its tokens, where a transformer must: the order of the
              sentence is the order of reading. Real code looks up all the tokens before it starts reading; the
              rows are the same.`;
            hint`Hover a cube in ${x}: the one tile it was copied from lights up in the table.`;

            WE.says`Row ${row} of the embedding table is the vector for that word.`;
            x.says`Number ${col} of the vector for ${live(() => tok(word))}, copied from the embedding table as it is.`;
            return x;
          });

          const a = step('What the token tells the gates', () => {
            const Wih = param('Wih', 'W_{ih}', [de, G], random(0.6));
            const a = matmul(x, Wih).as('a', 'a_{x}');

            say`The token's row ${x} is multiplied by a second weight matrix, ${Wih}. Again ${G.size} numbers
              come out, in the same ${GATES.length} blocks: the token's say in what the gates will do.

              The two matrices differ in height because they read different things. ${Wih} has ${de.size} rows,
              one for each number of the token; ${Whh} has ${dh.size}, one for each number of the hidden state.
              Both have ${G.size} columns.`;
            hint`Hover a cube in ${a}: the row ${x} and one column of ${Wih} light up.`;

            Wih.says`Weight: how much number ${row} of the token adds to raw number ${col}, which belongs to ${live(part)}.`;
            a.says`What the token adds to raw number ${col}, which belongs to ${live(part)}: the token's row
              dotted with one column of ${Wih}.`;
            return a;
          });

          const z = step('Add the two', () => {
            const z = add(a, r).as('z', 'z');

            say`${z} = ${a} + ${r}, cell by cell. Each of its ${G.size} numbers now depends on both the new
              token and the memory.

              These are still raw numbers of any size. Nothing in the sum makes one block a gate and another a
              candidate. The next two steps do that, by cutting ${z} into its blocks and treating them
              differently.`;
            aside`A real LSTM also adds two learned bias rows here. They are left out on this page.`;
            hint`Hover a cube in ${z}: one cube of ${a} and one of ${r} light up.`;

            z.says`Raw number ${col}, which belongs to ${live(part)}: what the token says plus what the memory says.`;
            return z;
          });

          const [zi, zf, zg, zo] = step('Cut into four', () => {
            const zs = GATES.map((g, k) => {
              const piece = columns(z, k * dh.size, dh).as(`z${g}`, `z_{${g}}`);
              if (k) piece.hint({ side: 'below' });
              return piece.says`Position ${col} of the raw numbers for ${live(() => PARTS[k])}, copied from ${z} as it is.`;
            });
            const [zi, zf, zg, zo] = zs;

            say`<b>No arithmetic.</b> The ${G.size} numbers of ${z} are cut into ${GATES.length} runs of
              ${dh.size}: ${zi}, ${zf}, ${zg} and ${zo}. Each run is as long as the cell state: one number for
              each position of the memory.`;
            hint`Hover a cube in any of the runs: the one cube of ${z} it was copied from lights up.`;
            return zs;
          });

          const [ig, fg, gg, og] = step('Three gates and a candidate', () => {
            const ig = sigmoid(zi).as('i', 'i').hint({ show: 'prob' });
            const fg = sigmoid(zf).as('f', 'f').hint({ show: 'prob' });
            const gg = map('tanh(x)', { x: zg }, { name: 'tanh', show: 'tanh' }).as('g', 'g');
            const og = sigmoid(zo).as('o', 'o').hint({ show: 'prob' });

            say`Each run is squashed, one number at a time.

              <ul>
                <li>${ig}, ${fg} and ${og} go through the <b>sigmoid</b>, which gives a number between 0 and 1.
                These are the <b>gates</b>: 0 is closed, 1 is open, and every position of the memory has its
                own.</li>
                <li>${gg} goes through <b>tanh</b>, which gives a number between −1 and 1. It is the
                <b>candidate</b>: what this token would write into the memory.</li>
              </ul>

              What a gate controls is decided by where it is used in the next two steps, not here: ${fg}
              scales the old cell state, ${ig} scales the candidate, and ${og} scales what is let out.`;
            aside`With random weights the gates open and close arbitrarily. In a trained LSTM they depend on what
              is being read, which is how it can hold on to something for many tokens and then let it go.`;
            hint`Hover a gate: the one raw number it was squashed from lights up.`;

            ig.says`Input gate, position ${col}: how much of the candidate is written into the memory here, <b>${pct(value)}</b>.`;
            fg.says`Forget gate, position ${col}: how much of the old cell state is kept here, <b>${pct(value)}</b>.`;
            gg.says`Candidate, position ${col}: what this token would write into the memory here, between −1 and 1.`;
            og.says`Output gate, position ${col}: how much of the memory is let out here, <b>${pct(value)}</b>.`;
            return [ig, fg, gg, og];
          });

          const cNew = step('The cell state: forget some, write some', () => {
            const cPrev =
              i > 0
                ? again(c).as('cin', `c_{${i}}`)
                : constant([ONE, dh], () => 0, { rule: 'all zeros, because nothing has been stored yet', py: 'torch.zeros(1, d_h)' }).as('c0', 'c_{0}');
            const cNew = map('f * c + i * g', { f: fg, c: cPrev, i: ig, g: gg }).as('c', `c_{${i + 1}}`).hint({ arrows: 'all' });

            say`The memory itself is rewritten: ${cNew} = ${fg} · ${cPrev} + ${ig} · ${gg}, position by
              position.

              <ul>
                <li><b>Forget:</b> the old cell state is multiplied by the forget gate. Where the gate is near 0
                the old number is wiped; where it is near 1 the old number is kept.</li>
                <li><b>Write:</b> the candidate is multiplied by the input gate and added.</li>
              </ul>

              Before the first token the old cell state ${cPrev} is all zeros (silver tiles again), so there is
              nothing to forget and ${cNew} is just the gated candidate. In the next time step ${cNew} comes in
              here as the old state.

              Notice what does <b>not</b> happen to the old cell state: it is not multiplied by a weight
              matrix, and it is not squashed. It is only scaled and added to.`;
            aside`That plain path from one time step to the next is the point of the design. When an LSTM is
              trained, the error signal travels back along it without passing through a weight matrix at every
              token, which is what makes it fade in a simpler recurrent network.`;
            hint`Hover a cube in ${cNew}: one cube each of ${fg}, ${cPrev}, ${ig} and ${gg} lights up, all at the same position.`;

            cPrev.says`Position ${col} of the cell state ${live(() =>
              i > 0 ? 'left by the time step before, brought in unchanged.' : 'before anything has been read: zero, by rule.')}`;
            cNew.says`Position ${col} of the memory after reading ${live(() => tok(word))}: the old number scaled
              by the forget gate, plus the candidate scaled by the input gate.`;
            return cNew;
          });

          const hNew = step('The hidden state: what is let out', () => {
            const hNew = map('o * tanh(c)', { c: cNew, o: og }).as('h', `h_{${i + 1}}`).hint({ arrows: 'all' });

            say`${hNew} = ${og} · tanh(${cNew}). The cell state is squashed into the range −1 to 1, and the
              output gate decides how much of each position is let out.

              The time step ends with two rows of ${dh.size} numbers: ${cNew}, the memory, and ${hNew}, the part
              of it that is shown. Both go on to the next time step. ${hNew} is also what a prediction reads;
              ${cNew} is read by nothing but the next time step.`;
            hint`Hover a cube in ${hNew}: the cell state and the output gate at the same position light up.`;

            hNew.says`Position ${col} of the hidden state after reading ${live(() => tok(word))}: the squashed
              cell state, scaled by the output gate.`;
            return hNew;
          });

          // A later time step is shown in one tour step; this is its text.
          if (i > 0) {
            say`The same cell again, now reading ${live(() => tok(word))}. Nothing about the calculation has
              changed, and neither have the weights: ${Whh} and the two other matrices hold the very same numbers
              as in Time step 1.

              What has changed is what comes in:

              <ul>
                <li>the hidden state ${hPrev} arrives from the left, so ${r} is no longer all zeros;</li>
                <li>the token's row is now ${x};</li>
                <li>the cell state of the time step before is brought in where it is used, next to ${cNew}.</li>
              </ul>

              The steps are the ones walked through for the first token: two multiplications, a sum, a cut into
              ${GATES.length} runs, the squashing, then the new cell state ${cNew} and the new hidden state
              ${hNew}.${live(() => (i === T - 1 ? ' This was the last token: the prediction reads this hidden state.' : ''))}`;
            aside`However many tokens are read, the model has the same number of weights. That is what lets
              one LSTM read sentences of any length.`;
            hint`Hover a cube of ${Whh}: the inspector names the same weight in Time step 1.`;
          }
          [h, c] = [hNew, cNew];
        });
      });
    });

    section('out', 'Next word', { summary: 'a score for every word → probabilities' }, () => {
      const logits = step('A score for every word', () => {
        const Wout = param('Wout', 'W_{out}', [dh, V], random(1.2));
        const logits = matmul(h, Wout).as('logits', 'logits');

        say`The sentence has been read. The last hidden state ${h} is multiplied by one more weight matrix,
          ${Wout}, which has one column for each word of the vocabulary. The result is ${V.size}
          <b>logits</b>: one raw score per word for being the next one.

          Everything the model knows about ${live(sentence)} at this point is in the ${dh.size} numbers of
          ${h}. The earlier tokens are not looked at again.`;
        aside`The PyTorch example computes these scores after every token, not only the last: during training
          each position's guess is compared with the word that really follows. It also adds a bias, and it can
          reuse the embedding table as this matrix when the two widths are equal.`;
        hint`Hover a logit: the row ${h} and that word's column of ${Wout} light up.`;

        Wout.says`Weight: how much position ${row} of the hidden state adds to the score of ${col}.`;
        logits.says`Score for ${col} as the next word: the last hidden state dotted with that word's column of ${Wout}.`;
        return logits;
      });

      step('Softmax: next-word probabilities', () => {
        const p = softmax(logits).as('p', 'p');
        const best = () => {
          const k = p.data.indexOf(Math.max(...p.data));
          return `${tok(VOCAB[k])} at ${Math.round(p.data[k] * 100)}%`;
        };

        say`The softmax turns the scores into probabilities that sum to 1: every score is exponentiated and
          divided by the total. The most likely next word here is ${live(best)}.

          To generate text, one word is picked from these probabilities and read as the next token: one more
          time step, starting from the hidden state and cell state already there. Nothing is computed again for
          the earlier tokens.`;
        aside`With random weights these probabilities are arbitrary; a trained model would put its weight on
          words that tend to follow. The PyTorch example returns the logarithm of the probabilities, which is
          what its loss function expects.`;
        hint`Hover a cube in ${p} to see the whole row go through the softmax.`;

        p.says`The model's probability that ${col} comes next: <b>${pct(value)}</b>. Random weights, so the number
          is meaningless; the calculation is the real one.`;
      });
    });

    say`Folded up again: ${T} boxes for ${T} tokens, each handing its hidden state and cell state to the next,
      then one box that turns the last hidden state into a guess for the next word.

      Every box holds the same cell with the same weights. Click any box to reopen it.`;
    aside`Also left out of this model: the batch axis, the biases, dropout between layers during training, and
      the second layer the PyTorch example stacks on top.`;
    hint`Click a box to open it. Drag to orbit, Shift-drag (or right-drag) to move sideways, scroll to zoom.`;
  });
