// A Stable Diffusion v1-style text-to-image model, written in the tensor language.
//
//   prompt -> token rows + position                               the context C
//   noise  -> the starting latent z
//   for each of 3 timesteps t, from noisy to clean:
//     t -> sines and cosines -> two linear layers                 the time vector τ
//     U-Net(z, τ, C)                                              the noise guess with the prompt
//     U-Net(z, τ, C∅)                                             ... and without it
//     guidance: push the guess away from the promptless one       ε
//     sampler step (DDIM): remove part of ε from z                the next, cleaner z
//   decoder(z)                                                    the picture, 16 x 16 x 3
//
// The U-Net: conv in -> ResBlock -> cross-attention -> conv down -> ResBlock ->
// upsample + conv -> concat the skip -> ResBlock -> norm, conv out. It is
// drawn in full once; its five other calls are computed with module() and
// shown by their result only.
//
// The decoder: scale -> 1x1 conv -> conv in -> ResBlock -> self-attention ->
// upsample + conv -> ResBlock (fewer channels) -> upsample + conv -> norm,
// SiLU, conv out -> colours between 0 and 1.

import { program, axis, ONE, input, param, random, shared, departure, section, step, repeat, part, say, aside, hint, mulberry32 } from '../lang/core.js';
import { lookup, rows, transpose, map, add, silu, mean, std, softmax, matmul, conv2d, constant, again, reshape, flatten, unflatten, upsample, concat, groupMean, groupStd, module } from '../lang/ops.js';
import { row, col, slab, value, pct, live, picture } from '../lang/prose.js';

export const VOCAB = ['[start]', '[end]', 'a', 'blue', 'disc', 'green', 'red', 'square'];
export const TOKENS = ['[start]', 'red', 'square', '[end]'];
export const EMPTY = ['[start]', '[end]', '[end]', '[end]']; // the prompt that says nothing
export const STEPS = [667, 334, 1]; // 3 sampling steps spread over the 1000 timesteps of the schedule
export const GUIDANCE = 7.5;
export const GROUPS = 2; // of group normalisation
export const LATENT_SCALE = 0.18215; // latents are multiplied by this before diffusion, and divided by it before decoding
const SIZE = 4; // the latent is SIZE x SIZE
const tok = (text) => `<span class="tok">${text}</span>`;

// Standard normal noise from a seed: the "random" starting point of generation.
export function gaussian(n, seed) {
  const rnd = mulberry32(seed);
  return Array.from({ length: n }, () => {
    let u = 0;
    while (u === 0) u = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  });
}

// The noise schedule: ᾱ at timestep t is the share of the clean latent's
// variance that is left when the latent has been noised up to t. Stable
// Diffusion's "scaled linear" schedule: β runs from 0.00085 to 0.012 (linear in
// its square root) over 1000 timesteps, and ᾱ_t is the product of (1 − β) up to t.
export function alphaBar(t) {
  const [b0, b1, n] = [Math.sqrt(0.00085), Math.sqrt(0.012), 1000];
  let a = 1;
  for (let i = 0; i <= t; i++) a *= 1 - (b0 + ((b1 - b0) * i) / (n - 1)) ** 2;
  return a;
}

// cos and sin of t times a few frequencies, from fast (1) to slow (1 / 10000).
export const sinusoid = (t, j, n) => {
  const half = n / 2;
  const w = Math.exp((-Math.log(10000) * (j % half)) / half);
  return j < half ? Math.cos(t * w) : Math.sin(t * w);
};

export const build = () =>
  program('Stable Diffusion-style text-to-image model', { seed: 5 }, () => {
    const V = axis('V', VOCAB.length, { labels: VOCAB });
    const T = axis('T', TOKENS.length, { labels: TOKENS });
    const T0 = axis('T_0', EMPTY.length, { sym: 'T', labels: EMPTY });
    const Tmax = axis('T_max', TOKENS.length, { labels: TOKENS.map((_, i) => `pos ${i}`) });
    const dc = axis('d_c', 6);
    const ds = axis('d_s', 4);
    const dt = axis('d_t', 8);
    const cz = axis('c_z', 4, { unit: 'channel', deck: true });
    const [H, W] = [axis('H', SIZE), axis('W', SIZE)];
    const c = axis('c', 6, { unit: 'channel', deck: true });
    const cU = c; // the U-Net's width, under a name its ResBlocks can default to
    const k = axis('k', 3);
    const N = axis('N', SIZE * SIZE, { sym: 'H·W', labels: Array.from({ length: SIZE * SIZE }, (_, i) => `${Math.floor(i / SIZE)},${i % SIZE}`) });
    const [H2, W2] = [axis('H_2', SIZE / 2, { sym: 'H/2' }), axis('W_2', SIZE / 2, { sym: 'W/2' })];
    const cc = axis('c_cat', 2 * c.size, { sym: '2c', unit: 'channel', deck: true });
    const per = c.size / GROUPS; // channels in one group
    // The decoder's own widths and grids.
    const c1 = axis('c_1', 6, { unit: 'channel', deck: true });
    const c2 = axis('c_2', 4, { unit: 'channel', deck: true });
    const cp = axis('c_p', 3, { sym: '3', unit: 'colour', deck: true });
    const [Ha, Wa] = [axis('H_a', SIZE * 2, { sym: '2H' }), axis('W_a', SIZE * 2, { sym: '2W' })];
    const [Hb, Wb] = [axis('H_b', SIZE * 4, { sym: '4H' }), axis('W_b', SIZE * 4, { sym: '4W' })];

    const ids = input('ids', 'tokens', [T], TOKENS.map((t) => VOCAB.indexOf(t)));
    const ids0 = input('ids0', 'empty prompt', [T0], EMPTY.map((t) => VOCAB.indexOf(t)));

    departure('untrained', 'Every table, filter and weight matrix is seeded random numbers, so the predicted noise means nothing; a trained model holds different numbers in the same places.');
    departure('U-Net drawn once', 'The U-Net is called six times with the same weights. It is drawn in full for the first call only; the other five calls are computed exactly and shown by their inputs and their result.');
    departure('three sampling steps', 'The loop runs 3 times, where 20 to 50 is usual. Real code also runs the two U-Net calls of a step together, as a batch of two.');
    departure('text encoder cut down', 'The prompt becomes token rows plus position rows and nothing more. Stable Diffusion then runs them through a 12-layer transformer (CLIP), like the one on the transformer page.');
    departure('U-Net trimmed', 'One level down and one up, with one ResBlock at each and cross-attention in one place. Stable Diffusion has four levels, two or three ResBlocks at each, and attention at three of them.');
    departure('attention block trimmed', 'Only cross-attention with a single head is shown. The real block has 8 heads and also a self-attention step before it and a feed-forward step after it.');
    departure('decoder trimmed', 'The decoder doubles the grid twice, with one ResBlock and one attention layer before the first doubling and one ResBlock before the second. Stable Diffusion\'s decoder doubles three times and has 14 ResBlocks: two around the attention layer and three at each of four sizes.');
    departure('no biases', 'Biases are left out. They add one number per channel after each convolution and matrix multiply and change no shapes.');
    departure('tiny sizes', 'A 4 × 4 latent with 6 channels inside the U-Net, a 4-token prompt and a 16 × 16 picture, where Stable Diffusion uses a 64 × 64 latent, 320 to 1280 channels, 77 tokens of 768 numbers and a 512 × 512 picture.');

    say`A text-to-image model of this kind starts from a small grid of <b>pure noise</b>, the <b>latent</b>, and
      removes the noise a little at a time, steered by the prompt. The network that does the removing is a
      <b>U-Net</b>. It does not output a picture: it is shown a noisy latent and returns its <b>guess of the
      noise</b> in it.

      <ul>
        <li><b>Text</b> and <b>Noise</b> prepare what stays fixed: the prompt as rows of numbers, and the starting latent.</li>
        <li><b>Each step</b> then does the same five things: turn the timestep into a vector, ask the U-Net for the
        noise <b>with</b> the prompt, ask it again <b>without</b>, combine the two answers (guidance), and take
        part of that noise out of the latent (the sampler step).</li>
        <li><b>Step ${1}</b> shows the U-Net in full, box by box. It is the same network with the same weights in
        every later call, so those calls are shown by their result only.</li>
        <li><b>The decoder</b>, a second and separate network, then turns the finished latent into a
        ${SIZE * 4} × ${SIZE * 4} colour picture. It runs once, after the last step.</li>
      </ul>`;
    hint`Press → to start with the prompt, or click a box. Drag to orbit, Shift-drag to move sideways, scroll to zoom.`;

    // ---- pieces used more than once ----------------------------------------

    // Group normalisation of (channels, H, W), written from the primitives.
    const groupNorm = (X, id, sub = '') => {
      const s = sub ? `_{${sub}}` : '';
      const ch = X.shape[0];
      const mu = groupMean(X, { groups: GROUPS }).as(`${id}mu`, `μ${s}`);
      const sd = groupStd(X, { groups: GROUPS, mean: mu, eps: 1e-5 }).as(`${id}sd`, `σ${s}`);
      const g = param(`${id}g`, `γ${s}`, [ch, ONE, ONE], random(0.2, { mean: 1 }));
      const n = map('(x - μ) / σ * γ', { x: X, μ: mu, σ: sd, γ: g }).as(`${id}n`, `n${s}`);

      mu.says`The mean of the whole group that ${slab} belongs to: ${ch.size / GROUPS} channels of ${X}, at every
        position. The channels of one group show the same number.`;
      sd.says`How spread out the group that ${slab} belongs to is around its mean. The small ε only guards against
        dividing by zero.`;
      g.says`Gain for ${slab}: after normalising, every number of this channel is multiplied by ${value}.`;
      n.says`${slab}, row ${row}, column ${col}: the number at the same place in ${X}, minus its group's mean,
        divided by its group's spread, times the channel's gain.`;
      return n;
    };

    // A residual block: norm, SiLU, conv, add the time vector, norm, SiLU, conv, add the input back.
    // `detail` walks it step by step with its explanation; otherwise the caller shows it in one step.
    let tau; // the time vector of the call being built, set by unet()
    // `to` is the number of channels it gives out; `time: false` leaves the time vector out (the decoder's blocks).
    const resBlock = (X, id, out, { hw, detail = false, to: c = cU, time = true } = {}) => {
      const cin = X.shape[0];
      const at = (title, fn) => (detail ? step(title, fn) : fn());

      const n1 = at('Group norm: one scale for each group of channels', () => {
        const n = groupNorm(X, `${id}1`, '1');
        const [mu, sd, g] = [n.op.inputs[1], n.op.inputs[2], n.op.inputs[3]];
        if (detail) {
          say`A <b>residual block</b> (ResBlock) is the U-Net's basic building block. It starts by normalising.

            The ${cin.size} channels of ${X} are taken in <b>${GROUPS} groups of ${per}</b>. For each group:

            <ul>
              <li>${mu}: the mean of the whole group, all ${per} channels at all ${H.size * W.size} positions.</li>
              <li>${sd}: how spread out the group is around that mean.</li>
              <li>Subtract the mean and divide by the spread. Each group now has mean 0 and variance 1.</li>
              <li>Multiply each channel by its own gain ${g}.</li>
            </ul>

            ${mu} and ${sd} are drawn with one cube per channel; the ${per} channels of a group hold the same number.

            <b>Why groups?</b> The statistics come from this one latent alone, so the result does not depend on
            what else is being generated at the same time.`;
          aside`Stable Diffusion uses 32 groups everywhere, over 320 to 1280 channels. A transformer's LayerNorm
            is the same idea with one group per token.`;
          hint`Hover a cube in ${mu}: the whole group it averages lights up, ${per} channels deep.`;
        }
        return n;
      });

      const f1 = at('SiLU, then a convolution', () => {
        const a = silu(n1).as(`${id}a1`, 'a_{1}');
        const K = param(`${id}K1`, 'K_{1}', [c, cin, k, k], random(cin === c ? 0.22 : 0.16));
        const f = conv2d(a, K, { pad: 1, to: hw }).as(`${id}f1`, 'f_{1}');
        if (detail) {
          say`<b>SiLU</b> bends every number on its own: x times sigmoid(x). Large positive numbers pass almost
            unchanged, negative ones are squashed towards 0.

            Then a convolution: each of the ${c.size} filters in ${K} is ${k.size} × ${k.size} cells across and
            ${cin.size} slices deep. It is laid over a window of ${a}, multiplied cell by cell and added up, at every
            position. Padding keeps the grid ${H.size} × ${W.size}.

            This is how neighbouring positions exchange information: every output number has seen a
            ${k.size} × ${k.size} patch of the grid, across all channels.`;
          hint`Hover a cube in ${f}: its window lights up in every channel of ${a}, with the one filter that made it.`;
        }
        a.says`${slab}, row ${row}, column ${col}: the number at the same place in ${n1}, times its own sigmoid.`;
        K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${a}.`;
        f.says`Output of filter ${live(({ d }) => d)} at row ${row}, column ${col}: one window of ${a}, across all
          its channels, multiplied with that filter and added up.`;
        return f;
      });

      const g1 = !time ? f1 : at('Add the time vector', () => {
        const Tc = again(tau).as(`${id}T`, 'τ');
        const Wt = param(`${id}Wt`, 'W_{τ}', [dt, c], random(0.3));
        const e = matmul(Tc, Wt).as(`${id}e`, 'e');
        const ec = reshape(e, [c, ONE, ONE]).as(`${id}ec`, 'e_{c}');
        const g = add(f1, ec).as(`${id}g`, 'g');
        if (detail) {
          say`The U-Net has to behave differently on a very noisy latent and on an almost clean one. So every
            ResBlock is told the timestep.

            The time vector ${Tc} made earlier is brought in. A matrix ${Wt} turns its ${dt.size} numbers into
            ${c.size}, <b>one per channel</b>: ${e}. Re-addressed as ${ec}, one cube per channel, it is added to
            ${f1}.

            <b>Broadcasting:</b> the one number of a channel is added to all ${H.size * W.size} positions of that
            channel. The timestep shifts whole channels up or down; it says nothing about where things are.`;
          aside`Each ResBlock has its own matrix for this, so each can use the timestep in its own way.`;
          hint`Hover a cube in ${g}: one cube of ${f1} and its channel's one number in ${ec} light up.`;
        }
        Tc.says`Entry ${col} of the time vector, the same for the whole call.`;
        Wt.says`Weight: how much entry ${row} of the time vector contributes to channel ${col}.`;
        e.says`The timestep's contribution to channel ${col}: the time vector dotted with column ${col} of ${Wt}.`;
        ec.says`The timestep's contribution to ${slab}, copied from ${e}. One number for the whole channel.`;
        g.says`${slab}, row ${row}, column ${col}: the convolution's output plus the channel's time number.`;
        return g;
      });

      const f2 = at('Norm, SiLU and convolution once more', () => {
        const n = groupNorm(g1, `${id}2`, '2');
        const a = silu(n).as(`${id}a2`, 'a_{2}');
        const K = param(`${id}K2`, 'K_{2}', [c, c, k, k], random(0.22));
        const f = conv2d(a, K, { pad: 1, to: hw }).as(`${id}f2`, 'f_{2}');
        if (detail) {
          say`The same three operations a second time, with new weights: group norm over ${GROUPS} groups, SiLU,
            and a ${k.size} × ${k.size} convolution with the filters ${K}.

            After two convolutions, each number of ${f} has been computed from a patch
            ${2 * (k.size - 1) + 1} cells across, which here is more than the whole ${H.size} × ${W.size} grid.`;
          hint`Hover a cube in ${f}, then the cubes it lights, to trace one number back through the block.`;
        }
        a.says`${slab}, row ${row}, column ${col}: the number at the same place in ${n}, times its own sigmoid.`;
        K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${a}.`;
        f.says`Output of filter ${live(({ d }) => d)} at row ${row}, column ${col}: one window of ${a}, across all
          its channels, multiplied with that filter and added up.`;
        return f;
      });

      return at('Add the input back', () => {
        let skip = X;
        if (cin !== c) {
          // The input has more channels than the output: a 1 x 1 convolution brings it to size first.
          const Ks = param(`${id}Ks`, 'K_{s}', [c, cin, ONE, ONE], random(0.25));
          skip = conv2d(X, Ks, { to: hw }).as(`${id}s`, 's');
          Ks.says`Weight: how much ${slab} of ${X} contributes to channel ${row} of the shortcut.`;
          skip.says`Shortcut, channel ${live(({ d }) => d)} at row ${row}, column ${col}: the ${cin.size} channels of
            ${X} at this one position, mixed down by ${Ks}.`;
        }
        const h = add(skip, f2).as(id, out);
        if (detail) {
          say`The block's input ${X} is added to what the block computed. So the block only has to produce a
            <b>correction</b>; whatever it leaves alone passes through unchanged along the orange arc.

            ${h} is what the ResBlock hands on.`;
          aside`This is the residual connection that gives the block its name, the same idea as the residual
            stream of a transformer.`;
          hint`Hover a cube in ${h} to see the two numbers that were added.`;
        }
        h.says`${slab}, row ${row}, column ${col}: the block's input plus the block's correction at the same place.`;
        return h;
      });
    };

    // ---- the three inputs ---------------------------------------------------

    let C, WE, WP;
    section('text', 'Text', { summary: 'the prompt as rows of numbers' }, () => {
      const E = step('The prompt becomes rows of a table', () => {
        WE = param('WE', 'W_{E}', [V, dc], random(0.8));
        const E = lookup(WE, ids).as('E', 'E');

        say`A tokenizer has already turned the prompt into <b>token ids</b>:
          ${live(() => TOKENS.map((t, i) => `${tok(t)} → ${ids.data[i]}`).join(', '))}. A start marker is put in
          front and an end marker behind, so the prompt always has the same length, ${T} = ${T.size}.

          Each id selects one row of the embedding table ${WE}: a lookup, not arithmetic.`;
        aside`Stable Diffusion always uses 77 tokens: the prompt is cut off or padded with end markers. That is
          why a prompt has a length limit.`;
        hint`Hover a cube in ${E}: the one cube it was copied from lights up in the table.`;

        WE.says`Row ${row} of the embedding table is the vector for that token.`;
        E.says`The vector for ${row}, copied from the embedding table as it is.`;
        return E;
      });

      C = step('Add position: the context', () => {
        WP = param('WP', 'W_{P}', [Tmax, dc], random(0.5)).hint({ side: 'below' });
        const P = rows(WP, T).as('P', 'P');
        const C = add(E, P).as('C', 'C');

        say`A second table holds one row per position; it is added to ${E} so that word order is not lost.

          The result ${C} is the <b>context</b>: ${T.size} rows of ${dc} = ${dc.size} numbers. It is everything the
          U-Net will ever know about the prompt. It is computed once and then read, unchanged, by every
          cross-attention layer.`;
        aside`Here the real model does much more: these rows go through a 12-layer transformer (the CLIP text
          encoder) so that each row also reflects the words around it. That part works like the transformer
          page and is left out. Its output has the same shape as its input.`;
        hint`Hover a cube in ${C} to see the two numbers that were added.`;

        WP.says`Row ${row} of the position table: the vector for that position, whatever token sits there.`;
        P.says`The position vector for ${row}, copied from the position table as it is. It does not depend on which
          tokens the prompt holds, only on where they sit.`;
        C.says`Entry ${col} of the context row for ${row}: its token embedding plus its position embedding.`;
        return C;
      });
    });

    let z0;
    section('noise', 'Noise', { summary: 'the starting latent' }, () => {
      z0 = step('Start from pure noise', () => {
        const z = input('z', `z_{${STEPS[0]}}`, [cz, H, W], gaussian(cz.size * SIZE * SIZE, 2024));

        say`Generation does not start from a blank canvas but from <b>random numbers</b>: ${z.size} independent
          draws from a bell curve around 0. A different seed gives a different picture for the same prompt.

          ${z} is the <b>latent</b>: ${cz} = ${cz.size} channels on a ${H.size} × ${W.size} grid. It is not a
          picture. The model does all its denoising on this small grid, and only at the very end does a separate
          decoder turn the cleaned-up latent into pixels.

          The small number in its name is the timestep it belongs to: pure noise is where the countdown starts.`;
        aside`Stable Diffusion's latent is 4 × 64 × 64 for a 512 × 512 picture: each latent cell stands for an
          8 × 8 patch of pixels. Working on the small grid is what "latent diffusion" means, and what makes it cheap.`;
        hint`Use Focus channel (top left) to look at one channel at a time. Hover a cube for its value.`;

        z.says`${slab}, row ${row}, column ${col} of the latent: a random draw, given as input.`;
        return z;
      });
    });

    // ---- pieces of one sampling step ---------------------------------------

    // The timestep as a vector: waves, then two linear layers.
    const timeVector = (ts, i) => {
      let out;
      section('time', 'Timestep', { summary: `t = ${ts} → a vector` }, () => {
        if (i > 0)
          say`The countdown has moved on: this step is at t = ${ts}, less noisy than the one before. The same rule
            gives new waves, and the same two matrices turn them into a new time vector.`;
        const s = step('The timestep becomes waves', () => {
          const s = constant([ONE, ds], (_, j) => sinusoid(ts, j, ds.size), {
            rule: `the timestep t = ${ts} as cos(t·ω) for ${ds.size / 2} frequencies ω, then sin(t·ω) for the same ones; ω runs from 1 down towards 1/10000`,
            py: `w = torch.exp(-math.log(10000) * torch.arange(${ds.size / 2}) / ${ds.size / 2}); torch.cat([torch.cos(${ts} * w), torch.sin(${ts} * w)])[None]`,
          }).as('s', 's', { note: 'cos and sin of t, fixed by rule' });

          say`Generation runs through a countdown of <b>timesteps</b>. A high timestep means "the latent is almost
            pure noise", a low one "almost clean". This step is at t = ${ts}.

            A bare integer is a poor input for a network, so t is spread over ${ds} = ${ds.size} numbers: the cosine
            and the sine of t times a fast and a slow frequency. Nearby timesteps get similar patterns; distant
            ones get different patterns.

            ${s} is neither learned nor computed from the prompt or the noise: it follows from t by a fixed rule.
            That is why it is drawn as flat tiles with a silver frame.`;
          aside`Stable Diffusion uses 320 numbers here, and its timesteps run from 999 down to 0. This is the same
            sine and cosine pattern the original Transformer used to encode position in a sentence.`;

          s.says`${live(({ c: j }) => (j < ds.size / 2 ? 'cos' : 'sin'))} of t = ${ts} times frequency
            ${live(({ c: j }) => Math.exp((-Math.log(10000) * (j % (ds.size / 2))) / (ds.size / 2)))}.`;
          return s;
        });

        out = step('Two linear layers: the time vector', () => {
          const W1 = param('Wt1', 'W_{1}', [ds, dt], random(0.6));
          const t1 = matmul(s, W1).as('t1', 'τ_{1}');
          const t2 = silu(t1).as('t2', 'τ_{2}');
          const W2 = param('Wt2', 'W_{2}', [dt, dt], random(0.45));
          const t = matmul(t2, W2).as('tau', 'τ');

          say`A small network turns the waves into the <b>time vector</b> ${t}, ${dt} = ${dt.size} numbers: a matrix
            multiply, SiLU (x times sigmoid(x), applied to each number), and a second matrix multiply.

            ${t} is computed once per step and handed to every ResBlock of the U-Net.`;
          aside`These cubes do not depend on the prompt or on the noise, only on the timestep and the weights.`;
          hint`Hover a cube in ${t1}: the ${ds.size} waves and one column of ${W1} light up.`;

          W1.says`Weight from wave ${row} to entry ${col} of ${t1}.`;
          t1.says`Entry ${col}: the ${ds.size} waves dotted with column ${col} of ${W1}.`;
          t2.says`Entry ${col} of ${t1}, times its own sigmoid.`;
          W2.says`Weight from entry ${row} of ${t2} to entry ${col} of the time vector.`;
          t.says`Entry ${col} of the time vector: ${t2} dotted with column ${col} of ${W2}.`;
          return t;
        });
      });
      return out;
    };

    // The U-Net: latent, time vector and context in, predicted noise out. Written once.
    // Called directly it lays out its sections and steps; inside module() it is only computed.
    let ctx; // the context of the call being built
    const unet = (z, tauIn, ctxIn, outLabel) => {
      tau = tauIn;
      ctx = ctxIn;
      let X = z;

      section('in', 'Convolution in', { summary: 'latent → feature channels' }, () => {
        X = step('Convolution in: more channels', () => {
          const zc = again(X).as('zin', X.label);
          const K = param('Kin', 'K_{in}', [c, cz, k, k], random(0.3));
          const h = conv2d(zc, K, { pad: 1, to: [H, W] }).as('h0', 'h_{0}');

          say`The U-Net begins here, with the latent ${zc} brought in. A first convolution takes it from ${cz.size} channels to
            ${c} = ${c.size} <b>feature channels</b>, keeping the ${H.size} × ${W.size} grid.

            Each of the ${c.size} filters in ${K} is ${k.size} × ${k.size} cells across and one slice deep for every
            latent channel. Laid over a window of ${zc}, multiplied cell by cell and added up, it gives one number
            of ${h}.`;
          aside`In Stable Diffusion this step goes from 4 channels to 320.`;
          hint`Hover a cube in ${h}: a window in each of the ${cz.size} latent channels lights up, with one filter.`;

          zc.says`${slab}, row ${row}, column ${col} of the latent this call works on, brought in unchanged.`;
          K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of the latent.`;
          h.says`Output of filter ${live(({ d }) => d)} at row ${row}, column ${col}: one window of the latent, across
            its ${cz.size} channels, multiplied with that filter and added up.`;
          return h;
        });
      });

      section('res1', 'ResBlock', { summary: 'norm → conv → + time → norm → conv → add' }, () => {
        X = resBlock(X, 'h1', 'h_{1}', { hw: [H, W], detail: true });
      });

      section('attn', 'Cross-attention', { summary: 'every grid cell reads the prompt' }, () => {
        const h1 = X;
        const p = step('Group norm and a 1 × 1 convolution', () => {
          const n = groupNorm(h1, 'a');
          const K = param('Kp', 'K_{p}', [c, c, ONE, ONE], random(0.4));
          const p = conv2d(n, K, { to: [H, W] }).as('p', 'p');

          say`This is where the prompt gets in. First some preparation: group norm, as in the ResBlock, then a
            convolution whose filters are a single cell across. With no window, it mixes the ${c.size} channels of
            each grid cell into ${c.size} new ones without looking at the neighbours.`;

          K.says`Weight: how much ${slab} of ${n} contributes to channel ${row} of ${p}.`;
          p.says`Channel ${live(({ d }) => d)} at row ${row}, column ${col}: the ${c.size} channels of ${n} at this one
            position, mixed by ${K}.`;
          return p;
        });

        const x = step('Flatten: the grid becomes a list', () => {
          const x = flatten(p, N).as('x', 'x');

          say`<b>No arithmetic.</b> Attention works on a list of vectors, so the grid is re-addressed: each of the
            ${N.size} grid cells becomes <b>one row</b>, and its ${c.size} channels become that row's columns.

            The rows are named by grid position, "row,column". From here until the grid is put back together,
            a grid cell plays the part that a token plays in a language model.`;
          hint`Hover a cube in ${x}: the one cube it was copied from lights up in ${p}.`;

          x.says`Channel ${col} of grid cell ${row}, copied from ${p}.`;
          return x;
        });

        const xn = step('LayerNorm: each grid cell on its own', () => {
          const mu = mean(x).as('xmu', 'μ_{x}', { note: 'mean of each row' });
          const sd = std(x, { mean: mu, eps: 1e-5 }).as('xsd', 'σ_{x}', { note: 'spread of each row' });
          const g = param('xg', 'γ_{x}', [c], random(0.2, { mean: 1 }));
          const xn = map('(x - μ) / σ * γ', { x, μ: mu, σ: sd, γ: g }).as('xn', 'x_{n}');

          say`Every row of ${x} is normalised on its own: subtract the row's mean ${mu}, divide by its spread
            ${sd}, multiply each column by its gain ${g}.

            Unlike group norm a moment ago, nothing is shared between grid cells here: one row, one mean.`;

          mu.says`The average of the ${c.size} numbers of grid cell ${row}.`;
          sd.says`How spread out the ${c.size} numbers of grid cell ${row} are around their mean.`;
          g.says`Gain for column ${col}, shared by all grid cells.`;
          xn.says`Column ${col} of grid cell ${row}, normalised: minus the row's mean, divided by the row's spread,
            times the column's gain.`;
          return xn;
        });

        const [K, Vv] = step('Keys and values come from the prompt', () => {
          const Cc = again(ctx).as('Cc', ctx.label);
          const WK = param('WK', 'W_{K}', [dc, c], random(0.4));
          const K = matmul(Cc, WK).as('K', 'K');
          const WV = param('WV', 'W_{V}', [dc, c], random(0.4)).hint({ side: 'below' });
          const Vv = matmul(Cc, WV).as('V', 'V');

          say`Now the context ${Cc} from the Text box is brought in, unchanged. Two matrices turn each of its
            ${T.size} rows into two new rows of ${c.size} numbers:

            <ul>
              <li>${K}, the <b>keys</b>: what each token can be found by.</li>
              <li>${Vv}, the <b>values</b>: what each token hands over when it is found.</li>
            </ul>

            <b>This is the only place the prompt enters the U-Net.</b> Keys and values come from the text; the
            queries, next, come from the image. That is what makes it <b>cross</b>-attention. In the self-attention
            of a language model all three come from the same rows.

            ${WK} and ${WV} also change the width, from the text's ${dc.size} numbers per token to the ${c.size}
            the image side uses. Here both happen to be ${c.size}; they are different axes all the same.`;
          aside`In Stable Diffusion the text side is 768 wide and the image side 320 to 1280, so these matrices are
            far from square.`;
          hint`Hover a cube in ${K}: one row of the context and one column of ${WK} light up.`;

          Cc.says`Entry ${col} of the context row for ${row}, as it left the Text section.`;
          WK.says`Weight from context entry ${row} to key dimension ${col}.`;
          K.says`Key of ${row}, dimension ${col}: its context row dotted with column ${col} of ${WK}.`;
          WV.says`Weight from context entry ${row} to value dimension ${col}.`;
          Vv.says`Value of ${row}, dimension ${col}: its context row dotted with column ${col} of ${WV}.`;
          return [K, Vv];
        });

        const Q = step('Queries come from the image', () => {
          const WQ = param('WQ', 'W_{Q}', [c, c], random(0.4));
          const Q = matmul(xn, WQ).as('Q', 'Q');

          say`Each of the ${N.size} grid cells makes a <b>query</b>: its normalised row times ${WQ}. A query says
            what this part of the image is looking for in the prompt.`;

          WQ.says`Weight from column ${row} of ${xn} to query dimension ${col}.`;
          Q.says`Query of grid cell ${row}, dimension ${col}: its row of ${xn} dotted with column ${col} of ${WQ}.`;
          return Q;
        });

        const S = step('Every grid cell meets every token', () => {
          const KT = transpose(K).as('KT', 'K^{T}').hint({ place: 'flow' });
          const S = matmul(Q, KT).as('S', 'S');

          say`The query of every grid cell is dotted with the key of every token. The result ${S} has one row per
            grid cell and one column per token: <b>${N.size} × ${T.size}</b>.

            ${K} is transposed first so that its token axis runs sideways and lines up with the columns of ${S}.

            In a language model this table is square, tokens against tokens. Here its two sides are different
            things: places in the image against words of the prompt.`;
          hint`Hover a score: the query row of its grid cell and the key column of its token light up.`;

          KT.says`The same number as in ${K}, with rows and columns swapped: the key of ${col} is now a column.`;
          S.says`The query of grid cell ${row} dotted with the key of ${col}.`;
          return S;
        });

        const A = step('Scale, then softmax: shares of attention', () => {
          const Ss = map('s / sqrt(c)', { s: S }).as('Ss', 'S / √c');
          const A = softmax(Ss).as('A', 'A');

          say`Every score is divided by √${c} = √${c.size}, which keeps the scores from growing with the width.
            Then each <b>row</b> goes through a softmax: exponentiate, divide by the row's total.

            Each grid cell now has ${T.size} <b>weights that sum to 1</b>: how much of its attention goes to each
            token of the prompt.`;
          aside`In a trained model these weights are where words land in the picture: the grid cells that will
            show the square put their weight on "square". With random weights the pattern here is arbitrary.`;
          hint`Hover a cube in ${A}: its whole row of scores lights up, because softmax compares a score with the others in its row.`;

          Ss.says`The score of grid cell ${row} for ${col}, divided by √${c}.`;
          A.says`Grid cell ${row} gives <b>${pct(value)}</b> of its attention to ${col}.`;
          return A;
        });

        const Z = step('Mix the values', () => {
          const Z = matmul(A, Vv).as('Z', 'Z');

          say`Each grid cell collects a <b>weighted average of the values</b> ${Vv}, using its own row of ${A} as
            the weights. This is the moment information moves from the prompt into the image: ${Z} has one row per
            grid cell, built entirely out of token values.`;
          hint`Hover a cube in ${Z}: one row of ${A} and one column of ${Vv} light up.`;

          Z.says`Grid cell ${row}, dimension ${col}: its attention weights dotted with column ${col} of the values.`;
          return Z;
        });

        const y = step('Output projection, and add back', () => {
          const WO = param('WO', 'W_{O}', [c, c], random(0.4));
          const O = matmul(Z, WO).as('O', 'O');
          const y = add(x, O).as('y', 'y');

          say`One more matrix, ${WO}, and the result is <b>added</b> to ${x}, the rows from before the LayerNorm.
            As in the ResBlock, attention only contributes a correction; the inner orange arc carries ${x} around it.`;

          WO.says`Weight from dimension ${row} of ${Z} to column ${col} of ${O}.`;
          O.says`Grid cell ${row}, column ${col}: its row of ${Z} dotted with column ${col} of ${WO}.`;
          y.says`Grid cell ${row}, column ${col}: what it held before attention plus what attention brought from the prompt.`;
          return y;
        });

        X = step('Back to a grid, and add back again', () => {
          const Y = unflatten(y, [H, W]).as('Y', 'Y');
          const K = param('Kq', 'K_{q}', [c, c, ONE, ONE], random(0.4));
          const q = conv2d(Y, K, { to: [H, W] }).as('q', 'q');
          const h = add(h1, q).as('h2', 'h_{2}');

          say`The ${N.size} rows go back to their places in the ${H.size} × ${W.size} grid, columns becoming channels
            again: no arithmetic. A second single-cell convolution mixes the channels, and the result is added to
            ${h1}, the input of this whole box.

            ${h} has the same shape as ${h1}. What has changed is that every grid cell now carries something
            taken from the prompt.`;
          aside`Stable Diffusion's block does three things between flattening and unflattening: self-attention
            (grid cells look at each other), this cross-attention, and a feed-forward step. Only the middle one is
            shown.`;

          Y.says`Channel ${live(({ d }) => d)} at row ${row}, column ${col}, copied from ${y}.`;
          K.says`Weight: how much ${slab} of ${Y} contributes to channel ${row} of ${q}.`;
          q.says`Channel ${live(({ d }) => d)} at row ${row}, column ${col}: the channels of ${Y} at this one position, mixed.`;
          h.says`${slab}, row ${row}, column ${col}: the input of the attention box plus its correction.`;
          return h;
        });
      });

      const skip = X; // saved for the way back up
      section('down', 'Down', { summary: 'half the grid' }, () => {
        X = step('Down: half the grid each way', () => {
          const K = param('Kd', 'K_{↓}', [c, c, k, k], random(0.14));
          const d = conv2d(X, K, { stride: 2, pad: 1, to: [H2, W2] }).as('d', 'd');

          say`A convolution that moves <b>two cells at a time</b>, so the grid shrinks from ${H.size} × ${W.size} to
            ${H2.size} × ${W2.size}. Each cell of ${d} now stands for a larger patch of the latent.

            This is the downward stroke of the U. On a small grid a ${k.size} × ${k.size} window covers a large
            part of the picture, so the layers at the bottom can settle <b>what goes where</b> in broad strokes.
            ${X} is kept aside: it will be needed again on the way back up.`;
          aside`Stable Diffusion goes down three times, from 64 × 64 to 8 × 8, and doubles the number of channels
            at two of those steps.`;
          hint`Hover a cube in ${d}: its window lights up in every channel of ${X}. Neighbouring cubes of ${d} have windows two cells apart.`;

          K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${X}.`;
          d.says`Output of filter ${live(({ d: f }) => f)} at row ${row}, column ${col} of the smaller grid.`;
          return d;
        });
      });

      section('mid', 'Middle ResBlock', { summary: 'a ResBlock on the small grid' }, () => {
        X = step('A ResBlock at the bottom of the U', () => {
          const d = X;
          const m = resBlock(d, 'm', 'm', { hw: [H2, W2] });

          say`The same ResBlock as before, shown in one step: group norm, SiLU, convolution, add the time vector,
            and all of that once more, then add the input ${d} back. It has its own weights and works on the
            ${H2.size} × ${W2.size} grid.

            Every shape pattern is the same as in the first ResBlock; every number is different.`;
          aside`Stable Diffusion has a ResBlock, an attention block and another ResBlock here.`;
          hint`Hover a cube in ${m} and follow the highlights back through the block.`;
          return m;
        });
      });

      section('up', 'Up and skip', { summary: 'double the grid, rejoin the skip' }, () => {
        const v = step('Up: double the grid each way', () => {
          const m = X;
          const u = upsample(m, { to: [H, W] }).as('u', 'u');
          const K = param('Ku', 'K_{↑}', [c, c, k, k], random(0.1));
          const v = conv2d(u, K, { pad: 1, to: [H, W] }).as('v', 'v');

          say`The upward stroke of the U. First every cell of ${m} is simply <b>copied into a block</b>
            ${H.size / H2.size} cells wide and ${H.size / H2.size} tall. The grid is ${H.size} × ${W.size} again, but
            blocky: no new information has appeared. A convolution then blends the blocks into each other.`;
          aside`Other networks enlarge with a learned "transposed convolution" instead. Copying and then
            convolving, as Stable Diffusion does, avoids the checkerboard patterns those can leave.`;
          hint`Hover a cube in ${u}: the one cube of ${m} it was copied from lights up. Four cubes share each source.`;

          u.says`${slab}, row ${row}, column ${col}: a copy of one cell of ${m}, the cell at half this row and half
            this column, rounded down.`;
          K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${u}.`;
          v.says`Output of filter ${live(({ d }) => d)} at row ${row}, column ${col}: one window of ${u}, across all
            its channels, multiplied with that filter and added up.`;
          return v;
        });

        X = step('The skip connection: rejoin what was set aside', () => {
          const sk = again(skip).as('sk', 'h_{2}');
          const w = concat(v, sk, cc).as('w', 'w');

          say`Going down to the small grid and back up loses fine detail: ${v} was rebuilt from a quarter as many
            cells. So the tensor that was set aside before going down, ${sk}, is brought in, and the two are laid
            on top of each other as channels: ${c.size} + ${c.size} = ${cc.size}.

            <b>Concatenated, not added:</b> no number changes, the stack just gets deeper. The next block's
            filters read all ${cc.size} channels and decide for themselves how to combine the coarse path with the
            fine one.

            This long connection from the left side of the U to the right side is what gives the U-Net its name.`;
          aside`Stable Diffusion makes such a connection for every block on the way down, twelve in all.`;
          hint`Hover a cube in ${w}: front channels come from ${v}, back channels from ${sk}.`;

          sk.says`${slab}, row ${row}, column ${col} of the tensor saved before going down, brought in unchanged.`;
          w.says`${slab}, row ${row}, column ${col}: copied from ${live(({ d }) => (d < c.size ? 'the upsampled path' : 'the skip connection'))}.`;
          return w;
        });
      });

      section('res2', 'ResBlock', { summary: 'merge the two paths: 2c → c' }, () => {
        X = step('A ResBlock that merges the two paths', () => {
          const w = X;
          const r = resBlock(w, 'r', 'r', { hw: [H, W] });
          const s = r.op.inputs[0];

          say`A third ResBlock, shown in one step. It differs from the first in one way: it takes in
            ${cc.size} channels and gives out ${c.size}.

            <ul>
              <li>Its group norm works on ${GROUPS} groups of ${cc.size / GROUPS} channels.</li>
              <li>Its first convolution has filters ${cc.size} slices deep, and brings the channels down to ${c.size}.</li>
              <li>Its input cannot be added back directly, because ${cc.size} channels do not fit onto ${c.size}. So
              the shortcut goes through a single-cell convolution first: ${s}.</li>
            </ul>`;
          hint`Hover a cube in ${s}: one position lights up through all ${cc.size} channels of ${w}.`;
          return r;
        });
      });

      section('out', 'Out', { summary: 'back to latent channels: the noise guess' }, () => {
        const a = step('Norm and SiLU, one last time', () => {
          const r = X;
          const n = groupNorm(r, 'o');
          const a = silu(n).as('oa', 'a');

          say`The way out mirrors the way in: group norm over ${GROUPS} groups of ${per} channels, then SiLU on
            every number.`;

          a.says`${slab}, row ${row}, column ${col}: the number at the same place in ${n}, times its own sigmoid.`;
          return a;
        });

        X = step('Convolution out: the predicted noise', () => {
          const K = param('Kout', 'K_{out}', [cz, c, k, k], random(0.2));
          const eps = conv2d(a, K, { pad: 1, to: [H, W] }).as('out', outLabel);

          say`A last convolution goes from ${c.size} channels back to ${cz.size}, the shape of the latent.

            ${eps} is the U-Net's answer: its <b>guess of the noise</b> that is mixed into the latent, number for
            number, given the prompt and the timestep. It is not a picture, and not a cleaner latent either.
            Subtracting a part of this guess from the latent is what makes the latent a little cleaner; that is
            the job of the sampler step, three boxes further on.`;
          aside`Predicting the noise instead of the clean result is how these models are trained: noise is added
            to a real latent, and the U-Net is rewarded for naming exactly what was added.`;
          hint`Hover a cube in ${eps} and keep following the highlights leftwards: every number here depends on the whole latent, the timestep and the prompt.`;

          K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${a}.`;
          eps.says`The predicted noise in ${slab} of the latent at row ${row}, column ${col}.`;
          return eps;
        });
      });

      return X;
    };

    // One more call of the U-Net, computed exactly and not drawn: its three
    // inputs are brought in, and only its answer appears.
    const callUnet = (z, tauIn, ctxIn, key, label) => {
      const zc = again(z).as(`${key}z`, z.label);
      const tc = again(tauIn).as(`${key}t`, 'τ');
      const Cc = again(ctxIn).as(`${key}C`, ctxIn.label);
      const e = module('U-Net', [zc, tc, Cc], () => unet(zc, tc, Cc, 'ε'), { prefix: `${key}.` }).as(`eps${key}`, label);

      zc.says`${slab}, row ${row}, column ${col} of this step's latent, brought in unchanged.`;
      tc.says`Entry ${col} of this step's time vector, brought in unchanged.`;
      Cc.says`Entry ${col} of the context row for ${row}, brought in unchanged.`;
      e.says`The noise the U-Net predicts in ${slab} of the latent at row ${row}, column ${col},
        ${key === 'c' ? 'given the prompt' : 'given the empty prompt'}.`;
      return e;
    };

    // ---- the denoising loop -------------------------------------------------

    let C0; // the context of the empty prompt: made in the first step, reused in the others
    let z = z0; // the latent, replaced by a cleaner one at the end of every step
    shared(() => {
      repeat(STEPS.length, { prefix: 's', name: 'Step', detail: 'first' }, (i) => {
        const ts = STEPS[i];
        const prev = i + 1 < STEPS.length ? STEPS[i + 1] : 0; // the timestep this step hands over to
        const zt = z;
        const tauT = timeVector(ts, i);

        // The noise guess with the prompt: the U-Net in full the first time, by its result afterwards.
        let epsC;
        if (i === 0) epsC = unet(zt, tauT, C, 'ε_{c}');
        else
          section('cond', 'U-Net, with the prompt', { summary: 'the same U-Net, shown by its result' }, () => {
            epsC = step('The U-Net again, with the prompt', () => callUnet(zt, tauT, C, 'c', 'ε_{c}'));

            say`The whole U-Net of Step 1 is run again: the same boxes in the same order, with the <b>same
              weights</b>. Only two of its inputs have changed. The latent is now ${zt}, cleaner than before, and
              the time vector says t = ${ts}. The prompt's context ${C} is what it always was.

              Drawing every tensor inside it again would show the same boxes with other numbers, so only the
              answer ${epsC} is drawn. It is computed in full, exactly.`;
            hint`Hover a cube in ${epsC}: all three inputs light up completely, because every number of the answer depends on every number that went in.`;
          });

        if (i === 0)
          section('empty', 'Empty prompt', { summary: 'a context that says nothing' }, () => {
            C0 = step('An empty prompt', () => {
              const WEa = again(WE).as('WE0', 'W_{E}');
              const WPa = again(WP).as('WP0', 'W_{P}');
              const E0 = lookup(WEa, ids0).as('E0', 'E_{∅}');
              const P0 = rows(WPa, T0).as('P0', 'P');
              const Cn = add(E0, P0).as('C0', 'C_{∅}');

              say`The next box runs the U-Net a second time <b>without the prompt</b>, to find out what the prompt
                changes. "Without" still needs a context of the right shape, so an <b>empty prompt</b> is used: a
                start marker followed by end markers, ${live(() => EMPTY.map(tok).join(' '))}.

                It goes through exactly the same two tables as the prompt did. ${WEa} and ${WPa} are the Text
                box's tables again, not new weights. The result ${Cn} has the same shape as ${C} and says nothing
                about what to draw.

                ${Cn} depends neither on the prompt nor on the latent, so it is made once and reused in every step.`;
              aside`A model can only do this if it has seen empty prompts before: during training the prompt is
                left out some of the time, on purpose.`;
              hint`Look at the rows of ${E0}: the end markers all pick the same row of the table. Adding position is what makes them differ in ${Cn}.`;

              WEa.says`Row ${row} of the embedding table: the same table as in the Text section, brought in unchanged.`;
              E0.says`The vector for ${row}, copied from the embedding table as it is.`;
              WPa.says`Row ${row} of the position table: the same table as in the Text section, brought in unchanged.`;
              P0.says`The position vector for this row, copied from the position table as it is.`;
              Cn.says`Entry ${col} of the empty prompt's context row for ${row}: its token embedding plus its position embedding.`;
              return Cn;
            });
          });

        // The noise guess without the prompt.
        let epsU;
        section('uncond', 'U-Net, without the prompt', { summary: 'the same U-Net, shown by its result' }, () => {
          epsU = step('The U-Net again, without the prompt', () => {
            const e = callUnet(zt, tauT, C0, 'u', 'ε_{∅}');

            say`The same U-Net, with the <b>same weights</b>, is run a second time on the <b>same latent</b> ${zt}
              and the same time vector. The only thing that differs is the context: the empty prompt's ${C0}
              instead of the prompt's.

              Its answer ${e} is the noise the U-Net would guess <b>if it knew nothing about the prompt</b>.

              The whole U-Net you just walked through is inside that one arrow. Drawing it all again would show the
              same boxes with other numbers, so only the answer is drawn. It is computed in full, exactly.`;
            aside`Real code does not make two separate calls. It stacks the two cases into a batch of two and runs
              the U-Net once; the arithmetic is identical.`;
            hint`Hover a cube in ${e}: all three inputs light up completely, because every number of the answer depends on every number that went in.`;
            return e;
          });
          if (i > 0)
            say`Then once more without the prompt: the same latent, the same time vector, and the empty prompt's
              context from Step 1. ${epsU} is this step's promptless guess.`;
        });

        // Guidance: exaggerate what the prompt changed.
        let eps;
        section('guide', 'Guidance', { summary: 'exaggerate what the prompt changed' }, () => {
          const d = step('What the prompt changes', () => {
            const ec = again(epsC).as('gc', 'ε_{c}');
            const d = map('p - u', { u: epsU, p: ec }).as('gd', 'Δ');

            say`There are now two guesses of the noise in the same latent: ${ec} with the prompt, brought in from
              where the U-Net made it, and ${epsU} without.

              Subtract one from the other, cell by cell. Where the two agree, ${d} is close to 0. Where the prompt
              made the U-Net change its mind, it is large. ${d} is <b>what the prompt changes</b>.`;
            hint`Hover a cube in ${d} to see the two guesses it compares.`;

            ec.says`${slab}, row ${row}, column ${col}: the noise guess with the prompt, brought in unchanged.`;
            d.says`${slab}, row ${row}, column ${col}: the guess with the prompt minus the guess without it.`;
            return d;
          });

          eps = step('Guidance: go further the same way', () => {
            const g = constant([ONE], [GUIDANCE], {
              rule: 'the guidance scale, a setting chosen by whoever runs the model',
              py: `torch.tensor([${GUIDANCE}])`,
            }).as('g', 'g', { note: 'guidance scale: a setting' });
            const e = map('u + g * d', { u: epsU, g, d }).as('eps', 'ε');

            say`The final noise guess ${e} starts from the promptless one and adds ${d} not once but
              <b>${String(GUIDANCE)} times</b>.

              Adding it once would give back exactly the guess with the prompt. Adding it several times
              <b>exaggerates</b> what the prompt changed, and so pushes the latent further towards the prompt
              than the U-Net alone would. This is <b>classifier-free guidance</b>.

              ${g} is a single tile in a silver frame: not learned, but chosen as a setting.`;
            aside`Higher values follow the prompt more literally and lose variety; lower ones do the opposite.
              7.5 is the usual default for Stable Diffusion.`;
            hint`Hover a cube in ${e}: its promptless guess, its difference and the one setting light up.`;

            g.says`The guidance scale: how many times what the prompt changed is added.`;
            e.says`${slab}, row ${row}, column ${col}: the promptless guess plus ${String(GUIDANCE)} times what the
              prompt changed there.`;
            return e;
          });
          if (i > 0)
            say`As in Step 1: the difference between the two guesses, then ${String(GUIDANCE)} times that difference
              added to the promptless guess. ${eps} is the noise this step will act on.`;
        });

        // The sampler step (DDIM): guess the clean latent, then put back the noise the next timestep expects.
        section('update', 'Sampler step', { summary: `t = ${ts} → t = ${prev}: a cleaner latent` }, () => {
          const fourDigits = (t) => live(() => t.data[0].toFixed(4));
          const x0 = step('Guess the clean latent', () => {
            const zc = again(zt).as('zt', zt.label);
            const a = constant([ONE], [alphaBar(ts)], {
              rule: `ᾱ at t = ${ts}, from the noise schedule: the share of a latent at this timestep that is still clean signal`,
              py: `alphas_cumprod[${ts}]`,
            }).as('at', 'ᾱ_{t}', { note: `noise schedule at t = ${ts}` });
            const x0 = map('(z - sqrt(1 - a) * e) / sqrt(a)', { e: eps, z: zc, a }).as('x0', 'ẑ_{0}');

            say`Now the noise guess is used to clean the latent. The sampler needs one number from the <b>noise
              schedule</b>, ${a} = ${fourDigits(a)}. The schedule says what a latent at timestep t = ${ts} is
              made of:

              <ul>
                <li>√${a} times the clean latent, plus</li>
                <li>√(1 − ${a}) times pure noise.</li>
              </ul>

              If ${eps} is that noise, the recipe can be solved for the clean latent: take the noise's share out
              of ${zc} and divide by √${a}. The result ${x0} is the model's <b>guess of the finished latent</b>,
              made from where it stands now.

              ${a} is fixed in advance: the schedule is a list of ${1000} such numbers, set by a formula and
              never learned. Hence the silver frame.`;
            aside`Early in the countdown this guess is rough: almost everything in the latent is noise, so a small
              error in the noise guess is blown up by the division. That is why the loop does not stop here.`;
            hint`Hover a cube in ${x0}: the latent's number, the noise guess for the same place and the schedule's number light up.`;

            zc.says`${slab}, row ${row}, column ${col} of this step's latent, brought in unchanged.`;
            a.says`The noise schedule's number for t = ${ts}: the share of a latent at this timestep that is still
              clean signal. Its exact value is ${fourDigits(a)}.`;
            x0.says`${slab}, row ${row}, column ${col} of the guessed clean latent: the latent minus the guessed
              noise's share, scaled back up.`;
            return x0;
          });

          z = step(prev ? 'Put noise back, as much as the next timestep expects' : 'The last step: the clean latent', () => {
            const ap = constant([ONE], [alphaBar(prev)], {
              rule: `ᾱ at t = ${prev}, from the noise schedule: the share of a latent at that timestep that is still clean signal`,
              py: `alphas_cumprod[${prev}]`,
            }).as('ap', 'ᾱ_{prev}', { note: `noise schedule at t = ${prev}` });
            const zn = map('sqrt(a) * x + sqrt(1 - a) * e', { x: x0, a: ap, e: eps }).as('zn', `z_{${prev}}`);

            say`The next step expects a latent at timestep t = ${prev}, with that timestep's amount of noise in it.
              The schedule's number for it is ${ap} = ${fourDigits(ap)}: more clean signal than before, less noise.

              So the guessed clean latent ${x0} and the guessed noise ${eps} are mixed again, in the proportions
              of the new timestep: √${ap} of the one and √(1 − ${ap}) of the other. The result ${zn} is the latent
              this step hands on.

              <b>Nothing random happens here.</b> The noise that goes back in is the U-Net's own guess, so the
              same starting noise and the same prompt always give the same result.`;
            aside`This update rule is the DDIM sampler. Other samplers use the same noise guess in a different
              rule, and some add fresh random noise at every step.`;
            hint`Hover a cube in ${zn}: its guessed clean value, its guessed noise and the schedule's number light up.`;

            ap.says`The noise schedule's number for t = ${prev}, the timestep this step hands over to. Its exact
              value is ${fourDigits(ap)}.`;
            zn.says`${slab}, row ${row}, column ${col} of the latent after this step: the guessed clean latent and
              the guessed noise, mixed in the proportions of t = ${prev}.`;
            return zn;
          });
          if (i > 0)
            say`As in Step 1: guess the clean latent from ${zt} and ${eps}, then mix that guess and the noise guess
              in the proportions of t = ${prev}. ${z} is ${prev ? 'handed to the next step' : 'the finished latent'}.`;
        });
      });
    });

    // ---- the decoder --------------------------------------------------------
    // A separate network with its own weights: it is not part of the loop and runs once.

    let X = z;
    let img;
    part('Decoder', () => {
    section('din', 'Convolution in', { summary: 'latent → feature channels' }, () => {
      const zs = step('The decoder begins: undo the scaling of the latent', () => {
        const zf = X;
        const f = constant([ONE], [LATENT_SCALE], {
          rule: 'the latent scaling factor, a number measured once on the training pictures and fixed since',
          py: `torch.tensor([${LATENT_SCALE}])`,
        }).as('Df', 'f', { note: 'scaling factor: fixed' });
        const zs = map('z / f', { z: zf, f }).as('Dzs', 'z_{s}');

        say`The loop is over. ${zf} is the finished latent, and from here on a <b>second network</b> takes over: the
          <b>decoder</b>. It has its own weights, knows nothing of the prompt or the timestep, and is run
          <b>once</b>.

          Its job is to turn ${cz.size} channels on a ${H.size} × ${W.size} grid into ${cp.size} colours on a
          ${Hb.size} × ${Wb.size} grid: every latent cell becomes a ${Hb.size / H.size} × ${Hb.size / H.size} patch of pixels.

          First a bookkeeping step. The U-Net works on latents that were multiplied by a fixed number,
          ${f} = ${String(LATENT_SCALE)}, so that their spread is about 1. The decoder expects latents
          without that factor, so every number is divided by it again.`;
        aside`The decoder is one half of an <b>autoencoder</b>, trained before the U-Net and frozen since. The other
          half, the encoder, squeezes pictures into latents. It is needed to train the U-Net and to start from an
          existing picture, but not to generate from text.`;
        hint`Hover a cube in ${zs}: its one number in ${zf} and the single scaling tile light up.`;

        f.says`The latent scaling factor, ${String(LATENT_SCALE)}: fixed in advance, the same for every picture.`;
        zs.says`${slab}, row ${row}, column ${col} of the latent, divided by the scaling factor.`;
        return zs;
      });

      X = step('Two convolutions: more channels', () => {
        const Kz = param('DKz', 'K_{z}', [cz, cz, ONE, ONE], random(0.5));
        const zq = conv2d(zs, Kz, { to: [H, W] }).as('Dzq', 'z_{q}');
        const K = param('DKin', 'K_{in}', [c1, cz, k, k], random(0.045));
        const d = conv2d(zq, K, { pad: 1, to: [H, W] }).as('Dd0', 'd_{0}');

        say`Two convolutions in a row.

          <ul>
            <li>${Kz} has filters a single cell across: it mixes the ${cz.size} latent channels of each grid cell
            into ${cz.size} new ones, without looking at the neighbours.</li>
            <li>${K} is an ordinary ${k.size} × ${k.size} convolution that widens the latent to
            ${c1} = ${c1.size} <b>feature channels</b>.</li>
          </ul>

          The grid is still ${H.size} × ${W.size}. The decoder does most of its work here, on the small grid, before
          it starts enlarging.`;
        aside`In Stable Diffusion the second convolution goes from 4 channels to 512. The decoder is widest where
          the grid is smallest and gets narrower as the grid grows.`;
        hint`Hover a cube in ${zq}: one position lights up through all ${cz.size} channels of ${zs}.`;

        Kz.says`Weight: how much ${slab} of ${zs} contributes to channel ${row} of ${zq}.`;
        zq.says`Channel ${live(({ d }) => d)} at row ${row}, column ${col}: the ${cz.size} channels of ${zs} at this one
          position, mixed by ${Kz}.`;
        K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${zq}.`;
        d.says`Output of filter ${live(({ d }) => d)} at row ${row}, column ${col}: one window of ${zq}, across its
          ${cz.size} channels, multiplied with that filter and added up.`;
        return d;
      });
    });

    section('dres', 'ResBlock', { summary: 'norm → conv → norm → conv → add' }, () => {
      X = step('A ResBlock with no time vector', () => {
        const d = X;
        const r = resBlock(d, 'Dd1', 'd_{1}', { hw: [H, W], to: c1, time: false });

        say`The same building block as in the U-Net, shown in one step: group norm, SiLU, convolution, all of that
          once more, then add the input ${d} back.

          <b>One thing is missing:</b> no time vector is added in the middle. The decoder is only ever shown a
          finished latent, so it has no timestep to be told about.`;
        aside`Stable Diffusion's decoder has two ResBlocks at this point, one before the attention layer that
          follows and one after it.`;
        hint`Hover a cube in ${r} and follow the highlights back through the block.`;
        return r;
      });
    });

    section('dattn', 'Self-attention', { summary: 'every grid cell reads every other' }, () => {
      const d1 = X;
      const x = step('Group norm, then flatten', () => {
        const n = groupNorm(d1, 'Da');
        const x = flatten(n, N).as('Dx', 'x');

        say`The decoder has one attention layer, here on the small grid. As in the U-Net it starts with group norm
          and then re-addresses the grid as a list: each of the ${N.size} grid cells becomes <b>one row</b> of
          ${c1.size} numbers. No arithmetic in the second half.`;
        hint`Hover a cube in ${x}: the one cube it was copied from lights up in ${n}.`;

        x.says`Channel ${col} of grid cell ${row}, copied from ${n}.`;
        return x;
      });

      const [Q, K, V] = step('Queries, keys and values: all from the same rows', () => {
        const WQ = param('DWQ', 'W_{Q}', [c1, c1], random(0.4));
        const Q = matmul(x, WQ).as('DQ', 'Q');
        const WK = param('DWK', 'W_{K}', [c1, c1], random(0.4)).hint({ side: 'above' });
        const K = matmul(x, WK).as('DK', 'K');
        const WV = param('DWV', 'W_{V}', [c1, c1], random(0.4)).hint({ side: 'below' });
        const V = matmul(x, WV).as('DV', 'V');

        say`Three matrices turn every row of ${x} into three new rows:

          <ul>
            <li>${Q}, the <b>query</b>: what this grid cell is looking for.</li>
            <li>${K}, the <b>key</b>: what this grid cell can be found by.</li>
            <li>${V}, the <b>value</b>: what this grid cell hands over when it is found.</li>
          </ul>

          <b>This is self-attention.</b> In the U-Net's cross-attention the keys and values came from the prompt.
          Here all three come from the image itself, so grid cells exchange information <b>with each other</b>,
          however far apart they lie. A convolution only ever reaches its ${k.size} × ${k.size} window.`;
        aside`The decoder has no prompt to read. It never sees the text: everything it knows is in the latent.`;
        hint`Hover a cube in ${K}: one row of ${x} and one column of ${WK} light up.`;

        WQ.says`Weight from column ${row} of ${x} to query dimension ${col}.`;
        Q.says`Query of grid cell ${row}, dimension ${col}: its row of ${x} dotted with column ${col} of ${WQ}.`;
        WK.says`Weight from column ${row} of ${x} to key dimension ${col}.`;
        K.says`Key of grid cell ${row}, dimension ${col}: its row of ${x} dotted with column ${col} of ${WK}.`;
        WV.says`Weight from column ${row} of ${x} to value dimension ${col}.`;
        V.says`Value of grid cell ${row}, dimension ${col}: its row of ${x} dotted with column ${col} of ${WV}.`;
        return [Q, K, V];
      });

      const A = step('Every grid cell meets every other: scores, then shares', () => {
        const KT = transpose(K).as('DKT', 'K^{T}').hint({ place: 'flow' });
        const S = matmul(Q, KT).as('DS', 'S');
        const Ss = map('s / sqrt(c_1)', { s: S }).as('DSs', 'S / √c_{1}');
        const A = softmax(Ss).as('DA', 'A');

        say`The query of every grid cell is dotted with the key of every grid cell. ${S} is square,
          <b>${N.size} × ${N.size}</b>: rows are the cells that ask, columns the cells that are asked.

          Every score is divided by √${c1} = √${c1.size}, and each <b>row</b> goes through a softmax. Each grid cell
          now has ${N.size} <b>weights that sum to 1</b>: how much of its attention goes to each cell of the grid,
          itself included.

          There is no mask: every cell may look at every other, in all directions.`;
        aside`In a trained decoder this lets distant parts of the picture agree with each other, on a colour or
          on the direction of an edge. With random weights the pattern is arbitrary.`;
        hint`Hover a cube in ${A}: its whole row of scores lights up, because softmax compares a score with the others in its row.`;

        KT.says`The same number as in ${K}, with rows and columns swapped: the key of grid cell ${col} is now a column.`;
        S.says`The query of grid cell ${row} dotted with the key of grid cell ${col}.`;
        Ss.says`The score of grid cell ${row} for grid cell ${col}, divided by √${c1}.`;
        A.says`Grid cell ${row} gives <b>${pct(value)}</b> of its attention to grid cell ${col}.`;
        return A;
      });

      X = step('Mix the values, back to a grid, add back', () => {
        const Z = matmul(A, V).as('DZ', 'Z').hint({ frame: 'operands' }); // the values stand far back, below the scores
        const WO = param('DWO', 'W_{O}', [c1, c1], random(0.4));
        const O = matmul(Z, WO).as('DO', 'O');
        const Y = unflatten(O, [H, W]).as('DY', 'Y');
        const d = add(d1, Y).as('Dd2', 'd_{2}');

        say`Each grid cell collects a <b>weighted average of the values</b> ${V}, using its own row of ${A} as the
          weights. One more matrix, ${WO}, and the ${N.size} rows go back to their places in the
          ${H.size} × ${W.size} grid, columns becoming channels again.

          The result is <b>added</b> to ${d1}, the input of this box: attention contributes a correction, as a
          ResBlock does.`;
        aside`This layer has a single head in Stable Diffusion too, and it is the only attention in the decoder.`;
        hint`Hover a cube in ${Z}: one row of ${A} and one column of ${V} light up.`;

        Z.says`Grid cell ${row}, dimension ${col}: its attention weights dotted with column ${col} of the values.`;
        WO.says`Weight from dimension ${row} of ${Z} to column ${col} of ${O}.`;
        O.says`Grid cell ${row}, column ${col}: its row of ${Z} dotted with column ${col} of ${WO}.`;
        Y.says`Channel ${live(({ d }) => d)} at row ${row}, column ${col}, copied from ${O}.`;
        d.says`${slab}, row ${row}, column ${col}: the input of the attention box plus its correction.`;
        return d;
      });
    });

    // Enlarge: copy every cell into a 2 x 2 block, then convolve. The channels stay as they are.
    const enlarge = (id, n, [h, w], ch, std, real) => {
      section(`dup${n}`, 'Up', { summary: `${h.size / 2} × ${w.size / 2} → ${h.size} × ${w.size}` }, () => {
        X = step(`Up: ${h.size / 2} × ${w.size / 2} becomes ${h.size} × ${w.size}`, () => {
          const d = X;
          const by = h.size / d.shape[1].size; // how many times larger each way
          const u = upsample(d, { to: [h, w] }).as(`Du${n}`, `u_{${n}}`);
          const K = param(`DKu${n}`, 'K_{↑}', [ch, ch, k, k], random(std));
          const v = conv2d(u, K, { pad: 1, to: [h, w] }).as(id, `d_{${id.slice(2)}}`);

          if (n === 1)
            say`Now the grid grows. Exactly as on the way up in the U-Net: every cell of ${d} is <b>copied into a
              ${by} × ${by} block</b>, which makes the grid ${h.size} × ${w.size} but blocky, and a convolution then blends
              the blocks into each other.`;
          else
            say`The second and last doubling, to the size of the picture: every cell of ${d} is copied into a
              ${by} × ${by} block, and a convolution blends the blocks.`;
          say`The number of channels does not change, ${ch.size} in and ${ch.size} out. ${u} holds
            ${live(() => u.size)} numbers, ${live(() => u.size / d.size)} times as many as ${d}, and not one of them
            is new.`;
          aside`${real}`;
          hint`Hover a cube in ${u}: the one cube of ${d} it was copied from lights up. Four cubes share each source.`;

          u.says`${slab}, row ${row}, column ${col}: a copy of one cell of ${d}, the cell at half this row and half
            this column, rounded down.`;
          K.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${u}.`;
          v.says`Output of filter ${live(({ d: f }) => f)} at row ${row}, column ${col}: one window of ${u}, across all
            its channels, multiplied with that filter and added up.`;
          return v;
        });
      });
    };

    enlarge('Dd3', 1, [Ha, Wa], c1, 0.1, 'Stable Diffusion\'s decoder does this three times, from 64 × 64 to 512 × 512, with three ResBlocks before each.');

    section('dres2', 'ResBlock', { summary: `fewer channels: ${c1.size} → ${c2.size}` }, () => {
      X = step('A ResBlock that narrows the channels', () => {
        const d = X;
        const r = resBlock(d, 'Dd4', 'd_{4}', { hw: [Ha, Wa], to: c2, time: false });
        const s = r.op.inputs[0];

        say`Another ResBlock in one step, now on the ${Ha.size} × ${Wa.size} grid. It takes in ${c1.size} channels
          and gives out ${c2} = ${c2.size}.

          <ul>
            <li>Its first convolution has filters ${c1.size} slices deep and brings the channels down to ${c2.size}.</li>
            <li>Its input cannot be added back directly, because ${c1.size} channels do not fit onto ${c2.size}. So
            the shortcut goes through a single-cell convolution first: ${s}.</li>
          </ul>

          <b>Larger grid, fewer channels:</b> as the picture gets more positions, each position is described by
          fewer numbers. At the end there will be only three, one per colour.`;
        aside`Stable Diffusion's decoder goes 512 → 512 → 256 → 128 channels while the grid goes
          64 → 128 → 256 → 512 cells each way.`;
        hint`Hover a cube in ${s}: one position lights up through all ${c1.size} channels of ${d}.`;
        return r;
      });
    });

    enlarge('Dd5', 2, [Hb, Wb], c2, 0.12, 'In Stable Diffusion three more ResBlocks follow at the full size before the way out. They are left out here.');

    section('dout', 'To pixels', { summary: 'three colours per pixel' }, () => {
      const a = step('Norm and SiLU, one last time', () => {
        const d = X;
        const n = groupNorm(d, 'Do');
        const a = silu(n).as('Doa', 'a');

        say`The way out is the same as the U-Net's: group norm over ${GROUPS} groups of ${c2.size / GROUPS} channels,
          then SiLU on every number. The grid is now ${Hb.size} × ${Wb.size}, so each group's statistics are taken
          over ${(c2.size / GROUPS) * Hb.size * Wb.size} numbers.`;

        a.says`${slab}, row ${row}, column ${col}: the number at the same place in ${n}, times its own sigmoid.`;
        return a;
      });

      const o = step('Convolution out: red, green and blue', () => {
        const K = param('DKout', 'K_{out}', [cp, c2, k, k], random(0.2));
        const o = conv2d(a, K, { pad: 1, to: [Hb, Wb] }).as('Dout', 'o');

        say`A last convolution with <b>${cp.size} filters</b>, one for each colour: red, green and blue. ${o} has one
          number per pixel and colour.

          These are not yet brightnesses. On the decoder's scale −1 stands for dark and 1 for bright, and
          nothing stops a number from landing outside that range.`;
        aside`That scale is set by training: the pictures a decoder learns from are stored with −1 for dark and 1
          for bright.`;
        hint`Hover a cube in ${o}: its ${k.size} × ${k.size} window lights up in every channel of ${a}, with the one filter that made it.`;

        K.says`A weight of the filter for ${live(({ r }) => ['red', 'green', 'blue'][Math.floor(r / k.size)])}, in the slice that reads ${slab} of ${a}.`;
        o.says`The raw output for ${slab} at row ${row}, column ${col}: one window of ${a}, across all its channels,
          multiplied with that colour's filter and added up.`;
        return o;
      });

      img = step('From numbers to a picture', () => {
        const SHIFT = 0.5;
        const im = map(`min(max(o / 2 + ${SHIFT}, 0), 1)`, { o }).as('img', 'image').hint({ show: 'image' });

        say`Every number is halved and moved up by ${String(SHIFT)}, which maps −1…1 onto 0…1, and whatever still falls
          outside is cut off at 0 or 1. Now each number is a <b>brightness</b>: 0 is dark, 1 is full.

          ${im} is the picture: three sheets of ${Hb.size} × ${Wb.size} numbers, tinted here in their own colours.
          A screen shows the three numbers of a pixel as one colour:

          ${picture(im)}

          <b>This is the end of generation.</b> With random weights the picture shows nothing: neither the U-Net
          nor the decoder has learned anything, so what comes out is coloured noise, shaped only by the random
          numbers that went in. Every step of the arithmetic, though, is the one a working model performs.`;
        aside`A real decoder's last step is the same rule. The result is then multiplied by 255 and rounded, to give
          the usual 8-bit colour values.`;
        hint`Use Focus (top left) to look at one colour's sheet at a time. Hover a cube: the one raw number it came from lights up.`;

        im.says`The brightness of ${slab} at row ${row}, column ${col} of the picture: the raw output halved, moved
          up by ${String(SHIFT)} and kept between 0 and 1.`;
        return im;
      });
    });
    }); // part

    say`Folded up again. Text and Noise on the left; then ${STEPS.length} steps, each ending in a latent with less
      noise in it than the one before; then the decoder, which turns the last latent into the picture:

      ${picture(img)}

      With random weights it shows nothing. The U-Net's noise guesses mean nothing, so the loop has moved the
      noise around without removing it, and the decoder has painted that latent with filters that have learned
      no shapes. The arithmetic, though, is exactly what a working model does.

      Click any box to reopen it.`;
    hint`Click a box to open it. Drag to orbit, Shift-drag (or right-drag) to move sideways, scroll to zoom.`;
  });
