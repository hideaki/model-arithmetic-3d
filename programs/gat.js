// A graph attention network (Veličković et al., 2018), one layer, written in
// the tensor language.
//
//   A -> A + I -> M                          the graph, and the mask it gives
//   X -> X W -> split into heads -> Wh        every node's own numbers, per head
//   Wh a1, Wh a2 -> e = LeakyReLU(s + tᵀ)     a score for every pair of nodes
//     -> e + M -> softmax -> α                only linked nodes get a share
//   α Wh -> heads side by side -> ELU         the layer's output
//
// The paper's hidden layer (equations 1 to 5): K heads, joined side by side,
// then ELU. As in PyG's GATConv, the K weight tables are held as one table
// whose columns are split into heads, and the attention vector a = [a1 ‖ a2]
// is held as its two halves. The example graph is the one of the graph
// convolutional network page (programs/gcn.js), so the two can be compared.

import { program, axis, ONE, input, param, random, departure, section, step, say, aside, hint } from '../lang/core.js';
import { again, constant, map, add, transpose, split, merge, matmul, softmax } from '../lang/ops.js';
import { row, col, slab, value, pct, live } from '../lang/prose.js';

// The example: 8 papers and the links between them (who cites whom, either way) ...
const NODES = 8;
const LINKS = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7]];
// ... and, for every paper, which of 6 words it contains.
const WORDS = 6;
const HAS = [[0, 1], [0, 2], [0, 1, 2], [2, 3], [3, 4], [3, 4, 5], [4, 5], [1, 5]];
const SLOPE = 0.2; // of the LeakyReLU below zero, as in the paper

const linked = (i, j) => LINKS.some(([a, b]) => (a === i && b === j) || (a === j && b === i));
const ADJACENCY = Array.from({ length: NODES * NODES }, (_, q) => (linked(Math.floor(q / NODES), q % NODES) ? 1 : 0));
// Each row of the word table divided by its total, as the paper's code prepares the features.
const FEATURES = Array.from({ length: NODES * WORDS }, (_, q) => {
  const words = HAS[Math.floor(q / WORDS)];
  return words.includes(q % WORDS) ? 1 / words.length : 0;
});

export const build = () =>
  program('Graph attention network, one layer', { seed: 47 }, () => {
    const N = axis('N', NODES);
    const F = axis('F', WORDS);
    const K = axis('K', 2, { unit: 'head' });
    const Fp = axis('Fp', 3, { sym: 'F′' });
    const KF = axis('KF', 6, { sym: 'K·F′' });

    departure('untrained', 'The weight table W and the attention vectors a₁ and a₂ are seeded random numbers, so the attention weights and the output mean nothing; a trained model holds different numbers in the same places.');
    departure('tiny sizes', '8 nodes, 6 features, 2 heads of 3 numbers each, where the paper’s Cora graph has 2,708 nodes with 1,433 features and its first layer has 8 heads of 8.');
    departure('one layer', 'Only the first layer is shown. On Cora the paper stacks two: this one with 8 heads joined side by side, then an output layer with one head and a softmax over the 7 classes.');
    departure('inference only', 'Training is not shown: no dropout (0.6 in the paper, on the layer’s input and on the attention weights), no loss, no weight decay.');
    departure('no bias', 'The paper’s equations have no bias, and neither has this page. Both its authors’ code and PyG add one learned number per output column after the heads are joined.');
    departure('dense table and mask', 'Every pair of nodes is scored here, and pairs that are not linked are then masked out with −∞. Real code (PyG) scores only the pairs that are linked and never holds the rest; the paper’s own code adds −10⁹ instead of −∞. The probabilities that come out are the same.');
    departure('features prepared beforehand', 'Each node’s row of features has already been divided by its total, as the paper’s code does before the model runs. The page starts from the divided rows.');
    departure('no drawing of the graph', 'The graph is shown only as its table of links. The dots and lines one would draw on paper are not on the page.');

    const links = () => LINKS.map(([a, b]) => `${a}–${b}`).join(', ');
    const heads = (d) => `head ${d}`;

    say`Some data is a <b>graph</b>: things, and links between them. A graph attention network gives every node new
      numbers by listening to the nodes it is linked to, and it <b>decides for itself how much to listen to
      each</b>.

      <ul>
        <li><b>The graph</b> says which nodes may listen to which: every node to its neighbours and to itself.</li>
        <li><b>Each node's own numbers</b> are mixed with a weight table, separately for each of ${K.size} heads.</li>
        <li><b>Attention scores</b>: for every linked pair, a score made from the two nodes' numbers, turned into
        shares that add up to 1 over each node's neighbours.</li>
        <li><b>Weighted sum</b>: each node's new numbers are its neighbours' numbers, weighted by those shares.
        The heads are laid side by side.</li>
      </ul>

      The example is the same small graph as on the graph convolutional network page, which averages over the
      neighbours with fixed weights. Here the weights are computed.`;
    hint`Press → to start with the graph, or click a box. Drag to orbit, Shift-drag to move sideways, scroll to zoom.`;

    let M;
    section('graph', 'Graph', { summary: 'who may listen to whom' }, () => {
      const A = step('The graph is a table of links', () => {
        const A = input('A', 'A', [N, N], ADJACENCY);

        say`The model never sees dots and lines, only numbers. The graph here has ${N.size} nodes, numbered from 0,
          and ${live(() => LINKS.length)} links: ${live(links)}. Think of ${N.size} papers, linked where one cites
          the other.

          ${A} holds the same thing as a table with one row and one column per node: <b>1 where two nodes are
          linked, 0 where they are not</b>. A link has no direction, so the table is the same when rows and columns
          are swapped.`;
        aside`This table is called the adjacency matrix. The whole graph is one example: all its nodes go through
          the layer together.`;
        hint`Hover a cube in ${A}: its row and its column are the two nodes it speaks about.`;

        A.says`${live(({ value: v }) => (v ? 'Linked' : 'Not linked'))}: node ${row} and node ${col}.`;
        return A;
      });

      const At = step('Every node also listens to itself', () => {
        const I = constant([N, N], (i, j) => (i === j ? 1 : 0), { rule: '1 on the diagonal, 0 everywhere else', py: 'torch.eye(N)' }).as('I', 'I');
        const At = add(A, I).as('At', 'Ã');

        say`Every node will collect numbers from the nodes it is linked to. With ${A} as it is, it would hear its
          neighbours and <b>forget itself</b>. So a 1 is added on the diagonal: in ${At} every node is also linked
          to itself, and its own numbers take part in its average like a neighbour's.

          ${I} is not learned and not computed from the graph, which is why it is drawn as flat tiles with a
          silver frame.`;
        aside`The paper writes this as the neighbourhood of node i including i itself; PyG's GATConv adds these
          self-links unless told not to.`;

        I.says`${live(({ r, c }) => (r === c ? 'On the diagonal: node ' + r + ' with itself.' : 'Off the diagonal.'))}`;
        At.says`${live(({ r, c, value: v }) => (r === c ? `Node ${r} with itself: the 1 that was added.` : `${v ? 'Linked' : 'Not linked'}: node ${r} and node ${c}, as in the table before.`))}`;
        return At;
      });

      M = step('The mask: who may not listen to whom', () => {
        const M = map('mask(a)', { a: At }, { name: 'mask', show: 'mask' }).as('M', 'M');

        say`Further on, every node will give a share of its attention to <b>every</b> node, linked or not, because
          that is how the table arithmetic works. ${M} is what takes the shares of unlinked nodes away again: it is
          <b>0 where ${At} has a 1, and −∞ where ${At} has a 0</b>.

          It will be added to the scores just before the softmax. A score plus 0 is unchanged; a score plus −∞
          is −∞, and the softmax turns −∞ into a share of exactly 0. The −∞ cells are drawn small and dark.`;
        aside`The paper's own code builds the same table with −10⁹ in place of −∞ and calls it a bias matrix.
          PyG never builds it: it computes scores only along the links.`;
        hint`Hover a cube in ${M}: the one cell of ${At} it was made from lights up.`;

        M.says`${live(({ r, c, value: v }) => (v === 0 ? `Node ${r} may listen to node ${c}` : `Node ${r} may not listen to node ${c}: they are not linked`))}.`;
        return M;
      });
    });

    let Wh;
    section('transform', 'Transform', { summary: 'every node’s own numbers, per head' }, () => {
      const X = step('Each node comes with a row of numbers', () => {
        const X = input('X', 'X', [N, F], FEATURES);

        say`The graph says who is linked to whom. It says nothing about what a node <b>is</b>. That comes
          separately: one row of ${F.size} numbers for each of the ${N.size} nodes.

          In the example these stand for words: a paper's row is spread evenly over the words of a fixed list
          that it contains, and is 0 for the others, so <b>each row adds up to 1</b>. Node 0 contains
          ${live(() => X.data.slice(0, F.size).filter((v) => v > 0).length)} of the ${F.size} words.`;
        aside`These rows are made up for this page, like the links. In the paper's Cora graph a row has 1,433
          entries, one per word of the vocabulary, and nearly all of them are 0.`;

        X.says`Node ${row}, word ${col}: ${live(({ r, value: v }) => (v ? `the node contains it, one of its ${HAS[r].length} words` : 'the node does not contain it'))}.`;
        return X;
      });

      const XW = step('Every node’s own numbers are mixed', () => {
        const W = param('W', 'W', [F, KF], random(0.8));
        const XW = matmul(X, W).as('XW', 'XW');

        say`Each node's row of ${F.size} numbers is multiplied with the weight table ${W} and becomes a row of
          ${KF.size}. The <b>same table</b> is used for every node, and each node is handled on its own: the graph
          plays no part yet.

          The ${KF.size} columns are really ${K.size} tables of ${Fp.size} columns side by side, one per
          <b>head</b>. Each head will do its own attention with its own numbers.`;
        aside`The paper gives each head k its own matrix W^{k}. Holding them as one wide table, as PyG does, is the
          same arithmetic in one multiplication.`;
        hint`Hover a cube in ${XW}: one node's row of ${X} and one column of ${W} light up.`;

        W.says`Weight: how much word ${row} contributes to column ${col}, for every node alike.`;
        XW.says`Column ${col} of node ${row}, from its own features only.`;
        return XW;
      });

      Wh = step('Split into heads', () => {
        const Wh = split(XW, KF, [K, Fp]).as('Wh', 'Wh');

        say`The ${KF.size} columns of ${XW} are re-addressed as ${K.size} heads of ${Fp.size}. Nothing is
          computed: the numbers are the same, now standing in ${K.size} slabs, one per head.

          From here on the two heads do exactly the same thing, each in its own slab, with nothing passing
          between them until the very end.`;
        hint`Hover a cube in a slab of ${Wh}: the very same number lights up in ${XW}.`;

        Wh.says`${slab}: number ${col} of node ${row}, the same number as in the flat table.`;
        return Wh;
      });
    });

    let alpha;
    section('scores', 'Attention scores', { summary: 'how much each node listens to each neighbour' }, () => {
      const [s, tT] = step('A score for each node, as listener and as speaker', () => {
        const a1 = param('a1', 'a_{1}', [K, Fp, ONE], random(0.8));
        const s = matmul(Wh, a1).as('s', 's');
        const a2 = param('a2', 'a_{2}', [K, Fp, ONE], random(0.8));
        const t = matmul(Wh, a2).as('t', 't');
        const tT = transpose(t).as('tT', 't^{T}');

        say`Each head has two small learned columns. ${a1} turns each node's ${Fp.size} numbers into one score
          ${s}: how the node does <b>as the one listening</b>. ${a2} turns them into a second score ${t}: how it
          does <b>as the one listened to</b>.

          ${tT} is the same column laid on its side, one number per column of the table to come. Nothing is
          computed there.`;
        aside`The paper writes one vector a of length 2F′ applied to the two nodes' rows placed end to end,
          aᵀ[Wh_{i} ‖ Wh_{j}]. Its first half meets node i and its second half node j, so the score is
          a₁ · Wh_{i} + a₂ · Wh_{j}: the two columns here, added in the next step.`;
        hint`Hover a cube in ${s}: one node's row of ${Wh} and the column ${a1} of its head light up.`;

        a1.says`${slab}: weight ${row} of the listener's half of the attention vector.`;
        a2.says`${slab}: weight ${row} of the speaker's half of the attention vector.`;
        s.says`${slab}: node ${row}'s score as a listener.`;
        t.says`${slab}: node ${row}'s score as a speaker.`;
        tT.says`${slab}: node ${col}'s score as a speaker, copied from ${t}.`;
        return [s, tT];
      });

      const e = step('A score for every pair of nodes', () => {
        const e = map(`max(s + t, ${SLOPE} * (s + t))`, { s, t: tT }, { name: 'leaky', show: 'LeakyReLU' }).as('e', 'e').hint({ arrows: 'all' });

        say`For every pair, node i listening and node j speaking, the two scores are added: ${s} of the row's node
          plus ${tT} of the column's node. That gives a table of ${N.size} × ${N.size} for each head.

          Then a <b>LeakyReLU</b>: a positive sum is kept, a negative one is multiplied by ${live(() => SLOPE)}.

          <b>Broadcasting:</b> ${s} has one column and is repeated across; ${tT} has one row and is repeated down.

          One consequence can be read off the table: adding the listener's score shifts a whole row by the same
          amount, and the LeakyReLU never swaps two numbers. So within a head, <b>every row ranks the columns in
          the same order</b>, the order of ${tT}. Rows differ in how sharply they rank, and later in which
          columns the graph lets through.`;
        aside`This was pointed out after the paper, by Brody, Alon and Yahav (2022), who called it static
          attention. Their GATv2 applies the LeakyReLU before the attention vector, so that the order can differ
          from node to node.`;
        hint`Hover a cube in ${e}: the listener's score and the speaker's score light up.`;

        e.says`${slab}: how much node ${row} wants to listen to node ${col}, before the graph is consulted.`;
        return e;
      });

      const em = step('Unlinked pairs are masked', () => {
        const Mg = again(M).as('M1', 'M');
        const em = add(e, Mg).as('em', 'ẽ');

        say`${Mg} is the mask from the Graph section. Added to ${e}, it leaves the score of every linked pair as it
          was and sets every other one to −∞.

          The mask has no head axis: the same plate is added to both heads, because the graph is the same for
          both.`;
        hint`Hover a cube in ${em}: its score and its cell of the mask light up.`;

        Mg.says`${live(({ r, c, value: v }) => (v === 0 ? `Node ${r} may listen to node ${c}` : `Node ${r} may not listen to node ${c}`))}.`;
        em.says`${slab}: ${live(({ r, c, value: v }) => (v === -Infinity ? `node ${r} and node ${c} are not linked: −∞` : `the score of node ${r} for node ${c}, unchanged`))}.`;
        return em;
      });

      alpha = step('Softmax over each node’s neighbours', () => {
        const alpha = softmax(em).as('alpha', 'α');
        const most = (d, r) => {
          let at = 0;
          for (let c = 1; c < N.size; c++) if (alpha.data[(d * N.size + r) * N.size + c] > alpha.data[(d * N.size + r) * N.size + at]) at = c;
          return at;
        };

        say`Each row of ${em} becomes shares that add up to 1: every score is exponentiated and divided by the row's
          total. A −∞ becomes exactly 0, so <b>a node shares its attention only among its neighbours and itself</b>.

          ${alpha} is computed from the nodes' numbers, not fixed by the graph. In ${heads(0)}, node 0 gives its
          largest share to node ${live(() => most(0, 0))}; in ${heads(1)}, to node ${live(() => most(1, 0))}.`;
        aside`With random weights these shares mean nothing. In a trained model they are learned, and the paper's
          motivation is that a node can then listen more to some neighbours than to others, where the graph
          convolutional network weighs them by their numbers of links alone.`;
        hint`Hover a cube in ${alpha}: the whole row of scores it was a share of lights up.`;

        alpha.says`${slab}: of everything node ${row} takes in, <b>${pct(value)}</b> comes from node ${col}.`;
        return alpha;
      });
    });

    section('blend', 'Weighted sum', { summary: 'neighbours’ numbers, weighted; heads side by side' }, () => {
      const H = step('Every node takes in its neighbours', () => {
        const Wh2 = again(Wh).as('Wh2', 'Wh');
        const H = matmul(alpha, Wh2).as('H', 'H');

        say`${Wh2} is each node's own numbers from the Transform section. Multiplied from the left by ${alpha},
          every node's new row is a <b>weighted sum of the rows of the nodes it listens to</b>, with the weights
          from its row of ${alpha}. A node that is not linked contributes nothing, because its share is 0.`;
        aside`The graph convolutional network computes the same kind of sum, with weights fixed by the numbers of
          links. That one difference is the paper's contribution.`;
        hint`Hover a cube in ${H}: a row of ${alpha} lights up, with the column of ${Wh2} it is dotted with.`;

        Wh2.says`${slab}: number ${col} of node ${row}, brought in from the Transform section.`;
        H.says`${slab}: number ${col} of node ${row}, a weighted sum over the node and its neighbours.`;
        return H;
      });

      const Hc = step('The heads side by side', () => {
        const Hc = merge(H, [K, Fp], KF).as('Hc', 'H', { shape: '(N, K·F′)' });

        say`The ${K.size} slabs of ${H} are laid side by side again, ${Fp.size} columns from each head: one row of
          ${KF.size} numbers per node. Nothing is computed.`;
        aside`The paper also gives a second way, used in its output layer: the heads' results are averaged instead
          of laid side by side.`;

        Hc.says`Column ${col} of node ${row}: the same number as in the slab of its head.`;
        return Hc;
      });

      step('ELU', () => {
        const Ho = map('relu(x) + exp(-relu(-x)) - 1', { x: Hc }, { name: 'elu', show: 'ELU' }).as('Ho', 'H′');

        say`Each number goes through the same rule, one at a time: a positive number is kept; a negative number x
          becomes e^{x} − 1, which lies between −1 and 0. Here ${live(() => Hc.data.filter((x) => x < 0).length)}
          of the ${Hc.size} numbers are negative.

          ${Ho} is what the layer hands on: ${KF.size} numbers per node, each made from the node's own numbers and
          its neighbours', weighted by attention the layer worked out itself.`;
        aside`The paper uses ELU after its first layer. PyTorch has it as one function, F.elu; the formula above is
          the same thing written with ReLU.`;

        Ho.says`Column ${col} of node ${row}: the number at the same place in ${Hc}, or e to that number minus 1 if
          it was negative.`;
      });
    });

    say`Folded up again: the graph gives a mask, each node's numbers are mixed per head, every linked pair gets a
      score, the scores become shares, and each node's new numbers are its neighbours' weighted by those shares.

      Click any box to reopen it.`;
    hint`Click a box to open it. Drag to orbit, Shift-drag (or right-drag) to move sideways, scroll to zoom.`;
  });
