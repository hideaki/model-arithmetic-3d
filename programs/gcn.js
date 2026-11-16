// A graph convolutional network (Kipf and Welling, 2017), written in the
// tensor language.
//
//   A -> A + I -> degrees -> 1 / sqrt -> Â          the graph, scaled once
//   X0 -> X                                         the features, each row divided by its total
//   X -> X W0 -> Â (X W0) -> ReLU                   layer 1
//     -> H1 W1 -> Â (H1 W1) -> softmax              layer 2
//
// Two layers, node classification, the paper's equation 9, in the order of
// its own code (github.com/tkipf/gcn): weights first, then the graph.
// Every tensor has two axes; nothing is drawn as slabs.

import { program, axis, ONE, input, param, random, departure, section, step, say, aside, hint } from '../lang/core.js';
import { again, constant, map, add, sum, transpose, matmul, softmax } from '../lang/ops.js';
import { row, col, value, pct, live } from '../lang/prose.js';

// The example: 8 papers and the links between them (who cites whom, either way) ...
const NODES = 8;
const LINKS = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7]];
// ... and, for every paper, which of 6 words it contains.
const WORDS = 6;
const HAS = [[0, 1], [0, 2], [0, 1, 2], [2, 3], [3, 4], [3, 4, 5], [4, 5], [1, 5]];

const linked = (i, j) => LINKS.some(([a, b]) => (a === i && b === j) || (a === j && b === i));
const ADJACENCY = Array.from({ length: NODES * NODES }, (_, q) => (linked(Math.floor(q / NODES), q % NODES) ? 1 : 0));
const BAG = Array.from({ length: NODES * WORDS }, (_, q) => (HAS[Math.floor(q / WORDS)].includes(q % WORDS) ? 1 : 0));

export const build = () =>
  program('Graph convolutional network, two layers', { seed: 31 }, () => {
    const N = axis('N', NODES);
    const F = axis('F', WORDS);
    const Hd = axis('H', 4);
    const C = axis('C', 3);

    departure('untrained', 'Both weight tables are seeded random numbers, so the class probabilities mean nothing; a trained model holds different numbers in the same places.');
    departure('tiny sizes', '8 nodes, 6 features, 4 hidden numbers and 3 classes, where the paper’s Cora graph has 2,708 nodes with 1,433 features, and the model uses 16 hidden numbers for 7 classes.');
    departure('inference only', 'Training is not shown: no dropout (0.5 in the paper), no loss on the few labelled nodes, no weight decay.');
    departure('dense tables', 'Real code stores the graph as a sparse list of its links and never holds the zeros. Here every cell of the table is a cube.');
    departure('preparation shown as steps', 'Real code scales the graph and the features once, before the model runs, and hands both in ready-made. Here those calculations are steps of the page.');
    departure('no drawing of the graph', 'The graph is shown only as its table of links. The dots and lines one would draw on paper are not on the page.');

    const links = () => LINKS.map(([a, b]) => `${a}–${b}`).join(', ');
    const degree = (t, i) => Math.round(t.data[i]);

    say`Some data is a <b>graph</b>: things, and links between them. Papers that cite one another, people who know
      one another, atoms bonded in a molecule. A graph convolutional network gives every node of a graph a class,
      and it uses two kinds of evidence: what the node itself looks like, and <b>which nodes it is linked to</b>.

      <ul>
        <li><b>The graph</b> is prepared once: a table of links, scaled so that well-connected nodes do not drown out the rest.</li>
        <li><b>The features</b> are one row of numbers per node.</li>
        <li><b>Each layer</b> does two things: it mixes every node's own numbers with a weight table, then replaces
        every node's row by a weighted average over the node and its neighbours.</li>
        <li><b>Two layers</b> in a row, so a node hears from the neighbours of its neighbours, and a softmax turns
        the result into one probability per class.</li>
      </ul>`;
    hint`Press → to start with the graph, or click a box. Drag to orbit, Shift-drag to move sideways, scroll to zoom.`;

    let Ah;
    section('graph', 'Graph', { summary: 'the table of links, scaled once' }, () => {
      const A = step('The graph is a table of links', () => {
        const A = input('A', 'A', [N, N], ADJACENCY);

        say`The model never sees dots and lines, only numbers. The graph here has ${N.size} nodes, numbered from 0,
          and ${live(() => LINKS.length)} links: ${live(links)}. Think of ${N.size} papers, linked where one cites
          the other.

          ${A} holds the same thing as a table with one row and one column per node: <b>1 where two nodes are
          linked, 0 where they are not</b>. A link has no direction, so the table is the same when rows and columns
          are swapped. No node is linked to itself yet: the diagonal is all 0.`;
        aside`This table is called the adjacency matrix. Real code carries no batch axis here: the whole graph is
          one example, and all its nodes go through the model together.`;
        hint`Hover a cube in ${A}: its row and its column are the two nodes it speaks about.`;

        A.says`${live(({ value: v }) => (v ? 'Linked' : 'Not linked'))}: node ${row} and node ${col}.`;
        return A;
      });

      const At = step('Every node is also linked to itself', () => {
        const I = constant([N, N], (i, j) => (i === j ? 1 : 0), { rule: '1 on the diagonal, 0 everywhere else', py: 'torch.eye(N)' }).as('I', 'I');
        const At = add(A, I).as('At', 'Ã');

        say`Later, every node's numbers will be replaced by an average over the nodes it is linked to. With ${A} as
          it is, a node would hear from its neighbours and <b>forget itself</b>.

          So a 1 is added on the diagonal: in ${At} every node is linked to itself as well. ${I} is not learned and
          not computed from the graph, which is why it is drawn as flat tiles with a silver frame.`;
        aside`The paper calls this, together with the scaling that follows, the renormalisation trick.`;

        I.says`${live(({ r, c }) => (r === c ? 'On the diagonal: node ' + r + ' with itself.' : 'Off the diagonal.'))}`;
        At.says`${live(({ r, c, value: v }) => (r === c ? `Node ${r} with itself: the 1 that was added.` : `${v ? 'Linked' : 'Not linked'}: node ${r} and node ${c}, as in the table before.`))}`;
        return At;
      });

      const deg = step('Count each node’s links', () => {
        const deg = sum(At).as('deg', 'd');

        say`Each row of ${At} is added up. The total is the node's <b>degree</b>: the number of nodes it is linked
          to, itself included. Here the smallest is ${live(() => Math.min(...deg.data))} and the largest
          ${live(() => Math.max(...deg.data))}.

          <b>Why count:</b> a node with many links would otherwise collect a larger total from its neighbours than
          a node with few, for no better reason than having more of them.`;
        hint`Hover a cube in ${deg}: the whole row of ${At} it was added up from lights up.`;

        deg.says`Node ${row} is linked to ${live(({ t, r }) => degree(t, r))} nodes, itself included.`;
        return deg;
      });

      const [dis, disT] = step('One over the square root', () => {
        const dis = map('1 / sqrt(d)', { d: deg }).as('dis', 's');
        const disT = transpose(dis).as('disT', 's^{T}');

        say`Each degree is turned into a scale: one divided by its square root. A node with more links gets a
          smaller scale.

          ${disT} is the same column of numbers laid on its side, one per column of the table. Nothing is
          computed there; it is needed in the next step, where every link is scaled once for the node at each
          of its two ends.`;

        dis.says`The scale of node ${row}: 1 / √${live(({ r }) => degree(deg, r))}.`;
        disT.says`The scale of node ${col}, copied from ${dis}.`;
        return [dis, disT];
      });

      Ah = step('Scale every link by both its ends', () => {
        const Ah = map('a * r * c', { a: At, r: dis, c: disT }).as('Ah', 'Â').hint({ arrows: 'all' });

        say`Every cell of ${At} is multiplied by <b>two</b> scales: the one of its row's node and the one of its
          column's node. A link between two nodes of degree d_{i} and d_{j} becomes 1 / √(d_{i} · d_{j}); a
          cell that was 0 stays 0.

          <b>Broadcasting:</b> ${dis} has one column and is repeated across every column of the table; ${disT} has
          one row and is repeated down every row.

          ${Ah} is the graph as the model uses it. It is computed once and used unchanged in both layers. Scaling
          by both ends keeps the table symmetric: a link weighs the same in both directions.`;
        aside`Dividing by the degree alone would give a plain average over the neighbours. The paper's two
          square roots come from the theory of filters on graphs, and are what its own code computes.`;
        hint`Hover a cube in ${Ah}: one cell of ${At} and the two scales light up.`;

        Ah.says`${live(({ r, c, value: v }) => (v ? `How much node ${r} hears from node ${c}` : `Node ${r} and node ${c} are not linked: nothing passes`))}: <b>${value}</b>.`;
        return Ah;
      });
    });

    let X;
    section('features', 'Features', { summary: 'one row of numbers per node' }, () => {
      const X0 = step('Each node comes with a row of numbers', () => {
        const X0 = input('X0', 'X_{0}', [N, F], BAG);

        say`The graph says who is linked to whom. It says nothing about what a node <b>is</b>. That comes
          separately: one row of ${F.size} numbers for each of the ${N.size} nodes.

          In the paper example these are words: a paper's row has a 1 for every word of a fixed list that it
          contains and a 0 for the others. Node 0 here contains
          ${live(() => X0.data.slice(0, F.size).filter((v) => v === 1).length)} of the ${F.size} words.`;
        aside`These rows are made up for this page, like the links. In the paper's Cora graph a row has 1,433
          entries, one per word of the vocabulary, and nearly all of them are 0.`;

        X0.says`Node ${row} ${live(({ value: v }) => (v ? 'contains' : 'does not contain'))} word ${col}.`;
        return X0;
      });

      X = step('Each row is divided by its total', () => {
        const rs = sum(X0).as('rs', 'r');
        const X = map('x / r', { x: X0, r: rs }).as('X', 'X').hint({ arrows: 'all' });

        say`Each row of ${X0} is added up, giving ${rs}: how many words that node has. Every number of the row is
          then divided by it, so that <b>each row of ${X} adds up to 1</b>.

          A node with many words and a node with few now carry the same total weight into the model.`;
        aside`The paper's code does this to the features before the model runs.`;
        hint`Hover a cube in ${X}: its cell of ${X0} and its row's total light up.`;

        rs.says`Node ${row} contains ${live(({ t, r }) => degree(t, r))} of the words.`;
        X.says`Node ${row}, word ${col}: ${live(({ r, c }) => (X0.data[r * F.size + c] ? `1 divided by the node's ${degree(rs, r)} words` : 'the node does not contain it'))}.`;
        return X;
      });
    });

    section('layer1', 'Layer 1', { summary: 'weights first, then the neighbours' }, () => {
      const M1 = step('Every node’s own numbers are mixed', () => {
        const W0 = param('W0', 'W_{0}', [F, Hd], random(0.8));
        const M1 = matmul(X, W0).as('M1', 'M_{1}');

        say`Each node's row of ${F.size} numbers is multiplied with the weight table ${W0} and becomes a row of
          ${Hd.size}. The <b>same table</b> is used for every node, and each node is handled on its own: the
          graph plays no part yet.

          This is the only place in the layer where anything is learned.`;
        hint`Hover a cube in ${M1}: one node's row of ${X} and one column of ${W0} light up.`;

        W0.says`Weight: how much word ${row} contributes to hidden number ${col}, for every node alike.`;
        M1.says`Hidden number ${col} of node ${row}, from its own features only.`;
        return M1;
      });

      const P1 = step('Every node takes in its neighbours', () => {
        const A1 = again(Ah).as('Ah1', 'Â');
        const P1 = matmul(A1, M1).as('P1', 'P_{1}');

        say`Now the graph comes in. ${A1} is the scaled table of links from the Graph section, multiplied from the
          left: every node's new row is a <b>weighted sum of the rows of the nodes it is linked to</b>, itself
          included, with the weights from its row of ${A1}.

          A node that is not linked contributes nothing, because its weight is 0. This is the whole
          “graph convolution”: a node's numbers are blended with those of its neighbours, and with no others.`;
        aside`The paper writes the layer as Â X W. Its code multiplies X with W first and with Â second,
          as here; the result is the same either way.`;
        hint`Hover a cube in ${P1}: a row of ${A1} lights up, with the column of ${M1} it is dotted with. Only the node's neighbours have a weight above 0.`;

        A1.says`${live(({ r, c, value: v }) => (v ? `How much node ${r} hears from node ${c}` : `Node ${r} and node ${c} are not linked: nothing passes`))}: <b>${value}</b>.`;
        P1.says`Hidden number ${col} of node ${row}, blended over the node and its neighbours.`;
        return P1;
      });

      X = step('ReLU: negatives become 0', () => {
        const H1 = map('relu(x)', { x: P1 }, { name: 'relu', show: 'ReLU' }).as('H1', 'H_{1}');

        say`Each number goes through the same rule, one at a time: a positive number is kept, a negative one
          becomes 0. Here ${live(() => P1.data.filter((x) => x < 0).length)} of the ${P1.size} numbers are
          negative.

          <b>Why it is needed:</b> without it the two layers would collapse into a single one, with one weight
          table and the graph applied twice.

          ${H1} is what the first layer hands on: ${Hd.size} numbers per node, each already holding something of
          the node's direct neighbours.`;

        H1.says`Hidden number ${col} of node ${row}: the number at the same place in ${P1}, or 0 if that was negative.`;
        return H1;
      });
    });

    section('layer2', 'Layer 2', { summary: 'the same, down to the classes' }, () => {
      const H1 = X;
      const M2 = step('Mixed again, down to one column per class', () => {
        const W1 = param('W1', 'W_{1}', [Hd, C], random(0.8));
        const M2 = matmul(H1, W1).as('M2', 'M_{2}');

        say`The same two moves again, with a second weight table. ${W1} turns each node's ${Hd.size} hidden
          numbers into ${C.size}: <b>one number per class</b>.`;
        hint`Hover a cube in ${M2}: one node's row of ${H1} and one column of ${W1} light up.`;

        W1.says`Weight: how much hidden number ${row} contributes to the raw score of class ${col}.`;
        M2.says`Raw score of node ${row} for class ${col}, before its neighbours are taken in.`;
        return M2;
      });

      const P2 = step('The neighbours once more', () => {
        const A2 = again(Ah).as('Ah2', 'Â');
        const P2 = matmul(A2, M2).as('P2', 'P_{2}');

        say`The <b>same</b> scaled table of links as in Layer 1 blends the scores over each node's neighbours.

          The rows that are blended here already held something of their own neighbours. So after two layers a
          node's scores depend on nodes up to <b>two links away</b>, and on none further.`;
        aside`More layers would reach further. The paper found two or three to work best; with many layers
          the rows of all nodes drift towards one another.`;
        hint`Hover a cube in ${P2}: a row of ${A2} and a column of ${M2} light up.`;

        A2.says`${live(({ r, c, value: v }) => (v ? `How much node ${r} hears from node ${c}` : `Node ${r} and node ${c} are not linked: nothing passes`))}: <b>${value}</b>.`;
        P2.says`Raw score of node ${row} for class ${col}, blended over the node and its neighbours.`;
        return P2;
      });

      step('Softmax: one probability per class', () => {
        const Z = softmax(P2).as('Z', 'Z');
        const best = (r) => {
          let at = 0;
          for (let c = 1; c < C.size; c++) if (Z.data[r * C.size + c] > Z.data[r * C.size + at]) at = c;
          return at;
        };

        say`Each node's ${C.size} raw scores are turned into probabilities: every score is exponentiated and
          divided by the total of its row, so each row of ${Z} adds up to 1.

          That is the model's answer: for every node at once, how likely it belongs to each class. Node 0 comes
          out as class ${live(() => best(0))}, with ${live(() => Math.round(Z.data[best(0)] * 100) + '%')}.`;
        aside`With random weights these probabilities are arbitrary. In training, only a few nodes have a known
          class; the weights are adjusted until those few come out right, and the links carry the answer to
          the rest.`;

        Z.says`The probability that node ${row} belongs to class ${col}: <b>${pct(value)}</b>.`;
      });
    });

    say`Folded up again: the graph is scaled once, the features go through two layers, and both layers use the
      same scaled graph. Out comes one probability per class for every node.

      Click any box to reopen it.`;
    hint`Click a box to open it. Drag to orbit, Shift-drag (or right-drag) to move sideways, scroll to zoom.`;
  });
