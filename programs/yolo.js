// A YOLOv3-style object detector, written in the tensor language.
//
//   image -> [ conv 3x3 stride 2 -> leaky ReLU ] x 2      the backbone
//         -> conv 1x1                                      one prediction per grid cell
//         -> decode: box centre, box size, objectness, class probabilities, scores
//
// Fully convolutional, one scale, one anchor box per grid cell. Every tensor
// is (channels, height, width): channels are drawn as slabs.

import { program, axis, ONE, input, param, random, departure, section, step, say, aside, hint } from '../lang/core.js';
import { conv2d, channels, constant, map, sigmoid } from '../lang/ops.js';
import { row, col, slab, value, pct, live } from '../lang/prose.js';

const STRIDE = 2; // of each backbone convolution
const SLOPE = 0.1; // of the leaky ReLU below zero
const CLASSES = ['cat', 'dog', 'robot'];
const ANCHOR = [6, 7]; // width and height of the anchor box, in pixels

// A 16 x 16 colour picture: a red square and a blue disc on a dark background.
const SIZE = 16;
const pixel = (c, y, x) => {
  const square = y >= 2 && y <= 7 && x >= 2 && x <= 7;
  const disc = (y - 11) ** 2 + (x - 11) ** 2 <= 10;
  if (square) return [0.9, 0.25, 0.15][c];
  if (disc) return [0.15, 0.35, 0.95][c];
  return 0.08 + 0.04 * ((x + y) % 2);
};
const IMAGE = Array.from({ length: 3 * SIZE * SIZE }, (_, i) => pixel(Math.floor(i / (SIZE * SIZE)), Math.floor(i / SIZE) % SIZE, i % SIZE));

export const build = () =>
  program('YOLO-style detector, one scale', { seed: 23 }, () => {
    const c0 = axis('c_0', 3, { unit: 'channel', deck: true });
    const [H, W] = [axis('H', SIZE), axis('W', SIZE)];
    const c1 = axis('c_1', 4, { unit: 'channel', deck: true });
    const [H1, W1] = [axis('H_1', SIZE / STRIDE), axis('W_1', SIZE / STRIDE)];
    const c2 = axis('c_2', 8, { unit: 'channel', deck: true });
    const S = axis('S', SIZE / STRIDE / STRIDE); // the grid: S x S cells
    const k = axis('k', 3);
    const C = axis('C', CLASSES.length, { unit: 'class', deck: true });
    const n = axis('n', 5 + CLASSES.length, { sym: '5 + C', unit: 'channel', deck: true });
    const two = axis('xy', 2, { sym: '2', unit: 'channel', deck: true });
    const cell = W.size / S.size; // pixels per grid cell
    const seen = 1 + (k.size - 1) + (k.size - 1) * STRIDE; // pixels across that one grid cell depends on

    departure('untrained', 'Every filter is seeded random numbers, so the boxes and scores mean nothing; a trained model holds different numbers in the same places.');
    departure('one scale, one anchor', 'YOLOv3 predicts at three grid sizes with three anchor boxes per cell. Here there is one grid and one anchor.');
    departure('no batch norm or bias', 'Real layers apply batch normalisation after each convolution. At inference that folds into the filters and a bias, and changes no shapes.');
    departure('no suppression step', 'A real detector then drops low scores and removes overlapping duplicates (non-maximum suppression). That last step is a loop that depends on the data and is not shown.');
    departure('tiny sizes', 'A 16 × 16 picture, 3 layers and 3 classes, where YOLOv3 takes 416 × 416, has 53 backbone layers and predicts 80 classes.');

    say`An object detector answers two questions at once: <b>what</b> is in the picture and <b>where</b>.
      YOLO does it in a single pass. The picture is shrunk by convolutions into a small grid, and <b>every grid
      cell predicts one box</b>: where it is, how sure the model is that it holds an object, and what the object is.

      <ul>
        <li><b>Every tensor is (channels, height, width)</b>. Channels are drawn as slabs standing behind one another.</li>
        <li><b>The backbone</b> turns the ${W.size} × ${H.size} picture into an ${S.size} × ${S.size} grid with more channels.</li>
        <li><b>The head</b> turns each grid cell's channels into ${n.size} raw numbers.</li>
        <li><b>Decoding</b> turns those raw numbers into box positions, sizes and scores.</li>
      </ul>`;
    hint`Press → to start with the picture, or click a box. Drag to orbit, Shift-drag to move sideways, scroll to zoom.`;

    const leaky = (x) => map(`max(x, ${SLOPE} * x)`, { x }, { name: 'leaky', show: 'LeakyReLU' });

    let X;
    section('backbone', 'Backbone', { summary: 'two convolutions, each halving the size' }, () => {
      const img = step('The picture is three sheets of numbers', () => {
        const img = input('img', 'image', [c0, H, W], IMAGE).hint({ show: 'image' });

        say`The model never sees a picture, only numbers: one per pixel and colour, between 0 (dark) and 1 (bright).
          A colour picture is three sheets, red, green and blue, each ${H.size} × ${W.size}. The sheets are the
          <b>channels</b>; they stand behind one another as slabs.

          This picture holds a red square and a blue disc. Look at one sheet at a time: the square is bright in the
          red sheet and dark in the blue one.`;
        aside`Real code carries a leading batch axis, (B, 3, H, W); every picture in the batch is treated
          independently, so B is left out here. YOLOv3 takes 416 × 416 pictures.`;
        hint`Use Focus channel (top left) to look at one colour sheet. Hover a cube for its value.`;

        img.says`${slab}: the brightness of this colour at row ${row}, column ${col} of the picture.`;
        return img;
      });

      const F1 = step('Convolution: slide filters over the picture', () => {
        const K1 = param('K1', 'K_{1}', [c1, c0, k, k], random(0.45));
        const F1 = conv2d(img, K1, { stride: STRIDE, pad: 1, to: [H1, W1] }).as('F1', 'F_{1}');

        say`${K1} holds ${c1.size} <b>filters</b>. Each filter is a small block of weights, ${k.size} × ${k.size} cells
          across and one slice deep for every input channel. It is laid over a ${k.size} × ${k.size} window of the
          picture, every weight is multiplied with the number under it, and the products are added up. That sum is
          <b>one number</b> of the output.

          Then the filter slides ${STRIDE} pixels along and does it again. Moving ${STRIDE} at a time is the
          <b>stride</b>: it is why the output is ${H1.size} × ${W1.size}, half the picture's size each way.

          Every filter makes its own output sheet, so ${c1.size} filters give ${c1.size} channels. The same filter
          is used at every position: a pattern is detected the same way wherever it appears.

          <b>Padding:</b> at the edges the window hangs over the side of the picture. The missing cells count as 0.`;
        aside`A trained filter responds to something specific, such as an edge in one direction or a patch of one
          colour. These are random, so each sheet of the output is an arbitrary mix of the picture.`;
        hint`Hover a cube in ${F1}: its window lights up in all three colour sheets, with the one filter that made it.`;

        K1.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of the picture.
          The brackets on the left group the rows by filter.`;
        F1.says`Output of filter ${live(({ d }) => d)} at row ${row}, column ${col}: one window of the picture, across
          all its channels, multiplied with that filter and added up.`;
        return F1;
      });

      const A1 = step('Leaky ReLU: shrink the negatives', () => {
        const A1 = leaky(F1).as('A1', 'A_{1}');

        say`Each number goes through the same simple rule, <b>one at a time</b>: a positive number is kept, a
          negative one is multiplied by ${SLOPE}. Here ${live(() => F1.data.filter((x) => x < 0).length)} of the
          ${F1.size} numbers are negative: watch the blue cubes fade.

          <b>Why it is needed:</b> without a step like this, two convolutions in a row would amount to one
          convolution with a bigger filter, and stacking layers would buy nothing.`;
        aside`Plain ReLU sets negatives to exactly 0. The small slope keeps a little signal flowing for negative
          values, which helps during training.`;

        A1.says`${slab}: the number at the same place in ${F1}, kept if positive and multiplied by ${SLOPE} if negative.`;
        return A1;
      });

      const F2 = step('A second convolution: fewer cells, more channels', () => {
        const K2 = param('K2', 'K_{2}', [c2, c1, k, k], random(0.22));
        const F2 = conv2d(A1, K2, { stride: STRIDE, pad: 1, to: [S, S] }).as('F2', 'F_{2}');

        say`The same operation again, now reading ${A1} instead of the picture. Each of the ${c2.size} filters is
          ${k.size} × ${k.size} across and ${c1.size} slices deep, one slice for each channel it reads, so every output
          number mixes all ${c1.size} channels.

          The stride of ${STRIDE} halves the size again: ${S.size} × ${S.size}. This is the <b>grid</b> the detector
          works on. Each grid cell stands for a ${cell} × ${cell} patch of the picture, and because the windows of the
          two convolutions overlap, its numbers were computed from a ${seen} × ${seen} patch around it.

          That is the trade the backbone makes: fewer positions, more channels. Position is traded for description.`;
        hint`Hover a cube in ${F2}: a window in each of the ${c1.size} channels of ${A1} lights up.`;

        K2.says`A weight of filter ${live(({ r }) => Math.floor(r / k.size))}, in the slice that reads ${slab} of ${A1}.`;
        F2.says`Output of filter ${live(({ d }) => d)} for grid cell (row ${row}, column ${col}).`;
        return F2;
      });

      X = step('Leaky ReLU again', () => {
        const A2 = leaky(F2).as('A2', 'A_{2}');

        say`The same rule as before. ${A2} is what the backbone hands on: for each of the ${S.size * S.size} grid
          cells, ${c2.size} numbers describing the patch of the picture around it.`;
        aside`YOLOv3's backbone, Darknet-53, repeats this pattern for 53 layers, with shortcut additions around
          pairs of layers much like a transformer's residual connections.`;

        A2.says`${slab} for grid cell (row ${row}, column ${col}).`;
        return A2;
      });
    });

    section('head', 'Head', { summary: 'one prediction per grid cell' }, () => {
      X = step('One prediction per grid cell', () => {
        const K3 = param('K3', 'K_{3}', [n, c2, ONE, ONE], random(0.22));
        const Y = conv2d(X, K3, { to: [S, S] }).as('Y', 'Y');

        say`A convolution whose filters are a single cell across. There is no window any more: each grid cell's
          ${c2.size} numbers are combined into ${n.size} new ones, using the same weights for every cell and without
          looking at its neighbours.

          Those ${n.size} numbers are <b>one prediction</b>. Reading down the channels of a single grid cell:

          <ul>
            <li>channels 0 and 1: where the box centre is,</li>
            <li>the next ${two.size}: how wide and tall the box is,</li>
            <li>the next 1: whether there is an object at all,</li>
            <li>the last ${C.size}: one number per class (${CLASSES.join(', ')}).</li>
          </ul>

          So the whole output is ${S.size * S.size} predictions at once, one per grid cell, from a single pass over
          the picture. That is the “you only look once”. These are still raw numbers of any size; the next steps
          turn them into positions, sizes and probabilities.`;
        aside`Nothing in the arithmetic makes channel 0 an x position. It becomes one because of how the numbers
          are decoded next, and because training rewards the decoded box for matching the true one.`;
        hint`Hover a cube in ${Y}: one column of ${c2.size} numbers through the slabs of ${X} lights up.`;

        K3.says`Weight: how much ${slab} of the backbone output contributes to output channel ${row}.`;
        Y.says`Raw output channel ${live(({ d }) => d)} for grid cell (row ${row}, column ${col}).`;
        return Y;
      });
    });

    section('decode', 'Boxes and scores', { summary: 'raw numbers → boxes and scores' }, () => {
      const Y = X;
      const [txy, twh, to, tc] = step('Split the channels by what they mean', () => {
        const txy = channels(Y, 0, two).as('txy', 't_{xy}');
        const twh = channels(Y, 2, two).as('twh', 't_{wh}');
        const to = channels(Y, 4, ONE).as('to', 't_{o}');
        const tc = channels(Y, 5, C).as('tc', 't_{c}');

        say`<b>No arithmetic.</b> The ${n.size} channels of ${Y} are taken apart into four groups, because each group
          is decoded by a different rule: ${txy} for the box centre, ${twh} for its size, ${to} for “is there an
          object?” and ${tc} for the classes.`;

        txy.says`Raw ${live(({ d }) => 'xy'[d])} offset of the box predicted by grid cell (row ${row}, column ${col}), copied from ${Y}.`;
        twh.says`Raw ${live(({ d }) => ['width', 'height'][d])} of the box predicted by grid cell (row ${row}, column ${col}), copied from ${Y}.`;
        to.says`Raw objectness of grid cell (row ${row}, column ${col}), copied from ${Y}.`;
        tc.says`Raw score for ${live(({ d }) => CLASSES[d])} at grid cell (row ${row}, column ${col}), copied from ${Y}.`;
        return [txy, twh, to, tc];
      });

      step('Box centres: stay inside your own cell', () => {
        const cxy = constant([two, S, S], (ch, i, j) => (ch === 0 ? j : i), {
          rule: 'each cell’s own column (channel 0) and row (channel 1) in the grid',
          py: 'torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing="xy"))',
        }).as('cxy', 'c_{xy}');
        const bxy = map('(sigmoid(t) + c) * W / S', { t: txy, c: cxy }).as('bxy', 'b_{xy}');

        say`Each raw number is squashed by the <b>sigmoid</b> into the range 0 to 1, and the grid cell's own position
          ${cxy} is added. So a cell in column j can only place its box centre between column j and column j + 1:
          <b>a cell can only predict a box centred inside itself</b>. Multiplying by ${W} / ${S} = ${cell} turns
          grid units into pixels.

          ${cxy} is not learned and not computed from the picture: it simply holds each cell's column in one channel
          and its row in the other. That is why it is drawn as flat tiles with a silver frame: fixed like a weight,
          but set by a rule instead of learned.`;
        aside`This is the idea that lets one pass work. Each object is the responsibility of the one grid cell
          its centre falls in, so the cells do not compete for it.`;
        hint`Hover a cube in ${bxy}: its raw number and its cell's own position light up.`;

        cxy.says`The ${live(({ d }) => ['column', 'row'][d])} number of this grid cell.`;
        bxy.says`The ${live(({ d }) => 'xy'[d])} position, in pixels, of the centre of the box predicted by grid cell
          (row ${row}, column ${col}).`;
      });

      step('Box sizes: stretch an anchor box', () => {
        const p = constant([two, ONE, ONE], ANCHOR, {
          rule: 'the anchor box’s width (channel 0) and height (channel 1) in pixels, chosen before training',
          py: `torch.tensor([${ANCHOR.join('., ')}.]).view(2, 1, 1)`,
        }).as('anchor', 'p');
        const bwh = map('p * exp(t)', { t: twh, p }).as('bwh', 'b_{wh}');

        say`The model does not predict a width and height from nothing. It starts from an <b>anchor box</b> ${p}, a
          typical box size fixed in advance, and predicts how much to stretch or shrink it: the anchor's width and
          height are multiplied by e raised to the raw numbers.

          A raw number of 0 keeps the anchor as it is; positive numbers enlarge it and negative ones shrink it, and
          the result can never be negative.

          The anchor's two numbers are silver-framed tiles: they were chosen by hand in advance, not learned.`;
        aside`Anchor sizes are picked by clustering the box sizes in the training data. YOLOv3 uses three anchors
          at each of three grid sizes, so each cell predicts three boxes at each scale. Later versions drop anchors
          and predict the distances to the box edges directly.`;
        hint`Hover a cube in ${bwh}: its raw number and the anchor's size light up.`;

        p.says`The anchor box's ${live(({ d }) => ['width', 'height'][d])} in pixels.`;
        bwh.says`The ${live(({ d }) => ['width', 'height'][d])}, in pixels, of the box predicted by grid cell
          (row ${row}, column ${col}): the anchor's, stretched.`;
      });

      const [obj, cls] = step('Objectness and class probabilities', () => {
        const obj = sigmoid(to).as('obj', 'o').hint({ show: 'prob' });
        const cls = sigmoid(tc).as('cls', 'c').hint({ show: 'prob' });

        say`Two more sigmoids turn raw numbers into numbers between 0 and 1.

          <ul>
            <li>${obj}, <b>objectness</b>: how sure the model is that this cell's box holds any object at all.</li>
            <li>${cls}, <b>class probabilities</b>: for each class separately, how likely the object is to be of that
            class. Each class is judged on its own, so they need not add up to 1.</li>
          </ul>`;
        aside`YOLOv3 judges classes independently so that an object can carry two labels at once, such as
          “dog” and “animal”. The earlier YOLOv2 used a softmax, which forces exactly one class.`;

        obj.says`How sure the model is that grid cell (row ${row}, column ${col}) holds an object: <b>${pct(value)}</b>.`;
        cls.says`Given an object at grid cell (row ${row}, column ${col}), how likely it is a ${live(({ d }) => CLASSES[d])}: <b>${pct(value)}</b>.`;
        return [obj, cls];
      });

      step('Scores: is it there, and is it this class?', () => {
        const score = map('o * c', { o: obj, c: cls }).as('score', 'score').hint({ show: 'prob' });
        const best = () => {
          let at = 0;
          for (let i = 1; i < score.size; i++) if (score.data[i] > score.data[at]) at = i;
          const [d, r, c] = score.addr(at);
          return `${CLASSES[d]} at grid cell (row ${r}, column ${c}), with ${Math.round(score.data[at] * 100)}%`;
        };

        say`The final score of a box for a class is objectness times class probability: the box counts only if the
          model thinks <b>something is there</b> and that it <b>is that class</b>.

          <b>Broadcasting:</b> ${obj} has one channel and ${cls} has ${C.size}; the one objectness sheet is multiplied
          into every class sheet.

          The detector's whole answer is now on screen: ${S.size * S.size} boxes, each with a centre, a size and
          ${C.size} scores. The highest score here is for ${live(best)}.`;
        aside`With random filters these boxes and scores are arbitrary. A real detector finishes by keeping only
          boxes whose score clears a threshold and removing near-duplicates of the same object; that step is
          left out here.`;
        hint`Hover a cube in ${score}: the objectness and the class probability it multiplies light up.`;

        score.says`Score for a ${live(({ d }) => CLASSES[d])} in the box of grid cell (row ${row}, column ${col}): <b>${pct(value)}</b>.`;
      });
    });

    say`Folded up again: the picture goes in on the left, and one pass later every grid cell has a box and a score
      for every class.

      Click any box to reopen it.`;
    hint`Click a box to open it. Drag to orbit, Shift-drag (or right-drag) to move sideways, scroll to zoom.`;
  });
