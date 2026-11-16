// The Stable Diffusion-style model (denoising loop, then decoder): its numbers
// against an independent calculation, its sources, its prose, its layout and
// what hovering lights up.

import test from 'node:test';
import assert from 'node:assert/strict';

import { build, VOCAB, TOKENS, EMPTY, STEPS, GUIDANCE, GROUPS, LATENT_SCALE } from '../programs/sd.js';
import { checkProgram } from '../lang/check.js';
import { layout } from '../lang/layout.js';
import { compile } from '../lang/scene.js';

const prog = build();
const data = (id) => Array.from(prog.t(id).data);
const close = (got, want, what) => {
  assert.equal(got.length, want.length, what);
  got.forEach((v, i) => assert.ok(Math.abs(v - want[i]) < 1e-9, `${what}[${i}]: ${v} is not ${want[i]}`));
};

// ---- plain loops, written without the language -----------------------------

function conv(X, [Ci, Hi, Wi], K, [Co, Kh, Kw], stride = 1, pad = 0) {
  const Ho = Math.floor((Hi + 2 * pad - Kh) / stride) + 1;
  const Wo = Math.floor((Wi + 2 * pad - Kw) / stride) + 1;
  const out = [];
  for (let o = 0; o < Co; o++)
    for (let y = 0; y < Ho; y++)
      for (let x = 0; x < Wo; x++) {
        let s = 0;
        for (let c = 0; c < Ci; c++)
          for (let dy = 0; dy < Kh; dy++)
            for (let dx = 0; dx < Kw; dx++) {
              const [iy, ix] = [y * stride + dy - pad, x * stride + dx - pad];
              if (iy >= 0 && iy < Hi && ix >= 0 && ix < Wi) s += X[(c * Hi + iy) * Wi + ix] * K[((o * Ci + c) * Kh + dy) * Kw + dx];
            }
        out.push(s);
      }
  return out;
}
const sig = (x) => 1 / (1 + Math.exp(-x));
const silu = (xs) => xs.map((x) => x * sig(x));
const plus = (a, b) => a.map((v, i) => v + b[i]);
// A (n x k) times B (k x m), both flat.
function mm(A, n, k, B, m) {
  const out = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let x = 0; x < k; x++) s += A[i * k + x] * B[x * m + j];
      out.push(s);
    }
  return out;
}
// Group norm of (C, H, W) with one gain per channel.
function groupNorm(X, C, hw, gain, groups = GROUPS, eps = 1e-5) {
  const per = C / groups;
  const out = new Array(X.length);
  for (let g = 0; g < groups; g++) {
    const cells = X.slice(g * per * hw, (g + 1) * per * hw);
    const mu = cells.reduce((a, b) => a + b, 0) / cells.length;
    const sd = Math.sqrt(cells.reduce((a, b) => a + (b - mu) ** 2, 0) / cells.length + eps);
    cells.forEach((v, i) => {
      const at = g * per * hw + i;
      out[at] = ((v - mu) / sd) * gain[Math.floor(at / hw)];
    });
  }
  return out;
}
// Add one number per channel to every position of that channel.
const addPerChannel = (X, hw, e) => X.map((v, i) => v + e[Math.floor(i / hw)]);

// A ResBlock: norm, SiLU, conv, + time, norm, SiLU, conv, + input (through a 1x1 conv when the widths differ).
function resBlock(X, Cin, [h, w], tau, id) {
  const C = 6;
  const f1 = conv(silu(groupNorm(X, Cin, h * w, data(`s0.${id}1g`))), [Cin, h, w], data(`s0.${id}K1`), [C, 3, 3], 1, 1);
  const g = addPerChannel(f1, h * w, mm(tau, 1, 8, data(`s0.${id}Wt`), C));
  const f2 = conv(silu(groupNorm(g, C, h * w, data(`s0.${id}2g`))), [C, h, w], data(`s0.${id}K2`), [C, 3, 3], 1, 1);
  const skip = Cin === C ? X : conv(X, [Cin, h, w], data(`s0.${id}Ks`), [C, 1, 1]);
  return { f1, g, f2, out: plus(skip, f2) };
}

// The whole U-Net as one function of its three inputs, reading the weights of the first call.
function unet(z, tau, ctx, keep = {}) {
  const [T, dc, C, N] = [4, 6, 6, 16];
  const w = (id) => data(`s0.${id}`);
  const h0 = conv(z, [4, 4, 4], w('Kin'), [C, 3, 3], 1, 1);
  const r1 = resBlock(h0, C, [4, 4], tau, 'h1');
  // Cross-attention.
  const p = conv(groupNorm(r1.out, C, 16, w('ag')), [C, 4, 4], w('Kp'), [C, 1, 1]);
  const x = Array.from({ length: N * C }, (_, i) => p[(i % C) * N + Math.floor(i / C)]);
  const xn = x.map((v, i) => {
    const rowCells = x.slice(i - (i % C), i - (i % C) + C);
    const mu = rowCells.reduce((a, b) => a + b, 0) / C;
    const sd = Math.sqrt(rowCells.reduce((a, b) => a + (b - mu) ** 2, 0) / C + 1e-5);
    return ((v - mu) / sd) * w('xg')[i % C];
  });
  const [Q, K, Vv] = [mm(xn, N, C, w('WQ'), C), mm(ctx, T, dc, w('WK'), C), mm(ctx, T, dc, w('WV'), C)];
  const A = [];
  for (let i = 0; i < N; i++) {
    const scores = Array.from({ length: T }, (_, j) => {
      let s = 0;
      for (let q = 0; q < C; q++) s += Q[i * C + q] * K[j * C + q];
      return s / Math.sqrt(C);
    });
    const ex = scores.map((v) => Math.exp(v - Math.max(...scores)));
    const total = ex.reduce((a, b) => a + b, 0);
    A.push(...ex.map((v) => v / total));
  }
  const y = plus(x, mm(mm(A, N, T, Vv, C), N, C, w('WO'), C));
  const Y = Array.from({ length: C * N }, (_, i) => y[(i % N) * C + Math.floor(i / N)]);
  const h2 = plus(r1.out, conv(Y, [C, 4, 4], w('Kq'), [C, 1, 1]));
  // Down, middle ResBlock, up, skip, merging ResBlock, out.
  const d = conv(h2, [C, 4, 4], w('Kd'), [C, 3, 3], 2, 1);
  const m = resBlock(d, C, [2, 2], tau, 'm').out;
  const u = Array.from({ length: C * 16 }, (_, i) => m[(Math.floor(i / 16) * 2 + Math.floor((Math.floor(i / 4) % 4) / 2)) * 2 + Math.floor((i % 4) / 2)]);
  const cat = [...conv(u, [C, 4, 4], w('Ku'), [C, 3, 3], 1, 1), ...h2];
  const r = resBlock(cat, 2 * C, [4, 4], tau, 'r').out;
  const eps = conv(silu(groupNorm(r, C, 16, w('og'))), [C, 4, 4], w('Kout'), [4, 3, 3], 1, 1);
  Object.assign(keep, { h0, r1, x, xn, A, y, h2, d, m, u, cat, r });
  return eps;
}

// The noise schedule, as diffusers builds it: betas = linspace(sqrt(0.00085), sqrt(0.012), 1000) ** 2.
const alphasCumprod = (() => {
  const out = [];
  let a = 1;
  for (let i = 0; i < 1000; i++) {
    const root = Math.sqrt(0.00085) + ((Math.sqrt(0.012) - Math.sqrt(0.00085)) * i) / 999;
    a *= 1 - root * root;
    out.push(a);
  }
  return out;
})();
const context = (tokens) => tokens.flatMap((t, i) => Array.from({ length: 6 }, (_, j) => data('WE')[VOCAB.indexOf(t) * 6 + j] + data('WP')[i * 6 + j]));

test('every number of the loop equals an independent calculation', () => {
  const ctx = context(TOKENS);
  const ctx0 = context(EMPTY);
  close(data('C'), ctx, 'C');
  close(data('s0.C0'), ctx0, 'C0');
  assert.deepEqual(STEPS, [667, 334, 1]);
  assert.ok(Math.abs(alphasCumprod[999] - 0.00466) < 1e-4 && Math.abs(alphasCumprod[0] - 0.99915) < 1e-12); // the published ends of the schedule

  let z = data('z');
  STEPS.forEach((t, i) => {
    const s = (id) => data(`s${i}.${id}`);
    // Timestep: waves, then two linear layers with SiLU between. The same two matrices at every step.
    const waves = [Math.cos(t), Math.cos(t / 100), Math.sin(t), Math.sin(t / 100)];
    close(s('s'), waves, `waves ${i}`);
    const tau = mm(silu(mm(waves, 1, 4, data('s0.Wt1'), 8)), 1, 8, data('s0.Wt2'), 8);
    close(s('tau'), tau, `tau ${i}`);

    // Two calls of the U-Net: with the prompt and with the empty prompt.
    const inside = {};
    const epsC = unet(z, tau, ctx, inside);
    const epsU = unet(z, tau, ctx0);
    close(i ? s('epsc') : s('out'), epsC, `eps with prompt ${i}`);
    close(s('epsu'), epsU, `eps without prompt ${i}`);
    if (i === 0) {
      // The call that is drawn in full: its stages one by one.
      for (const [id, want] of [['h0', inside.h0], ['h1f1', inside.r1.f1], ['h1g', inside.r1.g], ['h1f2', inside.r1.f2], ['h1', inside.r1.out], ['x', inside.x], ['xn', inside.xn],
        ['A', inside.A], ['y', inside.y], ['h2', inside.h2], ['d', inside.d], ['m', inside.m], ['u', inside.u], ['w', inside.cat], ['r', inside.r]]) close(s(id), want, id);
      for (let q = 0; q < 16; q++) assert.ok(Math.abs(inside.A.slice(q * 4, q * 4 + 4).reduce((a, b) => a + b, 0) - 1) < 1e-12, `row ${q} of A sums to 1`);
    } else {
      // The calls that are not drawn keep their intermediate tensors under a prefix: spot-check them.
      close(s('c.h2'), inside.h2, `undrawn h2 ${i}`);
      close(s('c.out'), epsC, `undrawn eps ${i}`);
    }

    // Guidance, then the DDIM update.
    const diff = epsC.map((v, q) => v - epsU[q]);
    close(s('gd'), diff, `difference ${i}`);
    const eps = epsU.map((v, q) => v + GUIDANCE * diff[q]);
    close(s('eps'), eps, `guided eps ${i}`);
    const [a, aPrev] = [alphasCumprod[t], alphasCumprod[i + 1 < STEPS.length ? STEPS[i + 1] : 0]];
    close([...s('at'), ...s('ap')], [a, aPrev], `schedule ${i}`);
    const x0 = z.map((v, q) => (v - Math.sqrt(1 - a) * eps[q]) / Math.sqrt(a));
    close(s('x0'), x0, `clean guess ${i}`);
    z = x0.map((v, q) => Math.sqrt(aPrev) * v + Math.sqrt(1 - aPrev) * eps[q]);
    close(s('zn'), z, `next latent ${i}`);
  });
  assert.deepEqual(prog.t('s2.zn').dims, prog.t('z').dims); // the finished latent has the shape of the noise it started as
});

test('the decoder: every number equals an independent calculation', () => {
  const w = (id) => data(`D${id}`);
  // A ResBlock without a time vector: norm, SiLU, conv, twice, + input (through a 1x1 conv when the widths differ).
  const block = (X, Cin, C, [h, wd], id) => {
    const f1 = conv(silu(groupNorm(X, Cin, h * wd, w(`${id}1g`))), [Cin, h, wd], w(`${id}K1`), [C, 3, 3], 1, 1);
    const f2 = conv(silu(groupNorm(f1, C, h * wd, w(`${id}2g`))), [C, h, wd], w(`${id}K2`), [C, 3, 3], 1, 1);
    return { f1, f2, out: plus(Cin === C ? X : conv(X, [Cin, h, wd], w(`${id}Ks`), [C, 1, 1]), f2) };
  };
  // Every cell of (C, h, h) copied into a 2 x 2 block.
  const up = (X, C, h) => {
    const out = [];
    for (let ch = 0; ch < C; ch++) for (let y = 0; y < 2 * h; y++) for (let x = 0; x < 2 * h; x++) out.push(X[(ch * h + (y >> 1)) * h + (x >> 1)]);
    return out;
  };
  assert.equal(LATENT_SCALE, 0.18215);
  assert.deepEqual(w('f'), [0.18215]);

  const zs = data('s2.zn').map((v) => v / 0.18215);
  close(w('zs'), zs, 'scaled latent');
  const zq = conv(zs, [4, 4, 4], w('Kz'), [4, 1, 1]);
  close(w('zq'), zq, 'after the 1 x 1 convolution');
  const d0 = conv(zq, [4, 4, 4], w('Kin'), [6, 3, 3], 1, 1);
  close(w('d0'), d0, 'convolution in');
  const r1 = block(d0, 6, 6, [4, 4], 'd1');
  close(w('d1f1'), r1.f1, 'first ResBlock, first convolution');
  close(w('d1'), r1.out, 'first ResBlock');

  // Self-attention over the 16 grid cells: queries, keys and values all from the same rows.
  const [N, C] = [16, 6];
  const n = groupNorm(r1.out, C, N, w('ag'));
  const x = Array.from({ length: N * C }, (_, i) => n[(i % C) * N + Math.floor(i / C)]);
  close(w('x'), x, 'flattened rows');
  const [Q, K, V] = ['WQ', 'WK', 'WV'].map((m) => mm(x, N, C, w(m), C));
  const A = [];
  for (let i = 0; i < N; i++) {
    const scores = Array.from({ length: N }, (_, j) => {
      let sum = 0;
      for (let q = 0; q < C; q++) sum += Q[i * C + q] * K[j * C + q];
      return sum / Math.sqrt(C);
    });
    const ex = scores.map((v) => Math.exp(v - Math.max(...scores)));
    const total = ex.reduce((a, b) => a + b, 0);
    A.push(...ex.map((v) => v / total));
  }
  close(w('A'), A, 'attention weights');
  for (let q = 0; q < N; q++) assert.ok(Math.abs(w('A').slice(q * N, q * N + N).reduce((a, b) => a + b, 0) - 1) < 1e-12, `row ${q} of A sums to 1`);
  const O = mm(mm(A, N, N, V, C), N, C, w('WO'), C);
  const d2 = plus(r1.out, Array.from({ length: C * N }, (_, i) => O[(i % N) * C + Math.floor(i / N)]));
  close(w('d2'), d2, 'after attention');

  // Up, a ResBlock that narrows 6 -> 4, up again, then out.
  const u1 = up(d2, 6, 4);
  close(w('u1'), u1, 'first upsampling');
  const d3 = conv(u1, [6, 8, 8], w('Ku1'), [6, 3, 3], 1, 1);
  close(w('d3'), d3, 'first up convolution');
  const r2 = block(d3, 6, 4, [8, 8], 'd4');
  close(w('d4s'), conv(d3, [6, 8, 8], w('d4Ks'), [4, 1, 1]), 'shortcut');
  close(w('d4'), r2.out, 'second ResBlock');
  const d5 = conv(up(r2.out, 4, 8), [4, 16, 16], w('Ku2'), [4, 3, 3], 1, 1);
  close(w('d5'), d5, 'second up convolution');
  const o = conv(silu(groupNorm(d5, 4, 256, w('og'))), [4, 16, 16], w('Kout'), [3, 3, 3], 1, 1);
  close(w('out'), o, 'convolution out');
  const img = o.map((v) => Math.min(1, Math.max(0, v / 2 + 0.5)));
  close(data('img'), img, 'picture');

  // A picture: three sheets of 16 x 16 brightnesses, all between 0 and 1, and not a flat or saturated one.
  assert.deepEqual(prog.t('img').dims, [3, 16, 16]);
  assert.ok(img.every((v) => v >= 0 && v <= 1));
  assert.ok(img.filter((v) => v > 0 && v < 1).length > img.length / 2);
});

test('declared sources are true for every operation, and the prose has no warnings', () => {
  assert.deepEqual(checkProgram(prog), { checked: 186, problems: [] });
  assert.deepEqual(prog.warnings, []);
  for (const s of prog.steps) assert.ok(s.prose.length, `step "${s.title}" has no prose`);
  for (const t of prog.tensors) assert.ok(t.hover || !t.section, `${t.id} has no hover sentence`);
  // Checking a module's sources perturbs its undrawn tensors; they are put back afterwards.
  assert.deepEqual(data('s1.c.out'), data('s1.epsc'));
});

test('sections: two fixed inputs, three steps, then the decoder', () => {
  const by = Object.fromEntries(prog.sections.map((s) => [s.id, s]));
  const first = ['time0', 'in0', 'res10', 'attn0', 'down0', 'mid0', 'up0', 'res20', 'out0', 'empty0', 'uncond0', 'guide0', 'update0'];
  const later = (i) => ['time', 'cond', 'uncond', 'guide', 'update'].map((id) => id + i);
  const decoder = ['din', 'dres', 'dattn', 'dup1', 'dres2', 'dup2', 'dout'];
  assert.deepEqual(prog.sections.map((s) => s.id), ['text', 'noise', ...first, ...later(1), ...later(2), ...decoder]);
  // Only the U-Net's own boxes, guidance and the sampler step continue the box before them.
  const streams = Object.fromEntries(prog.sections.filter((s) => s.stream).map((s) => [s.id, s.stream.id]));
  assert.deepEqual(streams, {
    res10: 's0.h0', attn0: 's0.h1', down0: 's0.h2', mid0: 's0.d', up0: 's0.m', res20: 's0.w', out0: 's0.r',
    guide0: 's0.epsu', update0: 's0.eps', guide1: 's1.epsu', update1: 's1.eps', guide2: 's2.epsu', update2: 's2.eps',
    // The decoder is one line from the finished latent to the picture.
    din: 's2.zn', dres: 'Dd0', dattn: 'Dd1', dup1: 'Dd2', dres2: 'Dd3', dup2: 'Dd4', dout: 'Dd5',
  });
  // It is a named part, not an instance of the loop, and it is walked in full.
  for (const id of decoder) assert.deepEqual([by[id].part.name, by[id].repeat, by[id].collapsed], ['Decoder', null, false], id);
  assert.ok(prog.sections.filter((s) => !decoder.includes(s.id)).every((s) => !s.part));
  assert.equal(prog.sections.at(-1).keep.id, 'img');
  assert.deepEqual([by.text.input.id, by.noise.input.id, by.empty0.input.id, by.time0.input], ['ids', 'z', 'ids0', null]);
  // Steps 2 and 3 are shown briefly: one tour step per box.
  for (const id of [...later(1), ...later(2)]) assert.deepEqual([by[id].collapsed, by[id].steps.length], [true, 1], id);
  // Each step starts from the latent the step before handed on.
  assert.deepEqual(['s0.zin', 's1.cz', 's2.cz', 's1.zt', 's2.zt'].map((id) => prog.t(id).op.inputs[0].id), ['z', 's0.zn', 's1.zn', 's0.zn', 's1.zn']);
  assert.deepEqual(['z', 's0.zn', 's1.zn', 's2.zn'].map((id) => prog.t(id).label), ['z_{667}', 'z_{334}', 'z_{1}', 'z_{0}']);
});

test('the U-Net is one set of weights, called six times', () => {
  // Drawn again in steps 2 and 3, the time matrices are the very same numbers in the same memory.
  for (const id of ['s1.Wt1', 's2.Wt2']) {
    const t = prog.t(id);
    assert.equal(t.alias, prog.t(id.replace(/^s\d/, 's0')), id);
    assert.equal(t.data, t.alias.data, id);
    assert.ok(t.section, id);
  }
  // Five calls are modules: computed, not drawn. Their weights are the drawn ones, their inputs the three brought-in copies.
  const modules = prog.tensors.filter((t) => t.op?.family === 'module');
  assert.deepEqual(modules.map((t) => t.id), ['s0.epsu', 's1.epsc', 's1.epsu', 's2.epsc', 's2.epsu']);
  for (const t of modules) {
    const hidden = t.op.hidden;
    assert.equal(hidden.filter((h) => h.op).length, 80, t.id);
    assert.ok(hidden.every((h) => !h.section && !prog.tensors.includes(h)), t.id);
    const weights = hidden.filter((h) => h.kind === 'param');
    assert.equal(weights.length, 29, t.id);
    for (const w of weights) assert.equal(w.data, prog.t(`s0.${w.id.split('.').at(-1)}`).data, w.id);
    assert.deepEqual(t.op.inputs.map((x) => x.op.name), ['again', 'again', 'again'], t.id);
    assert.deepEqual(t.op.sources(7).map(([x, cells]) => [x, cells.length]), t.op.inputs.map((x) => [x, x.size]), t.id);
  }
  // No weight is counted twice: 29 U-Net tensors, the two time matrices and the two text tables ...
  const own = prog.tensors.filter((t) => t.kind === 'param' && !t.alias);
  assert.equal(own.filter((t) => !t.id.startsWith('D')).length, 33);
  // ... and the decoder's 20, which belong to no other network: none is shared, none is a time matrix.
  const dec = own.filter((t) => t.id.startsWith('D'));
  assert.equal(dec.length, 20);
  assert.ok(dec.every((t) => t.section.part?.name === 'Decoder' && !/Wt$/.test(t.id)));
  assert.ok(prog.tensors.filter((t) => t.section?.part).every((t) => !t.alias && t.op?.family !== 'module' && t.op?.name !== 'again'));
  // The two guesses of a step differ only through the context.
  assert.deepEqual(prog.t('s1.epsc').op.inputs.map((x) => x.op.inputs[0].id), ['s0.zn', 's1.tau', 'C']);
  assert.deepEqual(prog.t('s1.epsu').op.inputs.map((x) => x.op.inputs[0].id), ['s0.zn', 's1.tau', 's0.C0']);
});

test('layout: nothing overlaps, data flows left to right, inputs enter on side lanes', () => {
  const L = layout(prog);
  const top = (p) => p.cy + (p.R - 1) / 2;
  const bot = (p) => p.cy - (p.R - 1) / 2;
  for (const sec of prog.sections) {
    const S = L.sections.get(sec);
    assert.equal(S.placed.length, sec.tensors.length, sec.id);
    for (const a of S.placed)
      for (const b of S.placed) {
        if (a === b) break;
        const apart =
          a.x0 + a.C <= b.x0 || b.x0 + b.C <= a.x0 || bot(a) > top(b) || bot(b) > top(a) ||
          Math.min(...a.zs) > Math.max(...b.zs) || Math.min(...b.zs) > Math.max(...a.zs);
        assert.ok(apart, `${a.t.id} and ${b.t.id} overlap`);
      }
    // Each section hands on its last tensor on the stream line, at its right edge.
    assert.equal(S.keep.cy, 0, sec.id);
    assert.equal(S.keep.x0 + S.keep.C, Math.max(...S.placed.map((p) => p.x0 + p.C)), sec.id);
  }
  for (const t of prog.tensors) {
    const p = L.at.get(t);
    if (!p || !t.op || p.role !== 'flow') continue;
    for (const o of t.op.inputs) {
      const q = o.section === t.section ? L.at.get(o) : null;
      if (q && q.role !== 'beside' && q.role !== 'over') assert.ok(q.x0 + q.C <= p.x0, `${t.id} is not right of ${o.id}`);
    }
  }
  const at = (id) => L.at.get(prog.t(id));
  // The context comes in below the flow; keys, values and the transposed keys stay down there.
  assert.ok(top(at('s0.Cc')) < bot(at('s0.xn')) && at('s0.Cc').x0 >= at('s0.xn').x0 + at('s0.xn').C);
  assert.deepEqual([at('s0.K').cy, at('s0.KT').cy, at('s0.KT').role], [at('s0.Cc').cy, at('s0.Cc').cy, 'flow']);
  assert.ok(at('s0.V').cy < at('s0.K').cy);
  assert.deepEqual([at('s0.Q').cy, at('s0.S').cy, at('s0.Z').cy], [0, 0, 0]);
  assert.ok(at('s0.S').x0 >= at('s0.KT').x0 + at('s0.KT').C);
  // The time vector does the same in every ResBlock; the saved skip tensor likewise.
  for (const id of ['s0.h1T', 's0.mT', 's0.rT', 's0.sk']) assert.ok(at(id).cy < 0, id);
  // A section that does not continue the one before it stands clear of that section's last tensor.
  assert.deepEqual([at('WE').x0, at('z').x0, at('s0.s').x0, at('s0.zin').x0, at('s0.uz').x0, at('s1.cz').x0], [0, 14, 14, 14, 14, 14]);
  // A module call: the latent on the line, the time vector and the context below it, the answer on the line.
  for (const k of ['s0.u', 's1.c', 's2.u']) {
    assert.deepEqual([at(`${k}z`).cy, at(`s${k[1]}.eps${k.at(-1)}`).cy], [0, 0], k);
    assert.ok(at(`${k}C`).cy < at(`${k}t`).cy && at(`${k}t`).cy < 0, k);
  }
  // The decoder's self-attention: queries on the line, keys above, values below, each clear of the others.
  assert.deepEqual([at('DQ').cy, at('DS').cy, at('DA').cy, at('DZ').cy, at('Dd2').cy], [0, 0, 0, 0, 0]);
  assert.ok(bot(at('DK')) > top(at('DQ')) && top(at('DV')) < bot(at('DQ')));
  assert.deepEqual([at('DKT').cy, at('DKT').role, at('DWK').cy, at('DWV').cy], [at('DK').cy, 'flow', at('DK').cy, at('DV').cy]);
  assert.deepEqual([at('Df').role, at('Df').of.id], ['over', 'Dzs']);
  // Per-channel gains sit over the channels they scale; a single setting sits over the result it enters.
  assert.deepEqual([at('s0.h11g').role, at('s0.h11g').zs], ['over', at('s0.h11n').zs]);
  for (const [k, of] of [['g', 'eps'], ['at', 'x0'], ['ap', 'zn']]) assert.deepEqual([at(`s0.${k}`).role, at(`s0.${k}`).of.id], ['over', `s0.${of}`]);
});

test('the scene: steps, kinds, arrows and hover', () => {
  const scene = compile(prog);
  assert.equal(scene.steps.length, 59);
  assert.equal(scene.tensors.length, 257);
  assert.equal(scene.slabs.colors.length, 12); // one colour per channel of the widest stack
  const kinds = Object.fromEntries(scene.tensors.map((t) => [t.id, t.kind]));
  assert.deepEqual(['s0.s', 's0.Kin', 's0.h11g', 'z', 's0.tau', 's0.A', 's0.out'].map((id) => kinds[id]), ['const', 'param', 'param', 'act', 'act', 'prob', 'act']);
  // The two tables are learned weights: tiles, also when they are brought in again. The rows taken out of them are cubes.
  assert.deepEqual(['WE', 'E', 'WP', 'P', 'C', 's0.WE0', 's0.E0', 's0.P0'].map((id) => kinds[id]), ['param', 'act', 'param', 'act', 'act', 'param', 'act', 'act']);
  // The guidance scale and the schedule's numbers are fixed in advance; shared weights are still weights; a module's answer is computed.
  assert.deepEqual(['s0.g', 's1.at', 's2.ap', 's1.Wt1', 's1.epsc', 's2.zn'].map((id) => kinds[id]), ['const', 'const', 'const', 'param', 'act', 'act']);
  // The decoder: its scaling factor is fixed in advance, its filters are weights, its result is drawn as a picture.
  assert.deepEqual(['Df', 'DKz', 'DWQ', 'DA', 'Dout', 'img'].map((id) => kinds[id]), ['const', 'param', 'param', 'prob', 'act', 'image']);
  for (const [i, s] of scene.steps.entries()) {
    assert.ok(s.title && s.chapter && s.view.ids.length, `step ${i}`);
    assert.equal(typeof s.body(), 'string');
  }
  const sec = (id) => scene.sections.find((s) => s.id === id);
  for (const s of scene.sections)
    for (const f of s.flows) for (const pt of f.pts || [f.at || [f.x, f.y, 0]]) assert.ok(pt.every(Number.isFinite), `${s.id}: ${JSON.stringify(f)}`);
  // Folded: the text box names the tokens going in, the noise box nothing (it is only an input), the U-Net boxes their stream.
  assert.ok(sec('text').flows.some((f) => f.closed && f.text === 'tokens'));
  assert.ok(!sec('noise').flows.some((f) => f.closed && f.type === 'op'));
  assert.equal(sec('attn0').lines[1], '(c, H, W) → (c, H, W)');
  // Open: the values reach Z by an arrow of their own, and both residual arcs of the attention box are drawn at different heights.
  const zStep = scene.steps.findIndex((s) => s.title === 'Mix the values');
  assert.equal(sec('attn0').flows.filter((f) => f.type === 'flow' && f.step === zStep).length, 2);
  assert.ok(scene.steps[zStep].view.ids.includes('s0.V'));
  const arcs = sec('attn0').flows.filter((f) => f.type === 'op' && /residual connection/.test(f.text)).map((f) => f.at[1]);
  assert.equal(arcs.length, 2);
  assert.notEqual(arcs[0], arcs[1]);
  // A module call: one arrow from each of its three inputs, the first one named.
  const call = sec('uncond0').flows.filter((f) => !f.closed);
  assert.equal(call.filter((f) => f.type === 'flow').length, 3);
  assert.deepEqual(call.filter((f) => f.type === 'op').map((f) => f.text), ['U-Net']);
  // One bar over each step, and the overview counts each shared weight once.
  assert.deepEqual(scene.bars.map((b) => [b.label, b.first, b.last]), [['STEP 1', 'time0', 'update0'], ['STEP 2', 'time1', 'update1'], ['STEP 3', 'time2', 'update2'], ['DECODER', 'din', 'dout']]);
  assert.match(scene.steps[0].body(), /257 tensors and 24,414 numbers are drawn, one cube each; 6,024 of them are parameters/);
  assert.match(scene.steps[0].body(), /<li><b>Decoder · To pixels<\/b>: three colours per pixel<\/li>/);

  // The decoder's chapter is named once, on its first box. Its mixing step frames the values, which stand far back.
  const titled = (title) => scene.steps.find((s) => s.title === title);
  assert.deepEqual(scene.steps.slice(45, 49).map((s) => s.chapter), ['Decoder · Convolution in', 'Decoder · Convolution in', 'ResBlock', 'Self-attention']);
  assert.ok(['DA', 'DV', 'DZ', 'Dd2'].every((id) => titled('Mix the values, back to a grid, add back').view.ids.includes(id)));
  // The last step and the closing overview show the picture itself: one square per pixel, coloured by its three numbers.
  const im = data('img');
  const level = (v) => Math.round(255 * v);
  for (const s of [titled('From numbers to a picture'), scene.steps.at(-1)]) {
    const rects = s.body().match(/<svg class="picture"[^>]*>(.*?)<\/svg>/s)[1].match(/<rect [^>]*>/g);
    assert.equal(rects.length, 256);
    for (const i of [0, 37, 255])
      assert.equal(rects[i], `<rect x="${i % 16}" y="${Math.floor(i / 16)}" width="1" height="1" fill="rgb(${level(im[i])},${level(im[256 + i])},${level(im[512 + i])})"/>`);
  }

  // The brief steps say what they repeat, whose weights they use, and exactly what they compute.
  const brief = (title) => scene.steps.find((s) => s.title === title).body();
  assert.match(brief('Step 2 · Timestep'), /this step is at t = 334.*The same 2 steps as Step 1 · Timestep, on this step's inputs and with the same 2 weight tensors/s);
  assert.match(brief('Step 2 · U-Net, with the prompt'), /whole U-Net is run again on these 3 inputs, with the same weights.*Its 80 intermediate tensors \(4,480 numbers\) are computed exactly and not drawn/s);
  assert.match(brief('Step 3 · Guidance'), /The same 2 steps as Step 1 · Guidance, on this step's inputs:/);
  assert.match(brief('Step 3 · Sampler step'), /proportions of t = 0.*the finished latent/s);
  assert.match(scene.steps.find((s) => s.title === 'Guidance: go further the same way').formula, /ε = ε<sub>∅<\/sub> \+ g × Δ/);

  // For every cube of every tensor, what lights up is exactly its sources.
  let cubes = 0;
  for (const spec of scene.tensors) {
    const t = prog.t(spec.id);
    for (let i = 0; i < t.size; i++) {
      const [d, r, c] = t.addr(i);
      const lit = new Set();
      for (const g of scene.explain(spec.id, d, r, c).groups) for (let rr = g.r0; rr <= g.r1; rr++) for (let cc = g.c0; cc <= g.c1; cc++) lit.add(`${g.id}:${g.d}:${rr}:${cc}`);
      const want = new Set();
      for (const [src, cells] of prog.sources(t, i)) for (const f of cells) want.add(`${src.id}:${src.addr(f).join(':')}`);
      assert.deepEqual([...lit].sort(), [...want].sort(), `${spec.id}[${d}, ${r}, ${c}]`);
      cubes++;
    }
  }
  assert.equal(cubes, 24414);

  // A group mean lights its whole group: 3 channels, every position. A score lights a query row and a key column.
  const mu = scene.explain('s0.h11mu', 4, 0, 0);
  assert.deepEqual(mu.groups.map((g) => [g.id, g.d, g.r0, g.r1, g.c0, g.c1]), [3, 4, 5].map((d) => ['s0.h0', d, 0, 3, 0, 3]));
  assert.match(mu.html, /The average of all 48 numbers in channels 3 to 5 of h_0/);
  assert.deepEqual(scene.explain('s0.S', 0, 5, 2).groups.map((g) => [g.id, g.r0, g.r1, g.c0, g.c1]), [['s0.Q', 5, 5, 0, 5], ['s0.KT', 0, 5, 2, 2]]);
  assert.match(scene.explain('s0.A', 0, 5, 2).html, /Grid cell <span class="tok">1,1<\/span> gives <b>\d+%<\/b> of its attention to <span class="tok">square<\/span>/);
  // The upsampled grid: four cubes share each source. The concatenated stack: front from the up path, back from the skip.
  assert.deepEqual(scene.explain('s0.u', 2, 3, 1).groups.map((g) => [g.id, g.d, g.r0, g.c0]), [['s0.m', 2, 1, 0]]);
  assert.equal(scene.explain('s0.w', 2, 1, 1).groups[0].id, 's0.v');
  assert.deepEqual(scene.explain('s0.w', 8, 1, 1).groups.map((g) => [g.id, g.d]), [['s0.sk', 2]]);
  // Brought-in copies light their original; the noise, the waves and the weights light nothing, and say so.
  assert.deepEqual(scene.explain('s0.Cc', 0, 2, 3).groups.map((g) => [g.id, g.r0, g.c0]), [['C', 2, 3]]);
  for (const id of ['z', 's0.s', 's0.Kin', 's1.Wt1', 's0.g']) {
    const ex = scene.explain(id, 0, 0, 0);
    assert.deepEqual(ex.groups, [], id);
    assert.ok(ex.html.includes('Nothing else lights up:'), id);
  }
  assert.match(scene.explain('s1.Wt1', 0, 2, 3).html, /The very same weight as W_1\[2, 3\] where W_1 first appears/);
  // A module's answer lights all three of its inputs, completely: 4 channels of the latent, the time vector, the context.
  const call2 = scene.explain('s1.epsu', 2, 1, 3);
  assert.deepEqual(call2.groups.map((g) => [g.id, g.d, g.r0, g.r1, g.c0, g.c1]), [...[0, 1, 2, 3].map((d) => ['s1.uz', d, 0, 3, 0, 3]), ['s1.ut', 0, 0, 0, 0, 7], ['s1.uC', 0, 0, 3, 0, 5]]);
  assert.match(call2.html, /running the whole U-Net again, with the same weights, on z_334, τ, C_∅\. All 96 of their numbers/);
  // The decoder. A pixel's brightness lights its one raw number; an upsampled cell its one source; a self-attention
  // score a query row and a key column of the same 16 grid cells; the scaled latent its number and the one factor.
  assert.deepEqual(scene.explain('img', 2, 5, 9).groups.map((g) => [g.id, g.d, g.r0, g.r1, g.c0, g.c1]), [['Dout', 2, 5, 5, 9, 9]]);
  assert.deepEqual(scene.explain('Du2', 3, 13, 6).groups.map((g) => [g.id, g.d, g.r0, g.c0]), [['Dd4', 3, 6, 3]]);
  assert.deepEqual(scene.explain('DS', 0, 5, 11).groups.map((g) => [g.id, g.r0, g.r1, g.c0, g.c1]), [['DQ', 5, 5, 0, 5], ['DKT', 0, 5, 11, 11]]);
  assert.match(scene.explain('DA', 0, 5, 11).html, /Grid cell <span class="tok">1,1<\/span> gives <b>\d+%<\/b> of its attention to grid cell <span class="tok">2,3<\/span>/);
  assert.deepEqual(scene.explain('Dzs', 1, 2, 3).groups.map((g) => g.id), ['s2.zn', 'Df']);
  assert.deepEqual(scene.explain('Dout', 0, 0, 0).groups.filter((g) => g.id === 'Doa').length, 4); // a corner window: 2 x 2 cells in each of 4 channels
  // The sampler's formula shows its schedule number to four digits, and lights the latent, the noise guess and that number.
  const x0 = scene.explain('s2.x0', 1, 2, 2);
  assert.deepEqual(x0.groups.map((g) => g.id), ['s2.eps', 's2.zt', 's2.at']);
  assert.match(x0.html, /√\(1 − 0\.9983\)/);
});
