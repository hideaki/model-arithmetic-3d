// A tiny but complete GPT-style transformer with two blocks, computed exactly.
// Every number the 3D scene shows comes from compute() below.
//
//   tokens -> embedding lookup + position
//          -> NB x { [ LayerNorm -> attention -> add ] -> [ LayerNorm -> MLP -> add ] }
//          -> LayerNorm -> unembedding -> softmax
//
// Architecture: GPT-2's layout (pre-LayerNorm, learned position embeddings,
// GELU, output weights tied to the embedding table) with biases switched off,
// as in nanoGPT's `bias=False` setting. Biases add a vector after each matrix
// multiply and change no shapes.
//
// The model is untrained: every table and weight matrix is seeded random
// numbers. Nothing here is arranged to be readable; a trained model would hold
// different numbers in the same places and run the same arithmetic.
//
// Convention: row vectors. A token is a row, and a linear map is `X @ W` with
// W of shape (in, out). (PyTorch's nn.Linear stores the transpose.)

export const VOCAB = ['.', 'The', 'a', 'and', 'cat', 'dog', 'happy', 'it', 'mat', 'on', 'ran', 'robot', 'said', 'sat', 'tired', 'was'];
export const TOKENS = ['The', 'robot', 'said', 'it', 'was', 'tired'];
export const IDS = TOKENS.map((t) => VOCAB.indexOf(t));

export const V = VOCAB.length; // vocabulary size
export const T = TOKENS.length; // sequence length
export const TMAX = 8; // context length: the most positions the model can take
export const D = 12; // d_model
export const H = 3; // number of heads
export const DK = D / H; // d_k = d_model / h
export const DFF = 4 * D; // MLP hidden width
export const NB = 2; // number of blocks
export const EPS = 1e-5; // LayerNorm epsilon

// Heads have no assigned roles; the colours only tell the slabs apart.
export const HEAD_INFO = [
  { name: 'Head 0', color: '#b79cff' },
  { name: 'Head 1', color: '#3fd6c0' },
  { name: 'Head 2', color: '#ff7d9c' },
];

// Standard deviations of the random numbers. They are chosen so the cubes
// show contrast; a freshly initialised real model uses much smaller weights
// (around 0.02), which would make every attention row nearly uniform.
const STD = { tokenEmb: 0.8, posEmb: 0.5, qk: 0.34, v: 0.35, o: 0.3, w1: 0.3, w2: 0.2, gamma: 0.2 };

const range = (n) => Array.from({ length: n }, (_, i) => i);
const zeros = (r, c) => range(r).map(() => new Array(c).fill(0));

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function matmul(A, B) {
  const n = A.length;
  const k = B.length;
  const m = B[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let x = 0; x < k; x++) s += A[i][x] * B[x][j];
      out[i][j] = s;
    }
  return out;
}

export const transpose = (A) => A[0].map((_, c) => A.map((row) => row[c]));
const add = (A, B) => A.map((row, i) => row.map((v, c) => v + B[i][c]));

export function softmax(row) {
  const mx = Math.max(...row);
  const ex = row.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}

// erf to ~1e-7 (Abramowitz & Stegun 7.1.26), for the exact GELU.
function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
// Standard normal CDF, and GELU(x) = x * Phi(x).
export const phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
export const gelu = (x) => x * phi(x);

// LayerNorm over the last axis: each row is shifted to mean 0, scaled to
// variance 1, then multiplied by the learned gain gamma (one number per column).
export function layerNorm(X, gamma) {
  const mu = X.map((row) => [row.reduce((a, b) => a + b, 0) / row.length]);
  const sd = X.map((row, i) => [Math.sqrt(row.reduce((a, b) => a + (b - mu[i][0]) ** 2, 0) / row.length + EPS)]);
  const out = X.map((row, i) => row.map((v, c) => ((v - mu[i][0]) / sd[i][0]) * gamma[c]));
  return { mu, sd, out };
}

function buildParams(seed = 11) {
  const rnd = mulberry32(seed);
  const gauss = (s) => {
    let u = 0;
    while (u === 0) u = rnd();
    return s * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  };
  const randn = (r, c, s) => range(r).map(() => range(c).map(() => gauss(s)));
  // A real model initialises LayerNorm gains at exactly 1; training moves them a little.
  const gain = () => range(D).map(() => 1 + gauss(STD.gamma));

  return {
    WE: randn(V, D, STD.tokenEmb), // token embedding table, one row per vocabulary word
    WP: randn(TMAX, D, STD.posEmb), // position embedding table, one row per position
    // Every block has the same shapes and its own numbers.
    blocks: range(NB).map(() => ({
      gA: gain(),
      WQ: randn(D, D, STD.qk),
      WK: randn(D, D, STD.qk),
      WV: randn(D, D, STD.v),
      WO: randn(D, D, STD.o),
      gM: gain(),
      W1: randn(D, DFF, STD.w1),
      W2: randn(DFF, D, STD.w2),
    })),
    gF: gain(),
  };
}

export const PARAMS = buildParams();

// One transformer block: (T, d_model) in, (T, d_model) out.
function block(X, p, M) {
  // ---- Part 1: attention ----
  const lnA = layerNorm(X, p.gA);
  const NA = lnA.out;

  // Linear projections: (T, d_model) @ (d_model, d_model) -> (T, d_model)
  const Q = matmul(NA, p.WQ);
  const K = matmul(NA, p.WK);
  const Vv = matmul(NA, p.WV);

  // Split heads: (T, d_model) -> (h, T, d_k)
  const split = (m) => range(H).map((h) => m.map((row) => row.slice(h * DK, (h + 1) * DK)));
  const Qh = split(Q);
  const Kh = split(K);
  const Vh = split(Vv);

  // Scores: (h, T, d_k) @ (h, d_k, T) -> (h, T, T), scaled by 1 / sqrt(d_k)
  const KT = Kh.map(transpose);
  const S = Qh.map((q, h) => matmul(q, KT[h]));
  const scale = Math.sqrt(DK);
  const Ss = S.map((s) => s.map((row) => row.map((v) => v / scale)));

  // Causal mask (T, T), broadcast over heads; softmax over the last axis
  const Sm = Ss.map((s) => s.map((row, i) => row.map((v, j) => v + M[i][j])));
  const A = Sm.map((s) => s.map(softmax));

  // Weighted sum of values, concatenate heads, output projection
  const Z = A.map((att, h) => matmul(att, Vh[h]));
  const Zc = range(T).map((i) => [].concat(...range(H).map((h) => Z[h][i])));
  const O = matmul(Zc, p.WO);

  // Residual: added to the stream as it was before LayerNorm
  const Xa = add(X, O);

  // ---- Part 2: MLP ----
  const lnM = layerNorm(Xa, p.gM);
  const NM = lnM.out;
  const Hh = matmul(NM, p.W1); // (T, 4 d_model)
  const G = Hh.map((row) => row.map(gelu));
  const F = matmul(G, p.W2); // (T, d_model)
  const Xm = add(Xa, F);

  return {
    in: X,
    gA: p.gA, muA: lnA.mu, sdA: lnA.sd, NA,
    WQ: p.WQ, WK: p.WK, WV: p.WV, WO: p.WO,
    Q, K, V: Vv, Qh, Kh, Vh, KT, S, Ss, Sm, A, Z, Zc, O, Xa,
    gM: p.gM, muM: lnM.mu, sdM: lnM.sd, NM,
    W1: p.W1, W2: p.W2, H: Hh, G, F, Xm,
  };
}

export function compute({ causal = true } = {}) {
  const p = PARAMS;

  // ---- Embedding: (T,) ids -> (T, d_model) ----
  const E = IDS.map((id) => p.WE[id].slice()); // look up one row per token
  const P = range(T).map((i) => p.WP[i].slice()); // first T rows of the position table
  const X0 = add(E, P);

  // ---- Blocks: each reads the stream the previous one left ----
  const M = range(T).map((i) => range(T).map((j) => (causal && j > i ? -Infinity : 0)));
  const b = [];
  let X = X0;
  for (let i = 0; i < NB; i++) {
    b.push(block(X, p.blocks[i], M));
    X = b[i].Xm;
  }

  // ---- Output ----
  const lnF = layerNorm(X, p.gF);
  const NF = lnF.out;
  const WET = transpose(p.WE); // tied weights: the embedding table, transposed
  const L = matmul(NF, WET); // logits (T, V)
  const Pr = L.map(softmax);

  return {
    causal, scale: Math.sqrt(DK),
    WE: p.WE, WP: p.WP, E, P, X0,
    M, b,
    Xlast: X, gF: p.gF, muF: lnF.mu, sdF: lnF.sd, NF,
    WET, L, Pr,
  };
}
