// Small arithmetic expressions for map(): numbers, names, + - * / ^, unary
// minus and function calls. One parsed formula drives both the arithmetic and
// the explanation shown to the viewer.

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

export const FUNCS = {
  sqrt: Math.sqrt, exp: Math.exp, log: Math.log, tanh: Math.tanh, abs: Math.abs,
  relu: (x) => Math.max(0, x), sigmoid: (x) => 1 / (1 + Math.exp(-x)), max: Math.max, min: Math.min, phi, gelu,
};

const TOKEN = /\s*(?:(\d+(?:\.\d+)?(?:e[-+]?\d+)?)|([\p{L}_][\p{L}\p{N}_]*)|([-+*/^(),]))/iuy;

export function parse(src) {
  const toks = [];
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < src.length) {
    const at = TOKEN.lastIndex;
    const m = TOKEN.exec(src);
    if (!m) {
      if (!src.slice(at).trim()) break;
      throw new Error(`formula "${src}": cannot read "${src.slice(at).trim()}"`);
    }
    toks.push(m[1] !== undefined ? { t: 'num', v: Number(m[1]) } : m[2] !== undefined ? { t: 'id', v: m[2] } : { t: m[3] });
  }
  let i = 0;
  const peek = () => toks[i]?.t;
  const eat = (t) => {
    if (peek() !== t) throw new Error(`formula "${src}": expected "${t}"`);
    return toks[i++];
  };
  const sum = () => {
    let a = product();
    while (peek() === '+' || peek() === '-') a = { t: 'bin', op: toks[i++].t, a, b: product() };
    return a;
  };
  const product = () => {
    let a = unary();
    while (peek() === '*' || peek() === '/') a = { t: 'bin', op: toks[i++].t, a, b: unary() };
    return a;
  };
  const unary = () => (peek() === '-' ? (i++, { t: 'neg', a: unary() }) : power());
  const power = () => {
    const a = atom();
    return peek() === '^' ? (i++, { t: 'bin', op: '^', a, b: unary() }) : a;
  };
  const atom = () => {
    if (peek() === 'num') return { t: 'num', v: toks[i++].v };
    if (peek() === '(') {
      i++;
      const a = sum();
      eat(')');
      return { t: 'group', a };
    }
    const name = eat('id').v;
    if (peek() !== '(') return { t: 'var', name };
    i++;
    const args = [sum()];
    while (peek() === ',') (i++, args.push(sum()));
    eat(')');
    return { t: 'call', name, args };
  };
  const ast = sum();
  if (i < toks.length) throw new Error(`formula "${src}": unexpected "${toks[i].v ?? toks[i].t}"`);
  return ast;
}

// Names the formula reads.
export function names(ast, out = new Set()) {
  if (ast.t === 'var') out.add(ast.name);
  for (const k of ['a', 'b']) if (ast[k]) names(ast[k], out);
  if (ast.args) ast.args.forEach((a) => names(a, out));
  return out;
}

// Turn the formula into a function of an array holding one value per operand.
export function compile(ast, operands, consts = {}) {
  const go = (n) => {
    switch (n.t) {
      case 'num': {
        const v = n.v;
        return () => v;
      }
      case 'group':
        return go(n.a);
      case 'var': {
        const k = operands.indexOf(n.name);
        if (k >= 0) return (x) => x[k];
        if (!(n.name in consts)) throw new Error(`formula uses "${n.name}", which is neither an operand nor a known constant`);
        const v = consts[n.name];
        return () => v;
      }
      case 'neg': {
        const a = go(n.a);
        return (x) => -a(x);
      }
      case 'call': {
        const f = FUNCS[n.name];
        if (!f) throw new Error(`formula calls "${n.name}", which is not a known function (${Object.keys(FUNCS).join(', ')})`);
        const args = n.args.map(go);
        return args.length === 1 ? (x) => f(args[0](x)) : (x) => f(...args.map((g) => g(x)));
      }
      default: {
        const a = go(n.a);
        const b = go(n.b);
        if (n.op === '+') return (x) => a(x) + b(x);
        if (n.op === '-') return (x) => a(x) - b(x);
        if (n.op === '*') return (x) => a(x) * b(x);
        if (n.op === '/') return (x) => a(x) / b(x);
        return (x) => a(x) ** b(x);
      }
    }
  };
  return go(ast);
}

// Write the formula out for people. `sub(name)` may return text to put in
// place of a name (a number, say); otherwise the name is kept.
const SIGN = { '+': ' + ', '-': ' − ', '*': ' × ', '/': ' / ', '^': '^' };
export function render(ast, sub = () => undefined) {
  const go = (n) => {
    switch (n.t) {
      case 'num': return String(n.v);
      case 'group': return `(${go(n.a)})`;
      case 'var': return sub(n.name) ?? n.name;
      case 'neg': return `−${go(n.a)}`;
      case 'call': {
        if (n.name !== 'sqrt') return `${n.name}(${n.args.map(go).join(', ')})`;
        const a = n.args[0];
        return a.t === 'num' || a.t === 'var' || a.t === 'group' ? `√${go(a)}` : `√(${go(a)})`;
      }
      default: return `${go(n.a)}${SIGN[n.op]}${go(n.b)}`;
    }
  };
  return go(ast);
}
