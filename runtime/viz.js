// The 3D runtime: draws a compiled scene (see lang/scene.js). Every tensor cell
// is one instanced cube, coloured by its value. Nothing here knows which model
// is being shown.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

function fmt(v, n = 2) {
  if (!Number.isFinite(v)) return v < 0 ? '−∞' : '∞';
  let s = v.toFixed(n);
  if (parseFloat(s) === 0) s = (0).toFixed(n);
  return s.replace('-', '−');
}

// The section a step belongs to (-1: none, everything is folded).
const sectionOfStep = (secs, step) => secs.findIndex((s) => step >= s.from && step <= s.to);
// Sections open at a step: its own, plus the one just finished, which stays
// open for one more step so that it folds behind the camera.
const openSections = (secs, step) => {
  const cur = sectionOfStep(secs, step);
  if (cur < 0) return [];
  return cur > 0 && step === secs[cur - 1].to + 1 ? [cur - 1, cur] : [cur];
};

const SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
const PX = 64; // label canvas pixels per world unit

const COL = {
  bg: 0x0b0e14,
  zero: new THREE.Color('#2a3142'),
  pos: new THREE.Color('#ff9838'),
  neg: new THREE.Color('#2f8cff'),
  att0: new THREE.Color('#232a38'),
  att1: new THREE.Color('#d4f95a'),
  blocked: new THREE.Color('#5a2233'),
  maskOpen: new THREE.Color('#4a556e'),
  frameAct: 0x66748f,
  frameParam: 0xe0aa45,
  frameFixed: 0xa9c4e6,
  flow: 0x7a879f,
  skip: 0xffb454,
  text: '#e8edf7',
  textDim: '#9aa7bf',
  param: '#f0c36a',
  fixed: '#b7cdea',
};

// Three kinds of numbers. Thickness says whether a number depends on the
// input: cubes are computed from it, flat tiles are fixed. The frame of a tile
// says where its numbers come from: gold is learned, silver is set by a rule
// or by hand (a constant, a mask).
const isFixed = (kind) => kind === 'const' || kind === 'mask';
const isFlat = (kind) => kind === 'param' || isFixed(kind);
const frameOf = (kind) => (kind === 'param' ? COL.frameParam : COL.frameFixed);

const UP = new THREE.Vector3(0, 1, 0);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const rad = THREE.MathUtils.degToRad;

// ---------------------------------------------------------------------------
// Text labels (canvas sprites). Markup: _{sub} and ^{sup}.
// ---------------------------------------------------------------------------

function parseMarkup(s) {
  const out = [];
  const re = /([_^])\{([^}]*)\}/g;
  let last = 0;
  let m;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push({ t: s.slice(last, m.index), m: 'n' });
    out.push({ t: m[2], m: m[1] === '_' ? 'sub' : 'sup' });
    last = re.lastIndex;
  }
  if (last < s.length) out.push({ t: s.slice(last), m: 'n' });
  return out;
}

const texCache = new Map();

function textTexture(lines, align) {
  const key = JSON.stringify([lines, align]);
  if (texCache.has(key)) return texCache.get(key);
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d');
  const fontOf = (ln, k) => `${ln.weight || 500} ${Math.round(ln.size * PX * k)}px ${ln.mono ? MONO : SANS}`;
  const rows = lines.map((ln) => {
    const segs = parseMarkup(ln.text);
    let w = 0;
    for (const s of segs) {
      ctx.font = fontOf(ln, s.m === 'n' ? 1 : 0.68);
      s.w = ctx.measureText(s.t).width;
      w += s.w;
    }
    return { ln, segs, w, h: ln.size * PX * 1.32 };
  });
  const pad = 6;
  const W = Math.ceil(Math.max(...rows.map((r) => r.w)) + pad * 2);
  const Ht = Math.ceil(rows.reduce((a, r) => a + r.h, 0) + pad * 2);
  cv.width = W;
  cv.height = Ht;
  let y = pad;
  for (const row of rows) {
    let x = align === 'center' ? (W - row.w) / 2 : align === 'right' ? W - pad - row.w : pad;
    const base = y + row.h * 0.74;
    ctx.fillStyle = row.ln.color || COL.text;
    ctx.textBaseline = 'alphabetic';
    for (const s of row.segs) {
      ctx.font = fontOf(row.ln, s.m === 'n' ? 1 : 0.68);
      const dy = s.m === 'sub' ? row.ln.size * PX * 0.2 : s.m === 'sup' ? -row.ln.size * PX * 0.36 : 0;
      ctx.fillText(s.t, x, base + dy);
      x += s.w;
    }
    y += row.h;
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const out = { tex, w: W / PX, h: Ht / PX };
  texCache.set(key, out);
  return out;
}

function makeLabel(lines, { align = 'center', anchor = [0.5, 0.5], rotation = 0 } = {}) {
  if (typeof lines === 'string') lines = [{ text: lines, size: 0.5 }];
  const { tex, w, h } = textTexture(lines, align);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, rotation });
  const sp = new THREE.Sprite(mat);
  sp.scale.set(w, h, 1);
  sp.center.set(anchor[0], anchor[1]);
  return sp;
}

// A small sprite whose text can be rewritten (used for values on highlighted cubes).
class NumberSprite {
  constructor() {
    this.cv = document.createElement('canvas');
    this.cv.width = 160;
    this.cv.height = 80;
    this.ctx = this.cv.getContext('2d');
    this.tex = new THREE.CanvasTexture(this.cv);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: this.tex, transparent: true, depthTest: false, depthWrite: false }),
    );
    this.sprite.scale.set(0.84, 0.42, 1);
    this.sprite.renderOrder = 20;
    this.sprite.visible = false;
    this.text = null;
  }
  set(text) {
    if (text === this.text) return;
    this.text = text;
    const c = this.ctx;
    c.clearRect(0, 0, 160, 80);
    c.font = `700 ${text.length > 5 ? 38 : 44}px ${MONO}`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.lineJoin = 'round';
    c.lineWidth = 9;
    c.strokeStyle = 'rgba(6, 9, 15, 0.9)';
    c.strokeText(text, 80, 42);
    c.fillStyle = '#ffffff';
    c.fillText(text, 80, 42);
    this.tex.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------

// Seconds. Reveals are slow on purpose, so the eye can follow where numbers come from.
const TIME = { appear: 3.0, hide: 0.7, camera: 1.8, section: 1.8 };
// When several tensors appear in one step they start one after another, left to
// right; LAG is the share of the reveal spent on those staggered starts.
const LAG = 0.45;
// When the tour enters another section, that section's box has to clear first:
// reveals wait for this share of LAG, and the whole reveal is stretched to make room.
const FOLD_WAIT = 0.25;
const FOLD_STRETCH = 1.15;
const lerp = (a, b, t) => a + (b - a) * t;

export class Viz {
  constructor(container, scene, hooks = {}) {
    this.container = container;
    this.sc = scene;
    this.hooks = hooks;
    this.step = 0;
    this.focusHead = -1;
    this.nSlabs = Math.max(1, scene.slabs.n);
    this.slabColors = scene.slabs.colors;
    this.headF = new Float32Array(this.nSlabs).fill(1);

    this.hoverId = -1;
    this.pinnedId = -1;
    this.hoverEx = null;
    this.pinnedEx = null;
    this.hoverSec = -1;
    this.hl = 0;
    this.hlTarget = 0;
    this.hlBoxes = [];
    this.hlIds = [];
    this.tween = null;
    this.dragging = false;
    this.pointerInside = false;
    this.pointerDirty = false;
    this.ndc = new THREE.Vector2();
    this.appearDur = TIME.appear;
    // The start of one section is pinned in space; the others are laid out
    // around it. Pinning the boundary between the section being left and the
    // one being entered keeps what the viewer is looking at from moving.
    this.anchor = { sec: 0, x: 0 };
    // Sections waiting to fold until the camera has moved away, and the wait left.
    this.deferred = [];
    this.foldTimer = 0;

    this.initRenderer();
    this.buildSections();
    this.buildTensors();
    this.refreshData();
    this.buildDecor();
    this.buildFlows();
    this.buildHighlight();
    this.bindEvents();
    this.resize();
    this.layout(0);

    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  // ---- setup -------------------------------------------------------------

  initRenderer() {
    const r = new THREE.WebGLRenderer({ antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.container.appendChild(r.domElement);
    this.renderer = r;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COL.bg);

    this.camera = new THREE.PerspectiveCamera(28, 1, 0.5, 14000);
    this.camera.position.set(6, 3, 40);
    this.scene.add(this.camera);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x55607a, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(0.7, 1.0, 0.4);
    sun.target.position.set(0, 0, -1);
    this.camera.add(sun);
    this.camera.add(sun.target);

    const c = new OrbitControls(this.camera, r.domElement);
    c.enableDamping = true;
    c.dampingFactor = 0.12;
    c.minDistance = 4;
    c.maxDistance = 5000;
    c.zoomSpeed = 0.9;
    c.addEventListener('start', () => {
      this.tween = null;
      this.dragging = true;
    });
    c.addEventListener('end', () => {
      this.dragging = false;
      this.pointerDirty = true;
    });
    c.addEventListener('change', () => {
      this.pointerDirty = true;
    });
    this.controls = c;

    this.raycaster = new THREE.Raycaster();
  }

  // One group per section. It slides along x as sections open and close.
  // openGroup holds the detailed arrows and labels, closedGroup the box view.
  buildSections() {
    this.secs = this.sc.sections.map((s, i) => {
      const group = new THREE.Group();
      const openGroup = new THREE.Group();
      const closedGroup = new THREE.Group();
      group.add(openGroup, closedGroup);
      this.scene.add(group);
      return { ...s, i, p: 0, open: 0, target: 0, x: 0, group, openGroup, closedGroup, stepGroups: new Map() };
    });
    this.secIndex = new Map(this.secs.map((s) => [s.id, s.i]));
    this.secBoxes = [];
  }

  buildTensors() {
    this.tensors = [];
    this.byId = new Map();
    let n = 0;
    this.sc.tensors.forEach((spec, ti) => {
      const data = spec.get();
      const Dn = data.length;
      const R = data[0].length;
      const C = data[0][0].length;
      const sec = this.secs[this.secIndex.get(spec.sec)];
      const keep = sec.keep === spec.id;
      const t = {
        spec, id: spec.id, ti, D: Dn, R, C, start: n, count: Dn * R * C,
        appear: 0, target: 0, vis: 0, lag: 0, batch: false, flies: false, hold: 0, quick: false,
        sec: sec.i,
        keep, // stays visible when its section is closed ...
        keepShift: keep ? sec.keepX - spec.x0 : 0, // ... moved this far along x
        ox: 0, // current x offset (section position + keepShift)
      };
      t.yTop = spec.cy + (R - 1) / 2;
      t.yBot = spec.cy - (R - 1) / 2;
      t.index = (d, r, c) => t.start + (d * R + r) * C + c;
      // Position local to the section, in its open layout.
      t.pos = (d, r, c, out) => out.set(spec.x0 + c, t.yTop - r, spec.zs[d]);
      t.box = new THREE.Box3(
        new THREE.Vector3(spec.x0 - 0.5, t.yBot - 0.5, Math.min(...spec.zs) - 0.5),
        new THREE.Vector3(spec.x0 + C - 0.5, t.yTop + 0.5, Math.max(...spec.zs) + 0.5),
      );
      n += t.count;
      this.tensors.push(t);
      this.byId.set(t.id, t);
    });
    const N = (this.N = n);

    this.tOf = new Uint16Array(N);
    this.dOf = new Uint16Array(N);
    this.rOf = new Uint16Array(N);
    this.cOf = new Uint16Array(N);
    this.Pl = new Float32Array(N * 3); // resting position, local to the section
    this.fromI = new Int32Array(N).fill(-1); // the cell this one flies in from
    this.delay = new Float32Array(N);
    this.val = new Float32Array(N);
    this.base = new Float32Array(N * 3);
    this.sxy = new Float32Array(N);
    this.sz = new Float32Array(N);
    this.emph = new Uint8Array(N);

    const v = new THREE.Vector3();
    for (const t of this.tensors) {
      for (let d = 0; d < t.D; d++)
        for (let r = 0; r < t.R; r++)
          for (let c = 0; c < t.C; c++) {
            const i = t.index(d, r, c);
            this.tOf[i] = t.ti;
            this.dOf[i] = d;
            this.rOf[i] = r;
            this.cOf[i] = c;
            t.pos(d, r, c, v);
            this.Pl.set([v.x, v.y, v.z], i * 3);
            this.delay[i] = (r + c) / (t.R + t.C);
            if (t.spec.from) {
              const [sid, sd, sr, sc] = t.spec.from(d, r, c);
              this.fromI[i] = this.byId.get(sid).index(sd, sr, sc);
            }
          }
    }

    const geo = new RoundedBoxGeometry(1, 1, 1, 2, 0.1);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    const mesh = new THREE.InstancedMesh(geo, mat, N);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
    mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    // Cells move every frame; give raycasting a fixed, generous bound.
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1e6);
    const arr = mesh.instanceMatrix.array;
    arr.fill(0);
    for (let i = 0; i < N; i++) arr[i * 16 + 15] = 1;
    this.scene.add(mesh);
    this.mesh = mesh;
  }

  // Reads values out of the scene and derives each cell's colour and size.
  refreshData() {
    const gmax = new Map();
    for (const t of this.tensors) {
      t.data = t.spec.get();
      const g = t.spec.group || t.id;
      let mx = gmax.get(g) || 1e-6;
      for (const slab of t.data) for (const row of slab) for (const v of row) if (isFinite(v)) mx = Math.max(mx, Math.abs(v));
      gmax.set(g, mx);
    }
    const col = new THREE.Color();
    for (const t of this.tensors) {
      const kind = t.spec.kind;
      const mx = gmax.get(t.spec.group || t.id);
      t.max = mx;
      for (let d = 0; d < t.D; d++)
        for (let r = 0; r < t.R; r++)
          for (let c = 0; c < t.C; c++) {
            const i = t.index(d, r, c);
            const v = t.data[d][r][c];
            this.val[i] = v;
            const blocked = v === -Infinity || (t.spec.blocked ? t.spec.blocked(d, r, c) : false);
            let sxy = 0.86;
            let sz = isFlat(kind) ? 0.26 : 0.86;
            if (blocked) {
              col.copy(COL.blocked);
              sxy = 0.3;
              sz = Math.min(sz, 0.3);
            } else if (kind === 'attn' || kind === 'prob') {
              col.copy(COL.att0).lerp(COL.att1, Math.pow(clamp(v, 0, 1), 0.6));
            } else if (kind === 'image') {
              // A picture: brightness is the value; three channels are tinted red, green, blue.
              const tint = t.D === 3 ? [[1, 0.3, 0.25], [0.35, 0.95, 0.4], [0.35, 0.55, 1]][d] : [1, 1, 1];
              const k = 0.1 + 0.9 * clamp(v, 0, 1);
              col.setRGB(tint[0] * k, tint[1] * k, tint[2] * k);
            } else if (kind === 'mask') {
              col.copy(COL.maskOpen);
            } else {
              const tt = clamp(v / mx, -1, 1);
              col.copy(COL.zero).lerp(tt >= 0 ? COL.pos : COL.neg, Math.pow(Math.abs(tt), 0.8));
            }
            this.base.set([col.r, col.g, col.b], i * 3);
            this.sxy[i] = sxy;
            this.sz[i] = sz;
          }
    }
  }

  // Frames, titles, row/column labels and brackets. Built in section-local
  // coordinates; each tensor's group is moved to its offset every frame.
  buildDecor() {
    const labelOf = (x) => (typeof x === 'string' ? { text: x } : x);
    const line = (pts, color) =>
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p))),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
      );

    for (const t of this.tensors) {
      const { spec } = t;
      const g = new THREE.Group();
      g.visible = false;
      this.scene.add(g);
      t.decor = g;
      t.sliceLabels = [];
      const multi = t.D > 1;
      const cx = spec.x0 + (t.C - 1) / 2;

      spec.zs.forEach((z, d) => {
        // Outline of the slab.
        const flat = isFlat(spec.kind);
        const depth = flat ? 0.5 : 1.14;
        const color = flat ? frameOf(spec.kind) : multi ? this.slabColors[d % this.slabColors.length] : COL.frameAct;
        const frame = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(t.C + 0.24, t.R + 0.24, depth)),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: flat ? 0.95 : 0.6 }),
        );
        frame.position.set(cx, spec.cy, z);
        g.add(frame);

        const lg = new THREE.Group();
        g.add(lg);
        t.sliceLabels.push(lg);
        if (spec.rowLabels)
          for (let r = 0; r < t.R; r++) {
            const l = labelOf(spec.rowLabels[r]);
            const sp = makeLabel([{ text: l.text, size: 0.52, color: l.dim ? '#5f6b82' : '#cfd8ea' }], { align: 'right', anchor: [1, 0.5] });
            sp.position.set(spec.x0 - 0.7, t.yTop - r, z);
            lg.add(sp);
          }
        if (spec.colLabels)
          for (let c = 0; c < t.C; c++) {
            const l = labelOf(spec.colLabels[c]);
            const sp = makeLabel([{ text: l.text, size: 0.46, color: l.dim ? '#5f6b82' : '#cfd8ea' }], {
              align: 'left',
              anchor: [0, 0.5],
              rotation: Math.PI / 4,
            });
            sp.position.set(spec.x0 + c - 0.22, t.yTop + 0.62, z);
            lg.add(sp);
          }
      });

      const z0 = spec.zs[0];
      if (spec.colBrackets) {
        const yb = t.yBot - 0.86;
        for (const b of spec.colBrackets) {
          const xa = spec.x0 + b.from - 0.4;
          const xb = spec.x0 + b.to + 0.4;
          g.add(line([[xa, yb + 0.2, z0], [xa, yb, z0], [xb, yb, z0], [xb, yb + 0.2, z0]], b.color));
          const sp = makeLabel([{ text: b.label, size: 0.44, color: b.color }], { anchor: [0.5, 1] });
          sp.position.set((xa + xb) / 2, yb - 0.08, z0);
          g.add(sp);
        }
      }
      if (spec.rowBrackets) {
        const xb = spec.x0 - 0.86;
        for (const b of spec.rowBrackets) {
          const ya = t.yTop - b.from + 0.4;
          const yb = t.yTop - b.to - 0.4;
          g.add(line([[xb + 0.2, ya, z0], [xb, ya, z0], [xb, yb, z0], [xb + 0.2, yb, z0]], b.color));
          const sp = makeLabel([{ text: b.label, size: 0.44, color: b.color }], { anchor: [0.5, 0], rotation: Math.PI / 2 });
          sp.position.set(xb - 0.1, (ya + yb) / 2, z0);
          g.add(sp);
        }
      }

      // Title: name, shape, optional note.
      const dims = multi ? [t.D, t.R, t.C] : [t.R, t.C];
      const lines = [
        { text: spec.label, size: 0.95, weight: 700, color: spec.kind === 'param' ? COL.param : isFixed(spec.kind) ? COL.fixed : COL.text },
        { text: `${spec.sym} = ${spec.num || `(${dims.join(', ')})`}`, size: 0.58, mono: true, color: '#b4c0d6' },
      ];
      if (spec.note) lines.push({ text: spec.note, size: 0.46, color: COL.textDim });
      const below = spec.title === 'below';
      const title = makeLabel(lines, { anchor: [0.5, below ? 1 : 0] });
      const ty = below
        ? t.yBot - 0.5 - (spec.colBrackets ? 1.75 : 0.3)
        : t.yTop + 0.5 + (spec.colLabels ? 1.95 : 0.3);
      title.position.set(cx, ty, z0);
      g.add(title);
      t.title = title;
    }
  }

  // ---- arrows and operator labels ------------------------------------------
  // Everything is added to the section being built (this._sec): either to the
  // group for one tour step of its open view, or to its closed view.

  stepGroup(step) {
    const sec = this._sec;
    let g = sec.stepGroups.get(step);
    if (!g) {
      g = new THREE.Group();
      g.visible = false;
      sec.openGroup.add(g);
      sec.stepGroups.set(step, g);
    }
    return g;
  }

  // Anything added while building one head's plane is hidden when that head is unfocused.
  addToFlow(group, obj) {
    group.add(obj);
    if (this._curHead != null) this.headObjs[this._curHead].push(obj);
  }

  flowMat(color) {
    this._mats ??= new Map();
    if (!this._mats.has(color)) this._mats.set(color, new THREE.MeshBasicMaterial({ color }));
    return this._mats.get(color);
  }

  addFlow(step, pts, { head = true, color = COL.flow, radius = 0.065, closed = false } = {}) {
    const group = closed ? this._sec.closedGroup : this.stepGroup(step);
    const g = { add: (o) => this.addToFlow(group, o) };
    const mat = this.flowMat(color);
    this._cyl ??= new THREE.CylinderGeometry(1, 1, 1, 10);
    this._cone ??= new THREE.ConeGeometry(0.25, 0.7, 14);
    this._ball ??= new THREE.SphereGeometry(1, 10, 8);
    const HL = 0.7;
    const P = pts.map((p) => new THREE.Vector3(p[0] + this._ox, p[1], p[2]));
    for (let i = 0; i < P.length - 1; i++) {
      const a = P[i];
      const b = P[i + 1];
      const dir = b.clone().sub(a);
      let len = dir.length();
      dir.normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(UP, dir);
      if (i === P.length - 2 && head) {
        const cone = new THREE.Mesh(this._cone, mat);
        cone.position.copy(b).addScaledVector(dir, -HL / 2);
        cone.quaternion.copy(q);
        g.add(cone);
        len -= HL * 0.9;
      }
      const cyl = new THREE.Mesh(this._cyl, mat);
      cyl.scale.set(radius, len, radius);
      cyl.position.copy(a).addScaledVector(dir, len / 2);
      cyl.quaternion.copy(q);
      g.add(cyl);
      if (i > 0) {
        const ball = new THREE.Mesh(this._ball, mat);
        ball.scale.setScalar(radius);
        ball.position.copy(a);
        g.add(ball);
      }
    }
  }

  addOp(step, text, at, { size = 0.62, color = '#dfe6f3', anchor = [0.5, 0], weight = 600, closed = false } = {}) {
    const sp = makeLabel([{ text, size, color, weight }], { anchor });
    sp.position.set(at[0] + this._ox, at[1], at[2]);
    this.addToFlow(closed ? this._sec.closedGroup : this.stepGroup(step), sp);
    return sp;
  }

  // An "add" node: a ring with a plus sign.
  addPlus(step, x, y, { color = COL.skip, closed = false } = {}) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.07, 10, 48), this.flowMat(color));
    ring.position.set(x + this._ox, y, 0);
    (closed ? this._sec.closedGroup : this.stepGroup(step)).add(ring);
    this.addOp(step, '+', [x, y, 0], { size: 1.3, anchor: [0.5, 0.52], color: `#${new THREE.Color(color).getHexString()}`, closed });
  }

  // A glass pane behind one slab: everything on it belongs to that slab.
  addPane({ step, slab, x0, x1, y0, y1, z, label }) {
    const color = new THREE.Color(this.slabColors[slab % this.slabColors.length]);
    const pane = new THREE.Mesh(
      new THREE.PlaneGeometry(x1 - x0, y1 - y0),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.03, side: THREE.DoubleSide, depthWrite: false }),
    );
    pane.position.set((x0 + x1) / 2, (y0 + y1) / 2, z);
    const border = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => new THREE.Vector3(x, y, z))),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 }),
    );
    const tag = makeLabel([{ text: label, size: 0.9, weight: 800, color: this.slabColors[slab % this.slabColors.length] }], { align: 'left', anchor: [0, 1] });
    tag.position.set(x0 + 0.6, y1 - 0.3, z);
    this.stepGroup(step).add(pane, border, tag);
    this.panes.push({ pane, border, tag, h: slab });
  }

  // Arrows, operator labels, add nodes and panes come from the scene as a
  // list of primitives per section.
  buildFlows() {
    this.headObjs = Array.from({ length: this.nSlabs }, () => []);
    this.panes = [];
    this._ox = 0;
    for (const sec of this.secs) {
      this._sec = sec;
      for (const f of sec.flows) {
        // Anything on one slab is hidden when another slab is focused.
        this._curHead = f.slab ?? null;
        if (f.type === 'flow') this.addFlow(f.step, f.pts, f);
        else if (f.type === 'op') this.addOp(f.step, f.text, f.at, f);
        else if (f.type === 'plus') this.addPlus(f.step, f.x, f.y, f);
        else if (f.type === 'pane') this.addPane(f);
      }
      this._curHead = null;
      this.closedBox(sec);
    }
    this._sec = null;

    // A bar over each repeated instance, spanning its sections. Positioned every frame.
    this.blockBars = this.sc.bars.map((b) => {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(1, 0.1, 0.1), this.flowMat(0x5b6780));
      const label = makeLabel(
        [
          { text: b.label, size: 0.85, weight: 800, color: '#b4c0d6' },
          { text: b.sub, size: 0.42, color: COL.textDim },
        ],
        { align: 'left', anchor: [0, 0] },
      );
      this.scene.add(bar, label);
      return { bar, label, first: this.secs[this.secIndex.get(b.first)], last: this.secs[this.secIndex.get(b.last)], yOpen: b.yOpen };
    });
  }

  // The box that stands for a closed section. Clicking it opens the section.
  closedBox(sec) {
    const [x0, x1] = sec.box;
    const depth = 2.4;
    const cx = (x0 + x1) / 2;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(x1 - x0, 8.4, depth),
      new THREE.MeshBasicMaterial({ color: 0x151b28, transparent: true, opacity: 0.92 }),
    );
    mesh.position.set(cx, 0, 0);
    mesh.userData.sec = sec.i;
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0xcfd8ea, transparent: true, opacity: 0.5 }),
    );
    edges.position.copy(mesh.position);
    const label = makeLabel([
      { text: sec.title, size: 1.3, weight: 800 },
      { text: sec.lines[0], size: 0.6, color: '#cfd8ea' },
      { text: sec.lines[1], size: 0.5, mono: true, color: COL.textDim },
      { text: 'click to open', size: 0.46, color: '#ffb454' },
    ]);
    label.position.set(cx, 0, depth / 2 + 0.15);
    // A large name under the section's output, readable from far away.
    const kept = this.byId.get(sec.keep);
    const big = makeLabel([{ text: sec.big, size: 2.2, weight: 800 }], { anchor: [0.5, 1] });
    big.position.set(sec.keepX + (kept.C - 1) / 2, kept.spec.title === 'below' ? -6.6 : -4.4, 0);
    sec.closedGroup.add(mesh, edges, label, big);
    sec.boxEdges = edges;
    sec.boxLabel = { sprite: label, cx, half: [(x1 - x0) / 2, 4.2, depth / 2] };
    this.secBoxes.push(mesh);
  }

  // A box's text always faces the camera. Seen from the side, the half of it
  // that points away would sink into the box. So the text is slid towards the
  // camera, along the line of sight (it stays centred on the box), just far
  // enough for the whole of it to be nearer than the box's nearest corner.
  clearBoxLabel(sec) {
    const { sprite, cx, half } = sec.boxLabel;
    const [v1, v2, v3] = this._labelVecs || (this._labelVecs = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()]);
    const centre = v1.set(sec.x + cx, 0, 0); // sections are only ever moved along x
    const toCam = v2.copy(this.camera.position).sub(centre);
    const dist = toCam.length();
    if (dist < 1e-6) return;
    toCam.divideScalar(dist);
    const n = this.camera.getWorldDirection(v3).negate(); // from the scene towards the camera's plane
    const reach = half[0] * Math.abs(n.x) + half[1] * Math.abs(n.y) + half[2] * Math.abs(n.z) + 0.15;
    const t = Math.min(reach / Math.max(toCam.dot(n), 0.2), dist * 0.5);
    sprite.position.set(cx + toCam.x * t, toCam.y * t, toCam.z * t);
  }

  buildHighlight() {
    this.boxes = [];
    const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    for (let i = 0; i < 40; i++) {
      const b = new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({ color: 0xffe066, transparent: true, depthTest: false }),
      );
      b.renderOrder = 15;
      b.visible = false;
      this.scene.add(b);
      this.boxes.push(b);
    }
    this.numbers = [];
    for (let i = 0; i < 100; i++) {
      const n = new NumberSprite();
      this.scene.add(n.sprite);
      this.numbers.push(n);
    }
  }

  bindEvents() {
    const el = this.renderer.domElement;
    const track = (e) => {
      const r = el.getBoundingClientRect();
      this.ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.pointerInside = true;
      this.pointerDirty = true;
    };
    el.addEventListener('pointermove', track);
    el.addEventListener('pointerleave', () => {
      this.pointerInside = false;
      this.pointerDirty = true;
    });
    let down = null;
    el.addEventListener('pointerdown', (e) => {
      track(e); // a tap has no preceding move
      down = [e.clientX, e.clientY];
    });
    el.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
      down = null;
      if (moved > 5) return;
      track(e);
      const hit = this.pickAt();
      if (hit.sec >= 0) {
        this.hooks.onSection?.(hit.sec);
        return;
      }
      this.setHover(hit.cube);
      this.pin(this.hoverId === this.pinnedId ? -1 : this.hoverId);
    });
    new ResizeObserver(() => this.resize()).observe(this.container);
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ---- public controls ---------------------------------------------------

  // Opens the section the step belongs to and reveals its tensors up to that
  // step. The section just finished stays open for one more step, so that its
  // folding happens behind the camera rather than in front of it.
  setStep(step, view, animate = true) {
    const prevOpen = sectionOfStep(this.secs, this.step);
    this.step = step;
    const open = sectionOfStep(this.secs, step);
    const openSet = new Set(openSections(this.secs, step));
    const moved = animate && open !== prevOpen; // the tour is changing section

    // Pin the boundary between the two sections involved (the start of the later
    // one), at wherever it is right now. Opening or closing a single section
    // pins that section's start.
    const pin = Math.max(prevOpen, open);
    if (pin >= 0) this.anchor = { sec: pin, x: animate ? this.secs[pin].x : 0 };

    // A section behind the one being worked on does not fold yet: it waits
    // until the camera has finished moving on (see tick). Folding ahead of the
    // camera, or folding everything for the map, happens straight away.
    this.deferred = [];
    for (const s of this.secs) {
      const want = openSet.has(s.i) ? 1 : 0;
      if (animate && s.target === 1 && want === 0 && open >= 0 && s.i < open) this.deferred.push(s);
      else s.target = want;
      if (!animate) s.p = s.target;
    }
    const fresh = [];
    for (const t of this.tensors) {
      const sec = this.secs[t.sec];
      const target = sec.target ? (t.spec.step <= step ? 1 : 0) : t.keep ? 1 : 0;
      // Only the section the step belongs to gets the slow, staged reveal.
      // Anything else that has to show up (off to the side) does so briskly.
      const here = sec.i === open;
      t.flies = !!t.spec.from && here;
      t.quick = !here;
      t.batch = false;
      t.lag = 0;
      // A section's output that is about to hide stays where it is while it
      // shrinks, instead of sliding away with the opening section.
      if (t.keep && target === 0 && t.target === 1) t.hold = sec.open;
      if (target === 1 && t.appear < 1 && here) fresh.push(t);
      t.target = target;
      if (!animate) t.appear = target;
    }
    // Tensors appearing together start one after another, in reading order.
    // After a change of section they also wait for the old section to fold.
    fresh.sort((a, b) => a.spec.step - b.spec.step || a.spec.x0 - b.spec.x0 || b.spec.cy - a.spec.cy);
    const wait = moved ? FOLD_WAIT : 0;
    if (animate && (fresh.length > 1 || moved))
      fresh.forEach((t, k) => {
        t.batch = true;
        t.lag = (wait + (1 - wait) * (fresh.length > 1 ? k / (fresh.length - 1) : 0)) * LAG;
      });
    this.appearDur = TIME.appear * (fresh.length > 10 ? 2 : 1) * (moved ? FOLD_STRETCH : 1);
    if (!animate) this.layout(0);
    if (view) this.frame(view, animate);
    this.foldTimer = this.deferred.length ? (this.tween ? this.tween.dur : TIME.camera) + 0.2 : 0;
  }

  // Folds the sections that were waiting for the camera to leave.
  foldDeferred() {
    for (const s of this.deferred) {
      s.target = 0;
      for (const t of this.tensors)
        if (t.sec === s.i) {
          t.target = t.keep ? 1 : 0;
          t.quick = true;
        }
    }
    this.deferred = [];
  }

  // Flips one of the program's switches; every number is recomputed.
  setToggle(name, on) {
    this.sc.set(name, on);
    this.refreshData();
    if (this.pinnedId >= 0) {
      this.pinnedEx = this.explain(this.pinnedId);
      this.applyHighlight(this.pinnedEx);
      this.hooks.onPin?.(this.pinnedEx);
    } else if (this.hoverId >= 0) {
      this.hoverEx = this.explain(this.hoverId);
      this.applyHighlight(this.hoverEx);
      this.hooks.onHover?.(this.hoverEx);
    }
  }

  setFocusHead(h) {
    this.focusHead = h;
    this.pointerDirty = true;
  }

  // Where each section will start once the current open/close animation ends.
  targetOffsets() {
    const xs = this.placeSections(this.secs.map((s) => (s.target ? s.wOpen : s.wClosed)));
    const last = this.secs[this.secs.length - 1];
    return { xs, start: xs[0], end: xs[last.i] + (last.target ? last.wOpen : last.wClosed) };
  }

  // Start x of every section, given their widths: the pinned section sits at
  // its pinned x, the ones after it follow on, the ones before it lead up to it.
  placeSections(widths) {
    const { sec: a, x: ax } = this.anchor;
    const xs = new Array(widths.length);
    xs[a] = ax;
    for (let i = a + 1; i < widths.length; i++) xs[i] = xs[i - 1] + widths[i - 1];
    for (let i = a - 1; i >= 0; i--) xs[i] = xs[i + 1] - widths[i];
    return xs;
  }

  // Places the camera so the given tensors fill the view from a chosen angle.
  // With a head focused (and view.perHead), only that head's slab is framed, from nearly head-on.
  // view.all frames the whole row of sections.
  frame(view, animate = true) {
    let { ids = [], az = 20, el = 12, zoom = 1 } = view;
    const fh = view.perHead ? this.focusHead : -1;
    if (fh >= 0) ({ az = 18, el = 12, zoom = 1 } = view.headView || {});
    const { xs, start, end } = this.targetOffsets();
    const box = new THREE.Box3();
    const tmp = new THREE.Box3();
    for (const id of ids) {
      const t = this.byId.get(id);
      const ox = xs[t.sec] + t.keepShift * (1 - this.secs[t.sec].target);
      tmp.copy(t.box);
      tmp.min.x += ox;
      tmp.max.x += ox;
      if (fh >= 0 && t.D > 1) {
        tmp.min.z = t.spec.zs[Math.min(fh, t.D - 1)] - 0.5;
        tmp.max.z = t.spec.zs[Math.min(fh, t.D - 1)] + 0.5;
      }
      box.union(tmp);
    }
    if (view.all) {
      box.expandByPoint(new THREE.Vector3(start, -7, -1));
      box.expandByPoint(new THREE.Vector3(end, 14, 1));
    }
    box.expandByVector(new THREE.Vector3(2.6, 2.8, 0.5));
    const c = box.getCenter(new THREE.Vector3());
    const a = rad(az);
    const e = rad(el);
    const dir = new THREE.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e));
    const right = new THREE.Vector3(Math.cos(a), 0, -Math.sin(a));
    const up = new THREE.Vector3().crossVectors(dir, right);
    // Keep the content clear of the header (top) and the step bar (bottom).
    const vh = this.container.clientHeight || 1;
    const padTop = Math.min(116, vh * 0.25);
    // The step bar grows with the number of steps; the page says how tall it is.
    const padBot = Math.max(Math.min(92, vh * 0.15), Math.min(this.hooks.bottomInset?.() ?? 0, vh * 0.4));
    const usable = (vh - padTop - padBot) / vh;
    const tvFull = Math.tan(rad(this.camera.fov) / 2);
    const tv = tvFull * usable;
    const th = tvFull * this.camera.aspect;
    let dist = 0;
    const p = new THREE.Vector3();
    for (let i = 0; i < 8; i++) {
      p.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).sub(c);
      const zz = p.dot(dir);
      dist = Math.max(dist, zz + Math.abs(p.dot(right)) / th, zz + Math.abs(p.dot(up)) / tv);
    }
    // A single small tensor is not blown up to fill the stage.
    if (fh < 0) dist = Math.max(dist, 9 / th);
    dist = (dist * 1.04) / zoom;
    // Look slightly above the content so it lands in the middle of the free area.
    c.addScaledVector(up, ((padTop - padBot) / vh) * dist * tvFull);
    const pos = c.clone().addScaledVector(dir, dist);
    if (!animate) {
      this.camera.position.copy(pos);
      this.controls.target.copy(c);
      this.tween = null;
    } else {
      // A long trip (pulling back to show a whole section, say) is given more
      // time, so it reads as one deliberate move instead of a jump.
      const travel = this.camera.position.distanceTo(pos);
      const dur = TIME.camera * clamp(1 + (travel - 60) / 200, 1, 1.9);
      this.tween = { p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: c, t: 0, dur };
    }
  }

  // ---- picking and highlighting -----------------------------------------

  // What is under the pointer: a cube, or the box of a closed section.
  pickAt() {
    const none = { cube: -1, sec: -1 };
    if (!this.pointerInside) return none;
    this.raycaster.setFromCamera(this.ndc, this.camera);
    let cube = -1;
    let dCube = Infinity;
    for (const h of this.raycaster.intersectObject(this.mesh, false)) {
      const i = h.instanceId;
      const t = this.tensors[this.tOf[i]];
      if (t.vis < 0.95) continue;
      if (t.D > 1 && this.headF[this.dOf[i]] < 0.5) continue;
      cube = i;
      dCube = h.distance;
      break;
    }
    const shown = this.secBoxes.filter((b) => this.secs[b.userData.sec].closedGroup.visible);
    const hb = this.raycaster.intersectObjects(shown, false)[0];
    if (hb && hb.distance < dCube) return { cube: -1, sec: hb.object.userData.sec };
    return { cube, sec: -1 };
  }

  explain(i) {
    const t = this.tensors[this.tOf[i]];
    return this.sc.explain(t.id, this.dOf[i], this.rOf[i], this.cOf[i]);
  }

  setHover(i) {
    if (i === this.hoverId) return;
    this.hoverId = i;
    this.hoverEx = i < 0 ? null : this.explain(i);
    if (this.pinnedId < 0) this.applyHighlight(this.hoverEx);
    this.hooks.onHover?.(this.hoverEx);
  }

  setHoverSec(k) {
    if (k === this.hoverSec) return;
    this.hoverSec = k;
    for (const s of this.secs) {
      s.boxEdges.material.opacity = s.i === k ? 1 : 0.5;
      s.boxEdges.material.color.set(s.i === k ? 0xffb454 : 0xcfd8ea);
    }
    this.renderer.domElement.style.cursor = k >= 0 ? 'pointer' : '';
  }

  pin(i) {
    this.pinnedId = i;
    this.pinnedEx = i < 0 ? null : this.explain(i);
    this.applyHighlight(this.pinnedEx || this.hoverEx);
    this.hooks.onPin?.(this.pinnedEx);
  }

  // Records what to highlight; layoutHighlight() places the outlines every
  // frame, because sections (and so cubes) can be moving.
  applyHighlight(ex) {
    this.emph.fill(0);
    this.hlBoxes = [];
    this.hlIds = [];
    this.hlTarget = ex ? 1 : 0;
    if (!ex) return;

    // lit = false draws only the outline: the region is context, not a source.
    const mark = (g, color, lit = true) => {
      const t = this.byId.get(g.id);
      for (let r = g.r0; lit && r <= g.r1; r++)
        for (let c = g.c0; c <= g.c1; c++) {
          const i = t.index(g.d, r, c);
          if (!this.emph[i]) this.hlIds.push(i);
          this.emph[i] = 1;
        }
      this.hlBoxes.push({ t, g, color });
    };
    mark({ id: ex.id, d: ex.d, r0: ex.r, r1: ex.r, c0: ex.c, c1: ex.c }, 0xffffff);
    for (const g of ex.groups) mark(g, 0xffe066);
    for (const g of ex.context || []) mark(g, 0x7f8ba3, false);
    this.hlIds.length = Math.min(this.hlIds.length, this.numbers.length);
    this.hlIds.forEach((i, k) => this.numbers[k].set(fmt(this.val[i])));
  }

  layoutHighlight() {
    const a = this._va || (this._va = new THREE.Vector3());
    const b = this._vb || (this._vb = new THREE.Vector3());
    this.boxes.forEach((box, k) => {
      const h = this.hlBoxes[k];
      box.visible = !!h && h.t.vis > 0.5;
      if (!box.visible) return;
      const { t, g } = h;
      t.pos(g.d, g.r0, g.c0, a);
      t.pos(g.d, g.r1, g.c1, b);
      box.position.copy(a).add(b).multiplyScalar(0.5);
      box.position.x += t.ox;
      box.scale.set(Math.abs(b.x - a.x) + 1.04, Math.abs(b.y - a.y) + 1.04, isFlat(t.spec.kind) ? 0.5 : 1.04);
      box.material.color.set(h.color);
    });
    this.numbers.forEach((n, k) => {
      const i = this.hlIds[k];
      const t = i == null ? null : this.tensors[this.tOf[i]];
      n.sprite.visible = !!t && t.vis > 0.5;
      if (n.sprite.visible) n.sprite.position.set(t.ox + this.Pl[i * 3], this.Pl[i * 3 + 1], this.Pl[i * 3 + 2]);
    });
  }

  // ---- per-frame ----------------------------------------------------------

  // Advances the open/close animation and moves every section to its place.
  layout(dt) {
    for (const s of this.secs) {
      if (s.p !== s.target) s.p = clamp(s.p + (Math.sign(s.target - s.p) * dt) / TIME.section, 0, 1);
      s.open = ease(s.p);
      s.width = lerp(s.wClosed, s.wOpen, s.open);
    }
    const xs = this.placeSections(this.secs.map((s) => s.width));
    for (const s of this.secs) {
      s.x = xs[s.i];
      s.group.position.x = s.x;
      // Arrows of a folding section go early and its box arrives late; the box
      // of an unfolding section goes early and its arrows arrive late. Neither
      // view lingers over the other.
      s.openGroup.visible = s.target ? s.open > 0.5 : s.open > 0.9;
      s.closedGroup.visible = s.target ? s.open < 0.1 : s.open <= 0.5;
      for (const [step, g] of s.stepGroups) g.visible = step <= this.step;
      if (s.closedGroup.visible) this.clearBoxLabel(s);
    }
    for (const t of this.tensors) {
      const s = this.secs[t.sec];
      const o = t.keep && t.target === 0 && t.vis > 0.001 ? t.hold : s.open;
      t.ox = s.x + t.keepShift * (1 - o);
      t.decor.position.x = t.ox;
    }
    for (const bb of this.blockBars) {
      const x0 = bb.first.x + 2;
      const x1 = bb.last.x + bb.last.width - 0.5;
      const y = lerp(12.5, bb.yOpen, Math.max(bb.first.open, bb.last.open));
      bb.bar.position.set((x0 + x1) / 2, y, 0);
      bb.bar.scale.x = x1 - x0;
      bb.label.position.set(x0, y + 0.25, 0);
    }
  }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);

    if (this.tween) {
      const tw = this.tween;
      tw.t = Math.min(1, tw.t + dt / tw.dur);
      const e = ease(tw.t);
      this.camera.position.lerpVectors(tw.p0, tw.p1, e);
      this.controls.target.lerpVectors(tw.t0, tw.t1, e);
      if (tw.t >= 1) this.tween = null;
      this.pointerDirty = true;
    }
    this.controls.update();
    if (this.foldTimer > 0 && (this.foldTimer -= dt) <= 0) this.foldDeferred();
    this.layout(dt);

    if (this.pointerDirty && !this.dragging) {
      this.pointerDirty = false;
      const hit = this.pickAt();
      this.setHoverSec(hit.sec);
      this.setHover(hit.cube);
    }

    for (const t of this.tensors) {
      if (t.appear !== t.target) {
        const up = t.target > t.appear;
        t.appear = clamp(t.appear + ((up ? 1 : -1) * dt) / (up && !t.quick ? this.appearDur : TIME.hide), 0, 1);
      }
      t.vis = t.batch ? clamp((t.appear - t.lag) / (1 - LAG), 0, 1) : t.appear;
      t.decor.visible = t.vis > 0.5;
    }
    for (let h = 0; h < this.nSlabs; h++) {
      const target = this.focusHead < 0 || this.focusHead === h ? 1 : 0;
      this.headF[h] += clamp(target - this.headF[h], -dt / 0.25, dt / 0.25);
    }
    this.hl += clamp(this.hlTarget - this.hl, -dt / 0.15, dt / 0.15);

    const fz = Math.max(this.focusHead, 0);
    for (const t of this.tensors) {
      if (t.D <= 1) continue;
      t.sliceLabels.forEach((g, d) => (g.visible = this.headF[d] > 0.5));
      t.title.position.z = t.spec.zs[Math.min(fz, t.D - 1)];
    }
    this.headObjs.forEach((list, h) => {
      const vis = this.headF[h] > 0.5;
      for (const o of list) o.visible = vis;
    });
    for (const p of this.panes) {
      const f = this.headF[p.h];
      p.border.material.opacity = 0.12 + 0.4 * f;
      p.pane.material.opacity = 0.03 * f;
      p.tag.material.opacity = 0.25 + 0.75 * f;
    }

    this.updateInstances();
    this.layoutHighlight();
    this.renderer.render(this.scene, this.camera);
  }

  updateInstances() {
    const arr = this.mesh.instanceMatrix.array;
    const col = this.mesh.instanceColor.array;
    const { Pl, base, tensors, tOf } = this;
    const hl = this.hl;
    for (let i = 0; i < this.N; i++) {
      const t = tensors[tOf[i]];
      const v = t.vis;
      // Each cube's own move takes half the reveal; a wave runs across the tensor.
      const e = v >= 1 ? 1 : v <= 0 ? 0 : ease(clamp(v * 2 - this.delay[i], 0, 1));
      const j = i * 3;
      let x = t.ox + Pl[j];
      let y = Pl[j + 1];
      let z = Pl[j + 2];
      let s = e;
      const src = t.flies ? this.fromI[i] : -1;
      if (src >= 0) {
        // Fly in from the cube this number was copied or computed from.
        const k = src * 3;
        const fx = tensors[tOf[src]].ox + Pl[k];
        x = fx + (x - fx) * e;
        y = Pl[k + 1] + (y - Pl[k + 1]) * e;
        z = Pl[k + 2] + (z - Pl[k + 2]) * e;
        s = Math.min(1, e * 6);
      }
      let k = 1;
      if (t.D > 1) {
        const hf = this.headF[this.dOf[i]];
        s *= 0.2 + 0.8 * hf;
        k *= 0.4 + 0.6 * hf;
      }
      if (hl > 0 && !this.emph[i]) k *= 1 - 0.68 * hl;
      s = Math.max(s, 1e-4);
      const o = i * 16;
      arr[o] = arr[o + 5] = this.sxy[i] * s;
      arr[o + 10] = this.sz[i] * s;
      arr[o + 12] = x;
      arr[o + 13] = y;
      arr[o + 14] = z;
      col[j] = base[j] * k;
      col[j + 1] = base[j + 1] * k;
      col[j + 2] = base[j + 2] * k;
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
  }
}
