// Layout: where every tensor sits, derived from the operations.
//
// Coordinates are local to a section, one unit per cube. Data flows left to
// right along y = 0, the stream; the previous section's last tensor is taken
// to end just left of x = 0. A tensor's x0 is its first column, cy its
// vertical centre, zs the depth of each slab (rank-3 tensors are drawn as one
// slab per entry of their first axis).
//
// Rules, by what a tensor is for:
//   flow      to the right of its operands, on the lane of the first one.
//             If that spot is taken it moves up or down a lane.
//   weight    a parameter multiplied on the right sits inline: A, W, result.
//   beside    an activation multiplied on the right sits directly above or
//             below the result, so its columns line up with the result's.
//   over      a per-column parameter (a gain) sits just above the result.
//   front     an operand with no depth axis, combined with slabs, is a plate
//             in front of them.
//   stack     a cell-by-cell function of a wide tensor goes underneath it.
//   ring      adding two streams leaves room for a "+" node.
//   lane      a tensor brought in from another section enters on a side lane
//             below (or above) the flow; what is computed from it stays there.

export const GEO = {
  gap: 8, // between an operand and what is computed from it
  ringGap: 12, // ... when a "+" node sits in between
  copyGap: 12, // ... when rows are copied out of a table
  colGap: 3, // ... for a one-column tensor
  lanePad: 3, // free space above and below every tensor
  slabGap: 16, // between a plate and the slabs behind it, when there is one slab
  slabClear: 10, // free depth between the slabs of a rank-3 tensor
  stackDepth: 60, // the most depth a stack of slabs may take
  deckGap: 3, // between slabs that stand close together
  wide: 24, // columns from which a cell-by-cell result is stacked underneath
  stackGap: 4,
  beside: 2,
  over: 1,
  sideGap: 8, // between the flow and a side lane that a tensor from another section enters on
  sideReach: 40, // how far back the flow is looked at when choosing that lane's height
  apart: 14, // before a section that does not continue the one before it
};

export const colsOf = (t) => t.view.C;
export const rowsOf = (t) => t.view.R;
export const slabsOf = (t) => t.view.D;
// The axis drawn as slabs, and whether its slabs stand close together.
export const slabAxisOf = (t) => (t.rank === 4 ? t.shape[1] : t.rank === 3 ? t.shape[0] : null);
export const isDeck = (t) => !!slabAxisOf(t)?.deck;
const sameDims = (a, b) => a.rank === b.rank && a.dims.every((d, i) => d === b.dims[i]);

// A parameter, or a rearranged view of one (a transposed table, say).
// Rows taken out of a table for this example are not: which rows (lookup) or how many (rows) depends on it.
const TAKEN = new Set(['lookup', 'rows']);
export const isParamLike = (t) => t.kind === 'param' || (t.op?.family === 'rearrange' && !TAKEN.has(t.op.name) && isParamLike(t.op.inputs[0]));

// Which operands are placed by the operation that uses them.
function findRoles(prog) {
  const roles = new Map();
  for (const t of prog.tensors) {
    const op = t.op;
    if (!op) continue;
    const claim = (o, type) => {
      if (!roles.has(o) && o.section === t.section) roles.set(o, { type, of: t });
    };
    if (op.family === 'contract') {
      const o = op.inputs[1];
      // A rearranged view (K transposed, V split into heads) stands beside the
      // result; anything that is itself computed here stays in the flow.
      if (isParamLike(o)) claim(o, 'weight');
      else if (o.op?.family === 'rearrange' && o.hints.place !== 'flow') claim(o, 'beside');
    }
    if (op.family === 'map')
      for (const o of op.inputs) {
        const fixed = !o.op?.inputs.length; // a parameter or a constant
        if (o.kind === 'param' && o.rank === 1 && t.rank > 1) claim(o, 'over');
        else if (!fixed || o === op.inputs[0]) continue;
        else if (o.size === 1) claim(o, 'over'); // a single fixed number: a setting, a schedule value
        else if (t.rank === 3 && o.rank === 2) claim(o, 'front');
        else if (o.view.R === 1 && o.view.C === 1 && o.view.D === t.view.D) claim(o, 'over'); // one number per slab
        else if (sameDims(o, t)) claim(o, 'beside');
      }
  }
  return roles;
}

export function layout(prog) {
  const roles = findRoles(prog);
  const at = new Map(); // tensor -> placement record
  const sections = new Map();

  for (const sec of prog.sections) {
    const placed = [];
    // The stream coming in: the previous section's last tensor, just left of 0.
    const stream = sec.stream && { t: sec.stream, x0: -colsOf(sec.stream), cy: 0, lane: 0, C: colsOf(sec.stream), R: rowsOf(sec.stream), zs: at.get(sec.stream).zs, stream: true };
    const pos = (t) => (t.section === sec ? at.get(t) : t === sec.stream ? stream : null);
    // A section that does not read the one before it stands clear of that section's last tensor.
    const lead = !stream && sec !== prog.sections[0] ? GEO.apart : 0;

    // Slabs stand far enough apart to see between them: further for larger ones.
    // ... but a stack of many slabs is squeezed so that it does not run away into depth.
    const zsOf = (t) => {
      const n = slabsOf(t);
      const gap = isDeck(t) ? GEO.deckGap : Math.min(Math.max(rowsOf(t), colsOf(t)) + GEO.slabClear, n > 1 ? GEO.stackDepth / (n - 1) : Infinity);
      return Array.from({ length: n }, (_, d) => ((n - 1) / 2 - d) * gap);
    };
    // An operand placed against a result with as many slabs shares its depths.
    const depthsOf = (o, rec) => (slabsOf(o) === rec.zs.length ? { zs: rec.zs } : {});
    const put = (t, x0, cy, extra = {}) => {
      const rec = { t, x0, cy, lane: cy, C: colsOf(t), R: rowsOf(t), zs: zsOf(t), role: 'flow', ...extra };
      at.set(t, rec);
      placed.push(rec);
      return rec;
    };
    const hits = (x0, cy, C, R) =>
      placed.some(
        (p) => p.role !== 'front' && x0 - 2.7 < p.x0 + p.C - 0.1 && p.x0 - 2.7 < x0 + C - 0.1 && Math.abs(cy - p.cy) < (R + p.R) / 2 + 2 * GEO.lanePad,
      );
    // The nearest free height to cy; a tie goes up unless hinted otherwise.
    const settle = (x0, cy, C, R, side) => {
      if (!hits(x0, cy, C, R)) return cy;
      let [up, down] = [cy, cy];
      while (hits(x0, up, C, R)) up += 1;
      while (hits(x0, down, C, R)) down -= 1;
      if (side === 'above') return up;
      if (side === 'below') return down;
      return up - cy <= cy - down ? up : down;
    };

    // Operands that the result positions: beside, in front of, or over it.
    const attach = (rec) => {
      for (const o of rec.t.op.inputs) {
        const role = roles.get(o);
        if (!role || role.of !== rec.t || at.has(o) || role.type === 'weight') continue;
        const R = rowsOf(o);
        if (role.type === 'beside') {
          const origin = o.op?.inputs.map(pos).find(Boolean);
          const up = !origin || origin.lane >= rec.lane;
          put(o, rec.x0, rec.cy + (up ? 1 : -1) * ((rec.R + R) / 2 + GEO.beside), { role: 'beside', side: up ? 'above' : 'below', of: rec.t, ...depthsOf(o, rec) });
        } else if (role.type === 'front') {
          put(o, rec.x0, rec.cy, { role: 'front', zs: [rec.zs[0] + (rec.zs.length > 1 ? rec.zs[0] - rec.zs[1] : GEO.slabGap)], of: rec.t });
        } else {
          put(o, rec.x0 + (rec.C - colsOf(o)) / 2, rec.cy + (rec.R + R) / 2 + GEO.over, { role: 'over', of: rec.t, ...depthsOf(o, rec) });
        }
      }
      return rec;
    };

    const place = (t) => {
      const [C, R] = [colsOf(t), rowsOf(t)];
      const side = t.hints.side;
      const op = t.op;
      // Operands that are in the flow here (not ones this result positions).
      const ins = op ? op.inputs.filter((o) => pos(o) && roles.get(o)?.of !== t) : [];
      if (!ins.length && op?.name === 'again' && (placed.length || stream)) {
        // Brought in from another section part-way through: it enters on a
        // side lane, level with where the flow has got to.
        const x0 = placed.length ? Math.max(...placed.map((p) => p.x0 + p.C)) + GEO.gap : 0;
        const near = placed.filter((p) => p.role !== 'front' && p.x0 + p.C > x0 - GEO.sideReach);
        if (!near.length && stream) near.push(stream);
        const up = side === 'above';
        const edge = up ? Math.max(...near.map((p) => p.cy + p.R / 2)) : Math.min(...near.map((p) => p.cy - p.R / 2));
        const cy = Math.round(edge + (up ? 1 : -1) * (GEO.sideGap + R / 2));
        return put(t, x0, settle(x0, cy, C, R, up ? 'above' : 'below'));
      }
      if (!ins.length) return put(t, lead, settle(lead, 0, C, R, side)); // a source starts at the left edge

      const recs = ins.map(pos);
      const right = Math.max(...recs.map((p) => p.x0 + p.C));
      const first = recs[0];

      if (op.family === 'contract') {
        const W = op.inputs[1];
        if (roles.get(W)?.of === t && roles.get(W).type === 'weight') {
          const x0 = right + GEO.gap;
          const wide = colsOf(W) + GEO.gap + C; // the weight and the result as one box
          let cy = settle(x0, first.lane, colsOf(W), rowsOf(W), W.hints.side ?? side);
          // A result taller than its weight needs the room too: move the pair until both are free.
          if (hits(x0 + colsOf(W) + GEO.gap, cy, C, R)) cy = settle(x0, first.lane, wide, Math.max(rowsOf(W), R), W.hints.side ?? side);
          const w = put(W, x0, cy, { role: 'weight', of: t, ...depthsOf(W, first) });
          return attach(put(t, w.x0 + w.C + GEO.gap, cy));
        }
      }
      if (op.family === 'map' && ins.length === 1 && sameDims(t, ins[0]) && first.C >= GEO.wide && !first.stream) {
        // Underneath its operand; the pair stays centred on the lane.
        first.cy += (R + GEO.stackGap) / 2;
        const rec = put(t, first.x0, first.cy - R - GEO.stackGap, { stackedUnder: ins[0] });
        rec.lane = first.lane;
        return attach(rec);
      }
      if (op.name === 'add' && ins.length === 2 && ins.every((o) => sameDims(o, t))) {
        const cy = (recs[0].cy + recs[1].cy) / 2;
        return attach(put(t, right + GEO.ringGap, cy, { ring: { x: right + GEO.ringGap - 7, y: cy } }));
      }
      const gap = C === 1 ? GEO.colGap : op.name === 'lookup' || op.name === 'rows' ? GEO.copyGap : GEO.gap;
      return attach(put(t, right + gap, settle(right + gap, first.lane, C, R, side)));
    };

    for (const t of sec.tensors) {
      if (at.has(t)) continue;
      if (roles.has(t)) continue; // placed with the operation that uses it
      place(t);
    }

    // The section's last tensor is the stream going out: it sits on y = 0.
    const keep = at.get(sec.keep);
    if (stream && keep.cy !== 0) {
      const free = !placed.some(
        (p) => p !== keep && p.role !== 'front' && keep.x0 - 2.7 < p.x0 + p.C - 0.1 && p.x0 - 2.7 < keep.x0 + keep.C - 0.1 && Math.abs(p.cy) < (keep.R + p.R) / 2 + 2 * GEO.lanePad,
      );
      if (free) keep.cy = keep.lane = 0;
    }
    if (!stream && keep.cy !== 0) {
      const dy = keep.cy;
      for (const p of placed) {
        p.cy -= dy;
        p.lane -= dy;
        if (p.ring) p.ring.y -= dy;
      }
    }

    // Titles go above unless something sits right there.
    for (const p of placed) {
      const top = p.cy + (p.R - 1) / 2;
      const crowded = placed.some(
        (q) => q !== p && q.role !== 'front' && q.x0 < p.x0 + p.C && p.x0 < q.x0 + q.C && q.cy > p.cy && q.cy - (q.R - 1) / 2 - top < 5.5,
      );
      const secondColumn = p.C === 1 && placed.some((q) => q.C === 1 && q.x0 < p.x0 && p.x0 - q.x0 <= GEO.colGap + 1 && !q.below);
      // Column labels stand above a tensor; its title then goes below, unless
      // something else is already there.
      const bottom = p.cy - (p.R - 1) / 2;
      const labelled = !!p.t.shape.at(-1).labels && p.C > 1;
      const under = placed.some(
        (q) => q !== p && q.role !== 'front' && q.x0 < p.x0 + p.C && p.x0 < q.x0 + q.C && q.cy < p.cy && bottom - (q.cy + (q.R - 1) / 2) < 5.5,
      );
      p.below = crowded || secondColumn || p.side === 'below' || !!p.stackedUnder || (labelled && !under);
    }

    sections.set(sec, {
      placed, stream, keep,
      width: keep.x0 + keep.C,
      top: Math.max(...placed.map((p) => p.cy + (p.R - 1) / 2)),
      bottom: Math.min(...placed.map((p) => p.cy - (p.R - 1) / 2)),
    });
  }
  return { at, roles, sections };
}
