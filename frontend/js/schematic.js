// Animated P&ID schematics — HMI v2, informed by high-performance-HMI practice
// (ISA-101): neutral greys are the NORMAL state, colour is a SIGNAL (amber =
// deviation, red = interlock), and analog values get moving-indicator bars with
// their acceptance/alarm bands drawn in, so "in band / off band" reads at a
// glance. Every actuator carries a tag badge (name + % + physical rate) — what
// the controller is DOING is first-class information. Media fills keep the
// temperature colour ramp (a deliberate teaching aid, kept low-saturation).
// A `compact` option strips badges and bars for small embeds.
import { t as L } from './i18n.js?v=36';   // aliased: `t` is used locally for tank refs

const SVG = 'http://www.w3.org/2000/svg';
function el(tag, attrs = {}, kids = []) {
  const n = document.createElementNS(SVG, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  (Array.isArray(kids) ? kids : [kids]).forEach((c) => c && n.appendChild(c));
  return n;
}
const txt = (s) => document.createTextNode(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---- palette (HP-HMI: grey = normal, colour = signal) ----
const INK = '#5A626C', INK2 = '#8A93A0', MONO = 'IBM Plex Mono, monospace';
const WARN = '#E4A11B', CRIT = '#D64545', SPG = '#73B200';
const WARN_BG = 'rgba(228,161,27,.28)', CRIT_BG = 'rgba(214,69,69,.30)', NORM_BG = 'rgba(115,178,0,.14)';

const STOPS = [[10, [30, 90, 200]], [38, [20, 170, 160]], [60, [230, 150, 40]], [88, [225, 60, 60]]];
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
function tempColor(T) {
  if (T <= STOPS[0][0]) return rgb(STOPS[0][1]);
  if (T >= STOPS[STOPS.length - 1][0]) return rgb(STOPS[STOPS.length - 1][1]);
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [t0, c0] = STOPS[i], [t1, c1] = STOPS[i + 1];
    if (T >= t0 && T <= t1) {
      const f = (T - t0) / (t1 - t0);
      return rgb(c0.map((c, j) => Math.round(c + f * (c1[j] - c))));
    }
  }
  return rgb(STOPS[0][1]);
}
// comfort colouring for occupied spaces: in-band = calm green, cold = blue, hot = orange
function comfortColor(T, lo = 20, hi = 24) {
  if (T >= lo && T <= hi) return '#BFE0C2';
  if (T < lo) { const f = clamp((lo - T) / 8, 0, 1); return rgb([191 - f * 60, 224 - f * 30, 194 + f * 46]); }
  const f = clamp((T - hi) / 8, 0, 1); return rgb([191 + f * 55, 224 - f * 60, 194 - f * 80]);
}
function setFlow(flowEl, q, qref) {
  if (q > 1e-7) {
    flowEl.setAttribute('opacity', Math.min(0.95, 0.35 + q / qref));
    flowEl.style.animationDuration = `${clamp(0.9 / (0.2 + q / qref), 0.25, 2.5)}s`;
  } else flowEl.setAttribute('opacity', 0);
}
function glowFilter(id, dev) {
  const f = el('filter', { id, x: '-60%', y: '-60%', width: '220%', height: '220%' });
  f.appendChild(el('feGaussianBlur', { stdDeviation: dev, result: 'b' }));
  const m = el('feMerge');
  m.appendChild(el('feMergeNode', { in: 'b' }));
  m.appendChild(el('feMergeNode', { in: 'SourceGraphic' }));
  f.appendChild(m);
  return f;
}
function defsBlock() {
  const defs = el('defs');
  defs.appendChild(glowFilter('glowHeat', 3.2));
  defs.appendChild(glowFilter('glowPump', 2.0));
  const m = el('marker', { id: 'arrow', viewBox: '0 0 8 8', refX: 6, refY: 4, markerWidth: 5.5, markerHeight: 5.5, orient: 'auto' });
  m.appendChild(el('path', { d: 'M0.5,0.5 L8,4 L0.5,7.5 Z', fill: 'context-stroke' }));
  defs.appendChild(m);
  return defs;
}
const HALO = { stroke: '#ffffff', 'stroke-width': 3.5, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const hoverable = (node, title) => { if (title) node.appendChild(el('title', {}, txt(title))); return node; };

// ---- moving analog indicator (the HP-HMI signature component) ----
// Vertical strip with alarm/acceptance bands painted on the span; a black PV
// pointer and a green SP caret move along it. zones: [lo, hi, 'norm'|'warn'|'crit'].
function analogBar(svg, x, y, h, { min, max, zones = [], ticks = [] }) {
  const g = el('g');
  const yOf = (v) => y + h - clamp((v - min) / (max - min), 0, 1) * h;
  g.appendChild(el('rect', { x, y, width: 9, height: h, rx: 2.5, fill: '#EFF2F4', stroke: '#D5DAE0', 'stroke-width': 0.8 }));
  const Z = { norm: NORM_BG, warn: WARN_BG, crit: CRIT_BG };
  for (const [lo, hi, kind] of zones) {
    const y1 = yOf(Math.min(hi, max)), y2 = yOf(Math.max(lo, min));
    g.appendChild(el('rect', { x: x + 0.8, y: y1, width: 7.4, height: Math.max(0, y2 - y1), fill: Z[kind] || NORM_BG }));
  }
  for (const [v, label, color] of ticks) {
    g.appendChild(el('line', { x1: x - 1.5, x2: x + 10.5, y1: yOf(v), y2: yOf(v), stroke: color || INK2, 'stroke-width': 1 }));
    if (label) g.appendChild(el('text', { x: x + 13, y: yOf(v) + 3, fill: color || INK2, 'font-size': 8.5, 'font-family': MONO }, txt(label)));
  }
  const pv = el('path', { d: 'M0,0 l-7,-4.5 v9 Z', fill: '#1F2733' });
  const sp = el('path', { d: 'M0,0 l7,-4 v8 Z', fill: SPG });
  g.appendChild(pv); g.appendChild(sp);
  svg.appendChild(g);
  return {
    g,
    set(pvV, spV) {
      pv.setAttribute('transform', `translate(${x},${yOf(pvV)})`);
      if (spV == null) sp.setAttribute('opacity', 0);
      else { sp.setAttribute('opacity', 1); sp.setAttribute('transform', `translate(${x + 9},${yOf(spV)})`); }
    },
  };
}

// ---- actuator/instrument tag badge: name over value, with a status edge ----
function tagBadge(svg, x, y, w, name, title) {
  const g = hoverable(el('g'), title || name);
  g.appendChild(el('rect', { x, y, width: w, height: 30, rx: 4, fill: '#FFFFFF', stroke: '#D9DDE2', 'stroke-width': 1 }));
  const edge = el('rect', { x, y, width: 3, height: 30, rx: 1.5, fill: '#C3C9D0' });
  g.appendChild(edge);
  g.appendChild(el('text', { x: x + 8, y: y + 12, fill: INK2, 'font-size': 8.5, 'font-weight': 600, 'letter-spacing': '.4' }, txt(name)));
  const val = el('text', { x: x + 8, y: y + 25, fill: '#1F2733', 'font-size': 11.5, 'font-weight': 700, 'font-family': MONO }, txt('--'));
  g.appendChild(val);
  svg.appendChild(g);
  return { g, set(text, level) { val.textContent = text; edge.setAttribute('fill', level === 'crit' ? CRIT : level === 'warn' ? WARN : '#C3C9D0'); } };
}

// ---- interlock badge (hidden until tripped) ----
function tripBadge(svg, cx, cy, label) {
  const g = el('g', { opacity: 0, 'pointer-events': 'none' });
  const wid = 12 * label.length * 0.9 + 30;
  g.appendChild(el('rect', { x: cx - wid / 2, y: cy - 13, width: wid, height: 26, rx: 6, fill: CRIT, stroke: '#fff', 'stroke-width': 1.5 }));
  g.appendChild(el('text', { x: cx, y: cy + 4.5, fill: '#fff', 'font-size': 12.5, 'font-weight': 800, 'text-anchor': 'middle' }, txt('⛔ ' + label)));
  svg.appendChild(g);
  return { show(on) { g.setAttribute('opacity', on ? 1 : 0); } };
}

// Reusable tank cell: water + heater coil + glass + readouts.
function tankCell(g, x, y, w, h, label) {
  const innerH = h - 16, bottomY = y + h - 8;
  const water = el('rect', { x: x + 4, y: bottomY, width: w - 8, height: 0, fill: '#9CC2F0', rx: 2, opacity: 0.55 });
  const cap = el('rect', { x: x + 4, y: bottomY, width: w - 8, height: 2.5, fill: '#050B14', opacity: 0.12 });
  const coil = el('path', { d: coilPath(x + 14, bottomY - 12, w - 28), stroke: '#C3C7CC', 'stroke-width': 3.2, fill: 'none', 'stroke-linecap': 'round' });
  const glass = el('rect', { x, y, width: w, height: h, rx: 6, fill: 'none', stroke: INK, 'stroke-width': 1.6 });
  const spLine = el('line', { x1: x, y1: bottomY, x2: x + w, y2: bottomY, stroke: SPG, 'stroke-width': 1.6, 'stroke-dasharray': '5 4', opacity: 0 });
  const tempT = el('text', { x: x + w / 2, y: y + h / 2 - 2, fill: '#0B1220', 'font-size': 21, 'font-weight': 700, 'text-anchor': 'middle', 'font-family': MONO, ...HALO }, txt('--'));
  const tspT = el('text', { x: x + w / 2, y: y + h / 2 + 16, fill: '#3F6B00', 'font-size': 10, 'text-anchor': 'middle', 'font-family': MONO, ...HALO }, txt(''));
  const lvlT = el('text', { x: x + w / 2, y: bottomY - 6, fill: '#0B1220', 'font-size': 11, 'text-anchor': 'middle', 'font-family': MONO, ...HALO }, txt('--'));
  const title = el('text', { x: x + w / 2, y: y - 7, fill: '#585C62', 'font-size': 11, 'font-weight': 600, 'text-anchor': 'middle' }, txt(label));
  [water, cap, coil, glass, spLine, tempT, tspT, lvlT, title].forEach((e) => g.appendChild(e));
  hoverable(g, label);
  return { water, cap, coil, spLine, tempT, tspT, lvlT, x, y, w, h, innerH, bottomY };
}
function coilPath(x, y, w) {
  const seg = w / 5;
  let d = `M${x},${y}`;
  for (let i = 0; i < 5; i++) d += ` q${seg / 2},${i % 2 ? 10 : -10} ${seg},0`;
  return d;
}
function paintTank(t, level, temp, hmax, tHigh) {
  const hpx = clamp(level / hmax, 0, 1) * t.innerH;
  t.water.setAttribute('y', t.bottomY - hpx);
  t.water.setAttribute('height', hpx);
  t.water.setAttribute('fill', tempColor(temp));
  t.cap.setAttribute('y', t.bottomY - hpx);
  t.cap.setAttribute('opacity', hpx > 2 ? 0.12 : 0);
  t.tempT.textContent = temp.toFixed(1);
  t.tempT.setAttribute('fill', temp >= (tHigh || 80) ? CRIT : '#0B1220');
  t.lvlT.textContent = `${level.toFixed(2)} m`;
}
function paintHeater(coil, frac, tripped) {
  if (tripped) { coil.setAttribute('stroke', CRIT); coil.setAttribute('filter', ''); coil.setAttribute('opacity', 0.95); coil.setAttribute('stroke-width', 3.6); }
  else if (frac > 0.01) {
    coil.setAttribute('stroke', `rgb(${Math.round(225 - frac * 30)},${Math.round(140 - frac * 95)},20)`);
    coil.setAttribute('filter', 'url(#glowHeat)');
    coil.setAttribute('opacity', 0.65 + frac * 0.35);
    coil.setAttribute('stroke-width', 3.2 + frac * 1.8);
  } else { coil.setAttribute('stroke', '#C3C7CC'); coil.setAttribute('filter', ''); coil.setAttribute('opacity', 1); coil.setAttribute('stroke-width', 3.2); }
}
function pumpSymbol(svg, x, y, label) {
  const g = hoverable(el('g'), label);
  const c = el('circle', { cx: x, cy: y, r: 15, fill: '#fff', stroke: '#B0B4B9', 'stroke-width': 1.8 });
  g.appendChild(c);
  g.appendChild(el('path', { d: `M${x - 6},${y - 7} L${x + 8},${y} L${x - 6},${y + 7} Z`, fill: SPG }));
  g.appendChild(el('text', { x, y: y + 28, fill: '#585C62', 'font-size': 10, 'text-anchor': 'middle' }, txt(label)));
  svg.appendChild(g);
  return c;
}
function flowPipe(svg, d, color = '#5B8DEF') {
  svg.appendChild(el('path', { d, stroke: '#DCE0E5', 'stroke-width': 6, fill: 'none', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  const f = el('path', {
    d, stroke: color, 'stroke-width': 3, fill: 'none', 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    class: 'pipe-flow', 'stroke-dasharray': '7 7', opacity: 0, 'marker-end': 'url(#arrow)',
  });
  svg.appendChild(f);
  return f;
}
const pct = (v) => `${Math.round(v * 100)}%`;

export function buildSchematic(host, meta, opts = {}) {
  return buildCSTR(host, meta, opts);
}

// ---------------- CSTR (exothermic reactor + cooling jacket) ----------------
function buildCSTR(host, meta, opts = {}) {
  host.innerHTML = '';
  const compact = !!opts.compact;
  const W = 720, H = 400;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'xMidYMid meet' });
  svg.appendChild(defsBlock());
  const rx = 280, ry = 86, rw = 180, rh = 214, cx = rx + rw / 2, rbot = ry + rh;
  const feed = flowPipe(svg, `M120,150 V${ry - 14} H${cx} V${ry}`, '#2563EB');
  const prod = flowPipe(svg, `M${cx},${rbot} V${rbot + 26} H${rx + rw + 70}`, '#5B8DEF');
  svg.appendChild(el('text', { x: 120, y: 138, fill: '#585C62', 'font-size': 11, 'text-anchor': 'middle' }, txt(L('进料', 'Feed', '供給'))));
  const feedPump = pumpSymbol(svg, 120, 150, meta.actuators.pumps[0]);
  const feedBadge = compact ? null : tagBadge(svg, 56, 176, 128, meta.actuators.pumps[0], L('进料 · 开度', 'Feed · opening', '供給・開度'));
  svg.appendChild(el('text', { x: rx + rw + 78, y: rbot + 30, fill: '#585C62', 'font-size': 11 }, txt(L('产品', 'Product', '製品'))));
  const jacket = el('rect', { x: rx - 12, y: ry - 8, width: rw + 24, height: rh + 16, rx: 14, fill: 'none', stroke: '#9AD3DA', 'stroke-width': 6 });
  svg.appendChild(jacket);
  const coolIn = flowPipe(svg, `M${rx - 70},${ry + 30} H${rx - 12}`, '#0EA5C0');
  const coolOut = flowPipe(svg, `M${rx - 12},${rbot - 30} H${rx - 70}`, '#0EA5C0');
  svg.appendChild(el('text', { x: rx - 78, y: ry + 22, fill: '#0E8aa0', 'font-size': 10 }, txt(L('冷却水', 'Coolant', '冷却水'))));
  const coolBadge = compact ? null : tagBadge(svg, rx - 200, ry + 44, 128, meta.actuators.heaters[0], L('冷却 · 开度与功率', 'Cooling · opening and duty', '冷却・開度と出力'));
  const liquid = el('rect', { x: rx + 3, y: ry + 3, width: rw - 6, height: rh - 6, rx: 8, fill: '#9CC2F0', opacity: 0.62 });
  svg.appendChild(liquid);
  svg.appendChild(el('rect', { x: rx, y: ry, width: rw, height: rh, rx: 10, fill: 'none', stroke: INK, 'stroke-width': 1.8 }));
  // stirrer with a gentle SMIL sweep (alive without being noisy)
  svg.appendChild(el('line', { x1: cx, y1: ry - 4, x2: cx, y2: ry + rh * 0.55, stroke: INK, 'stroke-width': 2 }));
  const paddle = el('line', { x1: cx - 22, y1: ry + rh * 0.55, x2: cx + 22, y2: ry + rh * 0.55, stroke: INK, 'stroke-width': 3, 'stroke-linecap': 'round' });
  paddle.appendChild(el('animate', { attributeName: 'x1', values: `${cx - 22};${cx - 6};${cx - 22}`, dur: '1.6s', repeatCount: 'indefinite' }));
  paddle.appendChild(el('animate', { attributeName: 'x2', values: `${cx + 22};${cx + 6};${cx + 22}`, dur: '1.6s', repeatCount: 'indefinite' }));
  svg.appendChild(paddle);
  svg.appendChild(el('text', { x: cx, y: ry - 14, fill: '#9fb0c2', 'font-size': 12, 'font-weight': 600, 'text-anchor': 'middle' }, txt(meta.tank_labels[0])));
  const tempT = el('text', { x: cx, y: ry + 92, fill: '#0B1220', 'font-size': 30, 'font-weight': 700, 'text-anchor': 'middle', 'font-family': MONO, ...HALO }, txt('--'));
  const caT = el('text', { x: cx, y: ry + 122, fill: '#3F6B00', 'font-size': 14, 'text-anchor': 'middle', 'font-family': MONO, ...HALO }, txt('Cₐ --'));
  const spT = el('text', { x: cx, y: ry + 142, fill: SPG, 'font-size': 11, 'text-anchor': 'middle', 'font-family': MONO, ...HALO }, txt('SP --'));
  svg.appendChild(tempT); svg.appendChild(caT); svg.appendChild(spT);
  // reactor-temp indicator: economics hug <88, 88-92 amber, >92 runaway trip
  const tBar = compact ? null : analogBar(svg, rx + rw + 26, ry + 4, rh - 8,
    { min: 30, max: 110, zones: [[30, 88, 'norm'], [88, 92, 'warn'], [92, 110, 'crit']], ticks: [[88, '88', WARN], [92, '92⛔', CRIT]] });
  const trip = tripBadge(svg, cx, ry + 172, L('进料联锁切断', 'FEED TRIPPED', '供給遮断'));
  host.appendChild(svg);

  return {
    update(f) {
      const s = f.state, act = f.actuators, il = f.interlocks || {}, lim = f.limits || {};
      const T = s.temps[0], Ca = (s.conc || [0])[0], uc = act.heaters[0];
      const fieldOnly = f.mode === 'rl' && f.rl?.observationProfile === 'field_no_concentration';
      liquid.setAttribute('fill', tempColor(T));
      tempT.textContent = T.toFixed(1); tempT.setAttribute('fill', T >= (lim.t_high || 80) ? CRIT : '#0B1220');
      caT.textContent = fieldOnly ? L('Cₐ -- 无在线分析仪', 'Cₐ -- no online analyzer', 'Cₐ -- オンライン分析計なし') : `Cₐ ${Ca.toFixed(3)}`;
      caT.setAttribute('fill', fieldOnly ? '#73777D' : '#3F6B00');
      caT.setAttribute('font-size', fieldOnly ? 11 : 14);
      spT.textContent = `SP ${f.setpoints.t_sp[0].toFixed(0)}°`;
      if (tBar) tBar.set(T, f.setpoints.t_sp[0]);
      if (uc > 0.02) { jacket.setAttribute('stroke', `rgb(${Math.round(150 - uc * 130)},${Math.round(200 - uc * 40)},220)`); jacket.setAttribute('filter', 'url(#glowPump)'); jacket.setAttribute('stroke-width', 6 + uc * 3); }
      else { jacket.setAttribute('stroke', '#cdd6da'); jacket.setAttribute('filter', ''); jacket.setAttribute('stroke-width', 6); }
      const feedOn = s.pump_flow[0] > 1e-9 && !il.pump_trip;
      feedPump.setAttribute('stroke', feedOn ? '#2dd4bf' : '#B0B4B9'); feedPump.setAttribute('filter', feedOn ? 'url(#glowPump)' : '');
      if (feedBadge) feedBadge.set(pct(act.pumps[0]), il.pump_trip ? 'crit' : null);
      if (coolBadge) coolBadge.set(`${pct(uc)} · ${(s.heater_power[0] / 1000).toFixed(1)} kW`, T >= 88 ? 'warn' : null);
      setFlow(feed, s.pump_flow[0], 0.02); setFlow(prod, s.pump_flow[0], 0.02);
      setFlow(coolIn, uc * 0.02, 0.02); setFlow(coolOut, uc * 0.02, 0.02);
      trip.show(!!il.pump_trip);
    },
  };
}
