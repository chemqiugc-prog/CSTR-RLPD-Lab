// Interactive control panels, driven entirely by the active model's metadata:
// manual actuator sliders (per pump/valve/heater), PID setpoints (only the
// controlled levels + every tank temperature) with gain tuning, the
// disturbance/fault toggles. All actions go to the engine via the bus.
import { t } from './i18n.js?v=36';
import { BUILTIN_POLICIES } from './sim/controllers.js?v=1';

function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const k in props) {
    if (k === 'class') e.className = props[k];
    else if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), props[k]);
    else e.setAttribute(k, props[k]);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}

// Bilingual labels for the disturbance catalogue (keyed by CATALOG key).
const DIST_LABEL = {
  cold_inlet: () => t('冷进料温度阶跃', 'Cold inlet temp step', '冷供給温度ステップ'),
  ambient: () => t('环境温度变化', 'Ambient temp change', '外気温度変化'),
  sensor_noise: () => t('传感器噪声', 'Sensor noise', 'センサーノイズ'),
  heater_fault: () => t('冷却执行器失效（卡低）', 'Cooling actuator failed low', '冷却操作器故障（低位固着）'),
  pump_trip: () => t('泵跳闸（无进料）', 'Pump trip (no inflow)', 'ポンプトリップ（供給なし）'),
  cooling_water_hot: () => t('冷却水入口升温', 'Cooling-water inlet rise', '冷却水入口温度上昇'),
  cooling_flow_loss: () => t('冷却循环流量损失', 'Cooling circulation loss', '冷却循環流量低下'),
  ua_fouling: () => t('夹套传热结垢', 'Jacket heat-transfer fouling', 'ジャケット伝熱汚れ'),
};

// The PRICED objective each RL policy optimizes (mirrors aiogym/env.py ECON and the
// scoring panel — same numbers everywhere): profit ¥/h = value − energy×price − penalty.
const REWARD_FN = {
  cstr:      { zh: '利润(¥/h) = 1575·产量 − 0.7·冷却功率(kW) − 14·超温(T>88°C)', en: 'profit(¥/h) = 1575·prod − 0.7·cooling(kW) − 14·over(T>88°C)', ja: '利益(¥/h) = 1575·生産量 − 0.7·冷却出力(kW) − 14·過温(T>88°C)',
               nZh: '最大化产量(贴 88°C 安全边界,92°C 失控);冷却为成本,越界按罚价扣。', nEn: 'Max production hugging the 88°C safe edge (92°C runaway); cooling is cost; over-cap billed at the penalty rate.', nJa: '88°C の安全限界ギリギリで生産量を最大化(92°C で暴走);冷却はコスト、超過はペナルティ課金。' },
  cstr_jacketed_p2: { zh: 'P2.4 独立口径：利润(¥/h) = 1575·出口产物 − 0.7·实际冷却流量等效功率(kW) − 14·max(T−88,0)/10 − 50·I(T>92)', en: 'P2.4 metric: profit(¥/h) = 1575·product − 0.7·actual-coolant-flow equivalent power(kW) − 14·max(T−88,0)/10 − 50·I(T>92)', ja: 'P2.4 独立指標：利益 = 製品価値 − 実冷却流量コスト − 過温・暴走ペナルティ',
               nZh: '界面利润只用于同一 P2.4 环境横向比较；固定 60°C PID 基准为 −4.26 ¥/h。物理泵功率单独显示，尚未计入 reward。', nEn: 'Profit is comparable only within the same P2.4 environment; the fixed-60°C PID baseline is −4.26 ¥/h. Physical pump power is shown separately and is not yet included in the reward.', nJa: '利益は同一 P2.4 環境内のみ比較可能。固定60°C PID基準は −4.26 ¥/h。物理ポンプ動力は別表示で、rewardには未算入です。' },
};

export function buildControls(bus, meta, catalog) {
  const dragging = new Set();
  let mode = 'manual';
  const n = meta.n_tanks;
  const A = meta.actuators;
  const heaterKW = (meta.heater_max || []).map((w) => w / 1000);

  function slider(kind, index, label, fmt, init, cls) {
    const sid = `${kind}${index}`;
    const val = h('span', { class: 'val' }, fmt(init));
    const inp = h('input', {
      type: 'range', min: 0, max: 1, step: 0.01, value: init, class: cls,
      oninput: (e) => { val.textContent = fmt(+e.target.value); bus.send({ type: 'manual_cmd', kind, index, value: +e.target.value }); },
      onpointerdown: () => dragging.add(sid), onpointerup: () => dragging.delete(sid),
    });
    inp.dataset.sid = sid;
    return h('div', { class: 'ctrl-row' }, h('div', { class: 'ctrl-label' }, h('span', { class: 'name' }, label), val), inp);
  }

  function manualPanel() {
    const w = h('div');
    w.append(h('div', { class: 'group-title' }, t('进料泵', 'Pumps', '供給ポンプ')));
    A.pumps.forEach((lab, i) => w.append(slider('pump', i, lab, (v) => `${(v * 100).toFixed(0)}%`, 0.3, '')));
    if (A.valves.length) {
      w.append(h('div', { class: 'divider' }), h('div', { class: 'group-title' }, t('出料阀', 'Valves', '排出バルブ')));
      A.valves.forEach((lab, i) => w.append(slider('valve', i, lab, (v) => `${(v * 100).toFixed(0)}%`, 0.5, 'valve')));
    }
    w.append(h('div', { class: 'divider' }), h('div', { class: 'group-title' }, t('冷却执行器', 'Cooling actuator', '冷却操作器')));
    A.heaters.forEach((lab, i) => w.append(slider('heater', i, lab, (v) => `${(v * (heaterKW[i] || 90)).toFixed(1)} kW`, 0, 'heater')));
    return w;
  }

  function pidPanel(frame) {
    const sp = frame?.setpoints || { h_sp: [], t_sp: [] };
    const cfg = frame?.pid;
    const ctrl = meta.controlled_levels || [];
    const w = h('div');

    // level setpoints (only controlled). A model may relabel/rescale the "level"
    // quantity via metadata.sp_spec (the fired heater stores flue O₂ there).
    const spec = meta.sp_spec || {};
    const lv = spec.level || { label: ['液位设定 (m)', 'Level SP (m)', '液位設定値 (m)'], min: 0, max: 0.8, step: 0.01 };
    const tv = spec.temp || { label: ['温度设定 (°C)', 'Temp SP (°C)', '温度設定値 (°C)'], min: 10, max: 90, step: 1 };
    const lvDp = lv.step < 0.1 ? 2 : 1, tvDp = tv.step < 1 ? 1 : 0;
    w.append(h('div', { class: 'group-title' }, t(...lv.label)));
    ctrl.forEach((idx) => {
      const inp = h('input', { type: 'number', step: lv.step, min: lv.min, max: lv.max, value: (sp.h_sp[idx] ?? lv.min).toFixed(lvDp), onchange: sendSP });
      inp.dataset.hsp = idx;
      w.append(h('div', { class: 'sp-row' }, h('label', {}, meta.tank_labels[idx]), inp, h('span')));
    });
    // temperature setpoints (every tank)
    w.append(h('div', { class: 'group-title', style: 'margin-top:10px' }, t(...tv.label)));
    for (let i = 0; i < n; i++) {
      const inp = h('input', { type: 'number', step: tv.step, min: tv.min, max: tv.max, value: (sp.t_sp[i] ?? 50).toFixed(tvDp), onchange: sendSP });
      inp.dataset.tsp = i;
      w.append(h('div', { class: 'sp-row' }, h('label', {}, meta.tank_labels[i]), inp, h('span')));
    }
    function sendSP() {
      const h_sp = Array(n).fill(0);
      w.querySelectorAll('[data-hsp]').forEach((e) => { h_sp[+e.dataset.hsp] = +e.value; });
      const t_sp = Array(n).fill(50);
      w.querySelectorAll('[data-tsp]').forEach((e) => { t_sp[+e.dataset.tsp] = +e.value; });
      bus.send({ type: 'set_setpoints', h_sp, t_sp });
    }

    // demand valve (only where the model declares one — e.g. cascade's V-3)
    if (meta.demand_valve_index != null) {
      const dv = cfg?.demand_valve ?? 0.5;
      const dval = h('span', { class: 'val' }, `${(dv * 100) | 0}%`);
      w.append(h('div', { class: 'divider' }), h('div', { class: 'group-title' }, t('需求阀（扰动）', 'Demand valve (disturbance)', '需要バルブ（外乱）')),
        h('div', { class: 'ctrl-row' }, h('div', { class: 'ctrl-label' }, h('span', { class: 'name' }, t('下游需求', 'Downstream demand', '下流需要')), dval),
          h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: dv, class: 'valve', oninput: (e) => { dval.textContent = `${(e.target.value * 100) | 0}%`; bus.send({ type: 'set_pid', demand_valve: +e.target.value }); } })));
    }

    // PID gains
    const tune = h('details', { class: 'tune' }, h('summary', {}, t('PID 整定', 'PID tuning', 'PID 整定')));
    const g = cfg?.gains || {};
    const grid = h('div', { class: 'gain-grid' }, h('span'), h('span', { class: 'gh' }, 'Kp'), h('span', { class: 'gh' }, 'Ki'), h('span', { class: 'gh' }, 'Kd'));
    const loops = A.valves.length
      ? [['level_pump', t('液位·泵', 'Level·pump', '液位·ポンプ')], ['level_valve', t('液位·阀', 'Level·valve', '液位·バルブ')], ['temp', t('温度', 'Temp', '温度')]]
      : [['level_pump', t('液位·泵', 'Level·pump', '液位·ポンプ')], ['temp', t('温度', 'Temp', '温度')]];
    for (const [key, label] of loops) {
      grid.append(h('label', {}, label));
      for (const p of ['kp', 'ki', 'kd']) {
        const gi = h('input', { type: 'number', step: 0.001, value: (g[key]?.[p] ?? 0), onchange: sendGains });
        gi.dataset.gk = key; gi.dataset.gp = p; grid.append(gi);
      }
    }
    function sendGains() {
      const gains = {};
      grid.querySelectorAll('[data-gk]').forEach((e) => { (gains[e.dataset.gk] = gains[e.dataset.gk] || {})[e.dataset.gp] = +e.value; });
      bus.send({ type: 'set_pid', gains });
    }
    tune.append(grid); w.append(tune);
    return w;
  }

  // disturbances — auto by default (header checkbox); manual injection collapsed below
  function disturbPanel() {
    const w = h('div'), cat = catalog.disturbances;
    const det = h('details', { class: 'tune' }, h('summary', {}, t('手动注入', 'Manual inject', '手動注入')));
    const box0 = h('div');
    for (const key in cat) {
      const def = cat[key];
      if (def.needs === 'jacketed_cstr' && !meta.scenario.startsWith('cstr_jacketed')) continue;
      const toggle = h('div', { class: 'toggle', 'data-dist': key, onclick: (e) => {
        const on = e.target.classList.toggle('on');
        if (on) bus.send({ type: 'set_disturbance', dtype: key, params: readParams(key) });
        else bus.send({ type: 'clear_disturbance', dtype: key });
      } });
      const label = (DIST_LABEL[key] ? DIST_LABEL[key]() : def.label);
      box0.append(h('div', { class: 'dist-item' }, h('div', { class: 'dn' }, label, h('small', {}, def.kind === 'fault' ? t('故障', 'fault', '故障') : t('扰动', 'disturbance', '外乱'))), toggle));
      const pb = paramInputs(key, def, () => { if (toggle.classList.contains('on')) bus.send({ type: 'set_disturbance', dtype: key, params: readParams(key) }); });
      if (pb) box0.append(pb);
    }
    det.append(box0); w.append(det);
    return w;
    function readParams(key) {
      const box = w.querySelector(`[data-pbox="${key}"]`); if (!box) return {};
      const out = {}; box.querySelectorAll('[data-pk]').forEach((e) => { out[e.dataset.pk] = +e.value; }); return out;
    }
  }
  function paramInputs(key, def, onchg) {
    const d = def.default; if (!d || !Object.keys(d).length) return null;
    const box = h('div', { class: 'dist-param', 'data-pbox': key });
    for (const pk in d) {
      if (pk === 'index') {
        const sel = h('select', { onchange: onchg }); sel.dataset.pk = pk;
        for (let i = 0; i < n; i++) sel.append(h('option', { value: i }, meta.tank_labels[i]));
        sel.value = d[pk]; box.append(h('span', {}, t('目标', 'Target', '対象')), sel);
      } else {
        const inp = h('input', { type: 'number', step: pk.includes('std') ? 0.01 : (key.includes('demand') ? 0.0001 : 0.5), value: d[pk], onchange: onchg });
        inp.dataset.pk = pk; box.append(h('span', {}, labelFor(pk)), inp);
      }
    }
    return box;
  }
  const labelFor = (pk) => ({ value: t('幅度', 'Amplitude', '振幅'), level_std: t('液位σ', 'Level σ', '液位σ'), temp_std: t('温度σ', 'Temp σ', '温度σ') }[pk] || pk);

  function fillPolicySummary(host, policy) {
    host.innerHTML = '';
    host.dataset.policyId = policy?.id || '';
    host.className = `rl-policy-summary ${policy?.uiRole || ''}`;
    if (!policy) return;

    const role = policy.uiRole === 'recommended'
      ? t('当前推荐', 'Recommended', '推奨')
      : policy.uiRole === 'reference'
        ? t('标准对照', 'Reference', '比較')
        : t('当前策略', 'Active policy', '現在の方策');
    const summary = t(policy.summaryZh || policy.noteZh, policy.summaryEn || policy.noteEn, policy.summaryJa || policy.noteJa);
    host.append(h('div', { class: 'rl-policy-head' },
      h('span', { class: 'rl-policy-role' }, role),
      h('span', { class: 'rl-policy-scope' }, t('同种子 100 × 600 秒', 'Same seeds · 100 × 600 s', '同一シード · 100 × 600秒'))),
    h('p', { class: 'rl-policy-desc' }, summary));

    if (!policy.benchmark) return;
    const labels = [
      [t('平均利润', 'Mean profit', '平均利益'), policy.benchmark.profit],
      [t('控制 KPI', 'Control KPI', '制御 KPI'), policy.benchmark.kpi],
      [t('超过 88°C', 'Over 88°C', '88°C 超過'), policy.benchmark.over88],
      [t('跳闸回合', 'Trip episodes', 'トリップ'), policy.benchmark.trips],
    ];
    const metrics = h('div', { class: 'rl-policy-metrics' });
    labels.forEach(([label, value]) => metrics.append(h('div', { class: 'rl-policy-metric' },
      h('span', {}, label), h('strong', { class: 'mono' }, value))));
    host.append(metrics);
  }

  function policySummary(policy) {
    const host = h('div', { id: 'rl-policy-summary' });
    fillPolicySummary(host, policy);
    return host;
  }

  // RL (supervisory): the scenario's default policy auto-loads and the inner PID
  // executes its targets. The selector is reserved for a small set of useful references.
  function rlPanel(frame) {
    const rl = frame?.rl || {};
    const scn = frame?.scenario;
    const policies = BUILTIN_POLICIES.filter((p) => p.scenario === scn);
    const selectedId = rl.policyId || policies[0]?.id;
    const cur = policies.find((p) => p.id === selectedId) || policies[0];
    const w = h('div');
    w.append(h('div', { class: 'group-title' }, t('RL 监督控制', 'RL supervisory control', 'RL 監督制御')));
    w.append(h('div', { class: 'rl-status', id: 'rl-status' }, rl.status || t('加载中…', 'Loading…', '読み込み中…')));
    const summary = policySummary(cur);
    if (policies.length > 1) {
      const sel = h('select', { id: 'rl-policy-select', class: 'rl-policy-select', onchange: (e) => {
        const policy = policies.find((p) => p.id === e.target.value);
        if (policy) {
          fillPolicySummary(summary, policy);
          bus.send({ type: 'set_rl_policy', src: policy.url, mode: policy.mode, policy });
        }
      } });
      for (const p of policies) sel.append(h('option', { value: p.id }, t(p.zh, p.en, p.ja)));
      sel.value = selectedId || policies[0].id;
      w.append(h('label', { class: 'rl-policy-row' }, h('span', {}, t('控制策略', 'Control policy', '制御方策')), sel));
    }
    w.append(summary);
    w.append(h('div', { class: 'hint rl-supervision-note' }, t(
      'RL 给出温度与进料目标，内层 PID 负责执行；PID 基准可在顶部直接切换。',
      'RL supplies temperature and feed targets; the inner PID executes them. Switch to the PID baseline from the top bar.',
      'RL が温度・供給目標を設定し、内層 PID が実行します。PID 基準は上部で切り替えられます。')));

    // Keep technical scoring details available without crowding the operating view.
    const rf = REWARD_FN[scn];
    if (rf) {
      const metric = h('details', { class: 'tune rl-details' }, h('summary', {}, t('评估口径', 'Evaluation metric', '評価指標')));
      metric.append(h('div', { class: 'reward-formula' }, t(rf.zh, rf.en, rf.ja)),
        h('div', { class: 'hint' }, t(rf.nZh, rf.nEn, rf.nJa)));
      w.append(metric);
    }

    // advanced (collapsed): load a custom .onnx policy by file or URL
    const adv = h('details', { class: 'tune rl-details' }, h('summary', {}, t('高级设置', 'Advanced settings', '詳細設定')));
    const file = h('input', { type: 'file', accept: '.onnx',
      onchange: (e) => { const f = e.target.files[0]; if (f) f.arrayBuffer().then((b) => bus.send({ type: 'set_rl_policy', src: new Uint8Array(b) })); } });
    const url = h('input', { type: 'text', placeholder: t('models/policy.onnx 或 URL', 'models/policy.onnx or URL', 'models/policy.onnx または URL') });
    adv.append(h('label', { class: 'rl-load' }, h('span', {}, t('文件', 'File', 'ファイル')), file),
               h('div', { class: 'dist-param' }, url, h('button', { class: 'mini', onclick: () => { if (url.value) bus.send({ type: 'set_rl_policy', src: url.value }); } }, t('加载 URL', 'Load URL', 'URL を読み込み'))));
    w.append(adv);
    return w;
  }

  return {
    setMode(m) { mode = m; },
    renderControl(host, m, frame, subEl) {
      mode = m; host.innerHTML = '';
      if (m === 'pid') { host.append(pidPanel(frame)); if (subEl) subEl.textContent = t('PID 自动', 'PID auto', 'PID 自動'); }
      else if (m === 'rl') { host.append(rlPanel(frame)); if (subEl) subEl.textContent = t('RL 监督', 'RL supervisory', 'RL 監督'); }
      else { host.append(manualPanel()); if (subEl) subEl.textContent = t('手动', 'Manual', '手動'); }
    },
    syncRL(frame) {
      if (mode !== 'rl') return;
      const el = document.getElementById('rl-status');
      if (el && frame.rl) el.textContent = frame.rl.status;
      const sel = document.getElementById('rl-policy-select');
      if (sel && frame.rl?.policyId && sel.value !== frame.rl.policyId) sel.value = frame.rl.policyId;
      const summary = document.getElementById('rl-policy-summary');
      const policy = BUILTIN_POLICIES.find((p) => p.scenario === frame?.scenario && p.id === frame?.rl?.policyId);
      if (summary && policy && summary.dataset.policyId !== policy.id) fillPolicySummary(summary, policy);
    },
    renderDisturb(host) { host.innerHTML = ''; host.append(disturbPanel()); },
    syncManual(frame) {
      if (mode !== 'manual') return;
      const c = frame.command;
      const setLab = (sid, v, fmt) => {
        if (dragging.has(sid)) return;
        const el = document.querySelector(`input[data-sid="${sid}"]`);
        if (el && Math.abs(+el.value - v) > 1e-3) { el.value = v; const lab = el.parentElement.querySelector('.val'); if (lab) lab.textContent = fmt(v); }
      };
      c.pumps.forEach((v, i) => setLab(`pump${i}`, v, (x) => `${(x * 100).toFixed(0)}%`));
      c.valves.forEach((v, i) => setLab(`valve${i}`, v, (x) => `${(x * 100).toFixed(0)}%`));
      c.heaters.forEach((v, i) => setLab(`heater${i}`, v, (x) => `${(x * (heaterKW[i] || 90)).toFixed(1)} kW`));
    },
    syncConfig(frame) {
      const cfg = frame?.meta?.config; if (!cfg) return;
      const tag = document.getElementById('phase-tag'); if (tag && cfg.phase) tag.textContent = cfg.phase;
    },
    syncDisturb(host, active) {
      host.querySelectorAll('[data-dist]').forEach((t) => t.classList.toggle('on', !!active[t.getAttribute('data-dist')]));
    },
  };
}
