// App orchestrator: runs the simulation engine in-browser (no server) and wires
// its telemetry to the schematic, charts and panels, and the top-bar controls
// back to the engine. Rebuilds the whole UI when the scenario changes.
import { Engine, CATALOG } from './sim/engine.js?v=1';
import { buildSchematic } from './schematic.js?v=52';
import { buildCharts } from './charts.js?v=36';
import { buildControls } from './controls.js?v=1';
import { t, applyStatic, toggleLang, lang, onLang, nextLang } from './i18n.js?v=36';

const $ = (s) => document.querySelector(s);
let schematic, charts, controls, catalog, meta;
let scenario = null, mode = 'manual', running = true, lastFrame = null, pendingHistory = null;

// Local engine; `bus.send` forwards commands to it (keeps controls.js unchanged).
const query = new URLSearchParams(window.location.search);
const validScenarios = new Set(['cstr', 'cstr_jacketed_p2']);
const validModes = new Set(['manual', 'pid', 'rl']);
const initialScenario = validScenarios.has(query.get('scenario')) ? query.get('scenario') : 'cstr_jacketed_p2';
const initialMode = validModes.has(query.get('mode')) ? query.get('mode') : 'manual';
const engine = new Engine(initialScenario);
if (initialMode !== 'manual') engine.setMode(initialMode);
const bus = { send: (m) => engine.handleCommand(m) };

function setConn(ok) {
  const c = $('#conn'); if (!c) return;
  c.className = 'conn ' + (ok ? 'on' : 'off');
  c.textContent = ok ? t('● 本地引擎', '● Local engine', '● ローカルエンジン') : t('● 已停止', '● Stopped', '● 停止しました');
}

function init() {
  catalog = { disturbances: CATALOG, n_tanks: 3 };
  wireTopbar();
  applyStatic();            // reflect the saved language on the static topbar/panels
  setLangBtn();
  onLang(relayout);         // re-render everything when the language changes
  window.addEventListener('resize', () => charts && charts.resize());
  setConn(true);
  engine.start(onFrame);   // 20 Hz frames, identical shape to the old WS stream
}

// Language switch: static text is already swapped by i18n; rebuild the dynamic
// panels with a fresh (re-localized) telemetry frame.
function relayout() {
  setLangBtn();
  setRunBtn();                                  // run button text is language-dependent
  scenario = null;                              // force rebuildUI with fresh meta
  onFrame(engine.telemetry());
}

function rebuildUI(f) {
  meta = f.meta;
  const inf = $('#pid-info');
  if (inf) { inf.hidden = !meta.blurb; inf.title = meta.blurb || ''; }
  if (charts) charts.destroy();
  schematic = buildSchematic($('#schematic-host'), meta);
  charts = buildCharts($('#charts-host'), meta.trends, meta.n_tanks);
  controls = buildControls(bus, meta, catalog);
  controls.renderDisturb($('#disturb-body'));
  controls.renderControl($('#control-body'), mode, f, $('#control-sub'));
  setTimeout(() => charts && charts.resize(), 60);
}

function onFrame(f) {
 try {
  if (f.type === 'history') { if (charts) charts.fromHistory(f.samples); else pendingHistory = f.samples; return; }
  if (f.type !== 'telemetry') return;
  setConn(true);
  lastFrame = f;
  if (f.scenario !== scenario) { mode = f.mode; rebuildUI(f); scenario = f.scenario; syncSegs(f); }

  schematic.update(f);
  charts.push(f);
  controls.syncManual(f);
  controls.syncConfig(f);
  controls.syncRL(f);
  controls.syncDisturb($('#disturb-body'), f.disturbances || {});
  notifyDisturb(f.disturbances || {});
  renderScore(f.score, f.episode);
  renderAlarms(f.alarms, f.interlocks);
  updateTopbar(f);
  renderEnvironmentProfile(f);
 } catch (e) { console.error('onFrame error:', e && e.stack || e); }
}

// ---------------- top bar ----------------
function wireTopbar() {
  // scenario switcher dropdown
  const scnSwitch = $('#scn-switch'), scnMenu = $('#scenario-seg');
  const closeScn = () => { scnMenu.hidden = true; scnSwitch.classList.remove('open'); };
  $('#scn-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const open = scnMenu.hidden; scnMenu.hidden = !open; scnSwitch.classList.toggle('open', open);
  });
  scnMenu.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    const url = new URL(window.location.href);
    url.searchParams.set('scenario', b.dataset.scenario);
    window.history.replaceState({}, '', url);
    bus.send({ type: 'set_scenario', scenario: b.dataset.scenario });
    closeScn();
  });
  document.addEventListener('click', (e) => { if (!scnSwitch.contains(e.target)) closeScn(); });
  $('#mode-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b || b.disabled) return;
    const url = new URL(window.location.href);
    url.searchParams.set('mode', b.dataset.mode);
    window.history.replaceState({}, '', url);
    mode = b.dataset.mode; bus.send({ type: 'set_mode', mode }); syncModeSeg(mode);
    controls.renderControl($('#control-body'), mode, lastFrame, $('#control-sub'));
  });
  $('#speed').addEventListener('input', (e) => { $('#speed-val').textContent = (+e.target.value).toFixed(1) + '×'; bus.send({ type: 'set_speed', speed: +e.target.value }); });
  $('#btn-run').addEventListener('click', () => { running = !running; bus.send({ type: 'set_running', running }); setRunBtn(); });
  $('#btn-reset').addEventListener('click', () => bus.send({ type: 'reset' }));
  $('#lang-btn').addEventListener('click', () => toggleLang());
  $('#auto-events').addEventListener('change', (e) => bus.send({ type: 'set_auto_events', on: e.target.checked }));
  setRunBtn();
}

const setRunBtn = () => { const b = $('#btn-run'); b.textContent = running ? t('暂停', 'Pause', '一時停止') : t('运行', 'Run', '実行'); b.classList.toggle('paused', !running); };
const setLangBtn = () => {
  const b = $('#lang-btn'); if (b) b.textContent = { zh: '中', en: 'EN', ja: '日本語' }[nextLang()];  // shows the lang you switch TO
  document.title = t('CSTR-RLPD 实验室', 'CSTR-RLPD Lab', 'CSTR-RLPD ラボ');
};
const syncModeSeg = (m) => $('#mode-seg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.mode === m));
function syncSegs(f) {
  $('#scenario-seg').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.scenario === f.scenario));
  syncModeSeg(f.mode);
}
function updateTopbar(f) {
  $('#clock').textContent = f.t.toFixed(1) + 's';
  if (f.mode !== mode) { mode = f.mode; syncModeSeg(mode); controls.renderControl($('#control-body'), mode, f, $('#control-sub')); }
  if (f.running !== running) { running = f.running; setRunBtn(); }
  const title = $('#scn-title'); if (title && f.meta) title.textContent = f.meta.name;
  const MK = { cstr: '⊚', cstr_jacketed_p2: '▣' };
  const mk = $('#scn-mark'); if (mk) mk.textContent = MK[f.scenario] || '▣';
  const p = $('#scn-path'); if (p) p.textContent = `cstr / ${f.scenario}`;
  const c = f.state.t_cold.toFixed(1), a = f.state.t_amb.toFixed(1);
  $('#env-readout').textContent = t(`进料 ${c}°C · 环境 ${a}°C`, `Feed ${c}°C · Amb ${a}°C`, `フィード ${c}°C · 環境 ${a}°C`);
  $('#score-mode').textContent = f.mode.toUpperCase();
}

function renderEnvironmentProfile(f) {
  const host = $('#env-profile');
  if (!host) return;
  const p = f.profile;
  host.hidden = !p;
  if (!p) { host.innerHTML = ''; return; }
  const regimeName = {
    normal: t('正常工况', 'Normal', '通常運転'), reactive_feed: t('高活性进料', 'Reactive feed', '高反応性供給'),
    fouled: t('换热结垢', 'Fouled', '伝熱汚れ'), utility_limited: t('公用工程受限', 'Utility limited', 'ユーティリティ制約'),
  }[p.regime] || p.regime;
  const eventNames = { utility_stress: t('公用工程压力', 'Utility stress', 'ユーティリティ負荷'), fouling_ramp: t('渐进结垢', 'Fouling ramp', '漸進汚れ'), feed_batch: t('进料批次变化', 'Feed batch shift', '供給バッチ変動'), pump_degradation: t('循环泵退化', 'Circulation-pump degradation', '循環ポンプ劣化') };
  const activeLong = p.events.filter((event) => event.progress > 0).map((event) => `${eventNames[event.kind] || event.kind} ${(100 * event.progress).toFixed(0)}%`).join(' · ');
  const microNames = { feed_temp_pulse: t('进料温度脉冲', 'Feed-temperature pulse', '供給温度パルス'), coolant_temp_pulse: t('冷却水温脉冲', 'Coolant-temperature pulse', '冷却水温パルス'), cooling_flow_dip: t('冷却流量轻降', 'Cooling-flow dip', '冷却流量低下') };
  const activeMicro = (p.micro_events || []).filter((event) => event.progress > 0).map((event) => `${microNames[event.kind] || event.kind} ${(100 * event.progress).toFixed(0)}%`).join(' · ');
  const active = [activeLong, activeMicro].filter(Boolean).join(' | ') || t('轻扰动间歇', 'Between mild disturbances', '軽外乱の間欠区間');
  const trueT = f.state.temps[0], measuredT = f.measurements?.temps?.[0] ?? trueT;
  const cmdFeed = f.controller_command?.pumps?.[0] ?? 0, actFeed = f.actuators?.pumps?.[0] ?? 0;
  const process = p.process || {};
  const coolantSupply = f.state.coolant_supply_temp?.[0] ?? process.t_jacket_in ?? 10;
  const coolantReturn = f.state.coolant_return_temp?.[0] ?? f.state.jacket_temp?.[0] ?? coolantSupply;
  const operatingHead = f.state.cooling_operating_head?.[0] ?? process.pump_head_factor ?? 0;
  const coolingPumpKw = (f.state.cooling_pump_power?.[0] ?? 0) / 1000;
  host.innerHTML = `
    <div class="env-profile-head"><div><b>${t('P2.4 冷却公用工程工业仿真环境', 'P2.4 industrial cooling-utility environment', 'P2.4 冷却ユーティリティ産業シミュレーション')}</b><span>${t('泵/系统曲线 · 有限供水热容 · 显式污垢热阻', 'pump/system curves · finite coolant inventory · explicit fouling resistance', 'ポンプ/系統曲線 · 有限冷却水熱容量 · 明示的汚れ抵抗')}</span></div><strong class="${p.sis_latched ? 'trip' : ''}">${p.sis_latched ? 'SIS TRIPPED' : 'SIS ARMED'}</strong></div>
    <div class="env-profile-grid">
      <div><span>${t('本回合工况', 'Episode regime', '運転条件')}</span><b>${regimeName}</b></div>
      <div><span>${t('当前扰动', 'Current disturbance', '現在の外乱')}</span><b>${active}</b></div>
      <div><span>${t('真实 / 测量温度', 'True / measured T', '真値 / 計測温度')}</span><b class="mono">${trueT.toFixed(2)} / ${measuredT.toFixed(2)} °C</b></div>
      <div><span>${t('进料指令 / 实际', 'Feed command / actual', '供給指令 / 実値')}</span><b class="mono">${(100 * cmdFeed).toFixed(1)} / ${(100 * actFeed).toFixed(1)}%</b></div>
      <div><span>${t('冷却供水目标 / 实际', 'Coolant target / actual supply', '冷却供水目標 / 実値')}</span><b class="mono">${(process.coolant_supply_target_temperature ?? coolantSupply).toFixed(1)} / ${coolantSupply.toFixed(1)}°C</b></div>
      <div><span>${t('有效换热能力 UA', 'Effective heat-transfer UA', '有効伝熱能力 UA')}</span><b class="mono">${(100 * (process.effective_ua_factor ?? 1)).toFixed(0)}%</b></div>
      <div><span>${t('进料浓度 / 活性因子', 'Feed concentration / activity', '供給濃度 / 活性係数')}</span><b class="mono">${(100 * (process.feed_concentration_factor ?? 1)).toFixed(1)} / ${(100 * (process.kinetics_factor ?? 1)).toFixed(1)}%</b></div>
      <div><span>${t('实际流量 / 工作点扬程', 'Actual flow / operating head', '実流量 / 運転点揚程')}</span><b class="mono">${(100 * (f.state.cooling_flow_fraction?.[0] ?? 0)).toFixed(1)} / ${(100 * operatingHead).toFixed(0)}%</b></div>
      <div><span>${t('泵可用扬程 / 目标', 'Available / target pump head', '利用可能 / 目標ポンプ揚程')}</span><b class="mono">${(100 * (process.pump_head_factor ?? 1)).toFixed(0)} / ${(100 * (process.pump_head_target_factor ?? 1)).toFixed(0)}%</b></div>
      <div><span>${t('冷却回水 / 循环泵功率', 'Coolant return / circulation power', '冷却戻水 / 循環ポンプ動力')}</span><b class="mono">${coolantReturn.toFixed(1)}°C / ${coolingPumpKw.toFixed(1)} kW</b></div>
      <div><span>${t('污垢热阻 / 实际UA', 'Fouling resistance / actual UA', '汚れ熱抵抗 / 実UA')}</span><b class="mono">${(1e6 * (process.fouling_resistance_k_per_w ?? 0)).toFixed(2)} µK/W / ${((f.state.effective_ua?.[0] ?? 0) / 1000).toFixed(1)} kW/K</b></div>
    </div>`;
}

// ---------------- score ----------------
function renderScore(sc, ep) {
  if (!sc) return;
  const k = sc.kpis, ec = sc.econ || {};
  const isProfit = ec.value === 'production';
  const rate = ec.profit_rate ?? 0;
  const vsPct = ec.vs_base_pct ?? 0;               // + = better than the PID baseline
  const better = vsPct >= 0;
  // money formatting: cost scenarios show a positive "cost" figure (|rate|)
  const shown = isProfit ? rate : Math.abs(rate);
  const money = (v) => { const a = Math.abs(v); const d = a < 10 ? 1 : 0; return (v < 0 ? '−' : '') + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); };
  const baselineMoney = (v) => (v < 0 ? '−' : '') + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const mcol = isProfit ? (rate >= 0 ? '#2E8B3D' : '#C0392B') : (better ? '#2E8B3D' : '#C77700');
  const tcol = sc.score >= 80 ? '#2E8B3D' : sc.score >= 55 ? '#C77700' : '#C0392B';
  const vsDelta = money(rate - (ec.base ?? 0));
  // profit scenarios: percent breaks across the zero (PID loses money, RL earns —
  // "164% saved" reads wrong), so state the absolute gap; costs keep the intuitive %.
  const vsTxt = isProfit
    ? (better ? t(`多赚 ${vsDelta} ¥/h`, `+${vsDelta} ¥/h`, `+${vsDelta} ¥/h`) : t(`少赚 ${vsDelta} ¥/h`, `${vsDelta} ¥/h`, `${vsDelta} ¥/h`))
    : (better ? t(`省 ${Math.abs(vsPct).toFixed(1)}%`, `${Math.abs(vsPct).toFixed(1)}% better`, `${Math.abs(vsPct).toFixed(1)}% 節約`)
              : t(`多花 ${Math.abs(vsPct).toFixed(1)}%`, `${Math.abs(vsPct).toFixed(1)}% worse`, `${Math.abs(vsPct).toFixed(1)}% 超過`));
  const econKpi = isProfit
    ? kpi(t('产量', 'Production', '生産量'), ec.production, 'mmol/s')
    : kpi(t('能耗功率', 'Power', '消費電力'), k.avg_power_kw, 'kW');
  ep = ep || { n: 1, elapsed: 0, length: 600, history: [] };
  const hist = ep.history || [];
  const last = hist[hist.length - 1];
  // sparkline of recent episodes: bar height = episode ¥/h mapped over [worst, best]
  const es = sc.econ_score ?? 0;                    // internal 0-100 mapping still drives the spark/bar
  const spark = hist.slice(-12).map((s2) => {
    const c = s2.econ >= 80 ? '#2E8B3D' : s2.econ >= 55 ? '#C77700' : '#C0392B';
    return `<i title="${money(s2.profit)} ¥/h" style="height:${Math.max(6, s2.econ)}%;background:${c}"></i>`;
  }).join('');
  $('#score-body').innerHTML = `
    <div class="score-big"><span class="score-num" style="color:${mcol}">${money(shown)}</span><span class="score-unit">¥/h ${isProfit ? t('利润', 'profit', '利益') : t('运行成本', 'operating cost', '運転コスト')}</span></div>
    <div class="score-sub" style="font-size:12px;margin-top:2px">${t('vs PID 基准', 'vs PID baseline', 'PID 基準比')}(${baselineMoney(ec.base ?? 0)}):<b style="color:${better ? '#2E8B3D' : '#C0392B'}"> ${vsTxt}</b>${isProfit ? '' : `(${better ? '−' : '+'}${money(Math.abs(rate - (ec.base ?? 0)))} ¥/h)`}</div>
    <div class="score-bar"><i style="width:${es}%;background:${mcol}"></i></div>
    <div class="score-sub">${t('回合', 'Episode', 'エピソード')} #${ep.n} · ${ep.elapsed}/${ep.length}s${last ? ` · ${t('上回合', 'last', '前回')} <b>${money(last.profit)}</b>` : ''}</div>
    ${spark ? `<div class="ep-spark">${spark}</div>` : ''}
    <div class="score-sub">${t('控制 KPI', 'Control KPI', '制御 KPI')} <b style="color:${tcol}">${sc.score.toFixed(0)}</b>/100</div>
    <div class="kpi-grid">
      ${econKpi}${kpi(t('累计能耗', 'Energy', '累積エネルギー'), k.energy_kwh, 'kWh')}
      ${kpi(t('联锁时长', 'Interlock', 'インターロック時間'), k.interlock_seconds, 's')}${kpi(t('跳闸次数', 'Trips', 'トリップ回数'), k.trip_events, t('次', '×', '回'))}
    </div>
    <div class="hint" style="margin-top:9px">${t('利润 = 产品收入 − 冷却能耗成本 − 超温/失控罚则。每回合 = 600s 仿真，按回合平均计。',
      'Profit = product credit − cooling-energy cost − over-temperature/runaway penalties. Episode = 600 s simulation, episode-average.',
      '利益 = 製品収入 − 冷却エネルギーコスト − 過温/暴走ペナルティ。1エピソード = 600s、エピソード平均。')}</div>`;
}
const kpi = (k, v, u) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}<small> ${u}</small></div></div>`;

// ---------------- disturbance toasts ----------------
// Pop a transient notification when a disturbance/fault fires, so it's legible.
let _prevDisturb = {};
const DTOAST = {
  cold_inlet: (p) => t(`冷进料温度 +${(+p.value).toFixed(1)}°C`, `Cold inlet +${(+p.value).toFixed(1)}°C`, `冷フィード温度 +${(+p.value).toFixed(1)}°C`),
  ambient: (p) => t(`环境温度 ${+p.value >= 0 ? '+' : ''}${(+p.value).toFixed(1)}°C`, `Ambient ${+p.value >= 0 ? '+' : ''}${(+p.value).toFixed(1)}°C`, `環境温度 ${+p.value >= 0 ? '+' : ''}${(+p.value).toFixed(1)}°C`),
  sensor_noise: () => t('传感器噪声注入', 'Sensor noise injected', 'センサーノイズ注入'),
  cooling_water_hot: (p) => t(`冷却水入口 +${(+p.value).toFixed(1)}°C`, `Coolant inlet +${(+p.value).toFixed(1)}°C`, `冷却水入口 +${(+p.value).toFixed(1)}°C`),
  cooling_flow_loss: (p) => t(`冷却流量损失 ${(100 * +p.value).toFixed(0)}%`, `Cooling-flow loss ${(100 * +p.value).toFixed(0)}%`, `冷却流量損失 ${(100 * +p.value).toFixed(0)}%`),
  ua_fouling: (p) => t(`换热 UA 损失 ${(100 * +p.value).toFixed(0)}%`, `Heat-transfer UA loss ${(100 * +p.value).toFixed(0)}%`, `伝熱 UA 損失 ${(100 * +p.value).toFixed(0)}%`),
  heater_fault: () => t('冷却执行器失效（卡低）', 'Cooling actuator failed low', '冷却操作器故障（低位固着）'),
  pump_trip: () => t('泵跳闸 · 无进料', 'Pump trip · no inflow', 'ポンプトリップ · 進料なし'),
};
function notifyDisturb(dist) {
  for (const k in dist) {
    if (!(k in _prevDisturb)) {
      const fn = DTOAST[k];
      showToast(fn ? fn(dist[k].params || {}) : (dist[k].label || k), dist[k].kind === 'fault');
    }
  }
  _prevDisturb = { ...dist };
}
function showToast(msg, isFault) {
  const host = $('#toast-host'); if (!host) return;
  const el = document.createElement('div');
  el.className = 'toast' + (isFault ? ' fault' : '');
  const tag = isFault ? t('故障', 'Fault', '故障') : t('扰动', 'Disturbance', '外乱');
  el.innerHTML = `<span class="tdot"></span><span><b>${tag}</b> · ${msg}</span>`;
  host.appendChild(el);
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 400); }, 3600);
}

// ---------------- alarms ----------------
// Alarms carry a stable type + tank + value (the raw `message` stays English for
// logging); the UI renders a localized string from those fields.
function alarmText(a) {
  const u = a.tank >= 0 ? a.tank + 1 : '';
  switch (a.type) {
    case 'level_high': return t(`T-${u} 液位偏高 (${a.value.toFixed(2)} m)`, `Tank ${u} level HIGH (${a.value.toFixed(2)} m)`, `T-${u} 液位高め (${a.value.toFixed(2)} m)`);
    case 'level_low': return t(`T-${u} 液位偏低 (${a.value.toFixed(2)} m)`, `Tank ${u} level LOW (${a.value.toFixed(2)} m)`, `T-${u} 液位低め (${a.value.toFixed(2)} m)`);
    case 'temp_high': return t(`机组 ${u} 温度偏高 (${a.value.toFixed(1)} °C)`, `Unit ${u} temperature HIGH (${a.value.toFixed(1)} °C)`, `ユニット ${u} 温度高め (${a.value.toFixed(1)} °C)`);
    case 'heater_interlock': return t(`机组 ${u} 加热器联锁跳闸`, `Unit ${u} heater TRIPPED`, `ユニット ${u} ヒーターがトリップ`);
    case 'pump_interlock': return t('泵联锁跳闸(溢流保护)', 'Pump TRIPPED (overflow protection)', 'ポンプがトリップ（オーバーフロー保護）');
    case 'overtemp_interlock': return t('进料联锁跳闸(超温/失控保护)', 'Feed TRIPPED (over-temp / runaway protection)', 'フィードがトリップ（過温/暴走保護）');
    default: return a.message;
  }
}
function renderAlarms(alarms, interlocks) {
  alarms = alarms || [];
  const body = $('#alarm-body'), crit = alarms.filter((a) => a.severity === 'critical').length;
  const badge = $('#alarm-count');
  badge.textContent = alarms.length;
  badge.className = 'badge ' + (crit ? 'crit' : alarms.length ? 'warn' : '');
  if (!alarms.length) { body.innerHTML = `<div class="no-alarm">${t('— 无报警 · 系统正常 —', '— No alarms · all normal —', '— アラームなし · システム正常 —')}</div>`; return; }
  alarms.sort((a, b) => (a.severity === 'critical' ? 0 : 1) - (b.severity === 'critical' ? 0 : 1));
  body.innerHTML = alarms.map((a) => `<div class="alarm ${a.severity}"><span class="dot"></span><span>${alarmText(a)}</span></div>`).join('');
}

init();
