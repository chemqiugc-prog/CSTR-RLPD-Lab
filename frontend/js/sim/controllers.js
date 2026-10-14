// Controllers: manual (passthrough), PID (pairing-driven, ports the Python
// multi-loop PID), and RL (loads an ONNX policy and runs it in-browser via
// onnxruntime-web). All share one interface: compute(state, setpoints, dt) ->
// {pumps, valves, heaters} in [0,1]. The mode buttons swap between them.
import { t } from '../i18n.js?v=36';

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const zeros = (n) => new Array(n).fill(0);
const fill = (n, v) => new Array(n).fill(v);
const copyAct = (a) => ({ pumps: a.pumps.slice(), valves: a.valves.slice(), heaters: a.heaters.slice() });
const ramp = (current, target, limit) => current + Math.max(-limit, Math.min(limit, target - current));

// Shared observation vector (the in-browser RL ONNX contract):
//   obs = [ levels(n), temps(n), t_sp(n), h_sp(controlled k), t_cold, t_amb, model extras... ]
// (level slots are 0 for scenarios without levels, e.g. CSTR/HVAC).
export function obsVector(model, state, sp) {
  const n = model.n, o = [];
  for (let i = 0; i < n; i++) o.push(state.levels[i] ?? 0);
  for (let i = 0; i < n; i++) o.push(state.temps[i]);
  for (let i = 0; i < n; i++) o.push(sp.t_sp[i]);
  for (const i of model.controlledLevels()) o.push(sp.h_sp[i]);
  o.push(state.t_cold, state.t_amb);
  if (model.observationExtras) o.push(...model.observationExtras(state));
  return o;
}

// Concentration-free P2.4 deployment contract. Every entry is available from
// ordinary temperature or flow instrumentation; concentration remains hidden.
export function fieldCstrObsVector(model, state, sp, tempRate = 0, jacketRate = 0) {
  const p = model.p || {};
  const clipFlow = (v) => Math.max(0, Math.min(1.5, v));
  return [
    state.temps?.[0] ?? 0,
    state.jacket_temp?.[0] ?? 0,
    sp.t_sp?.[0] ?? 60,
    state.t_cold ?? p.t_cold ?? 20,
    state.coolant_supply_temp?.[0] ?? p.Tj_in ?? 10,
    clipFlow((state.pump_flow?.[0] ?? 0) / Math.max(p.Fmax || 0, 1e-12)),
    clipFlow((state.cooling_flow?.[0] ?? 0) / Math.max(p.Fjmax || 0, 1e-12)),
    tempRate,
    jacketRate,
  ];
}

// ---------------- Manual ----------------
export class ManualController {
  constructor(model) { this.bind(model); }
  bind(model) {
    const [nP, nV, nH] = model.actuatorCounts();
    this.cmd = { pumps: fill(nP, 0.3), valves: fill(nV, 0.5), heaters: zeros(nH) };
  }
  reset() {}
  setCommand(pumps, valves, heaters) {
    if (pumps) this.cmd.pumps = pumps.map(Number);
    if (valves) this.cmd.valves = valves.map(Number);
    if (heaters) this.cmd.heaters = heaters.map(Number);
  }
  setSingle(kind, i, v) {
    const arr = { pump: this.cmd.pumps, valve: this.cmd.valves, heater: this.cmd.heaters }[kind];
    if (arr && i >= 0 && i < arr.length) arr[i] = +v;
  }
  compute() { return copyAct(this.cmd); }
  snapshot() { return copyAct(this.cmd); }
}

// ---------------- PID ----------------
class PIDLoop {
  constructor(g, reverse = false, bias = 0) { this.g = g; this.reverse = reverse; this.bias = +bias || 0; this.reset(); }
  reset() { this.i = this.bias; this.prev = null; }
  update(sp, meas, dt) {
    // reverse-acting (e.g. cooling): output rises when measurement is ABOVE setpoint.
    const e = this.reverse ? meas - sp : sp - meas;
    const dmeas = this.prev == null || dt <= 0 ? 0 : (meas - this.prev) / dt;
    this.prev = meas;
    const p = this.g.kp * e, d = (this.reverse ? 1 : -1) * this.g.kd * dmeas;
    const iCand = this.i + this.g.ki * e * dt;
    const raw = p + iCand + d, out = clamp01(raw);
    if (!((raw > 1 && e > 0) || (raw < 0 && e < 0))) this.i = iCand;
    return out;
  }
}

export class PIDController {
  constructor(model) { this.bind(model); }
  bind(model) {
    this.model = model;
    [this.nP, this.nV, this.nH] = model.actuatorCounts();
    this.gains = JSON.parse(JSON.stringify(model.defaultGains()));
    const pr = model.controlPairing();
    this.demandIdx = pr.demand_valve_index; this.demandValve = 0.5;
    this.holds = pr.holds || [];   // [[kind, idx, value], ...] actuators held at a fixed value
    this.levelLoops = pr.level.map(([kind, ai, li, rev]) => ({ kind, ai, li, loop: new PIDLoop(this.gains[kind === 'pump' ? 'level_pump' : 'level_valve'], !!rev) }));
    const tempBias = model.pidTempBias ? model.pidTempBias() : [];
    this.tempLoops = pr.temp.map(([hi, ti, rev]) => ({ hi, ti, loop: new PIDLoop(this.gains.temp, !!rev, tempBias[ti] ?? 0) }));
  }
  reset() { this.levelLoops.forEach((l) => l.loop.reset()); this.tempLoops.forEach((l) => l.loop.reset()); }
  compute(state, sp, dt) {
    const act = { pumps: zeros(this.nP), valves: zeros(this.nV), heaters: zeros(this.nH) };
    for (const [kind, idx, value] of this.holds) {
      const arr = { pump: act.pumps, valve: act.valves, heater: act.heaters }[kind];
      if (arr && idx < arr.length) arr[idx] = value;
    }
    for (const { kind, ai, li, loop } of this.levelLoops) {
      const out = loop.update(sp.h_sp[li], state.levels[li], dt);
      if (kind === 'pump') act.pumps[ai] = out; else act.valves[ai] = out;
    }
    if (this.demandIdx != null && this.nV) act.valves[this.demandIdx] = this.demandValve;
    for (const { hi, ti, loop } of this.tempLoops) act.heaters[hi] = loop.update(sp.t_sp[ti], state.temps[ti], dt);
    return act;
  }
  getConfig() { return { gains: this.gains, demand_valve: this.demandValve }; }
  setConfig(cfg) {
    if (cfg.gains) for (const k in cfg.gains) if (this.gains[k]) for (const p of ['kp', 'ki', 'kd']) if (cfg.gains[k][p] != null) this.gains[k][p] = +cfg.gains[k][p];
    if (cfg.demand_valve != null) this.demandValve = +cfg.demand_valve;
  }
}

// ---------------- RL (ONNX policy, in-browser) ----------------
// Observation / action contract (the offline Gym env must match this):
//   obs    = [ ...levels(n), ...temps(n), ...t_sp(n), ...h_sp(controlled k), t_cold, t_amb ]   (Float32, length 3n+k+2)
//   action = [ ...pumps(nP), ...valves(nV), ...heaters(nH) ]  in [0,1] (clamped)               (Float32, length nP+nV+nH)
// onnxruntime-web is loaded on demand (only when RL mode is used).
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';

// Policies bundled under frontend/models/. Each is scenario-specific (the obs/act
// contract differs per plant), so selecting one switches the sim to its scenario
// before loading. Drop a new .onnx in frontend/models/ and add a row here.
// One economic-objective RLPD policy per scenario. Each was trained on the economic
// KPI (value − energy-cost in soft bands) under 工况 (operating-regime) variation, and
// beats both fixed-SP PID and fixed-model MPC on economic performance (aiogym/runs).
// Supervisory (RL-on-PID / RTO) action layout — MUST mirror aiogym/env.py SUPERVISORY.
// The policy outputs SETPOINTS (normalized 0-1 -> [lo,hi]); an inner PID regulates the
// plant to them, so the plant is always controlled and RL only picks the economic
// optimum. ['t_sp',tank,lo,hi] | ['h_sp',level,lo,hi] | ['mv',kind,idx,lo,hi] (direct MV).
export const SUPERVISORY = {
  cstr: [['t_sp', 0, 45, 90], ['mv', 'pumps', 0, 0.3, 1.0]],
  cstr_jacketed_p2: [['t_sp', 0, 45, 90], ['mv', 'pumps', 0, 0.3, 1.0]],
};
const CSTR_TARGET_RATE = { t_sp: 0.3, pump: 0.015 };

// One supervisory RLPD policy per scenario: RL sets the setpoints, the inner PID
// regulates to them, and RL beats the fixed-SP PID / fixed-model MPC on economics by
// adapting its targets to 工况 (operating regime). All trained in aiogym/runs.
export const BUILTIN_POLICIES = [
  { id: 'rlpd_cstr_jacketed_p24_field_v1_guard_v4', scenario: 'cstr_jacketed_p2', url: './models/rlpd_cstr_jacketed_p24_field_v1.onnx', mode: 'setpoint',
    zh: '推荐 · P2.4 无浓度安全经济', en: 'Recommended · P2.4 field-observed safe-economic', ja: '推奨 · P2.4 濃度計なし安全経済',
    observationProfile: 'field_no_concentration',
    layout: [['t_sp', 0, 45, 85], ['mv', 'pumps', 0, 0.3, 1.0]], jacketSafetyGuard: true,
    jacketGuardTriggerTemp: 85.5, jacketGuardHardTemp: 86.5, jacketGuardHorizon: 5,
    jacketGuardTarget: 82, jacketGuardPumpCap: 0.80, jacketGuardFeedStop: 87.5,
    jacketGuardCoolingFloor: 0.75, jacketPredictiveEmergency: false,
    uiRole: 'recommended',
    summaryZh: '默认运行策略。Actor 不读取 Ca/Cb，仅使用现场温度、流量和趋势；温度风险上升时由 V4 限制进料并提高冷却。',
    summaryEn: 'Default policy. The actor uses field temperatures, flows, and rates without Ca/Cb; V4 limits feed and raises cooling as thermal risk rises.',
    summaryJa: '既定方策。Actor は Ca/Cb を使わず温度・流量・変化率を使用し、V4 が熱リスク時に供給制限と冷却強化を行います。',
    benchmark: { profit: '11.54 ¥/h', kpi: '97.79', over88: '0/100', trips: '0/100' },
    noteZh: 'P2.4 严格同种子 100×600 秒评估：利润 11.54 ¥/h、KPI 97.79，零越温、零跳闸。',
    noteEn: 'Strict paired P2.4 evaluation on 100 × 600-second seeds: 11.54 ¥/h, KPI 97.79, no over-temperature episodes or trips.',
    noteJa: 'P2.4 厳密同一シード100×600秒評価：利益11.54 ¥/h、KPI 97.79、過温・トリップなし。' },
  { id: 'rlpd_cstr_jacketed_p2_official_raw', scenario: 'cstr_jacketed_p2', url: './models/rlpd_cstr_jacketed_p24_official.onnx', mode: 'setpoint',
    zh: '对照 · P2.4 分析仪 RLPD 上限', en: 'Reference · P2.4 analyzer RLPD upper bound', ja: '比較 · P2.4 分析計 RLPD 上限',
    layout: [['t_sp', 0, 45, 90], ['mv', 'pumps', 0, 0.3, 1.0]],
    uiRole: 'reference',
    summaryZh: '保留快速 Ca/Cb 在线分析仪假设且不附加安全保护，仅作为理想观测下的利润上限和风险代价对照。',
    summaryEn: 'Keeps the fast online Ca/Cb analyzer assumption without a safety overlay, serving only as an ideal-observation profit and risk reference.',
    summaryJa: '高速 Ca/Cb オンライン分析計を仮定し安全保護を付けず、理想観測での利益上限とリスク比較にのみ使用します。',
    benchmark: { profit: '12.93 ¥/h', kpi: '98.39', over88: '100/100', trips: '54/100' },
    noteZh: 'P2.4 严格同种子 100×600 秒评估：利润 12.93 ¥/h、KPI 98.39；100/100 回合超过 88°C，54/100 跳闸。',
    noteEn: 'Strict paired P2.4 evaluation on 100 × 600-second seeds: 12.93 ¥/h, KPI 98.39; 100/100 exceed 88°C and 54/100 trip.',
    noteJa: 'P2.4 厳密同一シード100×600秒評価：利益12.93 ¥/h、KPI 98.39、100/100過温、54/100トリップ。' },
  { id: 'rlpd_cstr_upstream', scenario: 'cstr', url: './models/rlpd_cstr_upstream.onnx', mode: 'setpoint',
    zh: '官方保留 · Legacy CSTR RLPD', en: 'Upstream retained · Legacy CSTR RLPD', ja: '上流保存 · Legacy CSTR RLPD',
    uiRole: 'reference',
    summaryZh: '原 AIO-Gym 两状态 CSTR 的 RLPD 策略，用于复现上游环境；不与 P2.4 利润数值直接排名。',
    summaryEn: 'RLPD policy for the upstream two-state CSTR. It reproduces the legacy environment and is not ranked directly against P2.4 profit.',
    summaryJa: '上流の2状態 CSTR 用 RLPD。Legacy 環境の再現用で、P2.4 の利益と直接比較しません。',
    noteZh: '官方 Legacy CSTR 的完整对照模型。',
    noteEn: 'Complete upstream Legacy CSTR reference policy.',
    noteJa: '上流 Legacy CSTR の比較方策。' },
];

export class RLController {
  constructor(model) { this.session = null; this.ready = false; this._st = { k: 'idle' }; this.bind(model); }
  bind(model) {
    this.model = model;
    // a policy is scenario-specific (obs/act dims differ) — drop any loaded one on (re)bind
    this.session = null; this.ready = false; this._st = { k: 'idle' };
    const [nP, nV, nH] = model.actuatorCounts();
    this.nP = nP; this.nV = nV; this.nH = nH;
    this.ctrl = model.controlledLevels();
    this.obsLen = 3 * model.n + this.ctrl.length + 2 + (model.extraObsDim || 0);
    this.baseObsLen = this.obsLen;
    this.actLen = nP + nV + nH;
    // supervisory (RL-on-PID): policy outputs setpoints, an inner PID regulates to them
    this.scenario = model.metadata ? model.metadata().scenario : 'cstr_jacketed_p2';
    this.layout = SUPERVISORY[this.scenario] || null;
    this.pid = new PIDController(model);
    this.mode = 'actuator';
    this.lastAction = { pumps: fill(nP, 0.3), valves: fill(nV, 0.5), heaters: zeros(nH) };
    this.policy = null;
    this.observationProfile = 'analyzer';
    this.safetyObs = false;
    this.safetyShield = false;
    this.rateLimit = false;
    this.riskSensitivity = 1.0;
    this.shieldProfile = 'legacy';
    this.economicBias = false;
    this.jacketSafetyGuard = false;
    this.jacketGuardTriggerTemp = 85.5;
    this.jacketGuardHardTemp = 86.5;
    this.jacketGuardHorizon = 0;
    this.jacketGuardTarget = 75;
    this.jacketGuardPumpCap = 0.55;
    this.jacketGuardFeedStop = 87.2;
    this.jacketGuardCoolingFloor = 1;
    this.jacketPredictiveEmergency = false;
    this._shield = null;
    this._prevCstrTemp = null;
    this._cstrCoolingFaultLatch = false;
    this._prevJacketedTemp = null;
    this._prevFieldTemp = null;
    this._prevFieldJacket = null;
    this._fieldTempRate = 0;
    this._fieldJacketRate = 0;
    this._jacketGuardActive = false;
    this._jacketEmergencyActive = false;
    this._resetSp();
    this._busy = false;
  }
  _resetSp() {
    const [hsp, tsp] = this.model.defaultSetpoints();
    this.tsp = tsp.slice();
    this.hsp = Array.from({ length: this.model.n }, (_, i) => hsp[i] ?? 0);
    this.mv = {};
  }
  reset() {
    if (this.pid) this.pid.reset();
    this._resetSp();
    this._shield = null;
    this._prevCstrTemp = null;
    this._cstrCoolingFaultLatch = false;
    this._prevJacketedTemp = null;
    this._prevFieldTemp = null;
    this._prevFieldJacket = null;
    this._fieldTempRate = 0;
    this._fieldJacketRate = 0;
    this._jacketGuardActive = false;
    this._jacketEmergencyActive = false;
  }
  getSetpoints() { return { t_sp: this.tsp.slice(), h_sp: this.hsp.slice() }; }

  obs(state, sp) { return Float32Array.from(obsVector(this.model, state, sp)); }

  _cstrSafetyObsFeatures(state) {
    const s = this._shield || {};
    const T = Number.isFinite(s.T) ? s.T : (state.temps?.[0] ?? 50);
    const obsD = Number.isFinite(s.obs_dTdt) ? s.obs_dTdt : 0;
    const modelD = Number.isFinite(s.model_dTdt) ? s.model_dTdt : (Number.isFinite(s.dTdt) ? s.dTdt : 0);
    const Tpred = Number.isFinite(s.T_pred) ? s.T_pred : T;
    const thermalRisk = Number.isFinite(s.thermal_risk) ? s.thermal_risk : 0;
    const latentHeatRisk = Number.isFinite(s.cooling_loss_risk) ? s.cooling_loss_risk : 0;
    const risk = Math.max(thermalRisk, 0.75 * latentHeatRisk);
    const residual = obsD - modelD;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const attenPos = (v) => v > 0 ? v * this.riskSensitivity : v;
    const tpredFeature = (Tpred - 80) / 20;
    return [
      clamp(attenPos(obsD) / 4, -1, 1),
      clamp(attenPos(modelD) / 4, -1, 1),
      clamp(attenPos(tpredFeature), -2, 4),
      clamp((risk * this.riskSensitivity) / 4, 0, 4),
      s.cooling_fault_latched ? 1 : 0,
      clamp(attenPos(residual) / 4, -1, 1),
      clamp(this.lastAction?.pumps?.[0] ?? 0, 0, 1),
      clamp(this.lastAction?.heaters?.[0] ?? 0, 0, 1),
    ];
  }

  policyObs(state, sp) {
    if (this.observationProfile === 'field_no_concentration') {
      return Float32Array.from(fieldCstrObsVector(
        this.model, state, sp, this._fieldTempRate, this._fieldJacketRate,
      ));
    }
    const o = obsVector(this.model, state, sp);
    if (this.scenario === 'cstr' && this.safetyObs) o.push(...this._cstrSafetyObsFeatures(state));
    return Float32Array.from(o);
  }

  // Non-blocking: kick async inference, return an actuator command. Supervisory mode
  // runs the inner PID to RL's (async-updated) setpoints every tick; actuator mode
  // returns the last cached raw action.
  compute(state, sp, dt) {
    this._updateFieldRates(state, dt);
    const setpoint = this.mode === 'setpoint' && this.layout;
    if (this.session && !this._busy) {
      this._busy = true;
      const x = setpoint ? this.policyObs(state, { t_sp: this.tsp, h_sp: this.hsp })
                         : this.policyObs(state, sp);
      (setpoint ? this._inferSp(x) : this._infer(x)).finally(() => { this._busy = false; });
    }
    if (!setpoint) return copyAct(this.lastAction);
    if (this.economicBias && this.scenario === 'cstr') this._applyCstrLowRiskEconomicBias(state, dt);
    this._updateJacketedSafetyGuard(state, dt);
    if (this._jacketGuardActive) this.tsp[0] = Math.min(this.tsp[0], this.jacketGuardTarget);
    const act = this.pid.compute(state, { t_sp: this.tsp, h_sp: this.hsp }, dt);   // PID holds RL's targets
    for (const key in this.mv) { const [kind, idx] = key.split(':'); act[kind][+idx] = this.mv[key]; }
    if (this._jacketGuardActive) {
      if (act.pumps.length) act.pumps[0] = Math.min(act.pumps[0], this._jacketEmergencyActive ? 0 : this.jacketGuardPumpCap);
      if (act.heaters.length) act.heaters[0] = Math.max(act.heaters[0], this._jacketEmergencyActive ? 1 : this.jacketGuardCoolingFloor);
    }
    const out = this.safetyShield ? this._cstrSafetyFilter(state, act, dt) : act;
    this.lastAction = copyAct(out);
    return out;
  }

  _updateFieldRates(state, dt) {
    if (this.observationProfile !== 'field_no_concentration') return;
    const T = state.temps?.[0] ?? 0;
    const Tj = state.jacket_temp?.[0] ?? 0;
    if (this._prevFieldTemp != null) {
      const tau = -0.5 / Math.log(0.8);
      const alpha = 1 - Math.exp(-Math.max(dt, 1e-6) / tau);
      this._fieldTempRate = (1 - alpha) * this._fieldTempRate
        + alpha * (T - this._prevFieldTemp) / Math.max(dt, 1e-6);
      this._fieldJacketRate = (1 - alpha) * this._fieldJacketRate
        + alpha * (Tj - this._prevFieldJacket) / Math.max(dt, 1e-6);
    }
    this._prevFieldTemp = T;
    this._prevFieldJacket = Tj;
  }

  _updateJacketedSafetyGuard(state, dt) {
    if (!this.jacketSafetyGuard || !this.scenario.startsWith('cstr_jacketed')) { this._jacketGuardActive = false; this._jacketEmergencyActive = false; return; }
    const T = state.temps?.[0] ?? 0;
    const rate = this._prevJacketedTemp == null ? 0 : (T - this._prevJacketedTemp) / Math.max(dt, 1e-6);
    this._prevJacketedTemp = T;
    const predictedT = T + this.jacketGuardHorizon * Math.max(0, rate);
    this._jacketGuardActive = (T > this.jacketGuardTriggerTemp && rate > 0.02)
      || T > this.jacketGuardHardTemp || predictedT > this.jacketGuardHardTemp;
    this._jacketEmergencyActive = T > this.jacketGuardFeedStop
      || (this.jacketPredictiveEmergency && predictedT > this.jacketGuardFeedStop);
  }

  _cstrMetrics(state, act, dt) {
    const p = this.model.p || {};
    const Ca = Math.max(0, state.conc?.[0] ?? 0);
    const T = state.temps?.[0] ?? 0;
    const pump = act.pumps?.[0] ?? 0;
    const cooling = act.heaters?.[0] ?? 0;
    const D = pump * (p.Dmax || 0);
    const TK = T + 273.15;
    const rate = (p.k0 || 0) * Math.exp(-(p.EaR || 0) / TK) * Ca;
    const qGen = (p.Hr || 0) * rate;
    const qCool = (p.Uc || 0) * cooling * (T - (p.Tcool || 0));
    const feedCooling = D * Math.max(0, T - (state.t_cold ?? p.t_cold ?? 20));
    const dTdt = D * ((state.t_cold ?? p.t_cold ?? 20) - T) + qGen - qCool;
    const obsD = this._prevCstrTemp == null ? 0 : (T - this._prevCstrTemp) / Math.max(dt, 1e-6);
    const riskD = Math.max(dTdt, obsD);
    return {
      T, Ca, pump, cooling, dTdt, obs_dTdt: obsD, risk_dTdt: riskD,
      T_pred: T + dt * riskD,
      thermal_risk: Math.max(0, qGen - Math.max(0, qCool) - feedCooling),
      cooling_loss_risk: Math.max(0, qGen - feedCooling),
    };
  }

  _applyCstrLowRiskEconomicBias(state, dt) {
    const relax = 1 - this.riskSensitivity;
    if (relax <= 1e-6) return;
    const baseAct = this.lastAction || { pumps: [0.3], valves: [], heaters: [0] };
    const m = this._cstrMetrics(state, baseAct, dt);
    const lowRisk = !this._cstrCoolingFaultLatch
      && m.T < 82.0
      && m.T_pred < 84.0
      && m.obs_dTdt < 0.10
      && m.cooling_loss_risk < 0.90
      && m.thermal_risk < 0.90;
    if (!lowRisk) return;
    const econFloor = Math.min(82.0, 55.0 + 28.0 * relax);
    this.tsp[0] = Math.max(this.tsp[0], econFloor);
    const key = 'pumps:0';
    const current = this.lastAction?.pumps?.[0] ?? 0.3;
    const target = Math.max(this.mv[key] ?? 0, Math.min(0.65, 0.18 + 0.30 * relax));
    this.mv[key] = this.rateLimit ? ramp(current, target, CSTR_TARGET_RATE.pump) : target;
  }

  _cstrLateSafetyFilter(state, act, dt) {
    act = copyAct(act);
    let shielded = false;
    const relax = 1 - this.riskSensitivity;
    const capPump = (lim) => { if (act.pumps[0] > lim) { act.pumps[0] = lim; shielded = true; } };
    const floorCooling = (lim) => { if (act.heaters[0] < lim) { act.heaters[0] = lim; shielded = true; } };
    const capTsp = (lim) => { if (this.tsp[0] > lim) { this.tsp[0] = lim; shielded = true; } };
    let m = this._cstrMetrics(state, act, dt);

    if (this._cstrCoolingFaultLatch && m.T < 62.0 && m.obs_dTdt < 0.02) this._cstrCoolingFaultLatch = false;
    const suspectedCoolingFault = (m.cooling > 0.90 && m.T > 60.0 + 8.0 * relax && m.obs_dTdt > 0.05 + 0.04 * relax)
      || (m.T_pred > 82.0 + 4.0 * relax && m.obs_dTdt > 0.06 + 0.04 * relax)
      || (m.obs_dTdt > 0.25 + 0.10 * relax && m.T > 60.0 + 5.0 * relax);
    if (suspectedCoolingFault) this._cstrCoolingFaultLatch = true;

    if (this._cstrCoolingFaultLatch) {
      if (m.T > 66.0 + 4.0 * relax || m.obs_dTdt > 0.08 + 0.04 * relax) {
        act.pumps[0] = 0.0;
        shielded = true;
      } else {
        capPump(0.10);
      }
      floorCooling(1.0);
      capTsp(65.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 80.0 + 2.0 * relax || m.T_pred > 83.0 + 2.0 * relax) {
      capPump(0.25);
      floorCooling(0.85);
      capTsp(76.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 84.0 + 2.0 * relax || m.T_pred > 86.0 + 2.0 * relax) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(70.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 92.0) {
      act.pumps[0] = 0.0;
      act.heaters[0] = 1.0;
      capTsp(60.0);
      shielded = true;
    }
    this._shield = { active: shielded, T: m.T, T_pred: m.T_pred, dTdt: m.risk_dTdt, model_dTdt: m.dTdt, obs_dTdt: m.obs_dTdt, thermal_risk: m.thermal_risk, cooling_loss_risk: m.cooling_loss_risk, Ca: m.Ca, pump: act.pumps[0], cooling: act.heaters[0], cooling_fault_latched: this._cstrCoolingFaultLatch };
    this._prevCstrTemp = m.T;
    return act;
  }

  _cstrResidualSafetyFilter(state, act, dt, profile = 'residual_v2') {
    act = copyAct(act);
    let shielded = false;
    const capPump = (lim) => { if (act.pumps[0] > lim) { act.pumps[0] = lim; shielded = true; } };
    const floorCooling = (lim) => { if (act.heaters[0] < lim) { act.heaters[0] = lim; shielded = true; } };
    const capTsp = (lim) => { if (this.tsp[0] > lim) { this.tsp[0] = lim; shielded = true; } };
    let m = this._cstrMetrics(state, act, dt);

    if (profile === 'residual_v9') {
      if (this._cstrCoolingFaultLatch && m.T < 68.0 && m.obs_dTdt < 0.02) this._cstrCoolingFaultLatch = false;
      const residualD = m.obs_dTdt - m.dTdt;
      const suspectedCoolingFault = (m.cooling > 0.78 && m.T > 60.0 && m.obs_dTdt > -0.010 && residualD > 0.055)
        || (m.cooling > 0.88 && m.T > 65.0 && residualD > 0.035)
        || (m.cooling > 0.80 && m.T_pred > 78.0 && residualD > 0.050)
        || (m.T > 60.0 && residualD > 0.120);
      if (suspectedCoolingFault) this._cstrCoolingFaultLatch = true;

      if (this._cstrCoolingFaultLatch) {
        if (m.T > 68.0 || m.T_pred > 72.0 || m.obs_dTdt > 0.05) {
          act.pumps[0] = 0.0;
          shielded = true;
        } else {
          capPump(0.20);
        }
        floorCooling(1.0);
        capTsp(68.0);
        m = this._cstrMetrics(state, act, dt);
      }

      if (m.T > 90.5 || m.T_pred > 91.3) {
        const edge = Math.max(m.T - 90.5, m.T_pred - 91.3);
        capPump(Math.max(0.0, Math.min(0.70, 0.70 - 0.45 * edge)));
        floorCooling(1.0);
        capTsp(86.0);
        m = this._cstrMetrics(state, act, dt);
      }

      if (m.T > 92.0) {
        act.pumps[0] = 0.0;
        act.heaters[0] = 1.0;
        capTsp(76.0);
        shielded = true;
      }

      this._shield = { active: shielded, T: m.T, T_pred: m.T_pred, dTdt: m.risk_dTdt, model_dTdt: m.dTdt, obs_dTdt: m.obs_dTdt, thermal_risk: m.thermal_risk, cooling_loss_risk: m.cooling_loss_risk, Ca: m.Ca, pump: act.pumps[0], cooling: act.heaters[0], cooling_fault_latched: this._cstrCoolingFaultLatch };
      this._prevCstrTemp = m.T;
      return act;
    }

    if (this._cstrCoolingFaultLatch && m.T < 70.0 && m.obs_dTdt < 0.02) this._cstrCoolingFaultLatch = false;
    const suspectedCoolingFault = (m.cooling > 0.93 && m.T > 69.0 && m.obs_dTdt > 0.065)
      || (m.cooling > 0.88 && m.T_pred > 86.0 && m.obs_dTdt > 0.055)
      || (m.obs_dTdt > 0.26 && m.T > 68.0);
    if (suspectedCoolingFault) this._cstrCoolingFaultLatch = true;

    if (this._cstrCoolingFaultLatch) {
      if (m.T > 74.0 || m.T_pred > 78.0 || m.obs_dTdt > 0.08) {
        act.pumps[0] = 0.0;
        shielded = true;
      } else {
        capPump(0.15);
      }
      floorCooling(1.0);
      capTsp(68.0);
      m = this._cstrMetrics(state, act, dt);
    }

    if (m.T > 87.5 || m.T_pred > 89.0) {
      if (m.T > 89.0 || m.T_pred > 91.0 || m.obs_dTdt > 0.11) {
        act.pumps[0] = 0.0;
        shielded = true;
      } else {
        capPump(0.20);
      }
      floorCooling(1.0);
      capTsp(80.0);
      m = this._cstrMetrics(state, act, dt);
    }

    if (m.T > 92.0) {
      act.pumps[0] = 0.0;
      act.heaters[0] = 1.0;
      capTsp(70.0);
      shielded = true;
    }

    this._shield = { active: shielded, T: m.T, T_pred: m.T_pred, dTdt: m.risk_dTdt, model_dTdt: m.dTdt, obs_dTdt: m.obs_dTdt, thermal_risk: m.thermal_risk, cooling_loss_risk: m.cooling_loss_risk, Ca: m.Ca, pump: act.pumps[0], cooling: act.heaters[0], cooling_fault_latched: this._cstrCoolingFaultLatch };
    this._prevCstrTemp = m.T;
    return act;
  }

  _cstrSafetyFilter(state, act, dt) {
    if (this.scenario !== 'cstr') return act;
    if (this.shieldProfile === 'residual_v2' || this.shieldProfile === 'residual_v9') return this._cstrResidualSafetyFilter(state, act, dt, this.shieldProfile);
    if (this.shieldProfile === 'late') return this._cstrLateSafetyFilter(state, act, dt);
    act = copyAct(act);
    let shielded = false;
    const capPump = (lim) => { if (act.pumps[0] > lim) { act.pumps[0] = lim; shielded = true; } };
    const floorCooling = (lim) => { if (act.heaters[0] < lim) { act.heaters[0] = lim; shielded = true; } };
    const capTsp = (lim) => { if (this.tsp[0] > lim) { this.tsp[0] = lim; shielded = true; } };
    const k = Math.max(0, Math.round((state.t || 0) / Math.max(dt, 1e-6)));
    let m = this._cstrMetrics(state, act, dt);

    if (k < 40) {
      const startup = k / 40;
      capPump(0.10 + 0.05 * startup);
      capTsp(52.0 + 3.0 * startup);
      if (m.risk_dTdt > 1.0) floorCooling(0.90);
      m = this._cstrMetrics(state, act, dt);
    }
    if (k < 90) {
      capPump(0.15);
      capTsp(55.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 50.0) {
      capPump(0.15);
      capTsp(55.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.cooling_loss_risk > 0.90 && m.Ca > 0.30 && m.T > 42.0) {
      capPump(0.05);
      floorCooling(0.95);
      capTsp(52.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.cooling_loss_risk > 1.25 && m.Ca > 0.36 && m.T > 44.0) {
      capPump(0.02);
      floorCooling(1.0);
      capTsp(50.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 0.05 && m.T > 50.0) {
      capPump(0.12);
      floorCooling(0.85);
      capTsp(50.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 0.25 && m.T > 42.0 && m.cooling < 0.60) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(48.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 0.25 && m.T > 42.0 && m.T_pred > 54.0) {
      capPump(0.05);
      floorCooling(0.90);
      capTsp(50.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 0.55 && m.T > 42.0 && (m.cooling < 0.75 || m.T_pred > 58.0)) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(48.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 82.0 || m.T_pred > 85.0) {
      capPump(0.45);
      floorCooling(0.80);
      capTsp(78.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (this._cstrCoolingFaultLatch && m.T < 54.0 && m.obs_dTdt < 0.02) this._cstrCoolingFaultLatch = false;
    const suspectedCoolingFault = (m.cooling > 0.85 && m.T > 50.0 && m.obs_dTdt > 0.08)
      || (m.cooling > 0.95 && m.T > 58.0 && m.obs_dTdt > 0.0)
      || (m.T_pred > 78.0 && m.obs_dTdt > 0.18);
    if (suspectedCoolingFault) this._cstrCoolingFaultLatch = true;
    if (this._cstrCoolingFaultLatch) {
      if (m.obs_dTdt > 0.12 || m.T > 58.0) {
        act.pumps[0] = 0.0;
        shielded = true;
      } else {
        capPump(0.05);
      }
      floorCooling(1.0);
      capTsp(55.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 1.0 && m.T > 50.0) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(55.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 0.25 && m.T > 55.0) {
      capPump(0.10);
      floorCooling(1.0);
      capTsp(60.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.obs_dTdt > 0.35 && m.T > 55.0 && m.cooling > 0.75) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(58.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if ((m.risk_dTdt > 0.8 && m.T > 75.0) || (m.thermal_risk > 0.8 && m.T > 70.0)) {
      capPump(0.05);
      floorCooling(1.0);
      capTsp(62.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if ((m.risk_dTdt > 1.2 && m.T > 62.0) || m.thermal_risk > 1.2) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(60.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if ((m.risk_dTdt > 2.0 && m.T > 60.0) || m.thermal_risk > 2.0) {
      act.pumps[0] = 0.0;
      floorCooling(1.0);
      capTsp(60.0);
      shielded = true;
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 86.0 || m.T_pred > 88.0) {
      if (m.T > 88.0 || m.T_pred > 88.0) {
        act.pumps[0] = 0.0;
        shielded = true;
      } else {
        capPump(0.15);
      }
      floorCooling(1.0);
      capTsp(70.0);
      m = this._cstrMetrics(state, act, dt);
    }
    if (m.T > 92.0) {
      act.pumps[0] = 0.0;
      act.heaters[0] = 1.0;
      capTsp(60.0);
      shielded = true;
    }
    this._shield = { active: shielded, T: m.T, T_pred: m.T_pred, dTdt: m.risk_dTdt, model_dTdt: m.dTdt, obs_dTdt: m.obs_dTdt, thermal_risk: m.thermal_risk, cooling_loss_risk: m.cooling_loss_risk, Ca: m.Ca, pump: act.pumps[0], cooling: act.heaters[0], cooling_fault_latched: this._cstrCoolingFaultLatch };
    this._prevCstrTemp = m.T;
    return act;
  }

  async _infer(x) {
    try {
      const ort = window.ort;
      const input = new ort.Tensor('float32', x, [1, x.length]);
      const feeds = {}; feeds[this.session.inputNames[0]] = input;
      const out = await this.session.run(feeds);
      const a = out[this.session.outputNames[0]].data;
      const act = { pumps: [], valves: [], heaters: [] };
      let k = 0;
      for (let i = 0; i < this.nP; i++) act.pumps.push(clamp01(a[k++]));
      for (let i = 0; i < this.nV; i++) act.valves.push(clamp01(a[k++]));
      for (let i = 0; i < this.nH; i++) act.heaters.push(clamp01(a[k++]));
      this.lastAction = act;
    } catch (e) { this._st = { k: 'err', msg: e.message }; this.ready = false; }
  }

  // Supervisory inference: ONNX outputs normalized setpoints -> denormalize -> the SPs
  // the inner PID tracks (+ any direct economic MVs).
  async _inferSp(x) {
    try {
      const ort = window.ort;
      const input = new ort.Tensor('float32', x, [1, x.length]);
      const feeds = {}; feeds[this.session.inputNames[0]] = input;
      const out = await this.session.run(feeds);
      const a = out[this.session.outputNames[0]].data;
      const mv = {};
      this.layout.forEach((spec, i) => {
        const lo = spec[spec.length - 2], hi = spec[spec.length - 1];
        let val = lo + clamp01(a[i]) * (hi - lo);
        if (spec[0] === 't_sp') {
          if (this.rateLimit && this.scenario === 'cstr') val = ramp(this.tsp[spec[1]], val, CSTR_TARGET_RATE.t_sp);
          this.tsp[spec[1]] = val;
        }
        else if (spec[0] === 'h_sp') this.hsp[spec[1]] = val;
        else {
          const key = spec[1] + ':' + spec[2];             // ['mv', kind, idx, lo, hi]
          if (this.rateLimit && this.scenario === 'cstr' && spec[1] === 'pumps' && spec[2] === 0) {
            const current = this.lastAction?.pumps?.[0] ?? this.mv[key] ?? 0.3;
            val = ramp(current, val, CSTR_TARGET_RATE.pump);
          }
          mv[key] = val;
        }
      });
      this.mv = mv;
    } catch (e) { this._st = { k: 'err', msg: e.message }; this.ready = false; }
  }

  async loadPolicy(src, mode, opts = {}) {
    this._st = { k: 'loading' };
    this.mode = mode || 'actuator';
    this.policy = opts.id ? opts : null;
    this.observationProfile = opts.observationProfile || 'analyzer';
    if (!['analyzer', 'field_no_concentration'].includes(this.observationProfile)) {
      this.observationProfile = 'analyzer';
    }
    this.layout = opts.layout || SUPERVISORY[this.scenario] || null;
    this.safetyObs = !!opts.safetyObs;
    this.safetyShield = !!opts.safetyShield;
    this.rateLimit = !!opts.rateLimit;
    this.riskSensitivity = Math.max(0.05, Math.min(1.0, Number.isFinite(+opts.riskSensitivity) ? +opts.riskSensitivity : 1.0));
    this.shieldProfile = opts.shieldProfile || 'legacy';
    this.economicBias = !!opts.economicBias;
    this.jacketSafetyGuard = !!opts.jacketSafetyGuard;
    this.jacketGuardTriggerTemp = Number.isFinite(+opts.jacketGuardTriggerTemp) ? +opts.jacketGuardTriggerTemp : 85.5;
    this.jacketGuardHardTemp = Number.isFinite(+opts.jacketGuardHardTemp) ? +opts.jacketGuardHardTemp : 86.5;
    this.jacketGuardHorizon = Math.max(0, Number.isFinite(+opts.jacketGuardHorizon) ? +opts.jacketGuardHorizon : 0);
    this.jacketGuardTarget = Number.isFinite(+opts.jacketGuardTarget) ? +opts.jacketGuardTarget : 75;
    this.jacketGuardPumpCap = Number.isFinite(+opts.jacketGuardPumpCap) ? +opts.jacketGuardPumpCap : 0.55;
    this.jacketGuardFeedStop = Number.isFinite(+opts.jacketGuardFeedStop) ? +opts.jacketGuardFeedStop : 87.2;
    this.jacketGuardCoolingFloor = Math.max(0, Math.min(1, Number.isFinite(+opts.jacketGuardCoolingFloor) ? +opts.jacketGuardCoolingFloor : 1));
    this.jacketPredictiveEmergency = !!opts.jacketPredictiveEmergency;
    this.obsLen = this.observationProfile === 'field_no_concentration'
      ? 9 : this.baseObsLen + ((this.scenario === 'cstr' && this.safetyObs) ? 8 : 0);
    this._shield = null;
    this._prevCstrTemp = null;
    this._cstrCoolingFaultLatch = false;
    this._prevJacketedTemp = null;
    this._prevFieldTemp = null;
    this._prevFieldJacket = null;
    this._fieldTempRate = 0;
    this._fieldJacketRate = 0;
    this._jacketGuardActive = false;
    this._jacketEmergencyActive = false;
    this._resetSp();
    const expectOut = (this.mode === 'setpoint' && this.layout) ? this.layout.length : this.actLen;
    try {
      if (!window.ort) await loadScript(ORT_CDN);
      const ort = window.ort;
      const session = await ort.InferenceSession.create(src);
      // Validate obs/output dims against the scenario up front, so a mismatched policy
      // fails loudly once at load — not silently every tick.
      const probe = new ort.Tensor('float32', new Float32Array(this.obsLen), [1, this.obsLen]);
      const feeds = {}; feeds[session.inputNames[0]] = probe;
      const out = await session.run(feeds);
      const aLen = out[session.outputNames[0]].data.length;
      if (aLen !== expectOut) throw new Error(`__DIM__act ${aLen} ${expectOut}`);
      this.session = session; this.ready = true;
      this._st = { k: 'loaded' };
      return true;
    } catch (e) { this.session = null; this.ready = false; this._st = { k: 'fail', msg: this._hint(e.message) }; return false; }
  }
  // Turn an onnxruntime dimension error into actionable guidance (which scenario the policy fits).
  _hint(msg) {
    const dim = /Got:\s*(\d+)\s*Expected:\s*(\d+)/.exec(msg);       // obs mismatch: Got=scenario, Expected=policy
    if (dim) return t(`策略输入维度=${dim[2]}，与当前场景 obs=${dim[1]} 不匹配——请切到匹配场景或选用对应策略`,
                      `policy expects obs=${dim[2]} but this scenario is obs=${dim[1]} — switch scenario or pick a matching policy`,
                      `方策の入力次元=${dim[2]} は現在のシーン obs=${dim[1]} と不一致——一致するシーンに切り替えるか対応する方策を選んでください`);
    const am = /^__DIM__act (\d+) (\d+)/.exec(msg);                  // act mismatch
    if (am) return t(`策略输出维度=${am[1]}，与当前场景 act=${am[2]} 不匹配`,
                     `policy outputs act=${am[1]} but this scenario needs act=${am[2]}`,
                     `方策の出力次元=${am[1]} は現在のシーン act=${am[2]} と次元不一致`);
    return msg;
  }
  // Localize the status at read-time so a language toggle updates it immediately.
  getStatus() {
    const sup = this.mode === 'setpoint' && this.layout;
    const outN = sup ? this.layout.length : this.actLen;
    const loaded = sup ? t(`策略已加载 · obs=${this.obsLen} · RL→设定点(${outN})→PID`, `Loaded · obs=${this.obsLen} · RL→setpoints(${outN})→PID`, `方策読み込み済み · obs=${this.obsLen} · RL→設定値(${outN})→PID`)
                       : t(`策略已加载 (obs=${this.obsLen}, act=${this.actLen})`, `Policy loaded (obs=${this.obsLen}, act=${this.actLen})`, `方策読み込み済み (obs=${this.obsLen}, act=${this.actLen})`);
    const s = this._st, st =
      s.k === 'loading' ? t('加载中…', 'Loading…', '読み込み中…')
      : s.k === 'loaded' ? loaded
      : s.k === 'err' ? t('ONNX 推理出错', 'ONNX inference error', 'ONNX 推論エラー') + ': ' + s.msg
      : s.k === 'fail' ? t('加载失败', 'Load failed', '読み込み失敗') + ': ' + s.msg
      : t('未加载策略', 'No policy loaded', '方策が読み込まれていません');
    return {
      ready: this.ready, status: st, obsLen: this.obsLen, actLen: this.actLen,
      policyId: this.policy?.id || null,
      policyLabel: this.policy ? t(this.policy.zh, this.policy.en, this.policy.ja) : '',
      safetyObs: this.safetyObs,
      safetyShield: this.safetyShield,
      observationProfile: this.observationProfile,
    };
  }
}

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = () => rej(new Error('无法加载 onnxruntime-web (离线? 可改为本地 vendored)'));
    document.head.appendChild(s);
  });
}
