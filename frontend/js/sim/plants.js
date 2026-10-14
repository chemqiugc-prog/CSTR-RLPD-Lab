// CSTR-only browser plant models. Equations and parameters mirror cstrlab/plants.py.
import { t } from '../i18n.js?v=1';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const maxv = (a, b) => (a > b ? a : b);


class LegacyCSTR {
  constructor() {
    this.scenario = 'cstr';
    this.modelVersion = 'cstr_v1_legacy';
    this.n = 1;
    this.dtMicro = 0.01;
    this.p = {
      Dmax: 0.02, Caf: 1.0, k0: 1e8, EaR: 7000, Hr: 120, Uc: 0.05, Tcool: 10,
      cool_max: 80000, feed_power_max: 1200, t_cold: 20, t_amb: 20, h_floor: 1e-3,
    };
  }

  actuatorCounts() { return [1, 0, 1]; }
  get heightMax() { return [1]; }
  _rate(ca, temperature) {
    return this.p.k0 * Math.exp(-this.p.EaR / (temperature + 273.15)) * maxv(ca, 0);
  }
  derivatives(x, act, env) {
    const p = this.p, [ca, temperature] = x;
    const dilution = act.pumps[0] * p.Dmax;
    const rate = this._rate(ca, temperature);
    return [
      dilution * (p.Caf - ca) - rate,
      dilution * (env.t_cold - temperature) + p.Hr * rate
        - p.Uc * act.heaters[0] * (temperature - p.Tcool),
    ];
  }
  buildState(x, act, env, time) {
    const ca = maxv(x[0], 0), temperature = x[1];
    const feedFlow = act.pumps[0] * this.p.Dmax;
    return {
      t: time, levels: [], temps: [temperature], volumes: [], conc: [ca],
      heater_power: [act.heaters[0] * this.p.cool_max],
      pump_flow: [feedFlow], pump_power: [act.pumps[0] * this.p.feed_power_max],
      tank_outflow: [feedFlow], t_cold: env.t_cold, t_amb: env.t_amb,
    };
  }
  initialState() { return [0.5, 50]; }
  clampState(x) { x[0] = Math.max(0, x[0]); x[1] = Math.min(200, x[1]); return x; }
  controlledLevels() { return []; }
  defaultSetpoints() { return [{}, [60]]; }
  defaultGains() {
    return {
      level_pump: { kp: 0, ki: 0, kd: 0 },
      level_valve: { kp: 0, ki: 0, kd: 0 },
      temp: { kp: 0.20, ki: 0.01, kd: 0 },
    };
  }
  controlPairing() {
    return { level: [], temp: [[0, 0, true]], demand_valve_index: null, holds: [['pump', 0, 0.5]] };
  }
  idealPower() { return 0; }
  energyScored() { return false; }
  safety() { return { dryFire: false, overflow: false, overTempAction: 'pump' }; }
  setConfig() {}
  metadata() {
    return {
      scenario: this.scenario,
      topology: 'cstr',
      environment_version: this.modelVersion,
      name: t('官方 Legacy CSTR', 'Upstream Legacy CSTR', '上流 Legacy CSTR'),
      blurb: t(
        '保留官方两状态基准，用于复现原始CSTR环境和RLPD。',
        'Preserved two-state upstream benchmark for reproducing the original CSTR and RLPD.',
        '元のCSTRとRLPDを再現するための上流2状態ベンチマーク。',
      ),
      n_tanks: 1,
      tank_labels: [t('R-1 反应器', 'R-1 reactor', 'R-1 反応器')],
      actuators: {
        pumps: [t('进料', 'Feed', '供給')], valves: [], heaters: [t('冷却', 'Cooling', '冷却')],
      },
      controlled_levels: [], height_max: [1],
      pump_flow_max: [this.p.Dmax], heater_max: [this.p.cool_max], config: {},
      trends: [
        { label: t('反应器温度 (°C)', 'Reactor Temp (°C)', '反応器温度 (°C)'), field: 'temps', sp: 't_sp', spIdx: [0], fmt: 1 },
        { label: t('反应物浓度 Cₐ', 'Reactant Cₐ', '反応物 Cₐ'), field: 'conc', fmt: 3 },
        { label: t('冷却功率 (kW)', 'Cooling Power (kW)', '冷却出力 (kW)'), field: 'heater_power', scale: 0.001, fmt: 1 },
      ],
    };
  }
}


export const P4_COOLING_DEFAULTS = Object.freeze({
  valve_rangeability: 30,
  valve_pressure_drop_fraction: 0.25,
  reference_cooling_flow_fraction: 0.26422141637233987,
  ua_flow_exponent: 0.35,
  ua_flow_factor_min: 0.25,
  ua_flow_factor_max: 1.60,
  normal_fouling_ua_factor_min: 0.92,
  normal_fouling_ua_factor_max: 1.00,
  fouled_ua_factor_min: 0.70,
  fouled_ua_factor_max: 0.90,
  fouling_resistance_rate_k_per_w_s: 1e-10,
  pump_head_normal_min: 0.95,
  pump_head_normal_max: 1.05,
  pump_head_limited_min: 0.70,
  pump_head_limited_max: 0.90,
  pump_degradation_max: 0.25,
  pump_curve_droop: 0.25,
  system_static_head_fraction: 0.05,
  pump_head_response_tau_s: 8,
  cooling_header_volume_m3: 2,
  cooling_supply_recovery_tau_s: 45,
  cooling_return_recirc_fraction: 0.08,
  cooling_supply_temperature_min_c: 2,
  cooling_supply_temperature_max_c: 40,
  cooling_pump_design_head_m: 25,
  cooling_pump_design_efficiency: 0.72,
  cooling_pump_best_efficiency_flow_fraction: 0.65,
  cooling_pump_min_efficiency_fraction: 0.45,
  cooling_pump_efficiency_curve_coefficient: 1.20,
});


class LocalP24CSTR {
  constructor() {
    this.scenario = 'cstr_jacketed_p2';
    this.modelVersion = 'cstr_v2_jacketed_industrial_p2_4';
    this.isP2 = true;
    this.n = 1;
    this.dtMicro = 0.01;
    this.extraObsDim = 3;
    const rho = 1000, Cp = 4186, V = 1;
    this.p = {
      V, Vj: 0.20, rho, Cp, rhoj: rho, Cpj: Cp,
      Fmax: 0.020, Fjmax: 0.100, Caf: 1.0,
      k0: 1e8, EaR: 7000, deltaH: 120 * rho * Cp / 1000,
      UA: 110000, Tj_in: 10,
      cool_max: 80000, feed_power_max: 1200,
      t_cold: 20, t_amb: 20, h_floor: 1e-3,
    };
  }

  actuatorCounts() { return [1, 0, 1]; }
  get heightMax() { return [1]; }
  _rate(ca, temperature) {
    return this.p.k0 * Math.exp(-this.p.EaR / (temperature + 273.15)) * maxv(ca, 0);
  }
  coolingCharacteristics(opening, env = {}) {
    const p = this.p, u = clamp(opening, 0, 1);
    if (!env.cooling_hydraulics_enabled) {
      const flowFraction = u * Math.max(0, env.cooling_flow_factor ?? 1);
      return {
        flow: flowFraction * p.Fjmax,
        ua: p.UA * Math.max(0, env.ua_factor ?? 1),
        flowFraction, valveCapacityFraction: u, installedFlowFraction: u,
        operatingHeadFactor: 0, pumpEfficiency: 0, pumpPower: 0,
      };
    }
    const rangeability = Math.max(1.000001, env.valve_rangeability);
    const valveFraction = (rangeability ** u - 1) / (rangeability - 1);
    const pressureFraction = clamp(env.valve_pressure_drop_fraction, 1e-6, 1);
    const installedFraction = valveFraction / Math.sqrt(
      pressureFraction + (1 - pressureFraction) * valveFraction ** 2
    );
    const headFactor = Math.max(0, env.pump_head_factor ?? 1);
    let flowFraction, operatingHeadFactor;
    if (env.pump_curve_enabled && installedFraction > 1e-12) {
      const droop = Math.max(0, env.pump_curve_droop ?? 0);
      const staticHead = clamp(env.system_static_head_fraction ?? 0, 0, 0.999999);
      const numerator = headFactor * (1 + droop) - staticHead;
      const denominator = (1 - staticHead) / installedFraction ** 2 + droop;
      flowFraction = Math.sqrt(Math.max(0, numerator) / Math.max(denominator, 1e-12));
      operatingHeadFactor = staticHead + (1 - staticHead) * (flowFraction / installedFraction) ** 2;
    } else {
      flowFraction = installedFraction * Math.sqrt(headFactor);
      operatingHeadFactor = headFactor;
    }
    flowFraction = clamp(flowFraction, 0, 1.2);
    const reference = Math.max(1e-6, env.reference_cooling_flow_fraction);
    const rawFlowFactor = (Math.max(flowFraction, 1e-6) / reference) ** env.ua_flow_exponent;
    const flowFactor = clamp(rawFlowFactor, env.ua_flow_factor_min, env.ua_flow_factor_max);
    const cleanUA = p.UA * Math.max(0, env.clean_ua_factor ?? 1) * flowFactor;
    const resistance = Math.max(0, env.fouling_resistance_k_per_w ?? 0);
    let pumpEfficiency = 0, pumpPower = 0;
    if (env.pump_curve_enabled && flowFraction > 0) {
      const designEfficiency = clamp(env.cooling_pump_design_efficiency, 1e-6, 1);
      const efficiencyFactor = Math.max(
        env.cooling_pump_min_efficiency_fraction,
        1 - env.cooling_pump_efficiency_curve_coefficient
          * (flowFraction - env.cooling_pump_best_efficiency_flow_fraction) ** 2,
      );
      pumpEfficiency = Math.min(1, designEfficiency * efficiencyFactor);
      const headM = env.cooling_pump_design_head_m * Math.max(0, operatingHeadFactor);
      pumpPower = p.rhoj * 9.80665 * (flowFraction * p.Fjmax) * headM / pumpEfficiency;
    }
    return {
      flow: flowFraction * p.Fjmax,
      ua: 1 / (1 / Math.max(cleanUA, 1e-12) + resistance),
      flowFraction, valveCapacityFraction: valveFraction, installedFlowFraction: installedFraction,
      operatingHeadFactor, pumpEfficiency, pumpPower,
    };
  }
  coolingOpeningForFlow(targetFraction, env) {
    let lo = 0, hi = 1;
    for (let i = 0; i < 80; i++) {
      const mid = 0.5 * (lo + hi);
      if (this.coolingCharacteristics(mid, env).flowFraction < targetFraction) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
  derivatives(x, act, env) {
    const p = this.p, [ca, cb, temperature, jacketTemperature] = x;
    const feedFlow = act.pumps[0] * p.Fmax;
    const cooling = this.coolingCharacteristics(act.heaters[0], env);
    const jacketInlet = env.t_jacket_in ?? p.Tj_in;
    const feedConcentration = p.Caf * Math.max(0, env.feed_concentration_factor ?? 1);
    const rate = this._rate(ca, temperature) * Math.max(0, env.kinetics_factor ?? 1);
    const dilution = feedFlow / p.V;
    const heatTransfer = cooling.ua * (temperature - jacketTemperature);
    const reactionHeat = p.deltaH * Math.max(0, env.reaction_heat_factor ?? 1) * 1000 * p.V * rate;
    return [
      dilution * (feedConcentration - ca) - rate,
      -dilution * cb + rate,
      (p.rho * p.Cp * feedFlow * (env.t_cold - temperature) + reactionHeat - heatTransfer)
        / (p.rho * p.Cp * p.V),
      (p.rhoj * p.Cpj * cooling.flow * (jacketInlet - jacketTemperature) + heatTransfer)
        / (p.rhoj * p.Cpj * p.Vj),
    ];
  }
  buildState(x, act, env, time) {
    const p = this.p, ca = maxv(x[0], 0), cb = maxv(x[1], 0);
    const temperature = x[2], jacketTemperature = x[3];
    const feedFlow = act.pumps[0] * p.Fmax;
    const cooling = this.coolingCharacteristics(act.heaters[0], env);
    return {
      t: time, levels: [], temps: [temperature], volumes: [], conc: [ca], product_conc: [cb],
      jacket_temp: [jacketTemperature],
      cooling_duty: [Math.abs(cooling.ua * (temperature - jacketTemperature))],
      effective_ua: [cooling.ua], cooling_flow: [cooling.flow],
      cooling_flow_fraction: [cooling.flowFraction],
      heater_power: [cooling.flowFraction * p.cool_max],
      cooling_operating_head: [cooling.operatingHeadFactor],
      cooling_pump_power: [cooling.pumpPower],
      coolant_supply_temp: [env.t_jacket_in ?? p.Tj_in],
      coolant_return_temp: [jacketTemperature],
      pump_flow: [feedFlow], pump_power: [act.pumps[0] * p.feed_power_max],
      tank_outflow: [feedFlow], t_cold: env.t_cold, t_amb: env.t_amb,
    };
  }
  observationExtras(state) {
    return [state.conc?.[0] ?? 0, state.product_conc?.[0] ?? 0, state.jacket_temp?.[0] ?? 0];
  }
  productionRate(state, act) {
    return (act.pumps[0] || 0) * this.p.Fmax / this.p.V * (state.product_conc?.[0] ?? 0);
  }
  initialState() { return [0.11771023406521823, 0.8822897659347819, 60, 34.93165497960367]; }
  clampState(x) {
    x[0] = Math.max(0, x[0]); x[1] = Math.max(0, x[1]);
    x[2] = Math.min(200, x[2]); x[3] = Math.min(200, x[3]);
    return x;
  }
  controlledLevels() { return []; }
  defaultSetpoints() { return [{}, [60]]; }
  defaultGains() {
    return {
      level_pump: { kp: 0, ki: 0, kd: 0 },
      level_valve: { kp: 0, ki: 0, kd: 0 },
      temp: { kp: 0.12, ki: 0.004, kd: 0 },
    };
  }
  pidTempBias() {
    const c = P4_COOLING_DEFAULTS;
    return [this.coolingOpeningForFlow(c.reference_cooling_flow_fraction, {
      cooling_hydraulics_enabled: true,
      valve_rangeability: c.valve_rangeability,
      valve_pressure_drop_fraction: c.valve_pressure_drop_fraction,
      reference_cooling_flow_fraction: c.reference_cooling_flow_fraction,
      ua_flow_exponent: c.ua_flow_exponent,
      ua_flow_factor_min: c.ua_flow_factor_min,
      ua_flow_factor_max: c.ua_flow_factor_max,
      clean_ua_factor: 1, fouling_resistance_k_per_w: 0, pump_head_factor: 1,
      pump_curve_enabled: true, pump_curve_droop: c.pump_curve_droop,
      system_static_head_fraction: c.system_static_head_fraction,
      cooling_pump_design_head_m: c.cooling_pump_design_head_m,
      cooling_pump_design_efficiency: c.cooling_pump_design_efficiency,
      cooling_pump_best_efficiency_flow_fraction: c.cooling_pump_best_efficiency_flow_fraction,
      cooling_pump_min_efficiency_fraction: c.cooling_pump_min_efficiency_fraction,
      cooling_pump_efficiency_curve_coefficient: c.cooling_pump_efficiency_curve_coefficient,
    })];
  }
  controlPairing() {
    return { level: [], temp: [[0, 0, true]], demand_valve_index: null, holds: [['pump', 0, 0.5]] };
  }
  idealPower() { return 0; }
  energyScored() { return false; }
  safety() { return { dryFire: false, overflow: false, overTempAction: 'pump' }; }
  setConfig() {}
  metadata() {
    return {
      scenario: this.scenario,
      topology: 'cstr',
      environment_version: this.modelVersion,
      realism_profile: 'industrial_p4',
      name: t('本地工业夹套CSTR · P2.4', 'Local Industrial Jacketed CSTR · P2.4', 'ローカル産業用CSTR · P2.4'),
      blurb: t(
        '当前主线：四状态夹套、工业仪表与执行器、相关扰动、冷却水力和独立SIS。',
        'Current mainline: four-state jacket, industrial I/O, correlated disturbances, cooling hydraulics, and independent SIS.',
        '現在の主系統：4状態ジャケット、産業I/O、相関外乱、冷却水力、独立SIS。',
      ),
      n_tanks: 1,
      tank_labels: [t('R-1 夹套反应器', 'R-1 jacketed reactor', 'R-1 ジャケット反応器')],
      actuators: {
        pumps: [t('进料', 'Feed', '供給')], valves: [], heaters: [t('冷却水', 'Coolant', '冷却水')],
      },
      controlled_levels: [], height_max: [1],
      pump_flow_max: [this.p.Fmax], heater_max: [this.p.cool_max], config: {},
      trends: [
        { label: t('反应器温度 (°C)', 'Reactor Temp (°C)', '反応器温度 (°C)'), field: 'temps', sp: 't_sp', spIdx: [0], fmt: 1 },
        { label: t('夹套温度 (°C)', 'Jacket Temp (°C)', 'ジャケット温度 (°C)'), field: 'jacket_temp', fmt: 1 },
        { label: t('隐藏真值 Cₐ', 'Latent truth Cₐ', '隠れ真値 Cₐ'), field: 'conc', fmt: 3 },
        { label: t('隐藏真值 Cᵦ', 'Latent truth Cᵦ', '隠れ真値 Cᵦ'), field: 'product_conc', fmt: 3 },
        { label: t('实际移热量 (kW)', 'Heat Removal (kW)', '実移熱量 (kW)'), field: 'cooling_duty', scale: 0.001, fmt: 1 },
      ],
    };
  }
}


const MODELS = { cstr: LegacyCSTR, cstr_jacketed_p2: LocalP24CSTR };

export function makeModel(scenario = 'cstr_jacketed_p2') {
  const Model = MODELS[scenario];
  if (!Model) throw new Error(`Unknown CSTR scenario: ${scenario}`);
  return new Model();
}

export const SCENARIOS = Object.keys(MODELS);
