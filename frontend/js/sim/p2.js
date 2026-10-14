// Industrial P2 operating profile for the jacketed CSTR. This layer mirrors the
// Python evaluation environment: correlated plant regimes, slowly developing
// process disturbances, and an independent high-temperature SIS.

import { P4_COOLING_DEFAULTS } from './plants.js?v=1';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const uniform = (lo, hi) => lo + Math.random() * (hi - lo);
const gauss = () => {
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const smooth = (z) => { z = clamp(z, 0, 1); return z * z * (3 - 2 * z); };
const pulse = (z) => { z = clamp(z, 0, 1); return z <= 0.5 ? smooth(2 * z) : smooth(2 * (1 - z)); };
const chooseMicroKind = () => {
  const draw = Math.random();
  return draw < 0.45 ? 'feed_temp_pulse' : draw < 0.80 ? 'coolant_temp_pulse' : 'cooling_flow_dip';
};

const REGIMES = {
  normal: {
    weight: 0.60,
    ranges: { UA: [0.95, 1.05], k0: [0.95, 1.05], deltaH: [0.98, 1.02], Fjmax: [0.95, 1.05] },
    events: ['utility_stress', 'feed_batch', 'pump_degradation'],
  },
  reactive_feed: {
    weight: 0.20,
    ranges: { UA: [0.95, 1.05], k0: [1.08, 1.25], deltaH: [1.02, 1.08], Fjmax: [0.95, 1.05] },
    events: ['utility_stress', 'feed_batch', 'pump_degradation'],
  },
  fouled: {
    weight: 0.15,
    ranges: { UA: [0.95, 1.05], k0: [0.96, 1.06], deltaH: [0.98, 1.03], Fjmax: [0.95, 1.05] },
    events: ['utility_stress', 'feed_batch', 'pump_degradation'],
  },
  utility_limited: {
    weight: 0.05,
    ranges: { UA: [0.95, 1.05], k0: [0.96, 1.06], deltaH: [0.98, 1.03], Fjmax: [0.95, 1.05] },
    events: ['feed_batch', 'pump_degradation'],
  },
};

function chooseRegime() {
  let draw = Math.random();
  for (const [name, cfg] of Object.entries(REGIMES)) {
    draw -= cfg.weight;
    if (draw <= 0) return name;
  }
  return 'normal';
}

export class P2IndustrialProfile {
  constructor(model, params = {}) { this.params = this._config(params); this.bind(model); }

  _config(overrides = {}) {
    const unknown = Object.keys(overrides).filter((key) => !(key in P4_COOLING_DEFAULTS));
    if (unknown.length) throw new Error(`Unknown P2.4 parameters: ${unknown.join(', ')}`);
    const config = { ...P4_COOLING_DEFAULTS, ...Object.fromEntries(Object.entries(overrides).map(([k, v]) => [k, +v])) };
    if (config.pump_curve_droop < 0 || config.system_static_head_fraction < 0 || config.system_static_head_fraction >= 1) {
      throw new Error('Invalid P2.4 pump/system curve parameters');
    }
    if (config.pump_head_response_tau_s <= 0 || config.cooling_header_volume_m3 <= 0
      || config.cooling_supply_recovery_tau_s <= 0) throw new Error('Invalid P2.4 dynamic time scale');
    return config;
  }

  setConfig(overrides = {}) { this.params = this._config({ ...this.params, ...overrides }); this.reset(); }

  bind(model) {
    this.model = model;
    this.enabled = model?.metadata?.().scenario === 'cstr_jacketed_p2';
    this.nominal = this.enabled ? Object.fromEntries(Object.entries(model.p).map(([k, v]) => [k, Array.isArray(v) ? v.slice() : v])) : {};
    this.autoEvents = true;
    this.reset();
  }

  _restoreNominal() {
    for (const [key, value] of Object.entries(this.nominal)) this.model.p[key] = Array.isArray(value) ? value.slice() : value;
  }

  reset() {
    this.sisLatched = false;
    this.feedOU = 0;
    this.utilityOU = 0;
    this.events = [];
    this.microEvents = [];
    this.regime = this.enabled ? chooseRegime() : 'ideal';
    this.multipliers = {};
    if (!this.enabled) return;

    this._restoreNominal();
    const cfg = REGIMES[this.regime];
    for (const [key, bounds] of Object.entries(cfg.ranges)) {
      const mult = uniform(bounds[0], bounds[1]);
      this.multipliers[key] = mult;
      this.model.p[key] = this.nominal[key] * mult;
    }
    const c = this.params;
    const uaFactor = this.regime === 'fouled'
      ? uniform(c.fouled_ua_factor_min, c.fouled_ua_factor_max)
      : uniform(c.normal_fouling_ua_factor_min, c.normal_fouling_ua_factor_max);
    this.initialFoulingResistance = Math.max(0, (1 / uaFactor - 1) / this.model.p.UA);
    this.initialPumpHead = this.regime === 'utility_limited'
      ? uniform(c.pump_head_limited_min, c.pump_head_limited_max)
      : uniform(c.pump_head_normal_min, c.pump_head_normal_max);
    this.baseFeedTemp = clamp(this.nominal.t_cold + uniform(-5, 5), 2, 35);
    this.baseJacketTemp = this.nominal.Tj_in;
    this.pumpHead = this.initialPumpHead;
    this.pumpHeadTarget = this.initialPumpHead;
    this.supplyTemperature = this.baseJacketTemp;
    this.supplyTargetTemperature = this.baseJacketTemp;
    this.returnTemperature = this.baseJacketTemp;

    const available = cfg.events.slice();
    const count = Math.random() < 0.5 ? 1 : Math.min(2, available.length);
    for (let i = 0; i < count; i++) {
      const idx = Math.floor(Math.random() * available.length);
      this.events.push({
        kind: available.splice(idx, 1)[0],
        start: uniform(90, 330),
        duration: uniform(90, 210),
        severity: uniform(0.35, 1),
        direction: Math.random() < 0.5 ? -1 : 1,
      });
    }
    let cursor = uniform(8, 20);
    while (cursor < 600) {
      const kind = chooseMicroKind();
      const duration = uniform(12, 30);
      const amplitude = kind === 'feed_temp_pulse' ? uniform(0.5, 1.5)
        : kind === 'coolant_temp_pulse' ? uniform(0.4, 1.2) : uniform(0.02, 0.06);
      this.microEvents.push({ kind, start: cursor, duration, amplitude,
        direction: kind === 'cooling_flow_dip' || Math.random() < 0.5 ? 1 : -1 });
      cursor += duration + uniform(8, 25);
    }
    this.current = { t_cold: this.baseFeedTemp, t_jacket_in: this.baseJacketTemp,
      cooling_flow_factor: Math.sqrt(this.initialPumpHead), ua_factor: uaFactor,
      feed_concentration_factor: 1, kinetics_factor: 1, reaction_heat_factor: 1,
      cooling_hydraulics_enabled: true, valve_rangeability: c.valve_rangeability,
      valve_pressure_drop_fraction: c.valve_pressure_drop_fraction,
      reference_cooling_flow_fraction: c.reference_cooling_flow_fraction,
      ua_flow_exponent: c.ua_flow_exponent, ua_flow_factor_min: c.ua_flow_factor_min,
      ua_flow_factor_max: c.ua_flow_factor_max, clean_ua_factor: 1,
      fouling_resistance_k_per_w: this.initialFoulingResistance,
      pump_head_factor: this.initialPumpHead, pump_head_target_factor: this.initialPumpHead,
      pump_curve_enabled: true, pump_curve_droop: c.pump_curve_droop,
      system_static_head_fraction: c.system_static_head_fraction,
      cooling_pump_design_head_m: c.cooling_pump_design_head_m,
      cooling_pump_design_efficiency: c.cooling_pump_design_efficiency,
      cooling_pump_best_efficiency_flow_fraction: c.cooling_pump_best_efficiency_flow_fraction,
      cooling_pump_min_efficiency_fraction: c.cooling_pump_min_efficiency_fraction,
      cooling_pump_efficiency_curve_coefficient: c.cooling_pump_efficiency_curve_coefficient,
      coolant_supply_target_temperature: this.supplyTargetTemperature,
      coolant_return_temperature: this.returnTemperature };
  }

  initialState(x) {
    if (!this.enabled) return x;
    return x.map((value) => value * uniform(0.92, 1.08));
  }

  setAutoEvents(on) { this.autoEvents = !!on; }

  update(t, dt, manualEnv, state = null) {
    if (!this.enabled) return manualEnv;
    const af = Math.exp(-dt / 60), au = Math.exp(-dt / 90);
    this.feedOU = af * this.feedOU + 0.35 * Math.sqrt(1 - af * af) * gauss();
    this.utilityOU = au * this.utilityOU + 0.25 * Math.sqrt(1 - au * au) * gauss();

    let hotWater = 0, flowLoss = 0, feedShift = 0, pumpDegradation = 0;
    let feedComposition = 0, feedActivity = 0, feedHeat = 0;
    let microFeed = 0, microJacket = 0, microFlowLoss = 0;
    if (this.autoEvents) {
      for (const event of this.events) {
        const severity = event.severity * smooth((t - event.start) / event.duration);
        if (event.kind === 'utility_stress') {
          hotWater = Math.max(hotWater, 10 * severity);
          flowLoss = Math.max(flowLoss, 0.40 * severity);
        } else if (event.kind === 'feed_batch') {
          feedShift += event.direction * 5 * severity;
          feedComposition += event.direction * 0.06 * severity;
          feedActivity += event.direction * 0.04 * severity;
          feedHeat += event.direction * 0.02 * severity;
        } else if (event.kind === 'pump_degradation') pumpDegradation = Math.max(pumpDegradation, this.params.pump_degradation_max * severity);
      }
      for (const event of this.microEvents) {
        const value = event.amplitude * pulse((t - event.start) / event.duration);
        if (event.kind === 'feed_temp_pulse') microFeed += event.direction * value;
        else if (event.kind === 'coolant_temp_pulse') microJacket += event.direction * value;
        else if (event.kind === 'cooling_flow_dip') microFlowLoss += value;
      }
    }

    const p = this.model.p;
    const manualFeed = (manualEnv.t_cold ?? p.t_cold) - p.t_cold;
    const manualJacket = (manualEnv.t_jacket_in ?? p.Tj_in) - p.Tj_in;
    const manualFlow = manualEnv.cooling_flow_factor ?? 1;
    const manualUA = manualEnv.ua_factor ?? 1;
    const headLoss = flowLoss + microFlowLoss + 0.01 * Math.max(0, this.utilityOU);
    const pumpHeadTarget = clamp(this.initialPumpHead * (1 - pumpDegradation) * (1 - headLoss) * manualFlow ** 2, 0.15, 1.2);
    const headAlpha = 1 - Math.exp(-dt / this.params.pump_head_response_tau_s);
    this.pumpHead += headAlpha * (pumpHeadTarget - this.pumpHead);
    this.pumpHeadTarget = pumpHeadTarget;
    const utilityTemperatureTarget = clamp(
      this.baseJacketTemp + hotWater + microJacket + this.utilityOU + manualJacket,
      this.params.cooling_supply_temperature_min_c, this.params.cooling_supply_temperature_max_c,
    );
    const coolantFlow = state?.cooling_flow?.[0] ?? 0;
    const jacketReturn = state?.jacket_temp?.[0] ?? this.returnTemperature;
    const recovery = (utilityTemperatureTarget - this.supplyTemperature) / this.params.cooling_supply_recovery_tau_s;
    const recirculation = this.params.cooling_return_recirc_fraction * coolantFlow
      / this.params.cooling_header_volume_m3 * (jacketReturn - this.supplyTemperature);
    this.supplyTemperature = clamp(this.supplyTemperature + dt * (recovery + recirculation),
      this.params.cooling_supply_temperature_min_c, this.params.cooling_supply_temperature_max_c);
    this.supplyTargetTemperature = utilityTemperatureTarget;
    this.returnTemperature = jacketReturn;
    const foulingResistance = this.initialFoulingResistance + this.params.fouling_resistance_rate_k_per_w_s * t;
    const uaFactor = manualUA / (1 + this.model.p.UA * manualUA * foulingResistance);
    this.current = {
      ...manualEnv,
      t_cold: clamp(this.baseFeedTemp + feedShift + microFeed + this.feedOU + manualFeed, 2, 35),
      t_jacket_in: this.supplyTemperature,
      cooling_flow_factor: Math.sqrt(this.pumpHead),
      ua_factor: clamp(uaFactor, 0.25, 1.1),
      feed_concentration_factor: clamp(1 + feedComposition, 0.90, 1.10),
      kinetics_factor: clamp(1 + feedActivity, 0.92, 1.08),
      reaction_heat_factor: clamp(1 + feedHeat, 0.96, 1.04),
      cooling_hydraulics_enabled: true,
      valve_rangeability: this.params.valve_rangeability,
      valve_pressure_drop_fraction: this.params.valve_pressure_drop_fraction,
      reference_cooling_flow_fraction: this.params.reference_cooling_flow_fraction,
      ua_flow_exponent: this.params.ua_flow_exponent,
      ua_flow_factor_min: this.params.ua_flow_factor_min,
      ua_flow_factor_max: this.params.ua_flow_factor_max,
      clean_ua_factor: manualUA,
      fouling_resistance_k_per_w: foulingResistance,
      pump_head_factor: this.pumpHead,
      pump_head_target_factor: this.pumpHeadTarget,
      pump_curve_enabled: true,
      pump_curve_droop: this.params.pump_curve_droop,
      system_static_head_fraction: this.params.system_static_head_fraction,
      cooling_pump_design_head_m: this.params.cooling_pump_design_head_m,
      cooling_pump_design_efficiency: this.params.cooling_pump_design_efficiency,
      cooling_pump_best_efficiency_flow_fraction: this.params.cooling_pump_best_efficiency_flow_fraction,
      cooling_pump_min_efficiency_fraction: this.params.cooling_pump_min_efficiency_fraction,
      cooling_pump_efficiency_curve_coefficient: this.params.cooling_pump_efficiency_curve_coefficient,
      coolant_supply_target_temperature: this.supplyTargetTemperature,
      coolant_return_temperature: this.returnTemperature,
    };
    return this.current;
  }

  protect(act, trueTemperature) {
    if (!this.enabled) return act;
    if (trueTemperature >= 92) this.sisLatched = true;
    else if (this.sisLatched && trueTemperature <= 89) this.sisLatched = false;
    if (!this.sisLatched) return act;
    const out = { pumps: act.pumps.slice(), valves: act.valves.slice(), heaters: act.heaters.slice() };
    if (out.pumps.length) out.pumps[0] = 0;
    if (out.heaters.length) out.heaters[0] = 1;
    return out;
  }

  status(t = 0) {
    if (!this.enabled) return null;
    const process = { ...this.current };
    process.effective_ua_factor = (this.multipliers.UA ?? 1) * (process.ua_factor ?? 1);
    process.effective_cooling_factor = (this.multipliers.Fjmax ?? 1) * (process.cooling_flow_factor ?? 1);
    return {
      id: 'industrial_p2_4', regime: this.regime, sis_latched: this.sisLatched,
      multipliers: { ...this.multipliers }, parameters: { ...this.params }, process,
      events: this.events.map((event) => ({ ...event, progress: smooth((t - event.start) / event.duration) })),
      micro_events: this.microEvents.map((event) => ({ ...event, progress: pulse((t - event.start) / event.duration) })),
    };
  }
}
