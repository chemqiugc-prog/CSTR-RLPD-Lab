"""Gymnasium environment for the two CSTR baselines in this repository.

The legacy plant preserves the upstream benchmark. The jacketed plant supports
the local P2.4 industrial-realism profile and concentration-free observation.

Physics parity with the browser JS is enforced by tests/test_parity.py.

Observation contracts (kept in parity with the browser for ONNX interchange):
  analyzer: legacy generic observations plus model-specific analyzer signals.
  field_no_concentration (industrial jacketed CSTR):
    [T, Tj, Tsp, Tfeed, Tcoolant, Ffeed/Fmax, Fcool/Fjmax, dT/dt, dTj/dt]
  action = [pumps..., valves..., heaters...] in [0, 1]   (direct-actuator mode)

reward_mode:
  "kpi"      (default) reward = -(instantaneous KPI penalty) using the same
             tracking + excess-energy + safety scoring the gym/browser display
             (scoring.py) — so the RL optimises EXACTLY what it's judged on.
  "economic" CSTR production-maximisation (legacy economic demo).
  "track"    plain setpoint tracking (legacy).

dynamic=True injects within-episode disturbances (setpoint steps, cold-inlet
steps, ambient drift, demand surges) on top of domain-randomised start points.
The policy OBSERVES the changed conditions (t_cold / t_amb / setpoints are all in
obs), so this trains the online adaptation a fixed-tuning MPC can't match.
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .plants import SCENARIOS, make_model, obs_vector
from .kernel import Integrator
from .realism import IndustrialP2Realism, p3_config, p4_config
from .scoring import KPIScorer

# advisory/interlock limits — mirror frontend/js/sim/alarms.js (LIMITS)
T_HIGH, T_TRIP = 80.0, 92.0
H_HIGH_FRAC, H_LOW_FRAC, H_OVERFLOW_FRAC = 0.90, 0.15, 0.97
I_TEMP_MAX, I_LEVEL_MAX = 300.0, 8.0          # anti-windup clamp + obs-normaliser for ∫error

# Operating-regime (工况) variation: per-episode multipliers on plant parameters
# that SHIFT the dynamics (fouling, actuator ageing, gain drift). A fixed-tuned
# PID / fixed-model MPC degrades off-nominal; an RL policy trained across the
# distribution (or learning online) stays robust — that's the adaptation edge.
PLANT_REGIME = {
    "cstr":      {"Uc": (0.5, 1.6), "k0": (0.55, 1.7), "Hr": (0.85, 1.2)},
    "cstr_jacketed": {"UA": (0.5, 1.4), "k0": (0.55, 1.7), "deltaH": (0.85, 1.2), "Fjmax": (0.7, 1.2)},
}

# Economic objective per scenario: maximize value − energy-cost while keeping
# controlled vars inside SOFT acceptance bands (not on a fixed setpoint). Within
# the band the economics are optimised; the optimum hugs a band/constraint edge
# that DRIFTS with 工况 (feed/outdoor temp, plant efficiency), so a fixed-SP PID /
# fixed-model MPC is structurally suboptimal and an adaptive RL wins.
#   temp_band : per-tank (lo, hi) acceptance window  (None = unconstrained side)
#   value     : "production" (CSTR: hug the drifting safe-temp edge) | "none" (min-energy)
#   w_*       : reward weights (value reward, energy cost /kW, band violation, runaway)
# PRICED economics (2026-07 rework): weights are now REAL MONEY RATES, so the
# per-step "profit" IS ¥/h and the UI can show operating cost / profit directly.
#   w_energy : ¥/h per kW   — electricity 0.7 ¥/kWh; heater fuel gas 97 ¥/GJ = 0.35 ¥/h·kW
#   w_viol   : ¥/h per violation-unit (off-spec contractual penalty, scaled from the
#              old tuning by the SAME factor as w_energy — an order-preserving linear
#              rescale, so previously-trained policies remain optimal, no retrain)
#   w_value  : ¥ per product unit (CSTR fine-chemical credit, same rescale)
# Bands unchanged. 工况 drift still moves the optimum; fixed tuning stays suboptimal.
ECON = {
    "cstr":      {"temp_band": [(None, 88.0)], "level_band": [],
                  "value": "production", "w_value": 1575.0, "w_energy": 0.7, "w_viol": 14.0},
    "cstr_jacketed": {"temp_band": [(None, 88.0)], "level_band": [],
                      "value": "production", "w_value": 1575.0, "w_energy": 0.7, "w_viol": 14.0},
}

# Supervisory (RL-on-PID / RTO) action layout: RL outputs SETPOINTS that the inner
# PID tracks (the plant is always regulated -> always controlled, RL >= PID by
# construction; RL only picks the economically-best targets, adapting to 工况). Each
# entry: ("t_sp", tank, lo, hi) | ("h_sp", level, lo, hi) | ("mv", kind, idx, lo, hi)
# ("mv" = an unregulated economic MV set directly, e.g. CSTR feed rate). Levels not
# listed stay at the default SP (PID holds them).
SUPERVISORY = {
    "cstr":      [("t_sp", 0, 45, 90), ("mv", "pumps", 0, 0.3, 1.0)],
    "cstr_jacketed": [("t_sp", 0, 45, 90), ("mv", "pumps", 0, 0.3, 1.0)],
}

OBSERVATION_PROFILES = ("analyzer", "field_no_concentration")
FIELD_CSTR_OBS_DIM = 9
SUPERVISORY_GUARDS = ("none", "v4")
V4_GUARD_DEFAULTS = {
    "trigger_temp": 85.5,
    "hard_temp": 86.5,
    "horizon_s": 5.0,
    "target_temp": 82.0,
    "pump_cap": 0.80,
    "feed_stop_temp": 87.5,
    "cooling_floor": 0.75,
    "predictive_emergency": False,
}
RISK_BARRIER_DEFAULTS = {
    "trigger_temp": 85.5,
    "temperature_scale": 1.0,
    "horizon_s": 5.0,
    "jacket_horizon_s": 2.0,
    "barrier_weight": 2.0,
    "recovery_weight": 1.0,
    "trip_cost": 100.0,
    "recovery_clip": 5.0,
}


class CSTRGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario="cstr_jacketed", control_dt=0.5, episode_steps=600,
                 reward_mode="kpi", dynamic=True, randomize=True, randomize_setpoints=True,
                 randomize_plant=False, plant_drift=False, integral_obs=False, action_mode="actuator",
                 noise=False, noise_pct=0.01, custom_reward=None,
                 terminate_on_runaway=False, reward_scale=0.03, w_prod=1000.0, w_energy=2.0,
                 w_constraint=8.0, realism_profile="ideal", realism_params=None,
                 risk_observation=False, observation_profile="analyzer",
                 supervisory_t_sp_high=None, supervisory_guard="none",
                 supervisory_guard_params=None, risk_barrier_params=None):
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.scenario = scenario
        self.control_dt = float(control_dt)
        self.episode_steps = int(episode_steps)
        self.reward_mode = reward_mode
        if self.reward_mode == "economic_risk_barrier" and scenario != "cstr_jacketed":
            raise ValueError("economic_risk_barrier currently supports only cstr_jacketed")
        self.reward_scale = reward_scale          # keep Q-magnitudes sane -> stable critic
        # NOTE: ECON weights are now REAL ¥/h rates, so economic rewards are larger
        # (heater ~1e4). When (re)training, scale per scenario, e.g. heater 3e-4.
        self.dynamic = dynamic
        self.randomize_plant = randomize_plant    # per-episode 工况 (operating-regime) variation
        self.plant_drift = plant_drift            # slow within-episode parameter drift
        self.randomize = randomize
        self.randomize_setpoints = randomize_setpoints
        self.noise = noise                        # measurement noise on observed levels/temps
        self.noise_pct = noise_pct                # std as a fraction of the per-quantity scale
        self.realism_profile = str(realism_profile)
        if self.realism_profile not in ("ideal", "industrial_p2", "industrial_p3", "industrial_p4"):
            raise ValueError("realism_profile must be 'ideal', 'industrial_p2', 'industrial_p3', or 'industrial_p4'")
        if self.realism_profile in ("industrial_p2", "industrial_p3", "industrial_p4") and scenario != "cstr_jacketed":
            raise ValueError("industrial CSTR profiles currently support only cstr_jacketed")
        self.realism = IndustrialP2Realism() if self.realism_profile != "ideal" else None
        self.observation_profile = str(observation_profile)
        if self.observation_profile not in OBSERVATION_PROFILES:
            raise ValueError(f"observation_profile must be one of {OBSERVATION_PROFILES}")
        if self.observation_profile == "field_no_concentration" and (
                scenario != "cstr_jacketed" or self.realism is None):
            raise ValueError("field_no_concentration requires an industrial cstr_jacketed profile")
        if self.realism_profile == "industrial_p4":
            self.p3_params = p4_config(realism_params)
        elif self.realism_profile == "industrial_p3":
            self.p3_params = p3_config(realism_params)
        else:
            self.p3_params = {}
        self.risk_observation = bool(risk_observation)
        if self.risk_observation and self.realism is None:
            raise ValueError("risk_observation requires an industrial realism profile")
        self.supervisory_guard = str(supervisory_guard)
        if self.supervisory_guard not in SUPERVISORY_GUARDS:
            raise ValueError(f"supervisory_guard must be one of {SUPERVISORY_GUARDS}")
        if self.supervisory_guard != "none" and (
                scenario != "cstr_jacketed" or action_mode != "setpoint"):
            raise ValueError("the V4 supervisory guard requires setpoint-mode cstr_jacketed")
        unknown_guard_params = set(supervisory_guard_params or {}) - set(V4_GUARD_DEFAULTS)
        if unknown_guard_params:
            names = ", ".join(sorted(unknown_guard_params))
            raise ValueError(f"unknown supervisory guard parameters: {names}")
        self.supervisory_guard_params = dict(V4_GUARD_DEFAULTS)
        self.supervisory_guard_params.update(supervisory_guard_params or {})
        self.supervisory_guard_params["horizon_s"] = max(
            0.0, float(self.supervisory_guard_params["horizon_s"])
        )
        self.supervisory_guard_params["cooling_floor"] = float(np.clip(
            self.supervisory_guard_params["cooling_floor"], 0.0, 1.0
        ))
        self.supervisory_guard_params["predictive_emergency"] = bool(
            self.supervisory_guard_params["predictive_emergency"]
        )
        unknown_risk_params = set(risk_barrier_params or {}) - set(RISK_BARRIER_DEFAULTS)
        if unknown_risk_params:
            names = ", ".join(sorted(unknown_risk_params))
            raise ValueError(f"unknown risk barrier parameters: {names}")
        self.risk_barrier_params = dict(RISK_BARRIER_DEFAULTS)
        self.risk_barrier_params.update(risk_barrier_params or {})
        for key in ("temperature_scale", "horizon_s", "jacket_horizon_s",
                    "barrier_weight", "recovery_weight", "trip_cost", "recovery_clip"):
            self.risk_barrier_params[key] = float(self.risk_barrier_params[key])
        if self.risk_barrier_params["temperature_scale"] <= 0.0:
            raise ValueError("risk barrier temperature_scale must be positive")
        if any(self.risk_barrier_params[key] < 0.0 for key in (
                "horizon_s", "jacket_horizon_s", "barrier_weight",
                "recovery_weight", "trip_cost", "recovery_clip")):
            raise ValueError("risk barrier weights, horizons, and clip must be non-negative")
        self.custom_reward = custom_reward         # optional callable(env, levels, temps, act) -> float
        self.terminate_on_runaway = terminate_on_runaway
        # legacy economic-mode weights (CSTR)
        self.w_prod, self.w_energy, self.w_constraint = w_prod, w_energy, w_constraint

        self.model = make_model(scenario)
        self._p_nominal = {k: (list(v) if isinstance(v, list) else v) for k, v in self.model.p.items()}
        self._regime = PLANT_REGIME.get(scenario, {})
        self._econ = ECON[scenario]
        self.integ = Integrator(self.model)
        self.scorer = KPIScorer(self.model)
        nP, nV, nH = self.model.actuator_counts()
        self.nP, self.nV, self.nH = nP, nV, nH
        self.nu = nP + nV + nH
        hsp, tsp = self.model.default_setpoints()
        self._hsp0 = [hsp.get(i, 0.0) for i in range(self.model.n)]
        self._tsp0 = list(tsp)
        self._tcold0 = float(self.model.p.get("t_cold", 15.0))
        self._tamb0 = float(self.model.p.get("t_amb", 20.0))
        if self.realism_profile in ("industrial_p3", "industrial_p4"):
            nominal_hydraulics = {
                "cooling_hydraulics_enabled": True,
                "valve_rangeability": self.p3_params["valve_rangeability"],
                "valve_pressure_drop_fraction": self.p3_params["valve_pressure_drop_fraction"],
                "reference_cooling_flow_fraction": self.p3_params["reference_cooling_flow_fraction"],
                "ua_flow_exponent": self.p3_params["ua_flow_exponent"],
                "ua_flow_factor_min": self.p3_params["ua_flow_factor_min"],
                "ua_flow_factor_max": self.p3_params["ua_flow_factor_max"],
                "clean_ua_factor": 1.0,
                "fouling_resistance_k_per_w": 0.0,
                "pump_head_factor": 1.0,
            }
            if self.realism_profile == "industrial_p4":
                nominal_hydraulics.update(
                    pump_curve_enabled=True,
                    pump_curve_droop=self.p3_params["pump_curve_droop"],
                    system_static_head_fraction=self.p3_params["system_static_head_fraction"],
                    cooling_pump_design_head_m=self.p3_params["cooling_pump_design_head_m"],
                    cooling_pump_design_efficiency=self.p3_params["cooling_pump_design_efficiency"],
                    cooling_pump_best_efficiency_flow_fraction=self.p3_params["cooling_pump_best_efficiency_flow_fraction"],
                    cooling_pump_min_efficiency_fraction=self.p3_params["cooling_pump_min_efficiency_fraction"],
                    cooling_pump_efficiency_curve_coefficient=self.p3_params["cooling_pump_efficiency_curve_coefficient"],
                )
            self.model.pid_temp_bias = [self.model.cooling_opening_for_flow(
                self.p3_params["reference_cooling_flow_fraction"], nominal_hydraulics
            )]

        # integral-of-error obs (the I-term a memoryless policy otherwise lacks): lets
        # the RL policy do offset-free tracking like PID + adapt under 工况 drift.
        self.integral_obs = integral_obs
        self.nctrl = len(self.model.controlled_levels())
        if self.observation_profile == "field_no_concentration":
            obs_dim = FIELD_CSTR_OBS_DIM
        else:
            obs_dim = 3 * self.model.n + self.nctrl + 2 + int(getattr(self.model, "extra_obs_dim", 0))
        if integral_obs:
            obs_dim += self.model.n + self.nctrl       # ∫temp-err (all) + ∫level-err (controlled)
        if self.risk_observation:
            obs_dim += 5
        # supervisory (RL-on-PID): action = setpoints, an inner PID does the regulation.
        self.action_mode = action_mode
        self.layout = ([tuple(spec) for spec in SUPERVISORY.get(scenario, [])]
                       if action_mode == "setpoint" else None)
        self.supervisory_t_sp_high = None
        if self.layout is not None and supervisory_t_sp_high is not None:
            requested_high = float(supervisory_t_sp_high)
            replaced = False
            updated_layout = []
            for spec in self.layout:
                if not replaced and spec[0] == "t_sp":
                    if requested_high <= float(spec[-2]):
                        raise ValueError("supervisory_t_sp_high must exceed the temperature lower bound")
                    spec = (*spec[:-1], requested_high)
                    replaced = True
                updated_layout.append(spec)
            if not replaced:
                raise ValueError("supervisory_t_sp_high requires a supervisory temperature action")
            self.layout = updated_layout
        if self.layout is not None:
            temperature_specs = [spec for spec in self.layout if spec[0] == "t_sp"]
            if temperature_specs:
                self.supervisory_t_sp_high = float(temperature_specs[0][-1])
        if self.layout is not None:
            from .baselines import PIDAgent
            self.pid = PIDAgent(self.model)
            act_dim = len(self.layout)
        else:
            self.pid = None
            act_dim = self.nu
        self.action_space = spaces.Box(0.0, 1.0, (act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._k = 0

    # ---- helpers ----
    def _env(self):
        out = {"t_cold": self.t_cold, "t_amb": self.t_amb, "extra_outflow": self.extra_outflow}
        if self.scenario == "cstr_jacketed":
            out.update(t_jacket_in=self.t_jacket_in, cooling_flow_factor=self.cooling_flow_factor,
                       ua_factor=self.ua_factor,
                       feed_concentration_factor=self.feed_concentration_factor,
                       kinetics_factor=self.kinetics_factor,
                       reaction_heat_factor=self.reaction_heat_factor)
            if self.realism_profile in ("industrial_p3", "industrial_p4"):
                out.update(
                    cooling_hydraulics_enabled=True,
                    valve_rangeability=self.p3_params["valve_rangeability"],
                    valve_pressure_drop_fraction=self.p3_params["valve_pressure_drop_fraction"],
                    reference_cooling_flow_fraction=self.p3_params["reference_cooling_flow_fraction"],
                    ua_flow_exponent=self.p3_params["ua_flow_exponent"],
                    ua_flow_factor_min=self.p3_params["ua_flow_factor_min"],
                    ua_flow_factor_max=self.p3_params["ua_flow_factor_max"],
                    clean_ua_factor=self.clean_ua_factor,
                    fouling_resistance_k_per_w=self.fouling_resistance_k_per_w,
                    pump_head_factor=self.pump_head_factor,
                )
                if self.realism_profile == "industrial_p4":
                    out.update(
                        pump_curve_enabled=True,
                        pump_curve_droop=self.p3_params["pump_curve_droop"],
                        system_static_head_fraction=self.p3_params["system_static_head_fraction"],
                        cooling_pump_design_head_m=self.p3_params["cooling_pump_design_head_m"],
                        cooling_pump_design_efficiency=self.p3_params["cooling_pump_design_efficiency"],
                        cooling_pump_best_efficiency_flow_fraction=self.p3_params["cooling_pump_best_efficiency_flow_fraction"],
                        cooling_pump_min_efficiency_fraction=self.p3_params["cooling_pump_min_efficiency_fraction"],
                        cooling_pump_efficiency_curve_coefficient=self.p3_params["cooling_pump_efficiency_curve_coefficient"],
                    )
        return out

    def _split(self, action):
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        return {"pumps": list(a[:self.nP]),
                "valves": list(a[self.nP:self.nP + self.nV]),
                "heaters": list(a[self.nP + self.nV:])}

    def _true_measurement(self, act=None):
        levels, temps = self.model.levels_temps(self.integ.x)
        out = {"levels": levels, "temps": temps, "t_cold": self.t_cold, "t_amb": self.t_amb}
        if hasattr(self.model, "conc"):
            out["conc"] = self.model.conc(self.integ.x)
        if hasattr(self.model, "measurement_extras"):
            measurement_act = act if self.observation_profile == "field_no_concentration" else None
            out.update(self.model.measurement_extras(self.integ.x, measurement_act, self._env()))
        if self.observation_profile == "field_no_concentration":
            out.pop("conc", None)
            out.pop("product_conc", None)
        return out

    def _field_cstr_observation(self, measured):
        """Build the concentration-free policy input from ordinary field signals."""
        p = self.model.p
        pump_flow = float(measured.get("pump_flow", [0.0])[0]) / max(float(p["Fmax"]), 1e-12)
        cooling_flow = float(measured.get("cooling_flow", [0.0])[0]) / max(float(p["Fjmax"]), 1e-12)
        coolant_supply = measured.get("coolant_supply_temp", [self.t_jacket_in])
        return [
            float(measured["temps"][0]),
            float(measured["jacket_temp"][0]),
            float(self.t_sp[0]),
            float(measured["t_cold"]),
            float(coolant_supply[0]),
            float(np.clip(pump_flow, 0.0, 1.5)),
            float(np.clip(cooling_flow, 0.0, 1.5)),
            float(self._p2_filtered_temp_rate),
            float(self._p2_filtered_jacket_rate),
        ]

    def _obs(self):
        if self.realism is not None:
            measured = self._measurement_cache
            levels, temps = measured["levels"], measured["temps"]
            t_cold, t_amb = measured["t_cold"], measured["t_amb"]
        else:
            levels, temps = self.model.levels_temps(self.integ.x)
            t_cold, t_amb = self.t_cold, self.t_amb
        if self.noise and self.realism is None:    # P2 already carries calibrated per-channel noise
            rng = self.np_random
            levels = [l + float(rng.normal(0, self.noise_pct * 0.5)) for l in levels]
            temps = [t + float(rng.normal(0, self.noise_pct * 10.0)) for t in temps]
        if self.observation_profile == "field_no_concentration":
            o = self._field_cstr_observation(measured)
        else:
            o = obs_vector(self.model, levels, temps, t_cold, t_amb, self.h_sp, self.t_sp)
            if self.realism is not None:
                o += list(measured.get("conc", [])) + list(measured.get("product_conc", [])) + list(measured.get("jacket_temp", []))
            elif hasattr(self.model, "extra_observation"):
                o += list(self.model.extra_observation(self.integ.x))
        if self.integral_obs:
            o = o + [it / I_TEMP_MAX for it in self._itemp] + [il / I_LEVEL_MAX for il in self._ilevel]
        if self.risk_observation:
            predicted = float(temps[0]) + 10.0 * max(0.0, self._p2_filtered_temp_rate)
            o += [self._p2_filtered_temp_rate / 2.0,
                  self._p2_filtered_jacket_rate / 2.0,
                  (predicted - 80.0) / 20.0,
                  float(self._last_act["pumps"][0]),
                  float(self._last_act["heaters"][0])]
        return np.asarray(o, dtype=np.float32)

    def _accumulate_integral(self):
        levels, temps = self.model.levels_temps(self.integ.x)
        ctrl = self.model.controlled_levels()
        dt = self.control_dt
        self._itemp = [float(np.clip(self._itemp[i] + (self.t_sp[i] - temps[i]) * dt, -I_TEMP_MAX, I_TEMP_MAX))
                       for i in range(self.model.n)]
        self._ilevel = [float(np.clip(self._ilevel[j] + (self.h_sp[i] - levels[i]) * dt, -I_LEVEL_MAX, I_LEVEL_MAX))
                        for j, i in enumerate(ctrl)]

    # ---- operating-regime (工况) variation ----
    def _restore_nominal(self):
        for k, v in self._p_nominal.items():
            self.model.p[k] = (list(v) if isinstance(v, list) else v)

    def _apply_regime(self):
        """Scale plant params by per-episode multipliers (fouling / ageing / gain
        drift). Stored separately so the regime targets can be re-sampled for drift."""
        rng = self.np_random
        if self.realism_profile in ("industrial_p2", "industrial_p3", "industrial_p4") and self.scenario == "cstr_jacketed":
            names = ("normal", "reactive_feed", "fouled", "utility_limited")
            self._p2_regime_name = str(rng.choice(names, p=(0.60, 0.20, 0.15, 0.05)))
            if self.realism_profile in ("industrial_p3", "industrial_p4"):
                ranges = {
                    "normal": {"UA": (0.95, 1.05), "k0": (0.95, 1.05), "deltaH": (0.98, 1.02), "Fjmax": (0.95, 1.05)},
                    "reactive_feed": {"UA": (0.95, 1.05), "k0": (1.08, 1.25), "deltaH": (1.02, 1.08), "Fjmax": (0.95, 1.05)},
                    "fouled": {"UA": (0.95, 1.05), "k0": (0.96, 1.06), "deltaH": (0.98, 1.03), "Fjmax": (0.95, 1.05)},
                    "utility_limited": {"UA": (0.95, 1.05), "k0": (0.96, 1.06), "deltaH": (0.98, 1.03), "Fjmax": (0.95, 1.05)},
                }[self._p2_regime_name]
                self._regime_mult = {key: float(rng.uniform(*bounds)) for key, bounds in ranges.items()}
                self._apply_mult(self._regime_mult)
                cfg = self.p3_params
                if self._p2_regime_name == "fouled":
                    ua_factor = float(rng.uniform(cfg["fouled_ua_factor_min"], cfg["fouled_ua_factor_max"]))
                else:
                    ua_factor = float(rng.uniform(cfg["normal_fouling_ua_factor_min"],
                                                  cfg["normal_fouling_ua_factor_max"]))
                self._p3_initial_fouling_resistance = max(
                    0.0, (1.0 / ua_factor - 1.0) / self.model.p["UA"]
                )
                if self._p2_regime_name == "utility_limited":
                    self._p3_initial_pump_head = float(rng.uniform(
                        cfg["pump_head_limited_min"], cfg["pump_head_limited_max"]
                    ))
                else:
                    self._p3_initial_pump_head = float(rng.uniform(
                        cfg["pump_head_normal_min"], cfg["pump_head_normal_max"]
                    ))
                return
            ranges = {
                "normal": {"UA": (0.90, 1.10), "k0": (0.95, 1.05), "deltaH": (0.98, 1.02), "Fjmax": (0.90, 1.10)},
                "reactive_feed": {"UA": (0.92, 1.02), "k0": (1.08, 1.25), "deltaH": (1.02, 1.08), "Fjmax": (0.90, 1.05)},
                "fouled": {"UA": (0.70, 0.90), "k0": (0.96, 1.06), "deltaH": (0.98, 1.03), "Fjmax": (0.85, 1.00)},
                "utility_limited": {"UA": (0.90, 1.00), "k0": (0.96, 1.06), "deltaH": (0.98, 1.03), "Fjmax": (0.70, 0.90)},
            }[self._p2_regime_name]
            self._regime_mult = {key: float(rng.uniform(*bounds)) for key, bounds in ranges.items()}
            self._apply_mult(self._regime_mult)
            return
        self._regime_mult = {}
        for k, (lo, hi) in self._regime.items():
            if k in self._p_nominal:
                self._regime_mult[k] = float(rng.uniform(lo, hi))
        self._apply_mult(self._regime_mult)

    def _apply_mult(self, mult):
        for k, m in mult.items():
            nom = self._p_nominal[k]
            self.model.p[k] = [x * m for x in nom] if isinstance(nom, list) else nom * m

    # ---- disturbance scheduler (the "adaptation" dimension) ----
    def _schedule_disturbances(self):
        self._dist_events = []
        self._p2_micro_events = []
        if not self.dynamic:
            return
        rng = self.np_random
        if self.realism_profile in ("industrial_p2", "industrial_p3", "industrial_p4"):
            regime = getattr(self, "_p2_regime_name", "normal")
            if self.realism_profile in ("industrial_p3", "industrial_p4"):
                allowed = {
                    "normal": ("utility_stress", "feed_batch", "pump_degradation"),
                    "reactive_feed": ("utility_stress", "feed_batch", "pump_degradation"),
                    "fouled": ("utility_stress", "feed_batch", "pump_degradation"),
                    "utility_limited": ("feed_batch", "pump_degradation"),
                }[regime]
            else:
                allowed = {
                    "normal": ("utility_stress", "fouling_ramp", "feed_batch"),
                    "reactive_feed": ("utility_stress", "feed_batch"),
                    "fouled": ("utility_stress", "feed_batch"),
                    "utility_limited": ("fouling_ramp", "feed_batch"),
                }[regime]
            count = min(len(allowed), int(rng.integers(1, 3)))
            for kind in rng.choice(allowed, size=count, replace=False):
                start = int(rng.integers(int(0.15 * self.episode_steps), int(0.55 * self.episode_steps)))
                duration = int(rng.integers(max(10, int(0.15 * self.episode_steps)),
                                           max(11, int(0.35 * self.episode_steps))))
                severity = float(rng.uniform(0.35, 1.0))
                direction = float(rng.choice((-1.0, 1.0)))
                self._dist_events.append({"kind": str(kind), "start": start,
                                          "duration": duration, "severity": severity,
                                          "direction": direction})
            # Recurring, recoverable perturbations keep the whole episode alive.
            # They are intentionally mild and separate from the one-way fault ramps.
            cursor = int(rng.uniform(8.0, 20.0) / self.control_dt)
            while cursor < self.episode_steps:
                kind = str(rng.choice(("feed_temp_pulse", "coolant_temp_pulse", "cooling_flow_dip"),
                                      p=(0.45, 0.35, 0.20)))
                duration = max(2, int(rng.uniform(12.0, 30.0) / self.control_dt))
                amplitude = ({"feed_temp_pulse": rng.uniform(0.5, 1.5),
                              "coolant_temp_pulse": rng.uniform(0.4, 1.2),
                              "cooling_flow_dip": rng.uniform(0.02, 0.06)}[kind])
                direction = 1.0 if kind == "cooling_flow_dip" else float(rng.choice((-1.0, 1.0)))
                self._p2_micro_events.append({"kind": kind, "start": cursor,
                                              "duration": duration, "amplitude": float(amplitude),
                                              "direction": direction})
                cursor += duration + max(1, int(rng.uniform(8.0, 25.0) / self.control_dt))
            return
        for _ in range(int(rng.integers(1, 4))):
            t = int(rng.integers(int(0.15 * self.episode_steps), max(2, self.episode_steps)))
            n_kinds = 7 if self.scenario == "cstr_jacketed" else 4
            self._dist_events.append((t, int(rng.integers(0, n_kinds))))

    def _apply_disturbance(self, kind):
        rng = self.np_random
        if kind == 0:                                                  # cold-inlet step
            self.t_cold = float(np.clip(self._tcold0 + rng.uniform(-8, 8), 2, 35))
        elif kind == 1:                                                # ambient drift
            self.t_amb = float(np.clip(self._tamb0 + rng.uniform(-8, 12), 0, 40))
        elif kind == 2:                                                # demand surge
            self.extra_outflow = float(abs(rng.uniform(0, 8e-4)))
        elif kind == 3:                                                # setpoint move
            for i in self.model.controlled_levels():
                self.h_sp[i] = float(np.clip(self.h_sp[i] * (1 + 0.15 * rng.uniform(-1, 1)), 0.15, 0.70))
            self.t_sp = [float(np.clip(t * (1 + 0.10 * rng.uniform(-1, 1)), 15.0, 85.0)) for t in self.t_sp]
        elif kind == 4:                                                # cooling-water supply warms
            self.t_jacket_in = float(self.model.p["Tj_in"] + rng.uniform(2.0, 12.0))
        elif kind == 5:                                                # coolant circulation loss
            self.cooling_flow_factor = float(rng.uniform(0.35, 0.85))
        elif kind == 6:                                                # heat-transfer fouling
            self.ua_factor = float(rng.uniform(0.45, 0.90))

    @staticmethod
    def _smooth_progress(step, start, duration):
        z = float(np.clip((step - start) / max(duration, 1), 0.0, 1.0))
        return z * z * (3.0 - 2.0 * z)

    @classmethod
    def _pulse_progress(cls, step, start, duration):
        """Smooth 0 -> 1 -> 0 pulse for recoverable operating disturbances."""
        z = float(np.clip((step - start) / max(duration, 1), 0.0, 1.0))
        return cls._smooth_progress(2.0 * z, 0.0, 1.0) if z <= 0.5 else cls._smooth_progress(2.0 * (1.0 - z), 0.0, 1.0)

    def _update_p2_process(self):
        """Advance correlated, time-continuous P2 disturbances.

        The latent utility state simultaneously warms cooling water and reduces
        available circulation. Fault events ramp instead of changing parameters
        instantaneously. Normal fluctuations use seeded Ornstein-Uhlenbeck-like
        states, so consecutive samples remain correlated.
        """
        dt, rng = self.control_dt, self.np_random
        alpha_feed = float(np.exp(-dt / 60.0))
        alpha_utility = float(np.exp(-dt / 90.0))
        self._p2_feed_ou = alpha_feed * self._p2_feed_ou + float(
            rng.normal(0.0, 0.35 * np.sqrt(1.0 - alpha_feed ** 2)))
        self._p2_utility_ou = alpha_utility * self._p2_utility_ou + float(
            rng.normal(0.0, 0.25 * np.sqrt(1.0 - alpha_utility ** 2)))

        hot_water = flow_loss = fouling = feed_shift = pump_degradation = 0.0
        feed_composition = feed_activity = feed_heat = 0.0
        for event in self._dist_events:
            progress = self._smooth_progress(self._k, event["start"], event["duration"])
            severity = event["severity"] * progress
            if event["kind"] == "utility_stress":
                hot_water = max(hot_water, 10.0 * severity)
                flow_loss = max(flow_loss, 0.40 * severity)
            elif event["kind"] == "fouling_ramp":
                fouling = max(fouling, 0.45 * severity)
            elif event["kind"] == "feed_batch":
                feed_shift += event["direction"] * 5.0 * severity
                feed_composition += event["direction"] * 0.06 * severity
                feed_activity += event["direction"] * 0.04 * severity
                feed_heat += event["direction"] * 0.02 * severity
            elif event["kind"] == "pump_degradation":
                pump_degradation = max(
                    pump_degradation, self.p3_params["pump_degradation_max"] * severity
                )

        micro_feed = micro_jacket = micro_flow_loss = 0.0
        for event in getattr(self, "_p2_micro_events", []):
            pulse = self._pulse_progress(self._k, event["start"], event["duration"])
            value = event["amplitude"] * pulse
            if event["kind"] == "feed_temp_pulse":
                micro_feed += event["direction"] * value
            elif event["kind"] == "coolant_temp_pulse":
                micro_jacket += event["direction"] * value
            elif event["kind"] == "cooling_flow_dip":
                micro_flow_loss += value

        self.t_cold = float(np.clip(
            self._p2_base_t_cold + feed_shift + micro_feed + self._p2_feed_ou, 2.0, 35.0
        ))
        utility_temperature_target = float(np.clip(
            self._p2_base_t_jacket + hot_water + micro_jacket + self._p2_utility_ou,
            2.0, 35.0,
        ))
        if self.realism_profile in ("industrial_p3", "industrial_p4"):
            utility_head_loss = flow_loss + micro_flow_loss + 0.01 * max(0.0, self._p2_utility_ou)
            pump_head_target = float(np.clip(
                self._p3_initial_pump_head * (1.0 - pump_degradation) * (1.0 - utility_head_loss),
                0.15, 1.20,
            ))
            self.pump_head_target_factor = pump_head_target
            if self.realism_profile == "industrial_p4":
                head_tau = self.p3_params["pump_head_response_tau_s"]
                head_alpha = 1.0 - math.exp(-dt / head_tau)
                self.pump_head_factor += head_alpha * (pump_head_target - self.pump_head_factor)

                self.coolant_supply_target_temperature = float(np.clip(
                    utility_temperature_target,
                    self.p3_params["cooling_supply_temperature_min_c"],
                    self.p3_params["cooling_supply_temperature_max_c"],
                ))
                previous_act = getattr(self, "_last_act", {"heaters": [0.0]})
                previous_opening = previous_act["heaters"][0] if previous_act.get("heaters") else 0.0
                coolant_flow, _, _ = self.model.cooling_characteristics(previous_opening, self._env())
                jacket_return = float(self.integ.x[3])
                supply = self.coolant_supply_temperature
                recovery = (
                    self.coolant_supply_target_temperature - supply
                ) / self.p3_params["cooling_supply_recovery_tau_s"]
                recirculation = (
                    self.p3_params["cooling_return_recirc_fraction"]
                    * coolant_flow / self.p3_params["cooling_header_volume_m3"]
                    * (jacket_return - supply)
                )
                supply += dt * (recovery + recirculation)
                self.coolant_supply_temperature = float(np.clip(
                    supply,
                    self.p3_params["cooling_supply_temperature_min_c"],
                    self.p3_params["cooling_supply_temperature_max_c"],
                ))
                self.coolant_return_temperature = jacket_return
                self.t_jacket_in = self.coolant_supply_temperature
            else:
                self.pump_head_factor = pump_head_target
                self.t_jacket_in = utility_temperature_target
            self.cooling_flow_factor = math.sqrt(self.pump_head_factor)
            elapsed = self._k * dt
            self.fouling_resistance_k_per_w = (
                self._p3_initial_fouling_resistance
                + self.p3_params["fouling_resistance_rate_k_per_w_s"] * elapsed
            )
            self.ua_factor = 1.0 / (
                1.0 + self.model.p["UA"] * self.fouling_resistance_k_per_w
            )
        else:
            self.t_jacket_in = utility_temperature_target
            self.cooling_flow_factor = float(np.clip(1.0 - flow_loss - micro_flow_loss
                                                     - 0.01 * max(0.0, self._p2_utility_ou), 0.2, 1.0))
            self.ua_factor = float(np.clip(1.0 - fouling, 0.35, 1.0))
        self.feed_concentration_factor = float(np.clip(1.0 + feed_composition, 0.90, 1.10))
        self.kinetics_factor = float(np.clip(1.0 + feed_activity, 0.92, 1.08))
        self.reaction_heat_factor = float(np.clip(1.0 + feed_heat, 0.96, 1.04))

    def _reward_done(self, act):
        levels, temps = self.model.levels_temps(self.integ.x)
        hmax = self.model.height_max
        ctrl = self.model.controlled_levels()
        # tracking error, normalised (level scale 0.1 m, temp scale 10 °C) — for info/legacy
        track = sum(abs(levels[i] - self.h_sp[i]) / 0.1 for i in ctrl)
        track += sum(abs(temps[i] - self.t_sp[i]) / 10.0 for i in range(self.model.n))
        con = 0.0
        for i, h in enumerate(levels):
            if h > H_HIGH_FRAC * hmax[i]:
                con += (h - H_HIGH_FRAC * hmax[i]) / (0.1 * hmax[i])
            elif h < H_LOW_FRAC * hmax[i]:
                con += (H_LOW_FRAC * hmax[i] - h) / (0.1 * hmax[i])
        for T in temps:
            if T > T_HIGH:
                con += (T - T_HIGH) / 10.0
        runaway = any(T > T_TRIP for T in temps) or any(levels[i] > H_OVERFLOW_FRAC * hmax[i] for i in range(len(levels)))

        # Always accumulate the KPI scorer (independent of reward_mode) so any agent
        # — PID / MPC / RL — can be ranked by env.scorer.report()["score"].
        heat_w = self.model.heater_power(act, self._env())
        ideal_w = self.model.ideal_power(levels, temps, self.t_sp, self._env(), act)
        pen = self.scorer.step_penalty(levels, temps, self.h_sp, self.t_sp,
                                       heat_w, ideal_w, runaway, self.control_dt)

        prod = 0.0
        if self.reward_mode in ("economic", "economic_guarded", "economic_risk_barrier"):
            profit, prod = self._economic_profit(act, levels, temps, runaway)
            safety_cost = 0.0
            recovery_bonus = 0.0
            if self.reward_mode == "economic_guarded" and self.scenario == "cstr_jacketed":
                temperature = float(temps[0])
                rise_rate = (temperature - self._previous_true_temps[0]) / max(self.control_dt, 1e-9)
                warning = max(0.0, temperature - 82.0)
                critical = max(0.0, temperature - 88.0)
                fast_rise = max(0.0, rise_rate - 0.05)
                safety_cost = 3.0 * warning ** 2 + 30.0 * fast_rise ** 2 + 200.0 * critical ** 2
                if temperature > T_TRIP:
                    safety_cost += 500.0
            elif self.reward_mode == "economic_risk_barrier" and self.scenario == "cstr_jacketed":
                temperature = float(temps[0])
                jacket_temperature = float(self.integ.x[3])
                rise_rate = (temperature - self._previous_true_temps[0]) / max(self.control_dt, 1e-9)
                jacket_rise_rate = (
                    jacket_temperature - self._previous_true_jacket_temperature
                ) / max(self.control_dt, 1e-9)
                cfg = self.risk_barrier_params
                predicted_temperature = (
                    temperature
                    + cfg["horizon_s"] * max(0.0, rise_rate)
                    + cfg["jacket_horizon_s"] * max(0.0, jacket_rise_rate)
                )
                normalized_risk = (
                    predicted_temperature - cfg["trigger_temp"]
                ) / cfg["temperature_scale"]
                softplus = max(normalized_risk, 0.0) + math.log1p(
                    math.exp(-abs(normalized_risk))
                )
                risk_barrier = softplus ** 2
                safety_cost = cfg["barrier_weight"] * risk_barrier
                risk_reduction = self._previous_risk_barrier - risk_barrier
                recovery_bonus = cfg["recovery_weight"] * float(np.clip(
                    risk_reduction, -cfg["recovery_clip"], cfg["recovery_clip"]
                ))
                if temperature > T_TRIP:
                    safety_cost += cfg["trip_cost"]
            reward = (profit - safety_cost + recovery_bonus) * self.reward_scale
        elif self.reward_mode == "kpi":
            reward = -pen * self.reward_scale            # -(instantaneous KPI penalty)
            profit = 0.0
        else:
            energy = sum(act["heaters"]) + 0.3 * sum(act["pumps"])
            reward = -(track + 0.03 * energy + 5.0 * con)
            profit = 0.0

        if self.custom_reward is not None:               # user-supplied reward overrides
            reward = float(self.custom_reward(self, levels, temps, act))

        terminated = bool(self.terminate_on_runaway and runaway)
        if terminated:
            reward -= 50.0
        # per-constraint violation amounts (PC-Gym-style tracking for safe-RL benchmarking)
        cons_info = {"temp_high": max((T - T_HIGH for T in temps), default=0.0),
                     "temp_trip": max((T - T_TRIP for T in temps), default=0.0),
                     "level_high": max((levels[i] - H_HIGH_FRAC * hmax[i] for i in range(len(levels))), default=0.0),
                     "level_low": max((H_LOW_FRAC * hmax[i] - levels[i] for i in range(len(levels))), default=0.0)}
        info = {"track": track, "constraint": con, "prod": prod, "profit": profit,
                "runaway": runaway, "cons_info": cons_info, "cons_violated": any(v > 0 for v in cons_info.values()),
                "levels": levels, "temps": temps}
        if self.reward_mode in ("economic_guarded", "economic_risk_barrier"):
            info["safety_cost"] = safety_cost
        if self.reward_mode == "economic_risk_barrier":
            info.update(
                predicted_temperature=predicted_temperature,
                temperature_rise_rate=rise_rate,
                jacket_temperature_rise_rate=jacket_rise_rate,
                risk_barrier=risk_barrier,
                risk_recovery_bonus=recovery_bonus,
            )
            self._previous_risk_barrier = risk_barrier
        self._previous_true_temps = list(temps)
        if self.scenario == "cstr_jacketed":
            self._previous_true_jacket_temperature = float(self.integ.x[3])
        return float(reward), terminated, info

    def _economic_profit(self, act, levels, temps, runaway):
        """Economic objective: value − energy-cost − soft-band-violation. The optimum
        hugs a band/constraint edge that drifts with 工况, so fixed-SP control is
        suboptimal. Returns (profit, production)."""
        cfg = self._econ
        ctrl = self.model.controlled_levels()
        value = prod = 0.0
        if cfg["value"] == "production" and hasattr(self.model, "production"):
            prod = self.model.production(self.integ.x, act)
            value = prod
        energy_kw = self.model.heater_power(act, self._env()) / 1000.0
        viol = 0.0
        for i, (lo, hi) in enumerate(cfg["temp_band"]):
            if lo is not None and temps[i] < lo:
                viol += (lo - temps[i]) / 10.0
            if hi is not None and temps[i] > hi:
                viol += (temps[i] - hi) / 10.0
        l_scale = cfg.get("level_scale", 0.1)
        for j, i in enumerate(ctrl):
            lo, hi = cfg["level_band"][j]
            if lo is not None and levels[i] < lo:
                viol += (lo - levels[i]) / l_scale
            if hi is not None and levels[i] > hi:
                viol += (levels[i] - hi) / l_scale
        profit = cfg["w_value"] * value - cfg["w_energy"] * energy_kw - cfg["w_viol"] * viol
        if runaway:
            profit -= 50.0
        return profit, prod

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        sensor_seed = None if seed is None else (int(seed) ^ 0xA10C57)
        self._sensor_rng = np.random.default_rng(sensor_seed)
        self._restore_nominal()
        if self.randomize_plant:
            self._apply_regime()        # this episode's 工况 (operating regime)
        x0 = list(self.model.initial_state())
        self.h_sp = list(self._hsp0)
        self.t_sp = list(self._tsp0)
        self.t_cold = float(self._tcold0)
        self.t_amb = float(self._tamb0)
        self.extra_outflow = 0.0
        self.t_jacket_in = float(self.model.p.get("Tj_in", 10.0))
        self.cooling_flow_factor = 1.0
        self.ua_factor = 1.0
        self.feed_concentration_factor = 1.0
        self.kinetics_factor = 1.0
        self.reaction_heat_factor = 1.0
        self.clean_ua_factor = 1.0
        self.fouling_resistance_k_per_w = 0.0
        self.pump_head_factor = 1.0
        self.pump_head_target_factor = 1.0
        self.coolant_supply_target_temperature = self.t_jacket_in
        self.coolant_supply_temperature = self.t_jacket_in
        self.coolant_return_temperature = x0[3] if self.scenario == "cstr_jacketed" else self.t_jacket_in
        self.cooling_pump_power_w = 0.0
        if self.realism_profile in ("industrial_p3", "industrial_p4"):
            self._p3_initial_fouling_resistance = getattr(
                self, "_p3_initial_fouling_resistance", 0.0
            )
            self._p3_initial_pump_head = getattr(self, "_p3_initial_pump_head", 1.0)
        if self.randomize:
            for j in range(len(x0)):
                x0[j] *= 1.0 + 0.08 * float(rng.uniform(-1, 1))
            self.t_cold = float(np.clip(self._tcold0 + rng.uniform(-5, 5), 2, 35))
            self.t_amb = float(np.clip(self._tamb0 + rng.uniform(-5, 8), 0, 40))
        if self.randomize_setpoints:
            for i in self.model.controlled_levels():
                self.h_sp[i] = float(np.clip(self.h_sp[i] * (1 + 0.15 * rng.uniform(-1, 1)), 0.15, 0.70))
            self.t_sp = [float(np.clip(t * (1 + 0.10 * rng.uniform(-1, 1)), 15.0, 85.0)) for t in self.t_sp]
        self.integ.reset(x0)
        self.scorer.reset()
        if self.pid is not None:
            self.pid.reset()
        self._itemp = [0.0] * self.model.n
        self._ilevel = [0.0] * self.nctrl
        self._previous_true_temps = list(self.model.levels_temps(x0)[1])
        self._previous_true_jacket_temperature = (
            float(x0[3]) if self.scenario == "cstr_jacketed" else 0.0
        )
        self._previous_risk_barrier = 0.0
        self._k = 0
        if self.realism is not None:
            self._p2_base_t_cold = self.t_cold
            self._p2_base_t_jacket = self.t_jacket_in
            self._p2_feed_ou = 0.0
            self._p2_utility_ou = 0.0
            initial_action = {"pumps": [0.5] * self.nP, "valves": [0.5] * self.nV,
                              "heaters": list(getattr(self.model, "pid_temp_bias", [0.0] * self.nH))}
            initial_action["heaters"] += [0.0] * (self.nH - len(initial_action["heaters"]))
            self._measurement_cache = self.realism.reset(self._true_measurement(initial_action), initial_action)
            self._last_command = initial_action
            self._last_act = initial_action
            self._sis_latched = False
            self._p2_previous_measured_temp = float(self._measurement_cache["temps"][0])
            self._p2_previous_measured_jacket = float(self._measurement_cache["jacket_temp"][0])
            self._p2_filtered_temp_rate = 0.0
            self._p2_filtered_jacket_rate = 0.0
            if self.realism_profile in ("industrial_p3", "industrial_p4"):
                self.fouling_resistance_k_per_w = self._p3_initial_fouling_resistance
                self.pump_head_factor = self._p3_initial_pump_head
                self.pump_head_target_factor = self._p3_initial_pump_head
                if self.realism_profile == "industrial_p4":
                    self.coolant_supply_target_temperature = self.t_jacket_in
                    self.coolant_supply_temperature = self.t_jacket_in
                    self.coolant_return_temperature = float(x0[3])
                nominal_env = self._env()
                nominal_env["fouling_resistance_k_per_w"] = 0.0
                nominal_env["pump_head_factor"] = 1.0
                nominal_opening = self.model.cooling_opening_for_flow(
                    self.p3_params["reference_cooling_flow_fraction"], nominal_env
                )
                self.model.pid_temp_bias = [nominal_opening]
                if self.pid is not None:
                    for _, _, loop in self.pid.temp_loops:
                        loop.bias = nominal_opening
                        loop.reset()
                self.realism.actuators["heaters"][0].position = nominal_opening
                self._last_command["heaters"][0] = nominal_opening
                self._last_act["heaters"][0] = nominal_opening
            # Reinitialize field instruments after the P2.3/P2.4 hydraulic state and
            # nominal actuator positions have been established.
            self._measurement_cache = self.realism.reset(
                self._true_measurement(self._last_act), self._last_act
            )
        self._last_supervisory_guard = {
            "mode": self.supervisory_guard,
            "active": False,
            "emergency": False,
        }
        self._supervisory_guard_previous_temperature = (
            float(self._meas()["temps"][0]) if self.supervisory_guard != "none" else None
        )
        self._schedule_disturbances()
        return self._obs(), {}

    def default_sp_action(self):
        """Normalized supervisory action that reproduces the default setpoints (= the
        fixed-SP PID baseline) — the offline prior to learn from."""
        if self.layout is None:
            return None
        a = []
        for spec in self.layout:
            lo, hi = spec[-2], spec[-1]
            if spec[0] == "t_sp":
                v = self._tsp0[spec[1]]
            elif spec[0] == "h_sp":
                v = self._hsp0[spec[1]] or 0.45
            else:
                v = lo + 0.7 * (hi - lo)
            a.append(float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)))
        return np.array(a, np.float32)

    def _meas(self):
        """buildState-like dict read by the inner PID."""
        if self.realism is not None:
            return {key: (list(value) if isinstance(value, list) else value)
                    for key, value in self._measurement_cache.items()}
        lv, tp = self.model.levels_temps(self.integ.x)
        m = {"levels": lv, "temps": tp, "t_cold": self.t_cold, "t_amb": self.t_amb}
        if hasattr(self.model, "conc"):
            m["conc"] = self.model.conc(self.integ.x)
        if hasattr(self.model, "measurement_extras"):
            m.update(self.model.measurement_extras(self.integ.x))
        return m

    def _apply_supervisory_guard(self, mv):
        """Apply the configured guard to decoded physical setpoints and MVs.

        The actor still chooses the normalized replay-buffer action. The plant sees
        this guarded command, so the critic learns the economic consequence of any
        intervention instead of meeting V4 for the first time after training.
        """
        if self.supervisory_guard == "none":
            self._last_supervisory_guard = {
                "mode": "none", "active": False, "emergency": False,
            }
            return self._last_supervisory_guard

        cfg = self.supervisory_guard_params
        measured = self._meas()
        temperature = float(measured["temps"][0])
        previous = self._supervisory_guard_previous_temperature
        heating_rate = 0.0 if previous is None else (
            temperature - float(previous)
        ) / max(self.control_dt, 1e-9)
        self._supervisory_guard_previous_temperature = temperature
        predicted_temperature = temperature + float(cfg["horizon_s"]) * max(0.0, heating_rate)

        active = bool(
            (temperature > float(cfg["trigger_temp"]) and heating_rate > 0.02)
            or temperature > float(cfg["hard_temp"])
            or predicted_temperature > float(cfg["hard_temp"])
        )
        emergency = bool(
            temperature > float(cfg["feed_stop_temp"])
            or (
                bool(cfg["predictive_emergency"])
                and predicted_temperature > float(cfg["feed_stop_temp"])
            )
        )
        pump_key = ("pumps", 0)
        requested_t_sp = float(self.t_sp[0])
        requested_pump = float(mv.get(pump_key, 0.0))
        if active:
            self.t_sp[0] = min(self.t_sp[0], float(cfg["target_temp"]))
            if pump_key in mv:
                mv[pump_key] = min(mv[pump_key], float(cfg["pump_cap"]))
        if emergency and pump_key in mv:
            mv[pump_key] = 0.0

        self._last_supervisory_guard = {
            "mode": self.supervisory_guard,
            "active": active,
            "emergency": emergency,
            "temperature": temperature,
            "heating_rate": float(heating_rate),
            "predicted_temperature": float(predicted_temperature),
            "requested_t_sp": requested_t_sp,
            "effective_t_sp": float(self.t_sp[0]),
            "requested_pump": requested_pump,
            "effective_pump": float(mv.get(pump_key, requested_pump)),
            "t_sp_intervention": max(0.0, requested_t_sp - float(self.t_sp[0])),
            "pump_intervention": max(0.0, requested_pump - float(mv.get(pump_key, requested_pump))),
        }
        return self._last_supervisory_guard

    def _supervise(self, action):
        """Supervisory action = normalized setpoints -> set SPs, inner PID regulates
        to them; unregulated economic MVs ('mv') are applied directly."""
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        mv = {}
        for i, spec in enumerate(self.layout):
            lo, hi = spec[-2], spec[-1]
            val = lo + float(a[i]) * (hi - lo)
            if spec[0] == "t_sp":
                self.t_sp[spec[1]] = val
            elif spec[0] == "h_sp":
                self.h_sp[spec[1]] = val
            else:                                   # ("mv", kind, idx, lo, hi)
                mv[(spec[1], spec[2])] = val
        guard = self._apply_supervisory_guard(mv)
        act = self.pid.compute(self._meas(), {"h_sp": self.h_sp, "t_sp": self.t_sp}, self.control_dt)
        for (kind, idx), v in mv.items():
            act[kind][idx] = v
        requested_cooling = float(act["heaters"][0]) if act["heaters"] else 0.0
        if guard["active"] and act["heaters"]:
            act["heaters"][0] = max(
                act["heaters"][0], float(self.supervisory_guard_params["cooling_floor"])
            )
        if guard["emergency"] and act["heaters"]:
            act["heaters"][0] = 1.0
        guard["requested_cooling"] = requested_cooling
        guard["effective_cooling"] = float(act["heaters"][0]) if act["heaters"] else 0.0
        guard["cooling_intervention"] = max(
            0.0, guard["effective_cooling"] - requested_cooling
        )
        return {"pumps": list(act["pumps"]), "valves": list(act["valves"]), "heaters": list(act["heaters"])}

    def step(self, action):
        command = self._supervise(action) if self.pid is not None else self._split(action)
        if self.realism is not None:
            self._update_p2_process()
            act = self.realism.actuate(command, self.control_dt)
            true_temperature = float(self.model.levels_temps(self.integ.x)[1][0])
            if true_temperature >= T_TRIP:
                self._sis_latched = True
            elif self._sis_latched and true_temperature <= 89.0:
                self._sis_latched = False
            if self._sis_latched:
                if act["pumps"]:
                    act["pumps"][0] = 0.0
                if act["heaters"]:
                    act["heaters"][0] = 1.0
        else:
            act = command
            for (t, kind) in self._dist_events:
                if t == self._k:
                    self._apply_disturbance(kind)
        self.integ.step(self.control_dt, act, self._env())
        if self.realism_profile == "industrial_p4":
            details = self.model.cooling_hydraulic_details(act["heaters"][0], self._env())
            self.cooling_pump_power_w = float(details.get("pump_power_w", 0.0))
            self.coolant_return_temperature = float(self.integ.x[3])
        self._accumulate_integral()
        self._k += 1
        if self.realism is not None:
            self._measurement_cache = self.realism.measure(
                self._true_measurement(act), self._k * self.control_dt, self.control_dt, self._sensor_rng)
            measured_temp = float(self._measurement_cache["temps"][0])
            measured_jacket = float(self._measurement_cache["jacket_temp"][0])
            raw_temp_rate = (measured_temp - self._p2_previous_measured_temp) / self.control_dt
            raw_jacket_rate = (measured_jacket - self._p2_previous_measured_jacket) / self.control_dt
            self._p2_filtered_temp_rate = 0.8 * self._p2_filtered_temp_rate + 0.2 * raw_temp_rate
            self._p2_filtered_jacket_rate = 0.8 * self._p2_filtered_jacket_rate + 0.2 * raw_jacket_rate
            self._p2_previous_measured_temp = measured_temp
            self._p2_previous_measured_jacket = measured_jacket
            self._last_command = command
            self._last_act = act
        reward, terminated, info = self._reward_done(act)
        if self.supervisory_guard != "none":
            info["supervisory_guard"] = dict(self._last_supervisory_guard)
        if self.realism is not None:
            info.update(commanded_action=command, actual_action=act,
                        realism={"profile": self.realism_profile, "ua_factor": self.ua_factor,
                                 "cooling_flow_factor": self.cooling_flow_factor,
                                 "t_jacket_in": self.t_jacket_in,
                                 "feed_concentration_factor": self.feed_concentration_factor,
                                 "kinetics_factor": self.kinetics_factor,
                                 "reaction_heat_factor": self.reaction_heat_factor,
                                 "fouling_resistance_k_per_w": self.fouling_resistance_k_per_w,
                                 "pump_head_factor": self.pump_head_factor,
                                 "pump_head_target_factor": self.pump_head_target_factor,
                                 "coolant_supply_target_temperature": self.coolant_supply_target_temperature,
                                 "coolant_return_temperature": self.coolant_return_temperature,
                                 "cooling_pump_power_w": self.cooling_pump_power_w,
                                 "p3_parameters": dict(self.p3_params),
                                 "cooling_parameters": dict(self.p3_params),
                                 "operating_regime": getattr(self, "_p2_regime_name", "nominal"),
                                 "sis_latched": self._sis_latched,
                                 "active_micro_disturbances": [event["kind"] for event in self._p2_micro_events
                                                               if self._pulse_progress(self._k, event["start"], event["duration"]) > 0.0]})
        truncated = self._k >= self.episode_steps
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        pass
