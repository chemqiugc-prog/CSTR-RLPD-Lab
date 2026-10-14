"""CSTR plant models shared by training and evaluation.

Only two plants are public in this repository:

* ``legacy`` preserves the upstream two-state CSTR benchmark.
* ``p24`` is the local four-state jacketed CSTR with P2.4 cooling hydraulics.
"""
from __future__ import annotations

import math


def _maxv(a, b):
    return a if a > b else b


class LegacyCSTR:
    scenario = "cstr"
    model_version = "cstr_v1_legacy"
    n = 1
    dt_micro = 0.01

    def __init__(self):
        self.p = dict(
            Dmax=0.02, Caf=1.0, k0=1e8, EaR=7000.0, Hr=120.0,
            Uc=0.05, Tcool=10.0, cool_max=80000.0,
            feed_power_max=1200.0, t_cold=20.0, t_amb=20.0,
            h_floor=1e-3,
        )

    def actuator_counts(self):
        return (1, 0, 1)

    @property
    def height_max(self):
        return [1.0]

    def _rate(self, ca, temperature):
        p = self.p
        return p["k0"] * math.exp(-p["EaR"] / (temperature + 273.15)) * _maxv(ca, 0.0)

    def derivatives(self, x, act, env):
        p = self.p
        ca, temperature = x
        dilution = act["pumps"][0] * p["Dmax"]
        cooling = act["heaters"][0]
        rate = self._rate(ca, temperature)
        return [
            dilution * (p["Caf"] - ca) - rate,
            dilution * (env["t_cold"] - temperature)
            + p["Hr"] * rate
            - p["Uc"] * cooling * (temperature - p["Tcool"]),
        ]

    def levels_temps(self, x):
        return [], [x[1]]

    def conc(self, x):
        return [_maxv(x[0], 0.0)]

    def initial_state(self):
        return [0.5, 50.0]

    def clamp_state(self, x):
        x[0] = max(0.0, x[0])
        x[1] = min(200.0, x[1])
        return x

    def controlled_levels(self):
        return []

    def default_setpoints(self):
        return {}, [60.0]

    energy_scored = False

    def heater_power(self, act, env=None):
        return act["heaters"][0] * self.p["cool_max"]

    def pump_power(self, act):
        return act["pumps"][0] * self.p["feed_power_max"]

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0

    def production(self, x, act):
        ca = _maxv(x[0], 0.0)
        dilution = act["pumps"][0] * self.p["Dmax"]
        return dilution * (self.p["Caf"] - ca)


class JacketedCSTR:
    """Four-state CSTR used by the local P2.4 research environment."""

    scenario = "cstr_jacketed"
    model_version = "cstr_v2_jacketed"
    n = 1
    dt_micro = 0.01
    extra_obs_dim = 3
    pid_temp_bias = [0.26422141637233987]

    def __init__(self):
        rho, cp, volume = 1000.0, 4186.0, 1.0
        self.p = dict(
            V=volume, Vj=0.20, rho=rho, Cp=cp, rhoj=rho, Cpj=cp,
            Fmax=0.020, Fjmax=0.100, Caf=1.0,
            k0=1e8, EaR=7000.0, deltaH=120.0 * rho * cp / 1000.0,
            UA=110000.0, Tj_in=10.0,
            cool_max=80000.0, feed_power_max=1200.0,
            t_cold=20.0, t_amb=20.0, h_floor=1e-3,
        )

    def actuator_counts(self):
        return (1, 0, 1)

    @property
    def height_max(self):
        return [1.0]

    def _rate(self, ca, temperature):
        p = self.p
        return p["k0"] * math.exp(-p["EaR"] / (temperature + 273.15)) * _maxv(ca, 0.0)

    def cooling_characteristics(self, opening, env):
        """Return coolant flow, effective UA, and normalized flow."""
        p = self.p
        opening = min(1.0, max(0.0, float(opening)))
        if env.get("cooling_hydraulics_enabled", False):
            cache_key = (
                opening, p["Fjmax"], p["UA"], True,
                env["valve_rangeability"], env["valve_pressure_drop_fraction"],
                env["reference_cooling_flow_fraction"], env["ua_flow_exponent"],
                env["ua_flow_factor_min"], env["ua_flow_factor_max"],
                env.get("clean_ua_factor", 1.0),
                env.get("fouling_resistance_k_per_w", 0.0),
                env.get("pump_head_factor", 1.0),
                bool(env.get("pump_curve_enabled", False)),
                env.get("pump_curve_droop", 0.0),
                env.get("system_static_head_fraction", 0.0),
                env.get("cooling_pump_design_head_m", 0.0),
                env.get("cooling_pump_design_efficiency", 1.0),
                env.get("cooling_pump_best_efficiency_flow_fraction", 1.0),
                env.get("cooling_pump_min_efficiency_fraction", 1.0),
                env.get("cooling_pump_efficiency_curve_coefficient", 0.0),
            )
        else:
            cache_key = (
                opening, p["Fjmax"], p["UA"], False,
                env.get("cooling_flow_factor", 1.0), env.get("ua_factor", 1.0),
            )
        if getattr(self, "_cooling_cache_key", None) == cache_key:
            return self._cooling_cache_value

        if not env.get("cooling_hydraulics_enabled", False):
            flow_fraction = opening * max(0.0, env.get("cooling_flow_factor", 1.0))
            ua = p["UA"] * max(0.0, env.get("ua_factor", 1.0))
            result = (flow_fraction * p["Fjmax"], ua, flow_fraction)
            self._cooling_cache_details = {
                "valve_capacity_fraction": opening,
                "installed_flow_fraction": opening,
                "operating_head_factor": 0.0,
                "pump_efficiency": 0.0,
                "pump_power_w": 0.0,
            }
            self._cooling_cache_key, self._cooling_cache_value = cache_key, result
            return result

        rangeability = max(1.000001, float(env["valve_rangeability"]))
        valve_fraction = (rangeability ** opening - 1.0) / (rangeability - 1.0)
        pressure_fraction = min(1.0, max(1e-6, float(env["valve_pressure_drop_fraction"])))
        installed_fraction = valve_fraction / math.sqrt(
            pressure_fraction + (1.0 - pressure_fraction) * valve_fraction ** 2
        )
        head_factor = max(0.0, float(env.get("pump_head_factor", 1.0)))
        if env.get("pump_curve_enabled", False) and installed_fraction > 1e-12:
            droop = max(0.0, float(env.get("pump_curve_droop", 0.0)))
            static_head = min(0.999999, max(0.0, float(env.get("system_static_head_fraction", 0.0))))
            numerator = head_factor * (1.0 + droop) - static_head
            denominator = (1.0 - static_head) / installed_fraction ** 2 + droop
            flow_fraction = math.sqrt(max(0.0, numerator) / max(denominator, 1e-12))
            operating_head_factor = static_head + (1.0 - static_head) * (
                flow_fraction / installed_fraction
            ) ** 2
        else:
            flow_fraction = installed_fraction * math.sqrt(head_factor)
            operating_head_factor = head_factor
        flow_fraction = min(1.2, max(0.0, flow_fraction))

        reference_flow = max(1e-6, float(env["reference_cooling_flow_fraction"]))
        flow_factor = (max(flow_fraction, 1e-6) / reference_flow) ** float(env["ua_flow_exponent"])
        flow_factor = min(env["ua_flow_factor_max"], max(env["ua_flow_factor_min"], flow_factor))
        clean_ua = p["UA"] * max(0.0, float(env.get("clean_ua_factor", 1.0))) * flow_factor
        fouling_resistance = max(0.0, float(env.get("fouling_resistance_k_per_w", 0.0)))
        ua = 1.0 / (1.0 / max(clean_ua, 1e-12) + fouling_resistance)

        pump_power_w = pump_efficiency = 0.0
        if env.get("pump_curve_enabled", False) and flow_fraction > 0.0:
            design_efficiency = min(1.0, max(1e-6, float(env["cooling_pump_design_efficiency"])))
            best_flow = float(env["cooling_pump_best_efficiency_flow_fraction"])
            efficiency_factor = 1.0 - float(env["cooling_pump_efficiency_curve_coefficient"]) * (
                flow_fraction - best_flow
            ) ** 2
            efficiency_factor = max(float(env["cooling_pump_min_efficiency_fraction"]), efficiency_factor)
            pump_efficiency = min(1.0, design_efficiency * efficiency_factor)
            head_m = float(env["cooling_pump_design_head_m"]) * max(0.0, operating_head_factor)
            pump_power_w = p["rhoj"] * 9.80665 * (flow_fraction * p["Fjmax"]) * head_m / pump_efficiency

        result = (flow_fraction * p["Fjmax"], ua, flow_fraction)
        self._cooling_cache_details = {
            "valve_capacity_fraction": valve_fraction,
            "installed_flow_fraction": installed_fraction,
            "operating_head_factor": operating_head_factor,
            "pump_efficiency": pump_efficiency,
            "pump_power_w": pump_power_w,
        }
        self._cooling_cache_key, self._cooling_cache_value = cache_key, result
        return result

    def cooling_hydraulic_details(self, opening, env):
        self.cooling_characteristics(opening, env)
        return dict(getattr(self, "_cooling_cache_details", {}))

    def cooling_opening_for_flow(self, target_fraction, env):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, _, flow_fraction = self.cooling_characteristics(mid, env)
            if flow_fraction < target_fraction:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def derivatives(self, x, act, env):
        p = self.p
        ca, cb, temperature, jacket_temperature = x
        feed_flow = act["pumps"][0] * p["Fmax"]
        coolant_flow, ua, _ = self.cooling_characteristics(act["heaters"][0], env)
        jacket_inlet = env.get("t_jacket_in", p["Tj_in"])
        feed_concentration = p["Caf"] * max(0.0, env.get("feed_concentration_factor", 1.0))
        rate = self._rate(ca, temperature) * max(0.0, env.get("kinetics_factor", 1.0))
        dilution = feed_flow / p["V"]
        heat_transfer = ua * (temperature - jacket_temperature)
        reaction_heat = (
            p["deltaH"] * max(0.0, env.get("reaction_heat_factor", 1.0))
            * 1000.0 * p["V"] * rate
        )
        return [
            dilution * (feed_concentration - ca) - rate,
            -dilution * cb + rate,
            (
                p["rho"] * p["Cp"] * feed_flow * (env["t_cold"] - temperature)
                + reaction_heat - heat_transfer
            ) / (p["rho"] * p["Cp"] * p["V"]),
            (
                p["rhoj"] * p["Cpj"] * coolant_flow * (jacket_inlet - jacket_temperature)
                + heat_transfer
            ) / (p["rhoj"] * p["Cpj"] * p["Vj"]),
        ]

    def levels_temps(self, x):
        return [], [x[2]]

    def conc(self, x):
        return [_maxv(x[0], 0.0)]

    def extra_observation(self, x):
        return [_maxv(x[0], 0.0), _maxv(x[1], 0.0), x[3]]

    def measurement_extras(self, x, act=None, env=None):
        out = {
            "conc": [_maxv(x[0], 0.0)],
            "product_conc": [_maxv(x[1], 0.0)],
            "jacket_temp": [x[3]],
        }
        if act is not None:
            coolant_flow, ua, flow_fraction = self.cooling_characteristics(act["heaters"][0], env or {})
            details = self.cooling_hydraulic_details(act["heaters"][0], env or {})
            out.update(
                cooling_duty=[abs(ua * (x[2] - x[3]))],
                cooling_flow=[coolant_flow],
                cooling_flow_fraction=[flow_fraction],
                effective_ua=[ua],
                cooling_pump_power=[details.get("pump_power_w", 0.0)],
                cooling_operating_head=[details.get("operating_head_factor", 0.0)],
                coolant_supply_temp=[(env or {}).get("t_jacket_in", self.p["Tj_in"])],
                coolant_return_temp=[x[3]],
                pump_flow=[max(0.0, act["pumps"][0]) * self.p["Fmax"]],
            )
        return out

    def initial_state(self):
        return [0.11771023406521823, 0.8822897659347819, 60.0, 34.93165497960367]

    def clamp_state(self, x):
        x[0] = max(0.0, x[0])
        x[1] = max(0.0, x[1])
        x[2] = min(200.0, x[2])
        x[3] = min(200.0, x[3])
        return x

    def controlled_levels(self):
        return []

    def default_setpoints(self):
        return {}, [60.0]

    energy_scored = False

    def heater_power(self, act, env=None):
        if env and env.get("cooling_hydraulics_enabled", False):
            _, _, flow_fraction = self.cooling_characteristics(act["heaters"][0], env)
            return flow_fraction * self.p["cool_max"]
        return act["heaters"][0] * self.p["cool_max"]

    def pump_power(self, act):
        return act["pumps"][0] * self.p["feed_power_max"]

    def ideal_power(self, levels, temps, t_sp, env, act):
        return 0.0

    def production(self, x, act):
        feed_flow = act["pumps"][0] * self.p["Fmax"]
        return feed_flow / self.p["V"] * _maxv(x[1], 0.0)


MODELS = {"cstr": LegacyCSTR, "cstr_jacketed": JacketedCSTR}
SCENARIOS = tuple(MODELS)


def make_model(scenario="cstr_jacketed"):
    try:
        return MODELS[scenario]()
    except KeyError as exc:
        raise ValueError(f"unknown CSTR scenario: {scenario!r}; choose from {SCENARIOS}") from exc


def obs_vector(model, levels, temps, t_cold, t_amb, h_sp, t_sp):
    """Build the upstream analyzer observation contract."""
    values = []
    for i in range(model.n):
        values.append(levels[i] if i < len(levels) else 0.0)
    values.extend(temps)
    values.extend(t_sp)
    for i in model.controlled_levels():
        values.append(h_sp[i])
    values.extend((t_cold, t_amb))
    return values
