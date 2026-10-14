"""Industrial I/O realism used by the opt-in P2 CSTR training profile."""
from __future__ import annotations

from collections import deque
import math

import numpy as np


SENSOR = {
    "temperature": {"delay": 0.8, "tau": 1.5, "noise": 0.08, "drift": 0.0015, "quant": 0.05},
    "analyzer": {"delay": 2.5, "tau": 3.0, "noise": 0.004, "drift": 0.0, "quant": 0.001},
    "feed_flow": {"delay": 0.4, "tau": 0.8, "noise": 4.0e-5, "drift": 0.0, "quant": 1.0e-5},
    "cooling_flow": {"delay": 0.4, "tau": 0.8, "noise": 2.0e-4, "drift": 0.0, "quant": 5.0e-5},
}
ACTUATOR = {"travel_time": 1.5, "stiction": 0.008, "quant": 0.004}

# P2.3 cooling-loop assumptions. Values are engineering priors, not identified
# plant constants. CSTRGymEnv(realism_params={...}) can override any entry.
P3_DEFAULTS = {
    "valve_rangeability": 30.0,
    "valve_pressure_drop_fraction": 0.25,
    "reference_cooling_flow_fraction": 0.26422141637233987,
    "ua_flow_exponent": 0.35,
    "ua_flow_factor_min": 0.25,
    "ua_flow_factor_max": 1.60,
    "normal_fouling_ua_factor_min": 0.92,
    "normal_fouling_ua_factor_max": 1.00,
    "fouled_ua_factor_min": 0.70,
    "fouled_ua_factor_max": 0.90,
    "fouling_resistance_rate_k_per_w_s": 1.0e-10,
    "pump_head_normal_min": 0.95,
    "pump_head_normal_max": 1.05,
    "pump_head_limited_min": 0.70,
    "pump_head_limited_max": 0.90,
    "pump_degradation_max": 0.25,
}

# P2.4 extends the P2.3 valve/fouling model with a finite cooling-water
# utility. The pump curve and system curve are dimensionless around the design
# point; temperatures, volume, head, and efficiency keep their physical units.
P4_DEFAULTS = {
    **P3_DEFAULTS,
    "pump_curve_droop": 0.25,
    "system_static_head_fraction": 0.05,
    "pump_head_response_tau_s": 8.0,
    "cooling_header_volume_m3": 2.0,
    "cooling_supply_recovery_tau_s": 45.0,
    "cooling_return_recirc_fraction": 0.08,
    "cooling_supply_temperature_min_c": 2.0,
    "cooling_supply_temperature_max_c": 40.0,
    "cooling_pump_design_head_m": 25.0,
    "cooling_pump_design_efficiency": 0.72,
    "cooling_pump_best_efficiency_flow_fraction": 0.65,
    "cooling_pump_min_efficiency_fraction": 0.45,
    "cooling_pump_efficiency_curve_coefficient": 1.20,
}


def p3_config(overrides=None):
    config = dict(P3_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"unknown industrial_p3 parameters: {sorted(unknown)}")
        config.update({key: float(value) for key, value in overrides.items()})
    if config["valve_rangeability"] <= 1.0:
        raise ValueError("valve_rangeability must be greater than 1")
    if not 0.0 < config["valve_pressure_drop_fraction"] <= 1.0:
        raise ValueError("valve_pressure_drop_fraction must be in (0, 1]")
    if config["reference_cooling_flow_fraction"] <= 0.0:
        raise ValueError("reference_cooling_flow_fraction must be positive")
    if config["ua_flow_exponent"] < 0.0:
        raise ValueError("ua_flow_exponent must be non-negative")
    return config


def p4_config(overrides=None):
    config = dict(P4_DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"unknown industrial_p4 parameters: {sorted(unknown)}")
        config.update({key: float(value) for key, value in overrides.items()})

    # Reuse P2.3 validation for the shared valve, UA, fouling, and head fields.
    p3_config({key: config[key] for key in P3_DEFAULTS})
    if config["pump_curve_droop"] < 0.0:
        raise ValueError("pump_curve_droop must be non-negative")
    if not 0.0 <= config["system_static_head_fraction"] < 1.0:
        raise ValueError("system_static_head_fraction must be in [0, 1)")
    for key in ("pump_head_response_tau_s", "cooling_header_volume_m3",
                "cooling_supply_recovery_tau_s", "cooling_pump_design_head_m"):
        if config[key] <= 0.0:
            raise ValueError(f"{key} must be positive")
    if not 0.0 <= config["cooling_return_recirc_fraction"] <= 1.0:
        raise ValueError("cooling_return_recirc_fraction must be in [0, 1]")
    if config["cooling_supply_temperature_min_c"] >= config["cooling_supply_temperature_max_c"]:
        raise ValueError("cooling supply temperature bounds are invalid")
    if not 0.0 < config["cooling_pump_design_efficiency"] <= 1.0:
        raise ValueError("cooling_pump_design_efficiency must be in (0, 1]")
    if not 0.0 < config["cooling_pump_min_efficiency_fraction"] <= 1.0:
        raise ValueError("cooling_pump_min_efficiency_fraction must be in (0, 1]")
    if config["cooling_pump_efficiency_curve_coefficient"] < 0.0:
        raise ValueError("cooling_pump_efficiency_curve_coefficient must be non-negative")
    return config


def _quantize(value, resolution):
    return round(value / resolution) * resolution if resolution > 0 else value


class SensorChannel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = deque()
        self.value = None
        self.bias = 0.0

    def reset(self, value, time=0.0):
        self.buffer.clear()
        self.buffer.append((float(time), float(value)))
        self.value = float(value)
        self.bias = 0.0

    def update(self, value, time, dt, rng):
        cfg = self.cfg
        self.buffer.append((float(time), float(value)))
        cutoff = time - cfg["delay"]
        while len(self.buffer) > 1 and self.buffer[1][0] <= cutoff:
            self.buffer.popleft()
        delayed = self.buffer[0][1]
        alpha = 1.0 - math.exp(-dt / cfg["tau"]) if cfg["tau"] > 0 else 1.0
        self.value += alpha * (delayed - self.value)
        self.bias += float(rng.normal(0.0, cfg["drift"] * math.sqrt(max(dt, 1e-9))))
        measured = self.value + self.bias + float(rng.normal(0.0, cfg["noise"]))
        return float(_quantize(measured, cfg["quant"]))


class ActuatorChannel:
    def __init__(self, initial):
        self.position = float(initial)

    def update(self, command, dt):
        cfg = ACTUATOR
        target = float(np.clip(command, 0.0, 1.0))
        if abs(target - self.position) < cfg["stiction"]:
            target = self.position
        max_move = dt / cfg["travel_time"] if cfg["travel_time"] > 0 else 1.0
        self.position += float(np.clip(target - self.position, -max_move, max_move))
        self.position = float(np.clip(_quantize(self.position, cfg["quant"]), 0.0, 1.0))
        return self.position


class IndustrialP2Realism:
    """Sensor delay/lag/noise/drift plus actuator slew/stiction/quantization.

    The profile preserves the environment observation and action dimensions. It
    changes only the information and actuation quality seen by the controller.
    """

    def __init__(self):
        self.sensors = {}
        self.actuators = {}

    def reset(self, true_measurement, initial_action):
        self.sensors = {}
        for key, values in self._signals(true_measurement).items():
            cfg = self._sensor_config(key)
            channels = [SensorChannel(cfg) for _ in values]
            for channel, value in zip(channels, values):
                channel.reset(value)
            self.sensors[key] = channels
        self.actuators = {
            kind: [ActuatorChannel(value) for value in initial_action[kind]]
            for kind in ("pumps", "valves", "heaters")
        }
        return self._copy_measurement(true_measurement)

    @staticmethod
    def _sensor_config(key):
        if key in ("conc", "product_conc"):
            return SENSOR["analyzer"]
        if key == "pump_flow":
            return SENSOR["feed_flow"]
        if key == "cooling_flow":
            return SENSOR["cooling_flow"]
        return SENSOR["temperature"]

    @staticmethod
    def _signals(measurement):
        out = {
            "temps": measurement.get("temps", []),
            "t_cold": [measurement.get("t_cold", 20.0)],
            "t_amb": [measurement.get("t_amb", 20.0)],
        }
        for key in ("conc", "product_conc", "jacket_temp", "coolant_supply_temp",
                    "pump_flow", "cooling_flow"):
            if key in measurement:
                out[key] = measurement[key]
        return out

    @staticmethod
    def _copy_measurement(measurement):
        return {key: (list(value) if isinstance(value, (list, tuple)) else value)
                for key, value in measurement.items()}

    def measure(self, true_measurement, time, dt, rng):
        measured = self._copy_measurement(true_measurement)
        for key, values in self._signals(true_measurement).items():
            result = [channel.update(value, time, dt, rng)
                      for channel, value in zip(self.sensors[key], values)]
            measured[key] = result[0] if key in ("t_cold", "t_amb") else result
        return measured

    def actuate(self, command, dt):
        return {
            kind: [channel.update(value, dt)
                   for channel, value in zip(self.actuators[kind], command[kind])]
            for kind in ("pumps", "valves", "heaters")
        }
