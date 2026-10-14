#!/usr/bin/env python3
"""Compare normal, V4-intervention, and SIS-trip control paths.

The temperatures are deliberately forced after reset for teaching. This script
does not represent a naturally evolved episode and must not be used for scoring.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.learn_one_step import build_environment


ACTOR_ACTION = np.array([1.0, 1.0], dtype=np.float32)


def force_temperature_for_lesson(env, temperature):
    """Synchronize true state and sensor memory after a deliberate test override."""
    env.integ.x[2] = float(temperature)
    env._measurement_cache = env.realism.reset(
        env._true_measurement(env._last_act), env._last_act
    )
    measured_temperature = float(env._measurement_cache["temps"][0])
    measured_jacket = float(env._measurement_cache["jacket_temp"][0])
    env._p2_previous_measured_temp = measured_temperature
    env._p2_previous_measured_jacket = measured_jacket
    env._p2_filtered_temp_rate = 0.0
    env._p2_filtered_jacket_rate = 0.0
    env._supervisory_guard_previous_temperature = measured_temperature


def run_case(name, temperature, guard_enabled):
    env = build_environment(guard=guard_enabled)
    env.reset(seed=20260721)
    force_temperature_for_lesson(env, temperature)

    initial_actual = {
        "pump": float(env._last_act["pumps"][0]),
        "cooling": float(env._last_act["heaters"][0]),
    }
    measured_before = float(env._measurement_cache["temps"][0])
    _, reward, terminated, truncated, info = env.step(ACTOR_ACTION)
    guard = info.get("supervisory_guard", env._last_supervisory_guard)

    return {
        "name": name,
        "guard_mode": env.supervisory_guard,
        "true_temperature_before": float(temperature),
        "measured_temperature_before": measured_before,
        "requested_t_sp": 85.0,
        "requested_pump": 1.0,
        "guard_active": bool(guard.get("active", False)),
        "guard_emergency": bool(guard.get("emergency", False)),
        "effective_t_sp": float(env.t_sp[0]),
        "initial_pump": initial_actual["pump"],
        "initial_cooling": initial_actual["cooling"],
        "command_pump": float(info["commanded_action"]["pumps"][0]),
        "command_cooling": float(info["commanded_action"]["heaters"][0]),
        "actual_pump": float(info["actual_action"]["pumps"][0]),
        "actual_cooling": float(info["actual_action"]["heaters"][0]),
        "sis_latched": bool(info["realism"]["sis_latched"]),
        "runaway": bool(info["runaway"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward": float(reward),
        "temperature_after": float(info["temps"][0]),
    }


def print_case(case):
    print(f"\n{case['name']}")
    print("-" * 72)
    print(f"  guard mode                         = {case['guard_mode']}")
    print(f"  true/measured T before             = {case['true_temperature_before']:.3f} / {case['measured_temperature_before']:.3f} degC")
    print(f"  Actor request: Tsp / feed          = {case['requested_t_sp']:.3f} degC / {case['requested_pump']:.3f}")
    print(f"  V4 active / emergency              = {case['guard_active']} / {case['guard_emergency']}")
    print(f"  effective Tsp                      = {case['effective_t_sp']:.3f} degC")
    print(f"  initial actuator: feed / cooling   = {case['initial_pump']:.3f} / {case['initial_cooling']:.3f}")
    print(f"  command: feed / cooling            = {case['command_pump']:.3f} / {case['command_cooling']:.3f}")
    print(f"  actual: feed / cooling             = {case['actual_pump']:.3f} / {case['actual_cooling']:.3f}")
    print(f"  SIS latched / runaway              = {case['sis_latched']} / {case['runaway']}")
    print(f"  terminated / truncated             = {case['terminated']} / {case['truncated']}")
    print(f"  true T after one step              = {case['temperature_after']:.3f} degC")
    print(f"  reward                             = {case['reward']:.6f}")


def main():
    print("P2.4 CONTROL-CHAIN LESSON")
    print("Actor request -> V4 -> PID -> actuator -> SIS -> plant")
    print("The forced temperatures are diagnostic probes, not evaluation episodes.")

    cases = (
        run_case("CASE A: normal operation", 70.0, guard_enabled=True),
        run_case("CASE B: V4 intervention", 86.7, guard_enabled=True),
        run_case("CASE C: SIS authority with V4 disabled", 92.5, guard_enabled=False),
    )
    for case in cases:
        print_case(case)


if __name__ == "__main__":
    main()
