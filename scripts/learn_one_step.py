#!/usr/bin/env python3
"""Beginner-friendly trace of one P2.4 CSTR control step.

This is a diagnostic lesson, not a trainer. It uses a hand-written supervisory
action so the environment data flow can be inspected without loading an Actor.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Direct execution sets sys.path[0] to scripts/. Add the repository root so the
# package import below works in the same way as the main training entry point.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cstrlab.env import CSTRGymEnv


STATE_NAMES = ("Ca", "Cb", "T", "Tj")
OBSERVATION_NAMES = (
    "T_measured",
    "Tj_measured",
    "T_setpoint",
    "T_feed_measured",
    "T_coolant_supply_measured",
    "feed_flow_fraction_measured",
    "cooling_flow_fraction_measured",
    "dT_dt_filtered",
    "dTj_dt_filtered",
)


def print_values(title, names, values):
    """Print named scalar values in a compact table."""
    print(f"\n{title}")
    for name, value in zip(names, values):
        print(f"  {name:34s} = {float(value):12.6f}")


def print_action(title, action):
    print(f"\n{title}")
    print(f"  feed pump                         = {float(action['pumps'][0]):12.6f}")
    print(f"  cooling opening                   = {float(action['heaters'][0]):12.6f}")


def build_environment(guard):
    """Use the same environment contract as the current P2.4 mainline."""
    return CSTRGymEnv(
        scenario="cstr_jacketed",
        control_dt=0.5,
        episode_steps=3,
        reward_mode="economic",
        dynamic=True,
        randomize=True,
        randomize_setpoints=False,
        randomize_plant=True,
        plant_drift=True,
        action_mode="setpoint",
        realism_profile="industrial_p4",
        observation_profile="field_no_concentration",
        supervisory_t_sp_high=85.0,
        supervisory_guard="v4" if guard else "none",
        terminate_on_runaway=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--temperature-action", type=float, default=0.50,
        help="normalized temperature-setpoint request in [0, 1]",
    )
    parser.add_argument(
        "--feed-action", type=float, default=0.70,
        help="normalized feed request in [0, 1]",
    )
    parser.add_argument(
        "--no-guard", action="store_true",
        help="disable V4 for comparison; the default keeps current-mainline V4",
    )
    args = parser.parse_args()

    env = build_environment(guard=not args.no_guard)
    observation_before, reset_info = env.reset(seed=args.seed)

    print("P2.4 ONE-STEP TRACE")
    print("=" * 72)
    print(f"Python type of env                = {type(env).__name__}")
    print(f"Python type of observation       = {type(observation_before).__name__}")
    print(f"observation shape                = {observation_before.shape}")
    print(f"observation dtype                = {observation_before.dtype}")
    print(f"action space                     = {env.action_space}")
    print(f"reset info                       = {reset_info}")
    print(f"operating regime                 = {env._p2_regime_name}")
    print(f"control period                   = {env.control_dt:.3f} s")

    state_before = np.asarray(env.integ.x, dtype=np.float64).copy()
    print_values("1. Hidden true state after reset", STATE_NAMES, state_before)
    print_values("2. Actor-visible 9D observation", OBSERVATION_NAMES, observation_before)

    actor_action = np.clip(
        np.array([args.temperature_action, args.feed_action], dtype=np.float32),
        0.0,
        1.0,
    )
    requested_t_sp = 45.0 + float(actor_action[0]) * (85.0 - 45.0)
    requested_feed = 0.30 + float(actor_action[1]) * (1.00 - 0.30)
    print("\n3. Hand-written Actor request")
    print(f"  normalized action                  = {actor_action.tolist()}")
    print(f"  decoded temperature setpoint       = {requested_t_sp:.6f} degC")
    print(f"  decoded feed-pump request          = {requested_feed:.6f}")

    observation_after, reward, terminated, truncated, info = env.step(actor_action)
    state_after = np.asarray(env.integ.x, dtype=np.float64).copy()

    print_action("4. Command after setpoint decode, V4 and PID", info["commanded_action"])
    print_action("5. Actual action after actuator dynamics and SIS", info["actual_action"])

    guard = info.get("supervisory_guard", {})
    print("\n6. V4 decision")
    print(f"  active                             = {guard.get('active', False)}")
    print(f"  emergency                          = {guard.get('emergency', False)}")
    print(f"  effective temperature setpoint     = {env.t_sp[0]:.6f} degC")
    print(f"  SIS latched                        = {info['realism']['sis_latched']}")

    print_values("7. Hidden true state after one 0.5 s step", STATE_NAMES, state_after)
    print_values("8. True-state change", STATE_NAMES, state_after - state_before)
    print_values("9. Next Actor-visible observation", OBSERVATION_NAMES, observation_after)

    actual_action = info["actual_action"]
    energy_kw = env.model.heater_power(actual_action, env._env()) / 1000.0
    production_credit = env._econ["w_value"] * float(info["prod"])
    cooling_cost = env._econ["w_energy"] * energy_kw
    temperature_violation = max(0.0, float(info["temps"][0]) - 88.0) / 10.0
    violation_cost = env._econ["w_viol"] * temperature_violation

    print("\n10. Economic result")
    print(f"  production rate                    = {float(info['prod']):12.6f}")
    print(f"  production credit                  = {production_credit:12.6f} yuan/h")
    print(f"  cooling-flow-equivalent power      = {energy_kw:12.6f} kW")
    print(f"  cooling cost                       = {cooling_cost:12.6f} yuan/h")
    print(f"  temperature violation cost         = {violation_cost:12.6f} yuan/h")
    print(f"  profit                             = {float(info['profit']):12.6f} yuan/h")
    print(f"  reward = profit * reward_scale     = {float(reward):12.6f}")
    print(f"  reward_scale                       = {env.reward_scale:12.6f}")

    print("\n11. Gymnasium step return values")
    print(f"  next_observation type/shape         = {type(observation_after).__name__} {observation_after.shape}")
    print(f"  reward type                         = {type(reward).__name__}")
    print(f"  terminated                          = {terminated}")
    print(f"  truncated                           = {truncated}")
    print(f"  one-step KPI report                 = {env.scorer.report()}")


if __name__ == "__main__":
    main()
