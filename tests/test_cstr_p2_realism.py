"""Tests for the opt-in industrial P2 realism profile."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cstrlab.env import CSTRGymEnv


def make_env(dynamic=False, observation_profile="analyzer"):
    return CSTRGymEnv(
        "cstr_jacketed", reward_mode="economic", dynamic=dynamic,
        randomize=False, randomize_setpoints=False, randomize_plant=False,
        realism_profile="industrial_p2", observation_profile=observation_profile,
        episode_steps=80,
    )


def test_p2_preserves_interface_and_is_seed_reproducible():
    env_a, env_b = make_env(dynamic=True), make_env(dynamic=True)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert obs_a.shape == (8,)
    assert np.array_equal(obs_a, obs_b)
    action = np.array([0.8, 0.7], np.float32)
    for _ in range(30):
        obs_a, _, _, _, info_a = env_a.step(action)
        obs_b, _, _, _, info_b = env_b.step(action)
        assert np.array_equal(obs_a, obs_b)
        assert info_a["realism"] == info_b["realism"]


def test_field_observation_removes_concentration_and_uses_instruments():
    env = make_env(observation_profile="field_no_concentration")
    obs, _ = env.reset(seed=43)
    assert obs.shape == (9,)
    assert np.isfinite(obs).all()
    assert env.observation_space.shape == (9,)
    assert obs[0] == env._measurement_cache["temps"][0]
    assert obs[1] == env._measurement_cache["jacket_temp"][0]
    assert obs[5] > 0.0
    assert obs[6] > 0.0

    before = env._obs().copy()
    env._measurement_cache["conc"] = [999.0]
    env._measurement_cache["product_conc"] = [-999.0]
    after = env._obs()
    assert np.array_equal(before, after)


def test_field_observation_is_seed_reproducible():
    env_a = make_env(dynamic=True, observation_profile="field_no_concentration")
    env_b = make_env(dynamic=True, observation_profile="field_no_concentration")
    obs_a, _ = env_a.reset(seed=44)
    obs_b, _ = env_b.reset(seed=44)
    action = np.array([0.8, 0.7], np.float32)
    assert np.array_equal(obs_a, obs_b)
    for _ in range(30):
        obs_a, _, _, _, _ = env_a.step(action)
        obs_b, _, _, _, _ = env_b.step(action)
        assert np.array_equal(obs_a, obs_b)


def test_observation_profile_does_not_change_true_process_disturbances():
    analyzer = make_env(dynamic=True, observation_profile="analyzer")
    field = make_env(dynamic=True, observation_profile="field_no_concentration")
    analyzer.reset(seed=45)
    field.reset(seed=45)
    action = np.array([0.8, 0.7], np.float32)
    for _ in range(60):
        analyzer.step(action)
        field.step(action)
        assert np.array_equal(analyzer.integ.x, field.integ.x)
        assert analyzer.t_cold == field.t_cold
        assert analyzer.t_jacket_in == field.t_jacket_in
        assert analyzer.pump_head_factor == field.pump_head_factor


def test_p2_actuator_slew_separates_command_from_position():
    env = make_env()
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([1.0, 1.0], np.float32))
    command, actual = info["commanded_action"], info["actual_action"]
    assert command["pumps"][0] == 1.0
    assert command["heaters"][0] == 1.0
    assert 0.5 < actual["pumps"][0] < 1.0
    assert env.model.pid_temp_bias[0] < actual["heaters"][0] < 1.0


def test_p2_fouling_is_gradual_and_monotone():
    env = make_env()
    env.reset(seed=3)
    env._dist_events = [{"kind": "fouling_ramp", "start": 2, "duration": 12,
                         "severity": 1.0, "direction": 1.0}]
    values = []
    for step in range(18):
        env._k = step
        env._update_p2_process()
        values.append(env.ua_factor)
    assert values[2] == 1.0
    assert values[-1] == 0.55
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert len(set(values[3:14])) > 5


def test_p2_utility_stress_is_correlated():
    env = make_env()
    env.reset(seed=5)
    env._dist_events = [{"kind": "utility_stress", "start": 0, "duration": 10,
                         "severity": 1.0, "direction": 1.0}]
    for step in range(11):
        env._k = step
        env._update_p2_process()
    assert env.t_jacket_in > env._p2_base_t_jacket + 8.0
    assert env.cooling_flow_factor < 0.65


def test_p22_feed_batch_changes_composition_activity_and_heat_together():
    env = make_env()
    env.reset(seed=15)
    env._dist_events = [{"kind": "feed_batch", "start": 0, "duration": 10,
                         "severity": 1.0, "direction": 1.0}]
    for step in range(11):
        env._k = step
        env._update_p2_process()
    assert env.t_cold > env._p2_base_t_cold + 4.0
    assert np.isclose(env.feed_concentration_factor, 1.06)
    assert np.isclose(env.kinetics_factor, 1.04)
    assert np.isclose(env.reaction_heat_factor, 1.02)


def test_p22_richer_feed_increases_material_and_heat_source():
    env = make_env()
    env.reset(seed=16)
    state = list(env.integ.x)
    action = {"pumps": [0.7], "valves": [], "heaters": [0.5]}
    nominal = env.model.derivatives(state, action, env._env())
    env.feed_concentration_factor = 1.06
    env.kinetics_factor = 1.04
    env.reaction_heat_factor = 1.02
    richer = env.model.derivatives(state, action, env._env())
    assert richer[0] > nominal[0]
    assert richer[2] > nominal[2]


def test_p21_micro_disturbances_recur_and_recover():
    env = CSTRGymEnv(
        "cstr_jacketed", reward_mode="economic", dynamic=True,
        randomize=False, randomize_setpoints=False, randomize_plant=False,
        realism_profile="industrial_p2", episode_steps=400,
    )
    env.reset(seed=6)
    assert len(env._p2_micro_events) >= 3
    assert {event["kind"] for event in env._p2_micro_events}.issubset(
        {"feed_temp_pulse", "coolant_temp_pulse", "cooling_flow_dip"})
    event = env._p2_micro_events[0]
    start, duration = event["start"], event["duration"]
    assert env._pulse_progress(start, start, duration) == 0.0
    assert env._pulse_progress(start + duration // 2, start, duration) > 0.99
    assert env._pulse_progress(start + duration, start, duration) == 0.0


def test_p2_measurements_are_not_true_state_passthrough():
    env = make_env()
    env.reset(seed=7)
    for _ in range(12):
        obs, _, _, _, _ = env.step(np.array([0.9, 0.1], np.float32))
    measured_temp = float(obs[1])
    true_temp = float(env.integ.x[2])
    assert np.isfinite(obs).all()
    assert abs(measured_temp - true_temp) > 1e-4


def test_p2_sis_latches_feed_off_and_cooling_on():
    env = make_env()
    env.reset(seed=9)
    env.integ.x[2] = 93.0
    _, _, _, _, info = env.step(np.array([1.0, 0.0], np.float32))
    assert info["realism"]["sis_latched"] is True
    assert info["actual_action"]["pumps"][0] == 0.0
    assert info["actual_action"]["heaters"][0] == 1.0


def test_guarded_economic_reward_matches_profit_in_safe_region():
    env = CSTRGymEnv("cstr_jacketed", reward_mode="economic_guarded",
                          dynamic=False, randomize=False, randomize_setpoints=False)
    env.reset(seed=11)
    _, reward, _, _, info = env.step(np.array([0.5, 0.3], np.float32))
    assert info["temps"][0] < 82.0
    assert info["safety_cost"] == 0.0
    assert abs(reward - info["profit"] * env.reward_scale) < 1e-9


def test_guarded_economic_reward_penalizes_hot_fast_rise():
    env = CSTRGymEnv("cstr_jacketed", reward_mode="economic_guarded",
                          dynamic=False, randomize=False, randomize_setpoints=False)
    env.reset(seed=12)
    env.integ.x[2] = 90.0
    env._previous_true_temps = [88.0]
    _, reward, _, _, info = env.step(np.array([0.9, 0.0], np.float32))
    assert info["safety_cost"] > 0.0
    assert reward < info["profit"] * env.reward_scale


def test_risk_barrier_reward_matches_profit_far_from_thermal_boundary():
    env = CSTRGymEnv(
        "cstr_jacketed",
        reward_mode="economic_risk_barrier",
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env.reset(seed=121)
    _, reward, _, _, info = env.step(np.array([0.5, 0.3], np.float32))

    assert info["predicted_temperature"] < 80.0
    assert info["safety_cost"] < 1e-8
    assert abs(reward - info["profit"] * env.reward_scale) < 1e-8


def test_risk_barrier_reward_penalizes_predicted_heat_rise_before_trip():
    env = CSTRGymEnv(
        "cstr_jacketed",
        reward_mode="economic_risk_barrier",
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env.reset(seed=122)
    env.integ.x[2] = 86.0
    env.integ.x[3] = 45.0
    env._previous_true_temps = [85.8]
    env._previous_true_jacket_temperature = 44.9

    _, reward, _, _, info = env.step(np.array([0.9, 0.0], np.float32))

    assert info["temps"][0] < 92.0
    assert info["predicted_temperature"] > info["temps"][0]
    assert info["risk_barrier"] > 0.0
    assert info["safety_cost"] > 0.0
    assert reward < info["profit"] * env.reward_scale


def test_risk_barrier_reward_credits_actual_risk_recovery():
    env = CSTRGymEnv(
        "cstr_jacketed",
        reward_mode="economic_risk_barrier",
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env.reset(seed=123)
    env.integ.x[2] = 86.0
    env.integ.x[3] = 44.0
    env._previous_true_temps = [86.1]
    env._previous_true_jacket_temperature = 44.1
    env._previous_risk_barrier = 10.0
    act = {"pumps": [0.3], "valves": [], "heaters": [1.0]}

    _, _, info = env._reward_done(act)

    assert info["risk_barrier"] < 10.0
    assert info["risk_recovery_bonus"] > 0.0


def test_p2_risk_observation_adds_temporal_features():
    env = CSTRGymEnv("cstr_jacketed", reward_mode="economic_guarded",
                          dynamic=False, randomize=False, randomize_setpoints=False,
                          realism_profile="industrial_p2", risk_observation=True)
    obs, _ = env.reset(seed=13)
    assert obs.shape == (13,)
    obs, _, _, _, _ = env.step(np.array([0.9, 0.1], np.float32))
    assert np.isfinite(obs).all()
    assert np.isclose(obs[-2], env._last_act["pumps"][0])
    assert np.isclose(obs[-1], env._last_act["heaters"][0])
