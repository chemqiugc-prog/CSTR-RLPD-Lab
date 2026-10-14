"""Physical and configuration tests for the opt-in P2.3 cooling loop."""
from __future__ import annotations

import numpy as np
import pytest

from cstrlab.env import CSTRGymEnv


def make_env(**overrides):
    params = dict(
        scenario="cstr_jacketed",
        reward_mode="economic",
        dynamic=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        realism_profile="industrial_p3",
        episode_steps=100,
    )
    params.update(overrides)
    return CSTRGymEnv(**params)


def test_p3_preserves_policy_interface_and_exposes_configuration():
    env = make_env(realism_params={"valve_rangeability": 20.0})
    obs, _ = env.reset(seed=1)
    assert obs.shape == (8,)
    assert env.action_space.shape == (2,)
    assert env.p3_params["valve_rangeability"] == 20.0
    assert env._env()["cooling_hydraulics_enabled"] is True


def test_p3_rejects_unknown_configuration_keys():
    with pytest.raises(ValueError, match="unknown industrial_p3"):
        make_env(realism_params={"imaginary_parameter": 1.0})


def test_equal_percentage_valve_is_monotone_and_not_linear():
    env = make_env()
    env.reset(seed=2)
    process = env._env()
    flows = [env.model.cooling_characteristics(opening, process)[2]
             for opening in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a < b for a, b in zip(flows, flows[1:]))
    assert flows[0] == 0.0
    assert np.isclose(flows[-1], 1.0)
    assert not np.isclose(flows[2], 0.5, atol=0.05)


def test_reference_flow_has_a_reproducible_valve_opening():
    env = make_env()
    env.reset(seed=3)
    process = env._env()
    target = env.p3_params["reference_cooling_flow_fraction"]
    opening = env.model.cooling_opening_for_flow(target, process)
    flow = env.model.cooling_characteristics(opening, process)[2]
    assert np.isclose(flow, target, atol=1e-10)
    assert 0.3 < opening < 0.7


def test_fouling_resistance_and_pump_head_reduce_cooling_authority():
    env = make_env()
    env.reset(seed=4)
    process = env._env()
    nominal_flow, nominal_ua, _ = env.model.cooling_characteristics(0.7, process)
    fouled = dict(process, fouling_resistance_k_per_w=4.0e-6)
    weak_pump = dict(process, pump_head_factor=0.5)
    fouled_flow, fouled_ua, _ = env.model.cooling_characteristics(0.7, fouled)
    weak_flow, weak_ua, _ = env.model.cooling_characteristics(0.7, weak_pump)
    assert np.isclose(fouled_flow, nominal_flow)
    assert fouled_ua < nominal_ua
    assert weak_flow < nominal_flow
    assert weak_ua < nominal_ua


def test_normal_fouling_changes_slowly_over_a_ten_minute_episode():
    env = make_env(realism_params={"fouling_resistance_rate_k_per_w_s": 1.0e-10})
    env.reset(seed=5)
    initial = env.fouling_resistance_k_per_w
    env._k = int(600.0 / env.control_dt)
    env._update_p2_process()
    assert np.isclose(env.fouling_resistance_k_per_w - initial, 6.0e-8)
    assert env.ua_factor > 0.99


def test_p2_linear_cooling_path_is_preserved_for_cross_environment_tests():
    env = CSTRGymEnv(
        "cstr_jacketed", dynamic=False, randomize=False, randomize_setpoints=False,
        randomize_plant=False, realism_profile="industrial_p2",
    )
    env.reset(seed=6)
    flow = env.model.cooling_characteristics(0.5, env._env())[2]
    assert np.isclose(flow, 0.5)
