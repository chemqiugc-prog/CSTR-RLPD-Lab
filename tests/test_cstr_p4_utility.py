"""Physical and configuration tests for the P2.4 cooling utility."""
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
        realism_profile="industrial_p4",
        episode_steps=100,
    )
    params.update(overrides)
    return CSTRGymEnv(**params)


def test_p4_preserves_policy_interface_and_exposes_pump_curve():
    env = make_env(realism_params={"pump_curve_droop": 0.30})
    obs, _ = env.reset(seed=1)
    assert obs.shape == (8,)
    assert env.action_space.shape == (2,)
    assert env._env()["pump_curve_enabled"] is True
    assert env.p3_params["pump_curve_droop"] == 0.30


def test_p4_rejects_unknown_configuration_keys():
    with pytest.raises(ValueError, match="unknown industrial_p4"):
        make_env(realism_params={"imaginary_parameter": 1.0})


def test_pump_and_system_curves_intersect_at_the_reported_working_point():
    env = make_env()
    env.reset(seed=2)
    process = env._env()
    _, _, flow_fraction = env.model.cooling_characteristics(0.6, process)
    details = env.model.cooling_hydraulic_details(0.6, process)
    pump_head = process["pump_head_factor"] * (1.0 + process["pump_curve_droop"])
    pump_head -= process["pump_curve_droop"] * flow_fraction ** 2
    assert np.isclose(details["operating_head_factor"], pump_head, atol=1e-12)
    assert np.isclose(env.model.cooling_characteristics(1.0, process)[2], 1.0, atol=1e-12)


def test_static_head_can_remove_flow_when_available_head_is_too_low():
    env = make_env(realism_params={"system_static_head_fraction": 0.20})
    env.reset(seed=3)
    weak = dict(env._env(), pump_head_factor=0.10)
    assert env.model.cooling_characteristics(1.0, weak)[2] == 0.0


def test_circulation_pump_power_is_physical_and_zero_with_a_closed_valve():
    env = make_env()
    env.reset(seed=4)
    process = env._env()
    closed = env.model.cooling_hydraulic_details(0.0, process)
    open_ = env.model.cooling_hydraulic_details(0.6, process)
    assert closed["pump_power_w"] == 0.0
    assert open_["pump_power_w"] > 1000.0
    assert 0.0 < open_["pump_efficiency"] <= 1.0


def test_header_head_responds_with_a_finite_time_constant():
    env = make_env(dynamic=True)
    env.reset(seed=5)
    env._dist_events = [{"kind": "utility_stress", "start": 0, "duration": 1,
                         "severity": 1.0, "direction": 1.0}]
    env._p2_micro_events = []
    env._k = 2
    initial = env.pump_head_factor
    env._update_p2_process()
    assert env.pump_head_target_factor < 0.7
    assert env.pump_head_target_factor < env.pump_head_factor < initial


def test_return_water_load_warms_the_finite_supply_inventory():
    env = make_env(realism_params={
        "cooling_return_recirc_fraction": 0.20,
        "cooling_supply_recovery_tau_s": 1000.0,
    })
    env.reset(seed=6)
    env.integ.x[3] = 35.0
    env._dist_events = []
    env._p2_micro_events = []
    initial = env.coolant_supply_temperature
    for step in range(40):
        env._k = step
        env._update_p2_process()
    assert env.coolant_supply_temperature > initial + 0.2
    assert env.t_jacket_in == env.coolant_supply_temperature


def test_p3_constant_head_formula_is_unchanged():
    env = CSTRGymEnv(
        "cstr_jacketed", dynamic=False, randomize=False, randomize_setpoints=False,
        randomize_plant=False, realism_profile="industrial_p3",
    )
    env.reset(seed=7)
    process = env._env()
    opening = 0.6
    rangeability = process["valve_rangeability"]
    valve_fraction = (rangeability ** opening - 1.0) / (rangeability - 1.0)
    beta = process["valve_pressure_drop_fraction"]
    installed = valve_fraction / np.sqrt(beta + (1.0 - beta) * valve_fraction ** 2)
    expected = installed * np.sqrt(process["pump_head_factor"])
    assert np.isclose(env.model.cooling_characteristics(opening, process)[2], expected)
