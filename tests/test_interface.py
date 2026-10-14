"""Public CSTR-only package interface tests."""

import gymnasium as gym
import numpy as np
import pytest

import cstrlab
from cstrlab.env import CSTRGymEnv, SUPERVISORY
from cstrlab.plants import SCENARIOS


def test_only_two_cstr_scenarios_are_public():
    assert SCENARIOS == ("cstr", "cstr_jacketed")
    assert set(SUPERVISORY) == set(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("action_mode", ("actuator", "setpoint"))
def test_environment_reset_and_step(scenario, action_mode):
    realism = "industrial_p4" if scenario == "cstr_jacketed" else "ideal"
    env = CSTRGymEnv(
        scenario=scenario,
        reward_mode="economic",
        action_mode=action_mode,
        realism_profile=realism,
        randomize_setpoints=False,
        episode_steps=2,
    )
    observation, info = env.reset(seed=11)
    assert info == {}
    assert env.observation_space.contains(observation)
    next_observation, reward, terminated, truncated, step_info = env.step(
        np.full(env.action_space.shape, 0.5, dtype=np.float32)
    )
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "profit" in step_info


def test_registered_gymnasium_ids():
    legacy = gym.make("CSTRLab/Legacy-v0", episode_steps=2)
    p24 = gym.make("CSTRLab/P24-v0", episode_steps=2)
    legacy.reset(seed=12)
    p24.reset(seed=12)
    assert legacy.unwrapped.scenario == "cstr"
    assert p24.unwrapped.scenario == "cstr_jacketed"


def test_field_observation_is_nine_dimensional_and_has_no_concentration():
    env = cstrlab.CSTRGymEnv(
        scenario="cstr_jacketed",
        realism_profile="industrial_p4",
        observation_profile="field_no_concentration",
        action_mode="setpoint",
    )
    observation, _ = env.reset(seed=13)
    assert observation.shape == (9,)
    assert np.isfinite(observation).all()


def test_non_cstr_scenario_is_rejected():
    with pytest.raises(ValueError, match="scenario must be one of"):
        CSTRGymEnv(scenario="hvac")
