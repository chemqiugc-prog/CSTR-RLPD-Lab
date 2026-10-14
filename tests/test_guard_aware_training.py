import numpy as np

from cstrlab.env import CSTRGymEnv
from scripts.train_rlpd import checkpoint_rank, safety_checkpoint_feasible


def _p4_setpoint_env(**overrides):
    kwargs = {
        "scenario": "cstr_jacketed",
        "reward_mode": "economic",
        "control_dt": 0.5,
        "episode_steps": 12,
        "dynamic": True,
        "randomize": True,
        "randomize_setpoints": False,
        "randomize_plant": True,
        "plant_drift": True,
        "action_mode": "setpoint",
        "realism_profile": "industrial_p4",
        "observation_profile": "field_no_concentration",
    }
    kwargs.update(overrides)
    return CSTRGymEnv(**kwargs)


def test_default_supervisory_configuration_remains_unguarded_45_to_90():
    env = _p4_setpoint_env()
    assert env.supervisory_guard == "none"
    assert env.layout[0] == ("t_sp", 0, 45, 90)

    env.reset(seed=7101)
    command = env._supervise(np.array([1.0, 1.0], np.float32))

    assert env.t_sp[0] == 90.0
    assert command["pumps"][0] == 1.0


def test_v4_guard_uses_frontend_bounds_and_intervenes_before_inner_pid_output():
    env = _p4_setpoint_env(
        supervisory_t_sp_high=85.0,
        supervisory_guard="v4",
    )
    env.reset(seed=7102)
    env._measurement_cache["temps"][0] = 86.7
    env._supervisory_guard_previous_temperature = 86.6

    command = env._supervise(np.array([1.0, 1.0], np.float32))
    guard = env._last_supervisory_guard

    assert env.layout[0] == ("t_sp", 0, 45, 85.0)
    assert guard["active"] is True
    assert guard["emergency"] is False
    assert np.isclose(guard["heating_rate"], 0.2)
    assert np.isclose(guard["predicted_temperature"], 87.7)
    assert guard["requested_t_sp"] == 85.0
    assert guard["effective_t_sp"] == 82.0
    assert guard["requested_pump"] == 1.0
    assert guard["effective_pump"] == 0.8
    assert command["pumps"][0] == 0.8
    assert command["heaters"][0] >= 0.75


def test_v4_guard_status_is_returned_with_the_environment_transition():
    env = _p4_setpoint_env(
        supervisory_t_sp_high=85.0,
        supervisory_guard="v4",
    )
    env.reset(seed=7103)
    _, _, _, _, info = env.step(np.array([0.5, 0.5], np.float32))

    assert info["supervisory_guard"]["mode"] == "v4"
    assert "requested_t_sp" in info["supervisory_guard"]
    assert "effective_cooling" in info["supervisory_guard"]


def _report(score, *, trips=0, over88_rate=0.0, kpi=98.0):
    return {
        "score": float(score),
        "episodes_with_trip": int(trips),
        "over88_rate": float(over88_rate),
        "kpi": float(kpi),
    }


def test_safety_constrained_checkpoint_selection_prefers_feasibility_then_profit():
    aggressive = _report(20000.0, trips=1, over88_rate=0.80, kpi=99.0)
    feasible_low = _report(12000.0, trips=0, over88_rate=0.04, kpi=96.0)
    feasible_high = _report(13000.0, trips=0, over88_rate=0.02, kpi=97.0)

    assert not safety_checkpoint_feasible(aggressive)
    assert safety_checkpoint_feasible(feasible_low)
    assert checkpoint_rank(feasible_low, "safety_constrained") > checkpoint_rank(
        aggressive, "safety_constrained"
    )
    assert checkpoint_rank(feasible_high, "safety_constrained") > checkpoint_rank(
        feasible_low, "safety_constrained"
    )
    assert checkpoint_rank(aggressive, "metric") > checkpoint_rank(feasible_high, "metric")
