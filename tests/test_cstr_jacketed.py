"""Physics and interface tests for the P1 jacketed CSTR."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cstrlab.env import CSTRGymEnv
from cstrlab.kernel import Integrator
from cstrlab.plants import JacketedCSTR


def _env(model, **overrides):
    out = {
        "t_cold": model.p["t_cold"],
        "t_amb": model.p["t_amb"],
        "extra_outflow": 0.0,
        "t_jacket_in": model.p["Tj_in"],
        "cooling_flow_factor": 1.0,
        "ua_factor": 1.0,
    }
    out.update(overrides)
    return out


def test_version_state_and_observation_contract():
    model = JacketedCSTR()
    assert model.model_version == "cstr_v2_jacketed"
    assert len(model.initial_state()) == 4
    env = CSTRGymEnv("cstr_jacketed", dynamic=False, randomize=False, randomize_setpoints=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,)
    assert np.all(np.isfinite(obs))


def test_closed_batch_conserves_a_plus_b():
    model = JacketedCSTR()
    act = {"pumps": [0.0], "valves": [], "heaters": [0.0]}
    dx = model.derivatives([0.7, 0.3, 50.0, 50.0], act, _env(model))
    assert abs(dx[0] + dx[1]) < 1e-12


def test_internal_heat_transfer_conserves_total_sensible_energy():
    model = JacketedCSTR()
    model.p["k0"] = 0.0
    act = {"pumps": [0.0], "valves": [], "heaters": [0.0]}
    dx = model.derivatives([0.5, 0.5, 70.0, 20.0], act, _env(model))
    reactor_capacity = model.p["rho"] * model.p["Cp"] * model.p["V"]
    jacket_capacity = model.p["rhoj"] * model.p["Cpj"] * model.p["Vj"]
    assert abs(reactor_capacity * dx[2] + jacket_capacity * dx[3]) < 1e-7


def test_nominal_60c_steady_point_closes_all_balances():
    model = JacketedCSTR()
    p, temperature, feed = model.p, 60.0, 0.5
    flow = feed * p["Fmax"]
    dilution = flow / p["V"]
    k = p["k0"] * np.exp(-p["EaR"] / (temperature + 273.15))
    ca = dilution * p["Caf"] / (dilution + k)
    rate = k * ca
    cb = rate / dilution
    reaction_heat = p["deltaH"] * 1000.0 * p["V"] * rate
    feed_heat = p["rho"] * p["Cp"] * flow * (p["t_cold"] - temperature)
    duty = reaction_heat + feed_heat
    jacket_temperature = temperature - duty / p["UA"]
    cooling = duty / (p["rhoj"] * p["Cpj"] * p["Fjmax"] * (jacket_temperature - p["Tj_in"]))
    act = {"pumps": [feed], "valves": [], "heaters": [cooling]}
    dx = model.derivatives([ca, cb, temperature, jacket_temperature], act, _env(model))
    assert 0.0 < cooling < 1.0
    assert np.max(np.abs(dx)) < 1e-12
    assert abs(cooling - 0.2635) < 0.01


def test_cooling_faults_raise_temperature_over_a_trajectory():
    model = JacketedCSTR()
    action = {"pumps": [0.5], "valves": [], "heaters": [0.30]}

    def run(env):
        integ = Integrator(model)
        integ.reset([0.5, 0.5, 60.0, 35.0])
        for _ in range(240):
            integ.step(0.5, action, env)
        return integ.x[2]

    nominal = run(_env(model))
    flow_loss = run(_env(model, cooling_flow_factor=0.4))
    fouled = run(_env(model, ua_factor=0.5))
    hot_water = run(_env(model, t_jacket_in=model.p["Tj_in"] + 10.0))
    assert flow_loss > nominal
    assert fouled > nominal
    assert hot_water > nominal
