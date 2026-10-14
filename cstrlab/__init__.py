"""CSTR-RLPD-Lab public Python API."""

from .env import CSTRGymEnv
from .kernel import Integrator
from .plants import JacketedCSTR, LegacyCSTR, SCENARIOS, make_model

__all__ = [
    "CSTRGymEnv",
    "Integrator",
    "JacketedCSTR",
    "LegacyCSTR",
    "SCENARIOS",
    "make_model",
]

try:
    from gymnasium.envs.registration import register

    register(
        id="CSTRLab/Legacy-v0",
        entry_point="cstrlab.env:CSTRGymEnv",
        kwargs={"scenario": "cstr"},
    )
    register(
        id="CSTRLab/P24-v0",
        entry_point="cstrlab.env:CSTRGymEnv",
        kwargs={"scenario": "cstr_jacketed", "realism_profile": "industrial_p4"},
    )
except Exception:
    pass
