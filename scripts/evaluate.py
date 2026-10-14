#!/usr/bin/env python3
"""Evaluate the small set of CSTR policies shipped by this repository."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cstrlab.baselines import PIDAgent, make_meas
from cstrlab.env import CSTRGymEnv


ROOT = Path(__file__).resolve().parents[1]


class OnnxActor:
    def __init__(self, path):
        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

    @property
    def input_dim(self):
        value = self.input.shape[-1]
        return value if isinstance(value, int) else None

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if self.input_dim is not None and obs.shape[1] != self.input_dim:
            raise ValueError(
                f"{self.path.name} expects {self.input_dim} observations, got {obs.shape[1]}"
            )
        action = self.session.run([self.output.name], {self.input.name: obs})[0]
        return np.clip(np.asarray(action).reshape(-1), 0.0, 1.0).astype(np.float32)


def make_p24_env(episode_steps, observation_profile, guard="none", t_sp_high=90.0,
                 action_mode="setpoint"):
    return CSTRGymEnv(
        scenario="cstr_jacketed",
        reward_mode="economic",
        control_dt=0.5,
        episode_steps=episode_steps,
        dynamic=True,
        randomize=True,
        randomize_setpoints=False,
        randomize_plant=True,
        plant_drift=False,
        action_mode=action_mode,
        realism_profile="industrial_p4",
        observation_profile=observation_profile,
        supervisory_t_sp_high=t_sp_high if action_mode == "setpoint" else None,
        supervisory_guard=guard if action_mode == "setpoint" else "none",
        terminate_on_runaway=False,
    )


def make_legacy_env(episode_steps):
    return CSTRGymEnv(
        scenario="cstr",
        reward_mode="economic",
        control_dt=0.5,
        episode_steps=episode_steps,
        dynamic=True,
        randomize=True,
        randomize_setpoints=False,
        randomize_plant=True,
        action_mode="setpoint",
        realism_profile="ideal",
        observation_profile="analyzer",
        terminate_on_runaway=False,
    )


def _episode_summary(profits, temperatures, tripped, kpi):
    return {
        "profit_rate": float(np.mean(profits)),
        "kpi": float(kpi),
        "max_temperature": float(max(temperatures)),
        "over88": bool(max(temperatures) > 88.0),
        "tripped": bool(tripped),
    }


def evaluate_actor(name, actor, make_env, episodes, seed):
    rows = []
    for episode in range(episodes):
        env = make_env()
        observation, _ = env.reset(seed=seed + episode)
        profits, temperatures = [], []
        tripped = False
        done = False
        while not done:
            observation, _, terminated, truncated, info = env.step(actor.act(observation))
            profits.append(float(info["profit"]))
            temperatures.append(float(info["temps"][0]))
            tripped = tripped or bool(info.get("realism", {}).get("sis_latched", False))
            tripped = tripped or float(info["temps"][0]) > 92.0
            done = terminated or truncated
        rows.append(_episode_summary(profits, temperatures, tripped, env.scorer.report()["score"]))
    return summarize(name, rows)


def evaluate_pid(name, episodes, episode_steps, seed):
    rows = []
    for episode in range(episodes):
        env = make_p24_env(
            episode_steps, observation_profile="field_no_concentration", action_mode="actuator"
        )
        env.reset(seed=seed + episode)
        pid = PIDAgent(env.model)
        for _, _, loop in pid.temp_loops:
            loop.kp, loop.ki, loop.kd = 0.12, 0.004, 0.0
        pid.reset()
        profits, temperatures = [], []
        tripped = False
        done = False
        while not done:
            command = pid.compute(
                make_meas(env), {"h_sp": env.h_sp, "t_sp": env.t_sp}, env.control_dt
            )
            action = np.asarray(
                command["pumps"] + command["valves"] + command["heaters"], dtype=np.float32
            )
            _, _, terminated, truncated, info = env.step(action)
            profits.append(float(info["profit"]))
            temperatures.append(float(info["temps"][0]))
            tripped = tripped or bool(info["realism"]["sis_latched"])
            done = terminated or truncated
        rows.append(_episode_summary(profits, temperatures, tripped, env.scorer.report()["score"]))
    return summarize(name, rows)


def summarize(name, rows):
    return {
        "controller": name,
        "episodes": len(rows),
        "profit_rate": float(np.mean([row["profit_rate"] for row in rows])),
        "profit_std": float(np.std([row["profit_rate"] for row in rows])),
        "kpi": float(np.mean([row["kpi"] for row in rows])),
        "episodes_with_over88": int(sum(row["over88"] for row in rows)),
        "episodes_with_trip": int(sum(row["tripped"] for row in rows)),
        "worst_max_temperature": float(max(row["max_temperature"] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/latest_comparison.json"))
    args = parser.parse_args()

    official_p24 = OnnxActor(ROOT / "frontend/models/rlpd_cstr_jacketed_p24_official.onnx")
    current_p24 = OnnxActor(ROOT / "frontend/models/rlpd_cstr_jacketed_p24_field_v1.onnx")
    make_official = lambda: make_p24_env(
        args.episode_steps, observation_profile="analyzer", guard="none", t_sp_high=90.0
    )
    make_current = lambda: make_p24_env(
        args.episode_steps,
        observation_profile="field_no_concentration",
        guard="v4",
        t_sp_high=85.0,
    )

    p24 = [
        evaluate_pid("p24_pid", args.episodes, args.episode_steps, args.seed),
        evaluate_actor("p24_standard_rlpd", official_p24, make_official, args.episodes, args.seed),
        evaluate_actor("p24_field_v1_v4", current_p24, make_current, args.episodes, args.seed),
    ]
    legacy = None
    if not args.skip_legacy:
        legacy_actor = OnnxActor(ROOT / "frontend/models/rlpd_cstr_upstream.onnx")
        legacy = evaluate_actor(
            "upstream_legacy_rlpd",
            legacy_actor,
            lambda: make_legacy_env(args.episode_steps),
            args.episodes,
            args.seed,
        )

    payload = {
        "protocol": {
            "episodes": args.episodes,
            "episode_steps": args.episode_steps,
            "control_dt": 0.5,
            "seed": args.seed,
            "note": "Legacy and P2.4 results are separate environments and must not be ranked together.",
        },
        "p24_same_environment": p24,
        "legacy_environment": legacy,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
