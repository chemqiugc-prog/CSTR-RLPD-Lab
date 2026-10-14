#!/usr/bin/env python3
"""Compare one field-observed reward experiment with the official P2.4 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import OnnxActor, evaluate_actor, make_p24_env


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", default="field_risk_barrier_candidate_v4")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=202608000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/p24_risk_barrier_v1_vs_official_100ep.json"),
    )
    args = parser.parse_args()

    candidate_path = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    official_path = ROOT / "frontend/models/rlpd_cstr_jacketed_p24_official.onnx"
    official = OnnxActor(official_path)
    candidate = OnnxActor(candidate_path)

    official_result = evaluate_actor(
        "official_p24_analyzer_rlpd_raw",
        official,
        lambda: make_p24_env(
            args.episode_steps,
            observation_profile="analyzer",
            guard="none",
            t_sp_high=90.0,
        ),
        args.episodes,
        args.seed,
    )
    candidate_result = evaluate_actor(
        args.candidate_name,
        candidate,
        lambda: make_p24_env(
            args.episode_steps,
            observation_profile="field_no_concentration",
            guard="v4",
            t_sp_high=85.0,
        ),
        args.episodes,
        args.seed,
    )
    payload = {
        "protocol": {
            "environment": "cstr_jacketed industrial_p4",
            "episodes": args.episodes,
            "episode_steps": args.episode_steps,
            "control_dt_s": 0.5,
            "seed_range": [args.seed, args.seed + args.episodes - 1],
            "test_data_used_for_training_or_selection": False,
            "comparison_note": (
                "Same process seeds, but not an equal-information algorithm comparison: "
                "the official reference uses 8D analyzer observations without V4, while "
                "the candidate uses 9D field observations with V4."
            ),
        },
        "models": {
            "official": str(official_path.relative_to(ROOT)),
            "candidate": str(candidate_path.relative_to(ROOT)),
        },
        "results": [official_result, candidate_result],
        "comparison": {
            "candidate_profit_minus_official": (
                candidate_result["profit_rate"] - official_result["profit_rate"]
            ),
            "candidate_profit_retention": (
                candidate_result["profit_rate"] / official_result["profit_rate"]
                if official_result["profit_rate"] != 0.0 else None
            ),
            "over88_episode_reduction": (
                official_result["episodes_with_over88"]
                - candidate_result["episodes_with_over88"]
            ),
            "trip_episode_reduction": (
                official_result["episodes_with_trip"]
                - candidate_result["episodes_with_trip"]
            ),
        },
    }

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
