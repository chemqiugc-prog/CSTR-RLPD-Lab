#!/usr/bin/env python3
"""Train RLPD on either the upstream legacy CSTR or local P2.4 CSTR.

Pipeline (the offline->online story):
  1. roll out the existing PID controller -> an offline "historian" dataset
  2. offline-pretrain RLPD, then keep learning online (symmetric sampling)
  3. rank RLPD vs PID vs MPC by KPI score (scoring.py — the SAME composite the
     browser shows: tracking + excess-energy + safety) under dynamic disturbed
     conditions, so "RL beats MPC" is apples-to-apples
  4. save a checkpoint and export the Actor as ONNX for the browser HMI

    python scripts/train_rlpd.py --online-steps 30000 --out runs/my_p24
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cstrlab.env import CSTRGymEnv
from cstrlab.baselines import PIDAgent, MPCAgent, evaluate, make_meas
from cstrlab.plants import make_model
from cstrlab.rlpd import RLPD


def collect_offline(env, agent, episodes, seed=1000):
    """Prior data. Supervisory (RL-on-PID) env: roll the default-SP action (= fixed-SP
    PID) + exploration noise. Actuator env: roll the PID controller directly."""
    supervisory = getattr(env, "layout", None) is not None
    a0 = env.default_sp_action() if supervisory else None
    rng = np.random.default_rng(seed)
    data = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if agent is not None:
            agent.reset()
        done = False
        while not done:
            if supervisory:
                a = np.clip(a0 + rng.normal(0, 0.15, a0.shape), 0, 1).astype(np.float32)
            else:
                sp = {"h_sp": env.h_sp, "t_sp": env.t_sp}
                act = agent.compute(make_meas(env), sp, env.control_dt)
                a = np.array(list(act["pumps"]) + list(act["valves"]) + list(act["heaters"]), np.float32)
            o2, r, term, trunc, info = env.step(a)
            data.append((obs, a, r, o2, float(term)))
            obs = o2
            done = term or trunc
    return data


def eval_policy_report(rlpd, env, episodes=12, seed=5000):
    """Evaluate policy value and deployment-oriented thermal safety together."""
    econ = env.reward_mode.startswith("economic")
    guarded = env.reward_mode == "economic_guarded"
    scores = []
    profit_rates = []
    kpis = []
    episodes_with_over88 = 0
    episodes_with_trip = 0
    trip_count = 0
    worst_max_temp = -np.inf
    guard_active_steps = 0
    guard_emergency_steps = 0
    guard_t_sp_intervention = 0.0
    guard_pump_intervention = 0.0
    guard_cooling_intervention = 0.0
    total_steps = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s = 0.0
        profits = []
        over88 = False
        episode_tripped = False
        trip_latched = False
        max_temp = -np.inf
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(rlpd.act(obs, deterministic=True))
            if guarded:
                s += r / env.reward_scale
            elif econ:
                s += info["profit"]
            profits.append(float(info.get("profit", 0.0)))
            temperatures = info.get("temps", [])
            if temperatures:
                temperature = float(temperatures[0])
                max_temp = max(max_temp, temperature)
                over88 = over88 or temperature > 88.0
                if temperature > 92.0 and not trip_latched:
                    trip_count += 1
                    episode_tripped = True
                    trip_latched = True
                elif temperature < 89.0:
                    trip_latched = False
            guard = info.get("supervisory_guard", {})
            guard_active_steps += int(bool(guard.get("active", False)))
            guard_emergency_steps += int(bool(guard.get("emergency", False)))
            guard_t_sp_intervention += float(guard.get("t_sp_intervention", 0.0))
            guard_pump_intervention += float(guard.get("pump_intervention", 0.0))
            guard_cooling_intervention += float(guard.get("cooling_intervention", 0.0))
            total_steps += 1
            done = term or trunc
        kpi = float(env.scorer.report()["score"])
        scores.append(s if econ else kpi)
        profit_rates.append(float(np.mean(profits)))
        kpis.append(kpi)
        episodes_with_over88 += int(over88)
        episodes_with_trip += int(episode_tripped)
        worst_max_temp = max(worst_max_temp, max_temp)
    denominator = max(total_steps, 1)
    return {
        "score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "profit_rate": float(np.mean(profit_rates)),
        "kpi": float(np.mean(kpis)),
        "episodes": int(episodes),
        "episodes_with_over88": int(episodes_with_over88),
        "over88_rate": float(episodes_with_over88 / max(episodes, 1)),
        "episodes_with_trip": int(episodes_with_trip),
        "trip_count": int(trip_count),
        "worst_max_temp": float(worst_max_temp),
        "guard_active_rate": float(guard_active_steps / denominator),
        "guard_emergency_rate": float(guard_emergency_steps / denominator),
        "mean_t_sp_intervention": float(guard_t_sp_intervention / denominator),
        "mean_pump_intervention": float(guard_pump_intervention / denominator),
        "mean_cooling_intervention": float(guard_cooling_intervention / denominator),
    }


def eval_policy(rlpd, env, episodes=12, seed=5000):
    """Backward-compatible score/std wrapper around :func:`eval_policy_report`."""
    report = eval_policy_report(rlpd, env, episodes=episodes, seed=seed)
    return report["score"], report["score_std"]


def safety_checkpoint_feasible(report, max_trip_episodes=0, max_over88_rate=0.05,
                               min_control_kpi=95.0):
    return bool(
        report["episodes_with_trip"] <= max_trip_episodes
        and report["over88_rate"] <= max_over88_rate
        and report["kpi"] >= min_control_kpi
    )


def checkpoint_rank(report, selection="metric", max_trip_episodes=0,
                    max_over88_rate=0.05, min_control_kpi=95.0):
    """Rank checkpoints, with a deterministic least-violation fallback."""
    if selection == "metric":
        return (float(report["score"]),)
    feasible = safety_checkpoint_feasible(
        report, max_trip_episodes, max_over88_rate, min_control_kpi,
    )
    if feasible:
        return (1, float(report["score"]))
    return (
        0,
        -max(0, int(report["episodes_with_trip"]) - int(max_trip_episodes)),
        -max(0.0, float(report["over88_rate"]) - float(max_over88_rate)),
        -max(0.0, float(min_control_kpi) - float(report["kpi"])),
        float(report["score"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cstr_jacketed", choices=["cstr", "cstr_jacketed"])
    ap.add_argument(
        "--reward-mode",
        default="economic",
        choices=["kpi", "economic", "economic_guarded", "economic_risk_barrier", "track"],
    )
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--episode-steps", type=int, default=1200)
    ap.add_argument("--offline-episodes", type=int, default=40)
    ap.add_argument(
        "--baseline-episodes", type=int, default=16,
        help="PID/MPC episodes shown in the training log; final claims need a separate large evaluation",
    )
    ap.add_argument("--randomize-plant", action="store_true", default=True)
    ap.add_argument("--no-randomize-plant", dest="randomize_plant", action="store_false")
    ap.add_argument("--action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument(
        "--realism-profile", default="industrial_p4",
        choices=["ideal", "industrial_p2", "industrial_p3", "industrial_p4"],
    )
    ap.add_argument(
        "--observation-profile", default="field_no_concentration",
        choices=["analyzer", "field_no_concentration"],
        help="policy input contract; field_no_concentration uses only temperature and flow instruments",
    )
    ap.add_argument("--risk-observation", action="store_true")
    ap.add_argument(
        "--supervisory-t-sp-high", type=float,
        help="override the first supervisory temperature output upper bound",
    )
    ap.add_argument(
        "--supervisory-guard", default="v4", choices=["none", "v4"],
        help="optional deterministic guard executed inside training and validation",
    )
    ap.add_argument("--guard-trigger-temp", type=float, default=85.5)
    ap.add_argument("--guard-hard-temp", type=float, default=86.5)
    ap.add_argument("--guard-horizon", type=float, default=5.0)
    ap.add_argument("--guard-target", type=float, default=82.0)
    ap.add_argument("--guard-pump-cap", type=float, default=0.80)
    ap.add_argument("--guard-feed-stop", type=float, default=87.5)
    ap.add_argument("--guard-cooling-floor", type=float, default=0.75)
    ap.add_argument("--guard-predictive-emergency", action="store_true")
    ap.add_argument("--risk-barrier-trigger-temp", type=float, default=85.5)
    ap.add_argument("--risk-barrier-temperature-scale", type=float, default=1.0)
    ap.add_argument("--risk-barrier-horizon", type=float, default=5.0)
    ap.add_argument("--risk-barrier-jacket-horizon", type=float, default=2.0)
    ap.add_argument("--risk-barrier-weight", type=float, default=2.0)
    ap.add_argument("--risk-recovery-weight", type=float, default=1.0)
    ap.add_argument("--risk-barrier-trip-cost", type=float, default=100.0)
    ap.add_argument("--risk-recovery-clip", type=float, default=5.0)
    ap.add_argument(
        "--checkpoint-selection", default="safety_constrained",
        choices=["metric", "safety_constrained"],
        help="select by raw objective or maximize it inside explicit safety/KPI constraints",
    )
    ap.add_argument(
        "--min-checkpoint-step", type=int, default=0,
        help="exclude earlier (including random-initialization) policies from best-model selection",
    )
    ap.add_argument("--max-trip-episodes", type=int, default=0)
    ap.add_argument("--max-over88-rate", type=float, default=0.05)
    ap.add_argument("--min-control-kpi", type=float, default=95.0)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--pretrain-updates", type=int, default=5000)
    ap.add_argument("--online-steps", type=int, default=30000)
    ap.add_argument("--utd", type=int, default=5)
    ap.add_argument("--n-critics", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--offline-seed", type=int, default=1000,
                    help="fixed process and exploration seed for the prior dataset")
    ap.add_argument("--eval-seed", type=int, default=5000,
                    help="held-out seed range used only for checkpoint selection")
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--final-eval-seed", type=int,
                    help="optional second held-out seed range for the saved-run summary")
    ap.add_argument("--final-eval-episodes", type=int, default=24)
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="PyTorch CPU threads; 0 keeps the runtime default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.min_checkpoint_step < 0:
        ap.error("--min-checkpoint-step cannot be negative")

    guard_params = {
        "trigger_temp": args.guard_trigger_temp,
        "hard_temp": args.guard_hard_temp,
        "horizon_s": args.guard_horizon,
        "target_temp": args.guard_target,
        "pump_cap": args.guard_pump_cap,
        "feed_stop_temp": args.guard_feed_stop,
        "cooling_floor": args.guard_cooling_floor,
        "predictive_emergency": args.guard_predictive_emergency,
    }
    risk_barrier_params = {
        "trigger_temp": args.risk_barrier_trigger_temp,
        "temperature_scale": args.risk_barrier_temperature_scale,
        "horizon_s": args.risk_barrier_horizon,
        "jacket_horizon_s": args.risk_barrier_jacket_horizon,
        "barrier_weight": args.risk_barrier_weight,
        "recovery_weight": args.risk_recovery_weight,
        "trip_cost": args.risk_barrier_trip_cost,
        "recovery_clip": args.risk_recovery_clip,
    }

    def mkenv(mode=None, observation_profile=None):
        # RL uses the requested action_mode (setpoint = supervisory RL-on-PID); baselines
        # run on an actuator-mode env (they ARE fixed-SP controllers). integral_obs OFF
        # and randomize_setpoints OFF (RL/economic owns the SPs) keep obs = base contract.
        effective_mode = mode or args.action_mode
        return CSTRGymEnv(args.scenario, reward_mode=args.reward_mode, control_dt=args.control_dt,
                               episode_steps=args.episode_steps, dynamic=True, randomize=True,
                               randomize_setpoints=False, action_mode=effective_mode,
                               randomize_plant=args.randomize_plant, plant_drift=args.randomize_plant,
                               integral_obs=False, terminate_on_runaway=False,
                               realism_profile=args.realism_profile,
                               risk_observation=args.risk_observation,
                               observation_profile=observation_profile or args.observation_profile,
                               supervisory_t_sp_high=args.supervisory_t_sp_high,
                               supervisory_guard=(args.supervisory_guard
                                                  if effective_mode == "setpoint" else "none"),
                               supervisory_guard_params=guard_params,
                               risk_barrier_params=risk_barrier_params)

    env = mkenv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # baselines (the bar to beat) — FIXED-SP PID / fixed-model MPC on an actuator env,
    # the static operating point the supervisory RL must beat by adapting its setpoints.
    metric = "objective" if args.reward_mode == "economic_guarded" else (
        "profit" if args.reward_mode.startswith("economic") else "kpi")
    if args.baseline_episodes < 1:
        ap.error("--baseline-episodes must be at least 1")
    pid_raw = evaluate(
        PIDAgent(make_model(args.scenario)), mkenv("actuator"),
        episodes=args.baseline_episodes,
    )
    # The current full-state MPC reconstructs x from concentration analyzers. Keep it
    # as an analyzer-equipped oracle when training the field-only policy; it must not
    # silently regain concentration through the new deployment observation contract.
    mpc_observation_profile = (
        "analyzer" if args.observation_profile == "field_no_concentration"
        else args.observation_profile
    )
    mpc_raw = evaluate(
        MPCAgent(make_model(args.scenario)),
        mkenv("actuator", observation_profile=mpc_observation_profile),
        episodes=args.baseline_episodes,
    )
    if metric == "objective":
        pid = {metric: pid_raw["return"] / env.reward_scale}
        mpc = {metric: mpc_raw["return"] / env.reward_scale}
    else:
        pid, mpc = pid_raw, mpc_raw
    print(f"[baseline] PID {metric}={pid[metric]:.1f}   MPC {metric}={mpc[metric]:.1f}")

    # 1) offline historian from PID (fixed nominal PID is a fine prior)
    print(f"[offline] collecting {args.offline_episodes} PID episodes...")
    offline = collect_offline(
        mkenv(), PIDAgent(make_model(args.scenario)), args.offline_episodes,
        seed=args.offline_seed,
    )
    print(f"[offline] {len(offline)} transitions")

    rlpd = RLPD(obs_dim, act_dim, n_critics=args.n_critics, utd=args.utd,
                lr=args.lr, batch=256)
    rlpd.load_offline(offline)
    if args.init_checkpoint:
        print(f"[init] loading {args.init_checkpoint}")
        rlpd.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu", weights_only=True))

    # 1b) BC warm-start (tracking only): start near PID. For economic objectives the
    # optimum is FAR from the PID setpoint, so BC-to-PID is a bad init (and imperfect
    # clones run a nonlinear CSTR away → runaway) — skip it with --bc-steps 0 and let
    # pretrain update the actor (original RLPD offline pretrain).
    if args.bc_steps > 0:
        print(f"[bc] warm-starting actor ({args.bc_steps} steps)...")
        rlpd.bc_warmstart(args.bc_steps)
        bc_report = eval_policy_report(
            rlpd, mkenv(), episodes=args.eval_episodes, seed=args.eval_seed,
        )
        print(f"[bc] done  {metric}={bc_report['score']:.1f}  "
              f"(PID {pid[metric]:.1f} / MPC {mpc[metric]:.1f})")

    # 2a) offline pretrain. With BC: critic-only (hold the warm-started actor). Without
    # BC: full actor+critic (learn a policy from the PID prior data).
    pretrain_actor = args.bc_steps == 0
    print(f"[pretrain] {args.pretrain_updates} offline updates (actor={pretrain_actor})...")
    t0 = time.time()
    for i in range(args.pretrain_updates):
        rlpd.update(actor=pretrain_actor)
    pre_report = eval_policy_report(
        rlpd, mkenv(), episodes=args.eval_episodes, seed=args.eval_seed,
    )
    print(f"[pretrain] done in {time.time()-t0:.0f}s  {metric}={pre_report['score']:.1f}")

    # 2b) online learning (symmetric offline+online sampling). In constrained mode,
    # checkpoint selection is lexicographic: satisfy deployment constraints first,
    # then maximize the economic objective inside the feasible set.
    base = args.out or f"runs/rlpd_{args.scenario}"
    os.makedirs(os.path.dirname(base), exist_ok=True)
    selection_kwargs = {
        "selection": args.checkpoint_selection,
        "max_trip_episodes": args.max_trip_episodes,
        "max_over88_rate": args.max_over88_rate,
        "min_control_kpi": args.min_control_kpi,
    }
    best_report = None
    best_rank = None
    best_step = None
    best_path = base + "_best.pt"
    latest_path = base + "_latest.pt"
    if args.step_offset >= args.min_checkpoint_step:
        best_report = pre_report
        best_rank = checkpoint_rank(best_report, **selection_kwargs)
        best_step = args.step_offset
        torch.save(rlpd.state_dict(), best_path)
    torch.save(rlpd.state_dict(), latest_path)
    obs, _ = env.reset(seed=args.seed)
    hist = []
    t0 = time.time()
    for local_step in range(1, args.online_steps + 1):
        step = args.step_offset + local_step
        a = rlpd.act(obs, deterministic=False)
        o2, r, term, trunc, _ = env.step(a)
        rlpd.push(obs, a, r, o2, term)
        obs = o2 if not (term or trunc) else env.reset()[0]
        rlpd.update()
        if step % args.eval_every == 0:
            report = eval_policy_report(
                rlpd, mkenv(), episodes=args.eval_episodes, seed=args.eval_seed,
            )
            rank = checkpoint_rank(report, **selection_kwargs)
            eligible = step >= args.min_checkpoint_step
            if eligible and (best_rank is None or rank > best_rank):
                best_report = report
                best_rank = rank
                best_step = step
                torch.save(rlpd.state_dict(), best_path)
            torch.save(rlpd.state_dict(), latest_path)
            feasible = safety_checkpoint_feasible(
                report, args.max_trip_episodes, args.max_over88_rate, args.min_control_kpi,
            )
            hist.append({
                "step": step,
                metric: report["score"],
                "std": report["score_std"],
                "profit_rate": report["profit_rate"],
                "kpi": report["kpi"],
                "episodes_with_over88": report["episodes_with_over88"],
                "episodes_with_trip": report["episodes_with_trip"],
                "guard_active_rate": report["guard_active_rate"],
                "selection_feasible": feasible,
            })
            sps = local_step / (time.time() - t0)
            constraint_status = (
                f" safe={'Y' if feasible else 'N'}"
                f" over88={report['episodes_with_over88']}/{report['episodes']}"
                f" trip={report['episodes_with_trip']} kpi={report['kpi']:.1f}"
                if args.checkpoint_selection == "safety_constrained" else ""
            )
            best_label = (
                f"{best_report['score']:.1f}@{best_step}" if best_report else "pending"
            )
            print(f"[online] step {step:6d}  RLPD {metric}={report['score']:8.1f}"
                  f"±{report['score_std']:.1f}{constraint_status}  "
                  f"(PID {pid[metric]:.1f} / MPC {mpc[metric]:.1f})  "
                  f"best={best_label}  {sps:.0f} steps/s")

    if best_report is None or not os.path.exists(best_path):
        raise RuntimeError(
            "no checkpoint reached --min-checkpoint-step; increase --online-steps "
            "or lower --min-checkpoint-step"
        )
    rlpd.load_state_dict(torch.load(best_path))          # restore the selected checkpoint
    final_eval_seed = args.final_eval_seed if args.final_eval_seed is not None else args.eval_seed
    final_report = eval_policy_report(
        rlpd, mkenv(), episodes=args.final_eval_episodes, seed=final_eval_seed,
    )
    final = final_report["score"]
    final_std = final_report["score_std"]
    best_validation_feasible = safety_checkpoint_feasible(
        best_report, args.max_trip_episodes, args.max_over88_rate, args.min_control_kpi,
    )
    final_feasible = safety_checkpoint_feasible(
        final_report, args.max_trip_episodes, args.max_over88_rate, args.min_control_kpi,
    )
    result = {
        "scenario": args.scenario, "reward_mode": args.reward_mode,
        "realism_profile": args.realism_profile, "risk_observation": args.risk_observation,
        "observation_profile": args.observation_profile,
        "metric": metric,
        "PID": {metric: pid[metric]}, "MPC": {metric: mpc[metric]},
        "RLPD": {
            metric: final,
            "std": final_std,
            "best": best_report["score"],
            "best_step": best_step,
            "profit_rate": final_report["profit_rate"],
            "kpi": final_report["kpi"],
            "episodes_with_over88": final_report["episodes_with_over88"],
            "episodes_with_trip": final_report["episodes_with_trip"],
            "worst_max_temp": final_report["worst_max_temp"],
            "guard_active_rate": final_report["guard_active_rate"],
            "selection_feasible": final_feasible,
        },
        "best_validation_report": best_report,
        "final_evaluation_report": final_report,
        "history": hist,
        "beats_pid": final > pid[metric], "beats_mpc": final > mpc[metric],
        "margin_vs_mpc": final - mpc[metric], "margin_vs_pid": final - pid[metric],
        "training": {
            "control_dt": args.control_dt,
            "episode_steps": args.episode_steps,
            "offline_episodes": args.offline_episodes,
            "baseline_episodes": args.baseline_episodes,
            "bc_steps": args.bc_steps,
            "pretrain_updates": args.pretrain_updates,
            "online_steps": args.online_steps,
            "utd": args.utd,
            "n_critics": args.n_critics,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "offline_seed": args.offline_seed,
            "eval_seed": args.eval_seed,
            "eval_episodes": args.eval_episodes,
            "final_eval_seed": final_eval_seed,
            "final_eval_episodes": args.final_eval_episodes,
            "torch_threads": torch.get_num_threads(),
            "seed": args.seed,
            "init_checkpoint": args.init_checkpoint,
            "step_offset": args.step_offset,
            "randomize_plant": args.randomize_plant,
            "action_mode": args.action_mode,
            "mpc_observation_profile": mpc_observation_profile,
            "supervisory_t_sp_high": env.supervisory_t_sp_high,
            "supervisory_guard": args.supervisory_guard,
            "supervisory_guard_params": guard_params,
            "risk_barrier_params": risk_barrier_params,
            "checkpoint_selection": args.checkpoint_selection,
            "min_checkpoint_step": args.min_checkpoint_step,
            "max_trip_episodes": args.max_trip_episodes,
            "max_over88_rate": args.max_over88_rate,
            "min_control_kpi": args.min_control_kpi,
            "best_validation_feasible": best_validation_feasible,
            "final_evaluation_feasible": final_feasible,
        },
    }
    print(json.dumps({k: result[k] for k in ("scenario", "metric", "beats_pid", "beats_mpc",
                                             "margin_vs_mpc", "margin_vs_pid")}, indent=2))

    torch.save(rlpd.state_dict(), base + ".pt")
    rlpd.save_onnx(base + ".onnx")
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {base}.pt / .onnx / .json")


if __name__ == "__main__":
    main()
