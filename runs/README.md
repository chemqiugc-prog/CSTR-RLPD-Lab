# Published Research Snapshots

This directory contains curated experiment snapshots. New training runs and
duplicate latest checkpoints remain ignored by Git.

| Snapshot | Contents | Status |
|---|---|---|
| `p24_risk_barrier_v1_seed808` | Best `.pt`, Actor `.onnx`, experiment JSON | Strong risk barrier; not adopted |
| `p24_risk_barrier_v2_seed808` | Best `.pt`, Actor `.onnx`, training JSON | Weaker barrier and lower learning rate; candidate for multi-seed validation |

The frontend's deployed field V1 Actor and these reward-experiment V1/V2 Actors
are different models. The deployed model lives in `frontend/models/`.

The `.pt` files contain the Actor, five Critics, target Critics, and entropy
coefficient. They do not contain optimizer state, replay buffers, or RNG state.
Continuation reloads weights and reconstructs the buffers; it is not an exact
resume of an interrupted training process.

## Reproduce the Published V2 Evaluation

Run from the repository root after installing `requirements.txt`:

```bash
python scripts/evaluate_reward_candidate.py \
  --candidate runs/p24_risk_barrier_v2_seed808.onnx \
  --candidate-name field_risk_barrier_v2_v4 \
  --episodes 100 --episode-steps 1200 --seed 202608100 \
  --output results/p24_risk_barrier_v2_reproduction.json
```

The official reference uses eight analyzer observations and no V4; the candidate
uses nine field observations with V4. They share process seeds, but the result
is not an equal-information comparison of reward functions alone.

## Repeat the V2 Continuation Protocol

```bash
python scripts/train_rlpd.py \
  --reward-mode economic_risk_barrier \
  --episode-steps 1200 --offline-episodes 40 --baseline-episodes 1 \
  --bc-steps 0 --pretrain-updates 0 --online-steps 5000 \
  --utd 5 --n-critics 5 --lr 5e-5 \
  --eval-every 1250 --eval-episodes 12 --final-eval-episodes 24 \
  --min-checkpoint-step 1250 --step-offset 1250 \
  --supervisory-guard v4 --supervisory-t-sp-high 85 \
  --checkpoint-selection safety_constrained \
  --max-trip-episodes 0 --max-over88-rate 0.05 --min-control-kpi 95 \
  --torch-threads 4 --seed 808 --offline-seed 11000 \
  --eval-seed 92000 --final-eval-seed 93000 \
  --init-checkpoint runs/p24_risk_barrier_v1_seed808_best.pt \
  --risk-barrier-trigger-temp 86.5 --risk-barrier-weight 0.5 \
  --risk-recovery-weight 0.25 --out runs/reproduction_v2_seed808
```

The recorded run used Python 3.9, NumPy 2.0.2, PyTorch 2.8.0, Gymnasium 1.1.1,
ONNX 1.19.1, and ONNX Runtime 1.19.2 on CPU. Different software versions or
hardware can change optimization trajectories despite identical seeds.

The V1 run was stopped after the 5,000-step validation checkpoint because profit
declined. Its best checkpoint was at 1,250 steps. The V2 selected checkpoint is
at total step 3,750. Formal evaluations and protocol details are in `results/`
and `docs/cstr_model_change_log.md`.
