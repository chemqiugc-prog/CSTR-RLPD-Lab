# CSTR-RLPD Lab

一个独立的 CSTR 仿真、控制、RLPD 训练和浏览器 HMI 项目。本仓库从 AIO-Gym 中拆出，只保留 CSTR 主线，不修改原有 AIO-Gym 工作目录。

## 仓库保留了什么

| 路线 | 环境 | RL 模型 | 用途 |
|---|---|---|---|
| 官方保留 | AIO-Gym Legacy 两状态 CSTR | `rlpd_cstr_upstream.onnx` | 复现上游基准 |
| 本地主线 | P2.4 四状态夹套 CSTR | 分析仪参考模型、现场 9 维 V1 + V4 | 同环境比较利润、KPI 和安全性 |

Legacy 与 P2.4 的物理环境不同，其利润数值不能直接排名。P2.4 内部的 PID、分析仪 RLPD 和现场 V1 + V4 可以使用同环境、同过程 seed 评价；分析仪 RLPD 有额外浓度信息且没有 V4，因此不能将它与现场策略的差异全部归因于算法。

## 快速开始

### 1. Python 环境

```bash
git clone https://github.com/chemqiugc-prog/CSTR-RLPD-Lab.git
cd CSTR-RLPD-Lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

VS Code 中选择 `CSTR-RLPD-Lab/.venv/bin/python` 作为 Python Interpreter 或 Notebook Kernel。

### 2. 打开前端

```bash
./run.sh
```

访问 <http://127.0.0.1:8010/> 。页面默认打开 P2.4，可在顶部切换 Legacy/P2.4 和 Manual/PID/RL。

### 3. 运行测试

```bash
python -m pytest
```

### 4. 严格评价

```bash
python scripts/evaluate.py --episodes 100
```

输出写入 `results/latest_comparison.json`。P2.4 三个控制器使用同一批 seed；Legacy 结果单独列出。

### 5. 训练 RLPD

```bash
python scripts/train_rlpd.py --online-steps 200000 --out runs/p24_experiment
```

默认配置是 P2.4、无浓度现场观测、setpoint 动作与 V4 supervisory guard。建议先用小步数检查流程，再启动长训练。

## 已保存的研究实验

最新实验为非线性预测风险屏障 reward，代码模式为 `economic_risk_barrier`。
V2 放宽屏障并降低学习率，从 V1 最佳 checkpoint 续训。前端默认仍使用现场 V1 + V4；实验权重在 `runs/` 中单独保存。

| 100回合、600秒 P2.4 测试 | Profit (模拟计价 ¥/h) | KPI | T>88°C回合 | 跳闸回合 |
|---|---:|---:|---:|---:|
| V2 测试集上的标准分析仪 RLPD | 13.1858 | 98.3996 | 100/100 | 48/100 |
| 弱风险屏障 V2，现场9维 + V4 | 11.2034 | 98.6428 | 0/100 | 0/100 |

结果来自单训练种子；零事故仅表示该测试样本中的观察结果。训练后期仍有性能下降，V2 暂未通过多训练种子验收。
仿真参数采用工程先验，尚无真实装置数据校准；profit 采用简化经济口径。

直接评价已保存的 V2：

```bash
python scripts/evaluate_reward_candidate.py \
  --candidate runs/p24_risk_barrier_v2_seed808.onnx \
  --candidate-name field_risk_barrier_v2_v4 \
  --episodes 100 --seed 202608100 \
  --output results/p24_risk_barrier_v2_reproduction.json
```

实验文件与续训命令见 [runs/README.md](runs/README.md)，原始正式结果在 [results/](results/)。

## 目录地图

- `cstrlab/plants.py`: Legacy 和 P2.4 的物理方程。
- `cstrlab/env.py`: Gymnasium 环境，负责 observation、action、reward 和 episode。
- `cstrlab/realism.py`: 仪表、执行器、工况与 SIS 真实性层。
- `cstrlab/rlpd.py`: SAC/RLPD 网络、replay buffer 与更新逻辑。
- `scripts/train_rlpd.py`: 训练入口。
- `scripts/evaluate.py`: PID/参考 RLPD/当前 RLPD 同种子评价。
- `scripts/evaluate_reward_candidate.py`: reward 实验与分析仪 RLPD 的同过程 seed 评价。
- `runs/`: 经筛选的实验 Actor、最佳训练权重和训练摘要。
- `frontend/`: 纯浏览器仿真、ONNX 推理和 HMI。
- `docs/architecture.md`: 数据流和官方/本地边界。
- `docs/`: 环境、模型更新日志与学习资料。

## 当前模型边界

- P2.4 物理状态包含浓度，因为反应速率方程需要它。
- 现场 V1 actor 不读取浓度，使用温度、夹套温度、冷却/进料信号和温度趋势等 9 维可观测量。
- V4 是 actor 之外的确定性安全监督层，不是第四个神经网络。
- 分析仪 RLPD 保留 `Ca/Cb` 快速在线分析假设，只作为 P2.4 理想观测参考。

## 来源声明

仓库包含从 [supcon-international/AIO-Gym](https://github.com/supcon-international/AIO-Gym) 保留的 Legacy CSTR 部分，以及在其基础上开发的 P2.4 夹套 CSTR 环境、训练、评价和前端。详细差异见 `docs/upstream_vs_current_cstr_sim_and_training.md`。
