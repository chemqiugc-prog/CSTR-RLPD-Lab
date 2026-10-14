# CSTR-RLPD 代码陪读路线

这份路线专门面向 Python 基础较弱、希望摆脱 black box 的读者。不要从神经网络开始，先看懂一个动作如何变成下一个物理状态。

## 第 0 课：先跑通

```bash
cd /Users/qiuguanchen/Desktop/work/code/CSTR-RLPD-Lab
source .venv/bin/activate
python scripts/learn_one_step.py
python scripts/learn_control_chain.py
python -m pytest
```

如果尚未建立 `.venv`，先按根目录 `README.md` 安装环境。

## 第 1 课：认识物理对象

打开 `cstrlab/plants.py`，按这个顺序读：

1. `LegacyCSTR.__init__()`：查看原始参数。
2. `LegacyCSTR.derivatives()`：看 `Ca` 和 `T` 的微分方程。
3. `JacketedCSTR.__init__()`：比较 P2.4 多了哪些参数。
4. `JacketedCSTR.cooling_characteristics()`：看冷却泵和系统曲线如何决定实际流量。
5. `JacketedCSTR.derivatives()`：看 `Ca/Cb/T/Tj` 四个状态如何耦合。

核心问题：

- 为什么浓度是物理状态，却可以不是 actor observation？
- 进料泵开度同时怎样影响产量、浓度和温度？
- 冷却指令为 100% 时，为什么实际移热仍可能不足？

## 第 2 课：数值积分

打开 `cstrlab/kernel.py`。

`Integrator.step()` 的作用是将微分方程的变化率 `dx/dt` 积分成下一个状态。这里要区分：

```text
derivatives: 物理规律
integrator: 数值求解方法
```

物理公式正确不代表数值积分一定稳定，所以测试中会检查状态有界性和能量方向。

## 第 3 课：读懂 Gym 环境

打开 `cstrlab/env.py`，只先看三个入口：

```python
env = CSTRGymEnv(...)
observation, info = env.reset(seed=101)
next_observation, reward, terminated, truncated, info = env.step(action)
```

`reset()` 创建一个新 episode；`step()` 完成一次控制周期。在 `step()` 中寻找以下顺序：

1. 将 RL action 解码成设定点和进料目标。
2. 内层 PID 计算执行器命令。
3. supervisory guard 检查温度风险。
4. 真实性层修改测量和实际动作。
5. 物理模型积分到下一时刻。
6. 计算 reward、profit、KPI 和安全事件。

## 第 4 课：区分 true state 和 observation

打开 `cstrlab/realism.py` 和 `cstrlab/env.py` 中的 observation 构造函数。

P2.4 的物理模型必须保留 `Ca/Cb`，但现场 9 维 actor 只能看到：

```text
T, Tj, T_sp, T_feed, T_coolant_supply,
feed_flow, cooling_flow, dT/dt, dTj/dt
```

这是 sim-to-real 中非常重要的边界：仿真内部可以有不可测状态，但策略输入不能偷看它们。

## 第 5 课：PID 和 V4

打开 `cstrlab/baselines.py` 了解 PID，再在 `cstrlab/env.py` 中搜索 `supervisory_guard`。

- PID 解决“怎样跟踪当前目标”。
- RLPD actor 解决“当前该设什么经济目标”。
- V4 解决“风险上升时是否必须改写 actor 的请求”。
- SIS 解决“已到保护边界时是否必须跳闸”。

V4 不是神经网络，也不是第四个训练版本的简写；它是一组确定性监督规则。

## 第 6 课：reward 与 KPI

打开 `cstrlab/env.py` 中的 economic reward，再看 `cstrlab/scoring.py`。

```text
reward/profit: 训练时优化的标量
control KPI: 评价跟踪、能耗和联锁的综合指标
safety metrics: 超 88°C、跳闸、最高温度等独立指标
```

不要只看一个总分。安全和利润的 trade-off 必须用多列指标显式展示。

## 第 7 课：Actor、Critic 和 replay buffer

最后打开 `cstrlab/rlpd.py`。

- Actor: 输入 observation，输出 action 分布。
- Reward Critic: 估计当前 action 的长期收益。
- Replay buffer: 保存 transition，打破时间相关性并重复利用数据。
- SAC entropy: 鼓励探索，避免过早锁定单一 action。

在能够用自己的话解释前六课之前，不建议先修改 actor/critic 网络。否则很容易把环境问题、评估问题和网络问题混在一起。

## 第 8 课：训练入口

打开 `scripts/train_rlpd.py`，从 `main()` 开始。重点追踪：

1. 命令行参数如何创建 env。
2. 随机采样阶段如何填充 replay buffer。
3. actor 如何与 env 交互。
4. 每个 gradient step 抽取了哪些数据。
5. checkpoint 和 ONNX 在什么条件下保存。

先做烟雾训练：

```bash
python scripts/train_rlpd.py \
  --offline-episodes 2 \
  --bc-steps 10 \
  --pretrain-updates 10 \
  --online-steps 100 \
  --out runs/smoke
```

烟雾训练只验证程序链路，不能用来证明策略性能。

## 第 9 课：独立评价

打开 `scripts/evaluate.py`。正式评价命令：

```bash
python scripts/evaluate.py --episodes 100
```

代码将 P2.4 PID、P2.4 分析仪 RLPD 和 P2.4 现场 V1 + V4 放在同一 seed 集合中。Legacy 官方结果单独列出，不参与 P2.4 直接排名。

## 学会后你应该能回答

1. `state` 和 `observation` 有什么区别？
2. actor 为什么不能看 `Ca/Cb`，但 plant model 又必须有它们？
3. RLPD 输出的是最终冷却指令，还是监督设定点？
4. V4 和 SIS 的职责为什么不同？
5. 为什么两个不同环境的 profit 不能直接比？
6. 一个新 reward 如何做同种子、多指标、独立 holdout 评价？
