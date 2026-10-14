# CSTR-RLPD Lab 架构说明

## 1. 两套环境

### Legacy CSTR

- Python 名称: `cstr`
- 前端名称: `cstr`
- 状态: 反应物浓度 `Ca` 和反应器温度 `T`
- 操作: 进料泵和冷却开度
- 意义: 保留上游 AIO-Gym 的可复现基准

### Local P2.4 Jacketed CSTR

- Python 名称: `cstr_jacketed`
- 前端名称: `cstr_jacketed_p2`
- 物理状态: `Ca`/`Cb`、反应器温度、夹套温度
- 冷却子系统: 泵曲线/系统曲线工作点、有限冷却水热容、回水温度、污垢热阻与实际 UA
- 真实性: 工况分布、长时扰动、轻微扰动、仪表噪声/滞后/死区、执行器速率/卡滞、SIS
- 意义: 本地训练和同环境对比的主线

## 2. 一个控制步发生什么

```text
真实物理状态
  -> 传感器故障/噪声/滞后
  -> observation
  -> Manual / PID / RLPD actor
  -> 设定点和进料目标
  -> 内层 PID
  -> V4 安全监督（仅当前现场策略）
  -> 执行器卡滞/限速/故障
  -> SIS 保护动作
  -> 数值积分更新物理状态
  -> profit、KPI、超温和跳闸统计
```

## 3. 训练数据流

`scripts/train_rlpd.py` 创建 `CSTRGymEnv`，先收集随机/在线交互数据，将 transition 写入 replay buffer，然后使用 SAC/RLPD 更新 actor 和 critic。一条 transition 是：

```text
(observation, action, reward, next_observation, done)
```

RLPD 可同时从离线 replay 和新采集的在线 replay 抽样。没有离线数据也能训练，但会失去 RLPD 用旧经验提高样本效率的优势。

## 4. P2.4 策略边界

| 策略 | actor 观测 | 外部安全层 | 角色 |
|---|---|---|---|
| P2.4 analyzer reference | 包含快速 `Ca/Cb` | 无 | 理想信息下的收益/风险参考 |
| P2.4 field V1 + V4 | 9 维现场量，不含浓度 | V4 | 当前可部署假设下的主线 |

V4 读取实测温度与温升速度。当热风险上升时，它会压低温度目标、限制/切断进料并提高冷却。它不改变 actor 网络结构，但会改变最终送到装置的动作。

## 5. 比较原则

1. 只在同一物理环境、同一 observation 可用性声明、同一 seed 和同一 episode 长度下排名。
2. 同时报告利润、控制 KPI、超 88°C 回合、跳闸回合和最坏最高温度。
3. Legacy 只用于复现上游，不用它的利润值证明 P2.4 策略更好或更差。
4. 每次 reward、observation、物理方程或 guard 变更都应记录在相应 changelog 中。
