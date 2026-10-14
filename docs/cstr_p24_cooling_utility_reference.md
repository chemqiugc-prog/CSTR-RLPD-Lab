# CSTR P2.4 冷却公用工程参数参考

## 定位

P2.4在P2.3非线性阀门、流量相关UA和显式污垢热阻的基础上，新增循环泵曲线、系统阻力曲线、压力响应惯性以及有限冷却水热容。它是独立的`industrial_p4`环境；`industrial_p3`保持不变，用于回归对照。

本次不修改观测维度、动作维度、reward、ONNX或安全保护参数。供水总管温度、泵健康度等仍是装置隐藏状态。

## 默认参数

| 参数 | 默认值 | 单位 | 含义 |
|---|---:|---|---|
| `pump_curve_droop` | 0.25 | 1 | 关死点相对设计点的归一化扬程增量 |
| `system_static_head_fraction` | 0.05 | 1 | 系统静扬程占设计扬程比例 |
| `pump_head_response_tau_s` | 8 | s | 总管可用扬程一阶响应时间 |
| `cooling_header_volume_m3` | 2.0 | m3 | 供水总管与局部循环的等效热容积 |
| `cooling_supply_recovery_tau_s` | 45 | s | 公用工程将供水恢复到目标温度的时间常数 |
| `cooling_return_recirc_fraction` | 0.08 | 1 | 回水热负荷反馈到局部供水的比例 |
| `cooling_supply_temperature_min_c` | 2 | degC | 供水温度下限 |
| `cooling_supply_temperature_max_c` | 40 | degC | 供水温度上限 |
| `cooling_pump_design_head_m` | 25 | m | 循环泵设计扬程 |
| `cooling_pump_design_efficiency` | 0.72 | 1 | 设计效率 |
| `cooling_pump_best_efficiency_flow_fraction` | 0.65 | 1 | 最佳效率点流量比例 |
| `cooling_pump_min_efficiency_fraction` | 0.45 | 1 | 偏离最佳效率点后的最低效率比例 |
| `cooling_pump_efficiency_curve_coefficient` | 1.20 | 1 | 效率曲线曲率 |

P2.3的阀门、UA、污垢和工况参数全部继承，完整默认值位于`aiogym/realism.py`的`P4_DEFAULTS`。

## 泵与系统工作点

等百分比阀门先得到阀门容量比例`Cv`，再由P2.3安装特性得到定扬程参考流量`q0`。P2.4使用下列无量纲曲线：

```text
泵曲线: h_p(q) = h_available * (1 + d) - d * q^2
系统曲线: h_s(q) = s + (1 - s) * (q / q0)^2
```

其中`d`为泵曲线下垂，`s`为静扬程比例。联立后直接求工作点：

```text
q^2 = [h_available * (1 + d) - s] / [(1 - s) / q0^2 + d]
```

当可用扬程不足以克服静扬程时，流量为零。设计状态、阀门全开时仍满足`q=1`，便于与P2.3交叉验证。

## 总管压力与供回水动态

公用工程压力不再瞬时改变实际扬程：

```text
d h_available / dt = (h_target - h_available) / tau_head
```

冷却供水温度由公用工程恢复能力和回水热负荷共同决定：

```text
d T_supply / dt = (T_target - T_supply) / tau_recovery
                + recirculation * F_coolant / V_header
                  * (T_return - T_supply)
```

夹套采用完全混合模型，因此回水温度等于夹套状态温度`Tj`。该假设比固定入口温度更真实，但尚未描述整厂冷却塔、多个用户并网和长管网输送延迟。

## 循环泵功率

```text
P_pump = rho * g * F_coolant * H_operating / efficiency
```

效率按距最佳效率点的二次偏差计算并设置下限。P2.4前端单独显示循环泵功率；当前经济reward仍沿用P2.3的实际冷却流量计价，因此本次环境升级不会偷偷修改算法目标。

## Python接口

```python
env = AIOGymNativeEnv(
    "cstr_jacketed",
    realism_profile="industrial_p4",
    realism_params={
        "pump_curve_droop": 0.30,
        "cooling_header_volume_m3": 3.0,
        "cooling_supply_recovery_tau_s": 60.0,
    },
)
```

未知参数、非正时间常数、非法效率和温度边界会在环境构造时直接报错。

## 前端接口

```javascript
engine.handleCommand({
  type: 'set_p4_params',
  params: {
    pump_head_response_tau_s: 12,
    cooling_return_recirc_fraction: 0.12,
  },
});
```

旧的`set_p3_params`命令保留为兼容别名。

## 已知限制

- 当前泵曲线是归一化二次曲线，并非某台真实泵的厂家曲线。
- 冷却塔、换热站和其他并联用户被等效为目标温度与恢复时间常数。
- 未建模汽蚀、两相流、水锤和管网空间分布。
- 参数属于工程先验；有设备数据后应以泵曲线、压差、供回水温度和流量历史进行辨识。
