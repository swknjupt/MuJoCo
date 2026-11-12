# 信号字典 Schema v0.1-draft

> 状态：草案，**待 M0 三方评审**。机器可读定义见
> [`signal-dict.schema.json`](./signal-dict.schema.json)。

## 1. 版本规则

- Schema 使用语义化版本；新增可选字段升 minor，删除/改名/改单位升 major；
- v0 的首个实例是 `examples/output/rt_sil_act*_signals.csv`；
- CSV 字段顺序固定为下表顺序，JSON 记录字段名与 schema 一致；
- 时间统一使用整数微秒，字段名显式带单位，避免 `timestamp` 歧义。

## 2. 字段

| 字段 | 类型 | 单位 | 频率 | 语义 |
|---|---|---|---|---|
| `timestamp_us` | integer | µs | 1kHz | 本控制拍 deadline |
| `theta` | number | rad | 1kHz | 安全通道原始摆角 |
| `theta_dot` | number | rad/s | 1kHz | 安全通道原始角速度 |
| `torque_cmd` | number | N·m | 1kHz | governor/PD 后最终写入力矩 |
| `policy_target` | number | rad | 100Hz 快照 | 最近有效策略目标角 |
| `policy_latency_ms` | number | ms | 100Hz 快照 | 主循环读取时的输出年龄 |
| `envelope_ok` | boolean | — | 1kHz | 本拍安全包络结果 |
| `governor_state` | enum | — | 1kHz | `NORMAL/FALLBACK/SAFE/DISABLED` |
| `injected_fault_id` | string | — | 事件 | 用 `|` 连接同时发生的故障 |

`DISABLED` 只允许出现在 A/B 对照实验，不得出现在正式部署记录。

故障 token：`none`、`obs_delay`、`encoder_dropout`、`torque_attenuation`。

## 3. 追溯不变量

1. `timestamp_us` 在单个 run 内严格递增，间隔 1000µs；
2. `torque_cmd` 绝对值不超过电机配置上限；
3. `envelope_ok=false` 的下一拍或同拍必须出现 governor 接管事件；
4. `governor_state=FALLBACK` 后不得无 reset 回到 `NORMAL`；
5. 故障注入只改变策略/PD 功能通道，`theta/theta_dot` 仍来自安全通道。

## 4. Demo 覆盖

当前 schema 可完整描述 D1.8 四幕 CSV：

- 第 1 幕：常态；
- 第 2 幕：`obs_delay`；
- 第 3 幕：`obs_delay|encoder_dropout|torque_attenuation`，无保护；
- 第 4 幕：相同故障，governor 接管。

2026-11-10 已校验 host RT-SIL、T14 PREEMPT_RT 目标机及调参候选运行共
64,000 行九信号数据。

运行验证：

```bash
.venv/bin/python contracts/test_signal_dict.py
```

