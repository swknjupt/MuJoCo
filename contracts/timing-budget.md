# 时基与资源预算 v0.2-draft

> 状态：软件侧内部一致性检查通过，**待 FPGA/RTOS/软件三方核对**。
> 目标：冻结 1kHz 控制拍预算、推理配额和 demo 回填口径。

## 1. 时基

| 时基 | 用途 | 规则 |
|---|---|---|
| EtherCAT DC (`dc_time_ns`) | 采样、命令、跨设备追溯 | 唯一控制时基 |
| `CLOCK_MONOTONIC` | RT 调度 deadline、jitter | 不得写入控制算法语义 |
| Host wall clock | 报告展示、日志关联 | 不得用于实时判据 |

同一控制周期内，`deadline_ns` 必须同时写入命令帧与九信号日志。

## 2. 1kHz 单拍预算

| 阶段 | v0.1 最坏预算 |
|---|---:|
| FPGA 快照读取 + seqlock 校验 | 20µs |
| 故障注入与观测组装 | 10µs |
| governor 包络 + 仲裁 | 20µs |
| PD / 控制计算 | 30µs |
| 命令合法性检查 + 限幅 | 10µs |
| FPGA 命令写入 | 20µs |
| 结构化日志入环形区 | 20µs |
| 调度/调用开销预留 | 70µs |
| **总计** | **200µs** |
| 安全余量 | 800µs |

RTOS `step_fn` 的合同上限为 300µs，预留 100µs 给工具链与偶发分支；
超过下一拍 deadline 才计 overrun，超过 300µs 但仍在本拍内则记预算告警。

## 3. 推理配额

| 项目 | 配额 |
|---|---:|
| 推理频率 | 100Hz（每 10ms） |
| 正常推理 P99 | < 1ms |
| 单次硬 timeout | 5ms |
| 最近有效输出失效 | 30ms |
| 线程资源 | 独立 CPU；无 GPU 依赖；单线程 CPUExecutionProvider |
| 超时动作 | 丢弃迟到结果、保持旧值、timeout streak +1 |

推理永远不在 1kHz 拍内执行。主循环只读取 latest-value 快照。

## 4. Jitter 与 overrun 验收

| 场景 | Jitter | Overrun |
|---|---:|---:|
| Demo 带载 | 1kHz jitter < 100µs | 30min = 0 |
| M1 正式版 | P99.99 < 50µs，Max < 5ms，尖峰留痕 | 率 < 1e-6 |
| 当前 T14 隔离核 | P99.99=40.224µs / Max=114.946µs | D1.4 30min 通过；overrun=0 |

统计必须记录 P50/P99/P99.99/Max，不允许只报平均值。

## 5. 资源一致性

| 资源 | C1 FPGA API | C2 RTOS API | 本预算 |
|---|---|---|---|
| 控制周期 | 1kHz | 1kHz | 1ms |
| 读写各段预算 | 20µs + 20µs | 包含在 step | 40µs |
| 命令生效 | 下一 DC tick | 不补拍 | 同拍 deadline |
| 时间戳 | DC ns | deadline ns | 必须相等 |
| 热路径分配 | 禁止 | 0 byte | 禁止 |
| 故障降级 | stale/seq/io | 3 拍 governor；10 拍停机 | 一致 |

## 6. Demo 实测回填

数据日期：2026-11-10（目标机验收）。

| 项目 | 实测/状态 | 是否可作为目标机证据 |
|---|---|---|
| ONNX actor 数值误差 | max_abs_diff 1.43e-6 | 功能证据 |
| ONNX T14 RT 目标机 P99 | 0.0235ms（1000 次，单线程） | 是 |
| 推理壳 `update` | P99 0.00088ms，Max 0.0476ms | 功能证据 |
| 推理 timeout | 5ms，阻塞 50ms 时旧值保持 | 功能证据 |
| RT-SIL 第 4 幕 | 接管 0.000ms，到 SAFE 0.524s | 是，T14 RT 目标机 |
| generic WSL 1kHz | P99.99 1017.7µs，2 overrun/5s | 否，反例 |
| PREEMPT_RT 1kHz | P99.99=40.224µs，Max=114.946µs，overrun=0 | 是，隔离 CPU2 |

## 7. 冻结条件

- [ ] FPGA 确认 DC 时间戳边界与下一 tick 生效语义；
- [ ] RTOS 确认 300µs step budget、overrun 不补拍和停机路径；
- [ ] 软件确认 200µs 控制路径预算可实现；
- [ ] 三方确认 5ms timeout 与 30ms stale 的数字；
- [x] RT 盒完成 D1.4/D1.8 目标机复测并回填本表。

