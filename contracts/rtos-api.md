# RTOS 主循环 API 契约 v0.2-draft

> 状态：软件侧内部审查通过，**待 RTOS 团队评审**。评审签字前不得标记 M0 完成。
> 目标：把 1–2kHz 读—算—写循环、绝对截止时间与 overrun 语义冻结为 C ABI。

## 1. 调度与时间基准

- 默认周期 `period_ns=1_000_000`，频率 1kHz；
- 使用绝对单调时钟与绝对等待，不允许累加相对 sleep；
- 第 `n` 拍截止时间 `deadline_ns = epoch_ns + n * period_ns`；
- 控制主线程绑定专用隔离核，优先级由 RTOS 配置；
- 同一进程内推理线程不得与控制线程共享隔离核；
- overrun 发生后不补拍，下一拍重新对齐最近的未来 deadline。

## 2. C ABI

```c
#include <stddef.h>
#include <stdint.h>

#define MB_RT_ABI_VERSION 1u

typedef struct {
    uint64_t epoch_ns;
    uint64_t period_ns;
    uint64_t step_budget_ns;
    uint32_t control_cpu;
    uint32_t priority;
    size_t locked_memory_bytes;
} mb_rt_cycle_config_t;

typedef struct {
    uint64_t cycle_index;
    uint64_t deadline_ns;
    uint64_t actual_start_ns;
    uint64_t previous_finish_ns;
    uint32_t overrun_streak;
    uint32_t flags;
    void *user_data;
} mb_rt_cycle_context_t;

typedef int (*mb_rt_step_fn)(const mb_rt_cycle_context_t *ctx);

typedef struct {
    uint64_t missed_deadline_ns;
    uint32_t cycle_index;
    uint32_t overrun_streak;
    uint32_t reason;
} mb_rt_overrun_event_t;

typedef void (*mb_rt_overrun_fn)(const mb_rt_overrun_event_t *event);

int mb_rt_cycle_create(const mb_rt_cycle_config_t *config,
                       mb_rt_step_fn step_fn,
                       mb_rt_overrun_fn overrun_fn,
                       void *user_data,
                       void **handle_out);
int mb_rt_cycle_start(void *handle);
int mb_rt_cycle_stop(void *handle, uint32_t stop_timeout_ms);
void mb_rt_cycle_destroy(void *handle);
```

## 3. 回调语义

### 3.1 `step_fn`

- 每次调用只执行一拍完整“读→检→算→写”；
- `actual_start_ns` 是回调入口时刻，`deadline_ns` 是本拍理想起点；
- 返回值 0 表示本拍完成；非 0 由 RTOS 计入系统故障并请求停机；
- 回调禁止分配、阻塞、打印、访问网络或文件；
- 回调必须在 `step_budget_ns` 内结束。

### 3.2 `overrun_fn`

- 当回调完成时刻晚于 `deadline_ns + period_ns` 时触发；
- 回调记录 `missed_deadline_ns` 与连续次数，不参与正常控制计算；
- 实现必须是常数时间，推荐只写环形告警槽与原子计数；
- 连续 3 拍 overrun 由下一拍 `ctx->overrun_streak` 传给 governor；
- 连续 10 拍 overrun 或 5ms 无有效循环时，RTOS 请求停机，
  FPGA 独立看门狗进入安全保持。

## 4. 内存与算力配额

| 项目 | v0.1 配额 | 约束 |
|---|---:|---|
| 实时分区常驻内存 | 8MiB | 启动时 `mlockall`，运行期不增长 |
| 热路径堆分配 | 0 byte | 禁止 `malloc/free/new/delete` |
| 热路径栈 | 每线程 ≤ 64KiB | 启动时做 watermark 验证 |
| 单拍计算预算 | 300µs | 详见 `timing-budget.md` |
| 日志环形区 | 1MiB | 满时丢弃调试日志，保留 overrun/governor 事件 |
| 推理线程 | 独立 CPU，100Hz | 5ms timeout，30ms 失效 |

## 5. 降级路径

| 条件 | 动作 |
|---|---|
| 单拍 overrun 1–2 次 | 记账、继续；governor 可见 streak |
| 连续 3 次 overrun | 下一拍 governor 请求 FALLBACK |
| 策略输出过期 > 30ms | governor 请求 FALLBACK |
| 连续 10 次 overrun / 5ms 无循环 | RTOS 请求停机，FPGA 看门狗安全保持 |
| step 回调返回非 0 | 立即请求停机，记录返回码 |
| 停止超时 | 进程保留现场并触发硬件安全路径 |

## 6. 示例

```c
static int control_step(const mb_rt_cycle_context_t *ctx) {
    mb_fpga_io_sample_t sample;
    if (mb_fpga_io_read_latest(g_io, &sample, 0) == MB_E_STALE) {
        governor_request_fallback(MB_FAULT_IO_STALE);
    }

    governor_output_t out = governor_step(&g_gov,
                                          sample,
                                          ctx->overrun_streak,
                                          ctx->deadline_ns);
    return write_control_output(ctx->deadline_ns, &out);
}

static void on_overrun(const mb_rt_overrun_event_t *event) {
    atomic_store(&g_overrun_streak, event->overrun_streak);
}
```

## 7. 评审检查项

- [ ] `overrun` 定义统一为“完成时刻超过下一拍 deadline”；
- [ ] overrun 后不补拍，避免故障时产生连续突发； 
- [ ] `step_budget_ns=300_000` 覆盖 C3 最坏路径；
- [ ] 实时分区 8MiB 与推理线程内存配额可同时满足；
- [ ] 停机路径不依赖 governor 或日志线程继续存活；
- [ ] `mb_rt_cycle_context_t` 生命周期只在回调期间有效。

## 8. 软件侧内部审查（2026-11-10）

- 300µs `step_fn` 上限与 200µs 控制路径预算、100µs 工具链余量一致；
- overrun 不补拍、3 拍 governor 降级、10 拍停机与九信号时间线一致；
- D1.4 已验证 SCHED_FIFO、mlockall、隔离 CPU 和绝对 deadline 可实现；
- 仍需 RTOS 团队确认 ABI 布局、停机路径和推理线程资源配额。

