# FPGA I/O API 契约 v0.2-draft

> 状态：软件侧内部审查通过，**待 FPGA 团队评审**。评审签字前不得标记 M0 完成。
> 适用范围：M1 单关节台架；数据结构为后续多关节扩展预留 16 通道上限。
> 上游：[平台设计](../docs/mbd/robotics-mbd-platform-design.md) · [排期表](../docs/mbd/robotics-mbd-schedule.md)

## 1. 设计约束

- 1kHz 控制拍内的 `read_latest` / `write_command` 必须非阻塞；
- 热路径禁止动态分配、日志打印、文件访问和阻塞系统调用；
- 所有字段使用固定宽度整数与 IEEE-754 binary64，禁止依赖隐式 padding；
- FPGA 与 RTOS 共享时间基准为 EtherCAT Distributed Clock（DC）；
- API 只传递快照，不暴露 FPGA 寄存器地址；
- 读写失败必须返回错误码，不能通过全局变量或异常路径报告。

## 2. 时间语义

| 字段 | 语义 |
|---|---|
| `dc_time_ns` | 本帧第一个采样被 DC 锁存的 64-bit 纳秒时刻；只允许向前 |
| `sample_seq` | FPGA 每完成一帧采样递增；回绕按无符号比较 |
| `valid_mask` | 第 `i` 位为 1 表示第 `i` 关节数据有效 |
| `status_word` | FPGA 健康、总线同步、驱动报警等位域 |

RT 软件不得用本机墙钟替代 `dc_time_ns` 做控制排序。`sample_seq` 与
`dc_time_ns` 同时用于一致性检查；二者任一倒退即判接口失效。

## 3. C ABI

```c
#include <stddef.h>
#include <stdint.h>

#define MB_FPGA_MAX_JOINTS 16u
#define MB_FPGA_ABI_VERSION 1u
#define MB_READ_NONBLOCK 0x1u

enum {
    MB_OK = 0,
    MB_E_IO = 5,
    MB_E_AGAIN = 11,
    MB_E_STALE = 12,
    MB_E_SEQ = 13,
    MB_E_BUSY = 16,
    MB_E_INVAL = 22
};

typedef struct {
    uint32_t abi_version;
    uint32_t joint_count;
    uint32_t sample_timeout_us;   /* v0.1 建议 2000 */
    uint32_t flags;
} mb_fpga_io_config_t;

typedef struct {
    uint64_t dc_time_ns;
    uint64_t sample_seq;
    uint32_t valid_mask;
    uint32_t status_word;
    double position_rad[MB_FPGA_MAX_JOINTS];
    double velocity_rad_s[MB_FPGA_MAX_JOINTS];
    double torque_nm[MB_FPGA_MAX_JOINTS];
    double drive_temperature_c[MB_FPGA_MAX_JOINTS];
} mb_fpga_io_sample_t;

typedef struct {
    uint64_t dc_time_ns;
    uint64_t command_seq;
    uint32_t valid_mask;
    uint32_t control_mode;        /* v0.1: 1=torque */
    double torque_nm[MB_FPGA_MAX_JOINTS];
} mb_fpga_io_command_t;

typedef struct {
    uint64_t read_count;
    uint64_t no_new_sample_count;
    uint64_t stale_sample_count;
    uint64_t sequence_error_count;
    uint64_t write_count;
    uint64_t write_reject_count;
    uint32_t last_error;
} mb_fpga_io_stats_t;

int mb_fpga_io_open(const mb_fpga_io_config_t *config,
                    void **handle_out);
void mb_fpga_io_close(void *handle);

int mb_fpga_io_read_latest(void *handle,
                           mb_fpga_io_sample_t *sample_out,
                           uint32_t flags);
int mb_fpga_io_write_command(void *handle,
                             const mb_fpga_io_command_t *command,
                             uint32_t flags);
int mb_fpga_io_get_stats(void *handle,
                         mb_fpga_io_stats_t *stats_out);
```

## 4. 返回码与失败语义

| 返回码 | 值 | 语义 |
|---|---:|---|
| `MB_OK` | 0 | 成功取得新样本 / 命令已接收 |
| `MB_E_AGAIN` | 11 | 尚无更新帧；沿用旧样本并计入 `no_new_sample_count` |
| `MB_E_STALE` | 12 | 最新帧年龄 > `sample_timeout_us`，不得进入新控制计算 |
| `MB_E_SEQ` | 13 | `sample_seq` 或 `dc_time_ns` 倒退，接口失效 |
| `MB_E_BUSY` | 16 | 命令槽被下一 DC tick 占用，命令被拒绝 |
| `MB_E_INVAL` | 22 | ABI、通道数或字段非法 |
| `MB_E_IO` | 5 | FPGA/驱动层错误，取 `stats.last_error` |

`E_AGAIN` 允许零阶保持；`E_STALE`、`E_SEQ`、`E_IO` 必须进入降级路径。
共享内存读取采用 seqlock：读者前后校验版本号，禁止在写者更新期间取半帧。

## 5. 延迟与边界

- `read_latest`：调用至快照可用 ≤ 20µs，v0.1 预算；
- `write_command`：调用至 FPGA 接收 ≤ 20µs，v0.1 预算；
- 数据在下一 DC tick 边界生效，应用层不得假设立即生效；
- 命令时间戳必须等于对应周期 `deadline_ns`，偏差 > 100µs 拒收；
- FPGA 在连续 2ms 未收到合法命令时进入驱动层安全保持。

## 6. 调用示例

### 6.1 单关节读—算—写

```c
mb_fpga_io_sample_t sample;
mb_fpga_io_command_t command = {0};

int rc = mb_fpga_io_read_latest(handle, &sample, 0);
if (rc == MB_OK && (sample.valid_mask & 1u)) {
    double torque = controller_step(sample.position_rad[0],
                                    sample.velocity_rad_s[0],
                                    sample.dc_time_ns);
    torque = clamp(torque, -5.0, 5.0);

    command.dc_time_ns = current_cycle_deadline_ns;
    command.command_seq = next_command_seq++;
    command.valid_mask = 1u;
    command.control_mode = 1u;
    command.torque_nm[0] = torque;
    rc = mb_fpga_io_write_command(handle, &command, 0);
}
```

### 6.2 无新帧与过期帧降级

```c
int rc = mb_fpga_io_read_latest(handle, &sample, MB_READ_NONBLOCK);
if (rc == MB_E_AGAIN) {
    zero_order_hold_count++;
} else if (rc == MB_E_STALE || rc == MB_E_SEQ || rc == MB_E_IO) {
    governor_request_fallback(MB_FAULT_IO_INVALID);
    write_driver_safe_hold();
}
```

## 7. 评审检查项

- [ ] DC 时间戳对应首采样还是帧完成时刻，FPGA/软件解释一致；
- [ ] 结构体大小、字段对齐与字节序由双方静态断言验证；
- [ ] `sample_timeout_us=2000` 对 M1 总线周期足够；
- [ ] 命令槽生命周期与下一 DC tick 的覆盖规则一致；
- [ ] `E_AGAIN` 零阶保持上限不超过 governor 的 30ms 保鲜期；
- [ ] 16 通道上限满足 M1，并记录扩展到更多关节时的 ABI 升级规则。

## 8. 软件侧内部审查（2026-11-10）

- `MB_READ_NONBLOCK`、所有返回码和调用示例现已在同一 ABI 中定义；
- 时间戳、stale、sequence 回退与 RT-SIL 安全通道语义一致；
- 20µs + 20µs 热路径预算已纳入 `timing-budget.md`；
- 仍需 FPGA 团队确认结构体布局、DC 时间戳边界和命令槽覆盖规则。

