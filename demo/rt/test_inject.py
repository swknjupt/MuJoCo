"""D1.7 验收：三类注入独立生效且同种子逐拍复现。

运行（仓库根目录）：
    .venv/bin/python -m demo.rt.test_inject
"""

from __future__ import annotations

from demo.rt.inject import FaultId, InjectionConfig, SignalInjector, run_trace


def test_delay_only() -> None:
    """延迟 5ms：第 5 拍回放第 0 拍。"""
    injector = SignalInjector(InjectionConfig(delay_ms=5), seed=7)
    timeline = [
        injector.process_observation((float(tick),)).value[0] for tick in range(10)
    ]
    assert timeline == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    result = injector.process_observation((10.0,))
    assert result.fault_ids == (FaultId.OBS_DELAY,)
    print("[验收1] 5ms 延迟仅影响观测通道 ✔")


def test_dropout_only() -> None:
    """丢帧概率 1：首帧后始终保持上一帧。"""
    injector = SignalInjector(
        InjectionConfig(encoder_dropout_p=1.0), seed=11
    )
    outputs = [
        injector.process_observation((float(tick),)).value[0] for tick in range(6)
    ]
    assert outputs == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = injector.process_observation((99.0,))
    assert result.dropped and FaultId.ENCODER_DROPOUT in result.fault_ids
    print("[验收2] 编码器丢帧保持上一帧 ✔")


def test_torque_only() -> None:
    """力矩衰减只改变输出，不触碰观测。"""
    injector = SignalInjector(InjectionConfig(torque_scale=0.4), seed=13)
    observation = injector.process_observation((1.25, -0.5))
    torque = injector.apply_torque(5.0)
    assert observation.value == (1.25, -0.5)
    assert not observation.fault_ids
    assert torque.value == 2.0
    assert torque.fault_ids == (FaultId.TORQUE_ATTENUATION,)
    print("[验收3] 力矩衰减只影响输出通道 ✔")


def test_seed_reproducibility() -> None:
    """三类故障叠加时，同种子两次运行逐拍完全一致。"""
    config = InjectionConfig(
        delay_ms=7,
        encoder_dropout_p=0.25,
        torque_scale=0.6,
    )
    first = run_trace(config, seed=2026, ticks=1000)
    second = run_trace(config, seed=2026, ticks=1000)
    third = run_trace(config, seed=2027, ticks=1000)
    assert first == second
    assert first != third
    assert any("encoder_dropout" in row[2] for row in first)
    assert all("obs_delay" in row[2] for row in first)
    assert all("torque_attenuation" in row[2] for row in first)
    print("[验收4] 1000 拍同种子逐拍一致、异种子不同 ✔")


def main() -> None:
    test_delay_only()
    test_dropout_only()
    test_torque_only()
    test_seed_reproducibility()
    print("\n=== D1.7 故障注入全部验收通过 ===")


if __name__ == "__main__":
    main()
