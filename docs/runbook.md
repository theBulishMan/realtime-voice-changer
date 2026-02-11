# Runbook

## 1. 前置条件

- Windows 10/11
- NVIDIA GPU（推荐）
- 已安装 VB-CABLE
- Python 3.11+

## 2. 启动

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python scripts/check_env.py
python scripts/run_dev.py
```

访问 `http://127.0.0.1:8787`

## 3. 首次配置顺序

1. 打开页面后先检查 `health` 与 `WS` 状态。
2. 在 Realtime 控制区设置：
   - 输入设备（麦克风）
   - 虚拟麦设备（通常为 `CABLE Input`）
   - 监听设备（音箱/耳机）
   - 监听开关默认关闭
3. 先创建一个音色（设计或克隆）。
4. 选择该音色，点击“启动实时链路”。

## 4. 音色策略

- 设计音色：通过 `voice_prompt + preview_text` 生成参考音频，再转换为可复用 prompt。
- 克隆音色：上传样本音频与转写文本，生成可复用 prompt。
- 实时合成统一走 Base 模型，降低运行时模型切换抖动。

## 5. 延迟门禁

```powershell
python scripts/benchmark_latency.py --input data/logs/metrics.ndjson --output reports/latency.md --strict
```

目标：
- `P95 FAD <= 250ms`
- `P95 E2E <= 800ms`
- 样本数不足（默认 `<30`）会标记 `UNVERIFIED`

TTS 回灌链路延迟测试（自动生成语音并注入实时管线）：

```powershell
python scripts/benchmark_tts_loopback.py --strict --report-json reports/tts_loopback_latency.json
```

说明：
- 先通过当前音色生成一段 TTS 音频（preview）。
- 再调用 `/api/v1/realtime/inject` 将该音频按帧注入实时队列。
- 注入模式默认 `auto`：`fake mode -> segment`，真实模式 -> `queue`。
- 若找不到目标音色：fake 模式自动创建设计音色；真实模式自动创建克隆音色。
- 可用 `--vad-silence-ms` 与 `--max-segment-ms` 调整分段策略，做延迟调优对比。
- 输出门禁结果与 P95 延迟快照。

快速链路验证（推荐每次改动后执行）：

```powershell
# 先确保服务已启动：python scripts/run_dev.py
python scripts/smoke_realtime.py --create-voice-if-missing --strict
```

真实设备严格验证（建议交付前执行）：

```powershell
python scripts/smoke_realtime.py `
  --require-virtual-mic `
  --wait-for-min-samples `
  --min-samples 30 `
  --max-wait-seconds 120 `
  --strict `
  --report-json reports/smoke_realtime.json
```

说明：
- 默认会尝试复用已有音色；若没有则创建一个设计音色。
- 运行时请对麦克风说话，脚本会轮询 `metrics/current` 并给出门禁快照。
- 默认会先重置本进程内指标统计（不清空日志/数据库，除非显式开启对应参数）。
- 在 `fake mode` 下，脚本默认自动注入模拟段（`/api/v1/realtime/simulate`）来生成样本。
- 因此在 `fake mode` 下无需实际麦克风输入也可完成样本门禁链路验证。
- `--strict` 开启后，样本不足或门禁失败会返回非零退出码。
- `--wait-for-min-samples` 开启后会持续采样到 `min_samples` 或超时（`max_wait_seconds`）。

## 6. 质量门禁

准备 `reports/quality_input.json`（数组项包含 `language/reference/hypothesis`）后执行：

```powershell
python scripts/evaluate_quality.py --input reports/quality_input.json --output reports/quality.md --strict
```

目标：
- `zh CER <= 8%`
- `en WER <= 12%`
- 样本数不足（默认 `zh<5` 或 `en<5`）会标记 `UNVERIFIED`
- 主观 MOS 手工补充

## 7. 常见问题

- 无法识别 VB-CABLE：
  - 确认驱动已安装
  - 在 `GET /api/v1/audio/devices` 中检查是否有 `CABLE Input`
- 首次加载慢：
  - 首次会下载模型权重
  - 可先使用 `RVC_FAKE_MODE=1` 验证流程
- 无声音：
  - 确认已创建并选中音色
  - 确认输入设备有音频活动
  - 检查虚拟麦设备是否正确
