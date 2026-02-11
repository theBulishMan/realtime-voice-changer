# Realtime Voice Changer

全本地实时变声器（Qwen3-TTS + Qwen3-ASR + FastAPI + WebSocket + Desktop WebView）。

## 当前状态

- 已切换 ASR 主链路到 `Qwen3-ASR`。
- Real 模式要求 GPU（CUDA），并默认使用 `Qwen/Qwen3-ASR-0.6B`。
- 支持音色设计、音色克隆、试听、删除。
- 主控台与音色工作台分离：
  - `/` 主控台（设备、启动/停止、延迟、音量条）
  - `/voices` 音色工作台（文本设计 / 官方预置音色编排 / 克隆 / 试听 / 删除）
- 支持虚拟麦（VB-CABLE）输出。

## 目录

- `app/backend/` 后端 API 与实时引擎
- `app/frontend/` 前端页面
- `scripts/` 运行、环境检查、模型预下载、驱动安装
- `data/` 本地数据（SQLite + 音色文件）

## 一次性安装

```powershell
# 在项目根目录
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[desktop]"
```

## 环境检查

```powershell
python scripts/check_env.py
```

重点看：
- `nvidia_smi` 是否正常
- `audio.vb_cable_candidates` 是否能看到 CABLE 设备
- `sox` / `ffmpeg` 是否可用（缺失会影响部分功能）

## 安装虚拟麦（VB-CABLE）

以管理员权限运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_vbcable.ps1
```

安装后建议重启 Windows 一次，再执行：

```powershell
Get-CimInstance Win32_SoundDevice | Where-Object { $_.Name -match "CABLE|VB-Audio" } | Select-Object Name
```

## 启动方式（推荐）

直接双击：`start_test_ui.bat`

或命令行：

```powershell
.\start_test_ui.bat
```

### 启动模式

- 双击或直接执行 `start_test_ui.bat`：默认 `Real mode`（不再弹出 1/2 选择）。
- 可选参数：
  - `.\start_test_ui.bat --real`：真实模型模式。
  - `.\start_test_ui.bat --fake`：快速联调模式。

### Real 模式会自动做的事

- 强制 `RVC_REQUIRE_GPU=1`
- 强制 `RVC_ASR_BACKEND=qwen3`
- 自动检查/安装 `qwen-asr`
- 自动检查 `modelscope`、`sox`
- 自动预下载必要模型到本地（首次）
- 预下载失败会直接中止启动（保持严格全本地，不回退在线拉取）
- 启用 `GPU Turbo` 推理参数（TF32 + SDPA 回退 + 更高 TTS token 预算）
- 默认启用顺序推理（`RVC_TTS_INFER_WORKERS=1`，`RVC_TTS_MODEL_REPLICAS=1`，保证语音顺序）
- 默认启用 SiliconFlow 文本纠错（模型：`zai-org/GLM-4.5-Air`，`enable_thinking=false`）
- 为了更快打开界面，默认关闭启动预热（`RVC_STARTUP_WARMUP=0`，首次真实推理时再懒加载）
- 内嵌后端启动超时默认 180 秒（可用 `RVC_EMBED_BOOT_TIMEOUT_S` 调整）

## GPU 吃满说明

- 实时单人链路（麦克风 -> ASR -> TTS）天然是“短任务串行”，常见 GPU 占用会在 30%-60% 波动，这不代表没走 GPU。
- 可调高并发参数提高吞吐：`RVC_TTS_INFER_WORKERS`、`RVC_TTS_MODEL_REPLICAS`（建议先从 `2/2` 到 `3/2` 逐步试）。
- 如果要看更高占用，请在 Real 模式运行压测脚本：

```powershell
.\stress_gpu_tts.bat
```

- 压测报告输出到：`reports/gpu_stress_tts.json`

默认本地 ASR 目录：

- `.cache/models/Qwen3-ASR-0.6B`
- `.cache/models/Qwen3-TTS-12Hz-1.7B-Base`
- `.cache/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `.cache/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `.cache/models/Qwen2.5-1.5B-Instruct`（可选：ASR 文本大模型纠错）

SiliconFlow 配置（用于 ASR 文本纠错 + 音色 AI 编排）：

- `RVC_TEXT_CORRECTION_PROVIDER=siliconflow`
- `RVC_SILICONFLOW_MODEL=zai-org/GLM-4.5-Air`
- `RVC_SILICONFLOW_API_KEY=你的密钥`（或系统变量 `SILICONFLOW_API_KEY`）
- 也可在 `音色工作台 -> 配置 OpenAI 兼容模型` 弹窗中直接填写并保存到本地 SQLite（重启后保留）

统一模型根目录环境变量：

- `RVC_MODELS_DIR`（默认就是 `.cache/models`）

## 手动预下载模型（可选）

```powershell
python scripts/predownload_models.py --all --provider modelscope --models-dir .\.cache\models --report-json reports\model_predownload.json
```

如果网络中断，直接重复执行同一命令即可断点续传。

下载大模型纠错模型（可选）：

```powershell
python scripts/predownload_models.py --text-corrector --provider modelscope --models-dir .\.cache\models
```

## 页面地址

- 主控台：`http://127.0.0.1:8787/`（Fake）或 `http://127.0.0.1:8788/`（Real）
- 音色工作台：`/voices`
- AI 配置中心：`/settings`（OpenAI 兼容模型配置独立页面）

## 你应该怎么测试（建议顺序）

1. 先在音色工作台创建一个音色并试听。
2. 回主控台，选择输入麦克风、虚拟麦输出、监听输出。
3. 勾选/取消监听开关，点击“应用设置”。
4. 选择音色后点击“启动实时”。
5. 默认启用“按键说话 (PTT)”：
   - 按住“按住说话 (空格)”按钮，或按住键盘空格说话。
   - 松开后会触发一次分段并进入 ASR->纠错->TTS。
6. 说话并观察：
   - 输入麦克风电平条是否实时变化
   - 输出麦克风电平条是否实时变化
   - 事件日志是否出现 `ASR 最终结果`
   - 若开启“大模型纠错后再输出”，是否出现 `纠错后文本`
   - 延迟指标是否开始采样
7. 在其他软件（如会议软件）里把麦克风选成 `CABLE Output`，确认能听到变声结果。

## 音色一致性推荐流程（已内置）

1. 在音色工作台输入一句需求（例如“年轻女声，温柔、清晰、略带沙哑”），点“AI 生成描述与试听文本”。
2. 点“仅试听（不保存）”反复调到满意。
3. 满意后点“试听满意后固化音色”。
4. 固化后该音色会保存为可复用 prompt，实时链路会直接复用，稳定性更高。

说明：
- 设计区和试听区的文本框留空时，会自动使用一段内置的长通用试听文本。
- 仍然可以手动改文本；不改也能直接测试。
- 若切到“官方预置音色（CustomVoice）”，可直接选 `Vivian / Serena / Uncle_Fu / Dylan / Eric / Ryan / Aiden / Ono_Anna / Sohee` 先编排再固化。

## 常见问题

### 1) PowerShell 激活虚拟环境失败

错误示例：直接运行 `...\.venv\Scripts\activate`

正确方式：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Real 模式卡在健康检查

先看控制台是否有 GPU 检查报错、模型下载失败、设备打开失败。常见修复：
- 确认 CUDA 与 torch 匹配
- 重启后再试
- 先执行模型预下载命令

### 3) 看不到虚拟麦

- 重新以管理员安装 VB-CABLE
- 重启系统
- 再执行环境检查和设备检查命令

## API 概览

- `GET /api/v1/health`
- `GET /api/v1/audio/devices`
- `GET /api/v1/voices`
- `GET /api/v1/voices/custom/catalog`
- `POST /api/v1/voices/design`
- `POST /api/v1/voices/design/assist`
- `POST /api/v1/voices/custom`
- `POST /api/v1/voices/clone`
- `POST /api/v1/voices/{voice_id}/preview`
- `DELETE /api/v1/voices/{voice_id}`
- `POST /api/v1/realtime/start`
- `POST /api/v1/realtime/stop`
- `POST /api/v1/realtime/ptt`
- `PATCH /api/v1/realtime/settings`
- `GET /api/v1/settings/siliconflow`
- `PATCH /api/v1/settings/siliconflow`
- `GET /api/v1/metrics/current`
- `WS /ws/realtime`
