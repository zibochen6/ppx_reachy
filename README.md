# 🚐 柴火基地车 Reachy Mini 智能助手 — 皮皮虾

柴火基地车（MCV）是一台穿越中国的移动 AI 实验室。小柴是基地车上的 Reachy Mini
机器人智能助手，能听、会说、按需调用前后摄像头，并且只依据可追溯日记回答基地车事实。

## 快速开始

```bash
# 安装依赖
uv sync

# 设置环境变量
cp .env.example .env  # 编辑填入你的百炼 API Key

# 下载本地唤醒词模型（sherpa-onnx KWS，"皮皮虾"，一次性）
uv run python scripts/download_kws_model.py

# 下载本地 Silero VAD 模型（嘈杂环境端点，一次性）
uv run python scripts/download_vad_model.py

# 启动语音对话（默认本地唤醒；说"皮皮虾"即时响应）
uv run chaihuo-reachy

# 启动 Web Dashboard
uv run chaihuo-reachy dashboard

# 同步官方聚合页中的全部公开日记
uv run chaihuo-reachy index-journals

# 从语雀知识库入口完整同步正文、原始 HTML、图片和向量索引
uv run python scripts/sync_journals.py

# Agent/自动化调用：增量发现新日记并输出机器可读结果
uv run python scripts/sync_journals.py --json

# 强制重新校验所有正文与图片
uv run python scripts/sync_journals.py --refresh-all --json
```

同步结果保存在 `data/journals/`：

- `<slug>.md`：完整正文，图片在原位置使用本地相对路径。
- `raw/<slug>.html`：语雀 Lake 原始正文 HTML。
- `assets/<slug>/`：该篇日记的全部图片。
- `manifest.json`：来源、更新时间、正文哈希、图片哈希和完整性状态。

脚本只有在“目录篇数 = 完整 Markdown = 原始 HTML”且所有图片均存在并通过哈希校验时才返回成功。遇到未公开或无权限文档会明确以非零状态退出，不会生成占位内容。

## 架构

```
XVF3800 AEC/波束/降噪 → Silero VAD/DoA/本地端点 → ASR → 文本
  → 确定性来源路由 + 语义视觉规划
  → 日记在线校验 / observe_scene（前置或车外后视）/ 实时定位
  → VLM 结构化可见事实 → Qwen3.7 Responses（按需视觉/联网）→ 流式输出与来源
  → TTS (百炼 qwen3-tts-flash-realtime) → 扬声器
```

基地车、人物、路线和旅途事件必须有本轮日记证据；没有可靠记录时固定回答不知道。普通常识仍可由模型回答，但不能冒充基地车事实。

视觉默认使用 `REACHY_VISION_POLICY=semantic`：明确要求看东西时一定观察；“这是什么、
我手里拿的是什么、读一下这个牌子”等实时视觉问题由主模型按需调用摄像头；普通聊天
不会持续采集。未指定方向默认看 Reachy 面前，只有明确的车外/后方语义才使用后视。
可用 `semantic_shadow` 只记录决策不采集，或切换到 `explicit`/`off` 快速回滚。
最近一次视觉事实与图片只在内存保留 15 秒，用于“它是什么颜色”等连续追问。

生产定位优先级为 GPSD → 浏览器精确坐标 → 高德 IoT Wi-Fi → 高德 IP
城市级 → 不可用。Jetson 的 Wi-Fi 定位要求配置高德 Web Key 并能扫描到至少
两个固定周边热点；原始 BSSID 不写日志、不返回 Dashboard。手动位置不会再被当成
实时定位。

项目唯一入口是 `chaihuo-reachy dashboard`。它会启动 SDK 的无界面硬件 daemon，
但不会注册或启动 Reachy 官方 App/Control GUI；daemon 仅负责电机与媒体硬件。

现场上线时可先设置 `REACHY_AUDIO_FRONTEND_V2=false`：V2 仍记录 RMS/SNR/VAD
诊断，但不执行方向静音和本地强制断句；确认数据后切回 `true` 即启用，亦可用同一
开关快速回滚到旧端点行为。

## 唤醒词

默认在设备本地用 sherpa-onnx KWS 检测"皮皮虾"（约 0.5 秒响应，模型 33 MB，CPU 推理），
检测到后才连接云端 ASR。相关配置（均可用环境变量覆盖）：

- `REACHY_WAKE_ENGINE`：`local`（默认，本地 KWS）/ `cloud`（云端 ASR 转写文本匹配，旧路径）/ `off`
- `REACHY_KWS_THRESHOLD`：触发阈值（默认 0.35；误唤醒多就调高到 0.45-0.55）
- `REACHY_KWS_SCORE`：beam-search 增强分数（默认 1.0；漏唤醒多就调高）
- `REACHY_WAKE_LISTEN_TIMEOUT_S`：等待唤醒词的最长时间（默认 60s）

本地模型缺失或推理不可用时（如平台 wheel 损坏）自动回退云端文本匹配，服务不会挂掉；
`uv run chaihuo-reachy test` 会自检 KWS 是否可用。macOS 上如遇
`Library not loaded: libonnxruntime...` 报错，运行 `bash scripts/fix_mac_sherpa_dylib.sh`。

## 部署目标

- **macOS** (开发): 本地麦克风/扬声器 + Reachy Mini USB 摄像头
- **Jetson Orin** (生产): Reachy Mini 全套硬件 + Docker Compose

## 单手姿态交互

Dashboard 的“✋ 手势交互”按需启动 21 点手部姿态推理。五指张开会跟踪掌心，握拳会
随机选择一套现有舞蹈和同名音乐；两种姿态都经过 300 ms 确认，舞蹈中的张掌可抢占
旧舞蹈指令和音乐。推理接口不产生手框，PAF 解码器可解析多副骨架，但控制器只锁定
其中一只手。

先从 NVIDIA `trt_pose_hand` 发布页取得
`hand_pose_resnet18_baseline_att_224x224.pth`，安装与该权重对应的 `trt_pose` 和
PyTorch，然后导出公共制品：

```bash
uv run python scripts/export_hand_pose_models.py \
  --weights /path/to/hand_pose_resnet18_baseline_att_224x224.pth
```

Mac 默认加载 `models/hand_pose/hand_pose_resnet18.mlpackage` 并用 Core ML
`ComputeUnit.ALL`；模型缺失或 Core ML 加载失败时使用同权重的 TorchScript/MPS。
Mac 实机可先运行只读验证（只使用名为 `Reachy Mini Camera` 的摄像头，不驱动电机）：

```bash
uv run python scripts/verify_hand_pose_mac.py --seconds 30
```

在窗口中依次保持张掌和握拳；脚本只有在两者都稳定识别 300 ms 后才返回成功。

需要直接观察骨架并测试 Reachy 头部/身体跟随时，运行独立硬件脚本。它只启动
Reachy daemon、SDK 摄像头、Core ML 和运动控制，不启动 Dashboard、ASR、TTS、
唤醒词或对话服务：

```bash
uv run python scripts/test_hand_follow.py
```

张掌稳定 300 ms 后开始跟随；`Space` 暂停/恢复电机，`r` 回中，`q` 或 `Esc`
退出。退出时机器人先回中再休眠；如希望退出后保持唤醒，可加 `--keep-awake`。
Jetson 使用系统 Python 3.10 和 `jetson` extra。TensorRT Python 绑定来自 JetPack，
因此虚拟环境需要启用 `--system-site-packages`。必须在目标设备上按当前
JetPack/TensorRT 版本构建 engine：

```bash
uv venv --python /usr/bin/python3.10 --system-site-packages
uv sync --extra jetson

.venv/bin/python scripts/build_hand_pose_tensorrt.py \
  --onnx models/hand_pose/hand_pose_resnet18.onnx \
  --engine models/hand_pose/hand_pose_resnet18_fp16.engine
```

构建后执行软件验收：

```bash
.venv/bin/python scripts/verify_hand_pose_jetson.py \
  --engine models/hand_pose/hand_pose_resnet18_fp16.engine
```

Jetson 路径固定要求 Python 3.10、TensorRT 10.3 和可用的 PyTorch CUDA；engine
缺失、版本不兼容或 CUDA 不可用时会直接拒绝启动并把原因返回 Web，不会回退 CPU
或 `torch2trt`。主要配置见 `.env.example` 中的 `REACHY_GESTURE_*`；默认关闭，
只有网页按钮启动后才加载模型。

## License

MIT
