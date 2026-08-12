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
麦克风 → ASR (百炼 qwen3-asr-flash-realtime) → 文本
  → 确定性意图路由
  → 日记在线校验 / Reachy 前置 / 车外后视 / 实时定位
  → LLM (百炼 qwen-plus) → 流式输出
  → TTS (百炼 qwen3-tts-flash-realtime) → 扬声器
```

基地车、人物、路线和旅途事件必须有本轮日记证据；没有可靠记录时固定回答不知道。普通常识仍可由模型回答，但不能冒充基地车事实。

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

## License

MIT
