# 诊断报告：Reachy Mini 唤醒灵敏度、说话摇头与进程清理修复

> 生成于 2026-08-08 · 路径：软件 · 设备：NVIDIA Jetson AGX Orin Developer Kit / Reachy Mini

## 设备信息

- **SoC**：Jetson AGX Orin
- **L4T**：R36.4.4
- **Ubuntu**：22.04.5 LTS
- **Reachy Mini SDK**：1.9.0
- **应用目录**：/home/recomputer/chaihuo_reachy
- **音频**：Reachy Mini Audio hw:0,0（ALSA direct）
- **串口**：/dev/ttyACM0

## 问题描述

皮皮虾唤醒需要大声喊；TTS 说话时头部动作只持续一两下、提前结束或几乎不可见；启动脚本在非交互 SSH 环境找不到 SDK daemon；Ctrl+C 后曾遗留 daemon 并让机器人保持站立。

## 根因分析

### 假设 1：KWS 阈值与麦克风增益过于保守
- **依据**：原默认阈值 0.35、score 1.0、mic gain 1.0；真机输入链路可用，模型自检通过。
- **验证方法**：加载 KWS 后确认 threshold=0.20、score=1.50，并通过合成唤醒词和真实 USB 麦克风链路检查。

### 假设 2：说话动作错误地按最后一个 TTS 分块的短期限结束
- **依据**：TTS 会提前排队数秒 PCM，旧逻辑在最后回调后约 1.2 秒停止动作，因此声音继续而头已停止。
- **验证方法**：把生命周期绑定到真实播放队列排空，并逐样本比较 speech_audio_playing 与 talk_motion_active。

### 假设 3：高频 set_target 流被实体颈部控制器滤掉
- **依据**：软件线程持续运行但初次真机采样每段实际 yaw 变化仅约 2–6°；SDK min-jerk goto 探针可稳定达到约 +11°/-12°。
- **验证方法**：改用完整的左右 min-jerk 轨迹后，启动问候前中后三段均达到约 25° 峰峰值，动作全程为 active。

### 假设 4：非交互 PATH 与子进程所有权处理不完整
- **依据**：SSH/nohup PATH 不含 .venv/bin，daemon 可执行文件存在但 shutil.which 找不到；旧退出路径可能未回收子进程。
- **验证方法**：启动器从 sys.executable 同目录寻找 reachy-mini-daemon；SIGINT 验证 daemon 执行 Putting Reachy Mini to sleep，两个端口均释放。

## 解决方案

### 步骤 1：调整唤醒与回声插话参数
```bash
REACHY_MIC_GAIN=1.5
REACHY_KWS_THRESHOLD=0.20
REACHY_KWS_SCORE=1.50
REACHY_BARGE_IN_SENSITIVITY=0.04
```
> 提高唤醒灵敏度，同时避免扬声器回声被当成用户插话。

### 步骤 2：将动作生命周期绑定到真实语音播放
```bash
cd /home/recomputer/chaihuo_reachy && .venv/bin/python scripts/verify_talk_motion_runtime.py
```
> 首个 PCM 前启动动作，播放队列完全排空后才停止，并区分普通提示音和真实 TTS。

### 步骤 3：改用可见且匹配语速的平滑左右轨迹
```bash
# yaw target: 14-18 degrees; half-sway cadence: 0.42-0.48 seconds
```
> 只转头部 yaw，不驱动身体、不触发舞蹈；完整周期约 0.9 秒并带轻微速度变化。

### 步骤 4：运行远程回归并保留服务
```bash
cd /home/recomputer/chaihuo_reachy && ~/.local/bin/uv run --extra dev pytest -q
```
> Jetson 端 133 项通过；当前 dashboard 与 owned daemon 正常运行。

## 风险与回滚

**风险**：
- 继续提高 yaw 幅度或缩短半摆时间可能显得机械，并增加颈部执行器负担。
- REACHY_BARGE_IN_SENSITIVITY 过低会把扬声器回声当成用户插话。
- 语雀日记仍有 1 篇因上游 401 Unauthorized 无法同步，此问题与本次运动/唤醒修复无关。

**回滚**：
- 远程备份位于 /home/recomputer/chaihuo_reachy/state/backups/2026-08-08-talk-motion/。
- 回退节奏可恢复 motion.before-faster-cadence.py；回退可见幅度可恢复 motion.before-visible-sway.py。
- 回退完整说话轨迹可恢复 motion.before-goto-talk-sway.py 与对应 engine 备份。

## 引用资源

- **source**：/Users/chenzibo/data/project/Jetson/chaihuo_reachy/src/chaihuo_reachy/motion.py
- **source**：/Users/chenzibo/data/project/Jetson/chaihuo_reachy/src/chaihuo_reachy/engine.py
- **runtime verifier**：/Users/chenzibo/data/project/Jetson/chaihuo_reachy/scripts/verify_talk_motion_runtime.py
- **tests**：/Users/chenzibo/data/project/Jetson/chaihuo_reachy/tests/test_motion.py

---

_本报告由 jetson-troubleshooter skill 生成。_