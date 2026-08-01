# 诊断报告：Reachy Mini 启动、视觉抓拍、站立、跳舞与日记检索异常

> 生成于 2026-07-31 · 路径：复合（应用启动时序 + USB 硬件枚举 + 日记数据链路） · 设备：Reachy-Mini-on-macOS

## 设备信息

- **开发主机**：macOS
- **Reachy SDK**：reachy-mini 1.9.0
- **进程模型**：仅使用 reachy-mini SDK daemon，不自动启动桌面 Control 应用
- **当前硬件状态**：daemon 报告电机通信错误且 backend 未运行；遗留 daemon 已停止
- **日记状态**：2026-07-30 正文已完整下载并索引

## 问题描述

应用启动 Reachy SDK daemon 后连接时序不稳定，失败分支会误杀仍在初始化的 8000 端口进程；Ctrl+C 又只关闭主程序，没有让机器人休眠或停止脱离父进程会话的 daemon。启动链路还曾包含不必要的 macOS Control GUI 兜底，现已移除，统一由 SDK 管理 daemon。视觉问句“你能看到什么”因词表漏掉“能”而被误判为普通聊天，导致没有抓拍也生成摄像头画面描述。相对日期回答会被模型改写成相邻日期；日记同步又会因聚合页滞后或单篇 401 将其他已验证正文全部判为不可用。

## 根因分析

### 假设 1：daemon 首次连接超时后被旧逻辑误杀
- **依据**：SDK 先启动子进程再连接；旧代码仅等待约 5 秒，随后按端口 SIGKILL。Reachy 初始化电机和媒体管线可能超过首次连接超时。
- **验证方法**：检查修复后的启动日志：应只出现一次拉起动作，并持续轮询就绪；代码中不再存在按 8000 端口杀进程的分支。

### 假设 2：当前实机 USB 设备未被 macOS 枚举
- **依据**：真实 daemon 日志报告 No Reachy Mini serial port found、No camera found、No Reachy Mini Audio Source card found；system_profiler 未找到相应 USB 设备。
- **验证方法**：重新插紧 Reachy USB 数据线和电源后运行 system_profiler SPUSBDataType，并确认串口、摄像头和音频设备出现。

### 假设 3：日记目录源滞后且全局完整性门槛过严
- **依据**：2026-07-30 日记已在语雀发布并下载，但聚合页仍滞后；知识库另有一个目录项返回 401，旧引擎因此拒绝其余 53 篇完整正文。
- **验证方法**：查询“昨天发生了什么”应命中 2026-07-30 的 cecmz7p6dyevu4pm；查询“大前天”应命中 2026-07-28。

### 假设 4：Ctrl+C 未覆盖机器人与 daemon 生命周期
- **依据**：SDK 原生 spawn_daemon 使用 start_new_session=True；主程序退出不会向该进程传递 Ctrl+C，旧清理逻辑也没有调用 goto_sleep 或持有精确的子进程句柄。
- **验证方法**：退出测试必须按顺序观察到 goto_sleep、media close、owned daemon terminate/wait，并确认 8000 和 8640 端口均无监听。

### 假设 5：视觉问句路由漏词且幻觉拦截规则过窄
- **依据**：“你能看到什么”未命中 FRONT_CAMERA，仅“你看到什么”能命中；普通聊天回复“我正用摄像头看着”也未命中旧的未抓拍视觉声明正则。
- **验证方法**：该问句必须调用一次前置抓拍；黑帧应直接返回“画面太暗”，普通聊天中的摄像头所见声明应被硬拦截。

## 解决方案

### 步骤 1：确认 Reachy USB 数据与供电连接
```bash
system_profiler SPUSBDataType | grep -i -C 3 'reachy\|pollen\|xmos'
```
> 插拔前避免机器人运动区域有障碍；必须使用可传数据的 USB 线。

### 步骤 2：启动 Dashboard 并观察 daemon 就绪状态
```bash
uv run chaihuo-reachy dashboard -v
```
> 修复后会只拉起一次 daemon、等待最多 30 秒、识别终止性硬件错误，并对站立重试三次。

### 步骤 3：验证日记与相对日期
```bash
uv run pytest -q tests/test_engine.py tests/test_journals.py
```
> 7 月 30 日正文已缓存；单个私有目录项的 401 不再阻断其他逐篇验证过的正文。

### 步骤 4：验证 Ctrl+C 完整清理
```bash
uv run pytest -q tests/test_sdk_contract.py && lsof -nP -iTCP:8000 -sTCP:LISTEN
```
> 程序显式持有自己启动的 daemon 进程；所有退出路径先休眠机器人，再关闭媒体并停止该精确进程。外部已有 daemon 不会被误杀。

### 步骤 5：验证视觉问句会抓拍且拒绝黑帧
```bash
uv run pytest -q tests/test_intent.py tests/test_engine.py tests/test_camera_service.py
```
> “你能看到什么”会进入 FRONT_CAMERA；没有新鲜帧或画面过暗时不会调用视觉模型编造物体。

## 风险与回滚

**风险**：
- 站立和跳舞会驱动头部、身体与天线，执行前清理机器人周围空间。
- 不要再按端口号强杀未知进程；修复后的代码已移除该行为。
- 当前未检测到 USB 硬件时不可强行动作。

**回滚**：
- 如新启动流程异常，可使用 --standalone 跳过 daemon，仅运行本地音视频模式。
- 本次发现的孤儿 daemon 已先请求休眠，但因 backend 未运行无法执行；随后仅对明确的遗留 PID 发送 SIGTERM 并停止，未删除任何数据。

## 引用资源

- **软件诊断模式**：`references/sw-diagnostic-patterns.md`
- **项目测试**：/Users/chenzibo/data/project/Jetson/chaihuo_reachy/tests
- **日记清单**：/Users/chenzibo/data/project/Jetson/chaihuo_reachy/data/journals/manifest.json

---

_本报告由 jetson-troubleshooter skill 生成。_