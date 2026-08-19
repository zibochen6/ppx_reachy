# 诊断报告：Jetson Reachy Mini 对话长时间思考后空回复

> 生成于 2026-08-14 · 路径：软件 · 设备：reComputer Jetson AGX Orin

## 设备信息

- **SoC**：Jetson AGX Orin
- **JetPack**：6.2
- **Python**：3.10
- **service**：chaihuo-reachy.service active
- **runtime**：robot_ready=true, daemon_health=healthy
- **LLM**：qwen3.7-plus
- **validation**：269 tests passed; Dashboard first delta 1.13s, final 2.94s

## 问题描述

Dashboard 最近 11 轮对话中有 6 轮助手消息被保存为 status=done 但 text 为空。空回复每次在页面停留约 60 秒，随后日志出现 Qwen Realtime TTS did not finish within 60s。另外，Web 文字消息会被被动唤醒词监听占用的 turn lock 阻塞最多约 60 秒。两条路径均已修复并部署。

## 根因分析

### 假设 1：Responses API 的 SSE 解析器只接受 data: 后带空格的格式，漏掉百炼实际返回的 data:{...}
- **依据**：远端原始响应使用 data:{JSON}；llm_client.py 仅判断 line.startswith("data: ")，因此 response.failed 和 output_text 事件都被忽略。相同问题经现有 response_stream 得到 0 字，经 Chat API 得到 15 字。
- **验证方法**：用只读诊断请求记录 SSE 行前缀；确认出现 event:response.failed 与 data:{...}，而不是 data: {...}。

### 假设 2：Responses 请求参数组合无效：enable_thinking=false 时仍启用了 web_extractor
- **依据**：百炼原始 response.failed 明确返回 InvalidParameter: Normal mode does not support web_extractor. Please set enable_thinking to true。移除 web_extractor、保留 web_search 并兼容 data: 解析后，1.47 秒返回 59 字。
- **验证方法**：分别发送 web_search+web_extractor 与仅 web_search 的诊断请求，前者失败、后者正常产生 response.output_text.delta。

### 假设 3：空 LLM 输出仍无条件启动并 flush TTS，放大为 60 秒假死
- **依据**：engine.py 在任何首 token 出现前创建 TTS worker，队列为空也调用 flush；tts_client.py 的 finish 等待 complete_event 60 秒。日志在 15:12:26、15:16:49、15:18:07、15:19:19、15:20:25 重复超时。
- **验证方法**：构造零 token 的 LLM 流并观察 TTS；当前实现会等待 60 秒，修复后应直接返回可见错误或回退回答。

### 假设 4：被动唤醒词监听长期持有 turn lock，导致 Dashboard 消息排队
- **依据**：实机第二轮 Dashboard 验收在 wake_listening 期间等待约 25 秒仍未完成；代码显示 _run_turn 在 _listen_for_speech 外层持有 _turn_lock，而唤醒超时可达 60 秒。
- **验证方法**：将被动监听移出响应锁并允许 Dashboard 取消监听后，实机首字 1.13 秒、整轮 2.94 秒完成。

## 解决方案

### 步骤 1：兼容标准 SSE 的可选空格
```bash
将两个 Responses 流解析点统一改为接受 line.startswith("data:")，并用 line[5:].lstrip() 解析 JSON；补充 data:{...} 与 data: {...} 两种单元测试。
```
> 已实施；response.failed 和 output_text 事件均可正确解析。

### 步骤 2：修正联网工具参数
```bash
在 enable_thinking=false 模式下只发送 web_search；如确需 web_extractor，则显式启用 thinking 并验证延迟与输出协议。
```
> 已实施；目标 Jetson 真实 Responses 请求 0.97 秒返回28字。

### 步骤 3：为零文本输出增加快速失败和 Chat 回退
```bash
LLM 流结束且未产生 token 时抛出明确异常或回退 chat/completions；TTS 仅在首个非空文本块出现后建立/flush 会话。
```
> 已实施；零 token 回归测试确认不会创建或 flush TTS。

### 步骤 4：让 Dashboard 抢占被动语音监听
```bash
唤醒词/ASR 等待不再持有 turn lock；Dashboard 消息先取消被动监听，再独占执行 LLM 与 TTS。
```
> 已实施；端到端验收首字1.13秒、整轮2.94秒。

### 步骤 5：单独处理左天线电机过载
```bash
在断电并确认天线无机械卡阻后检查 left_antenna；恢复前避免继续驱动该关节。
```
> 硬件安全项，与空回复根因无关；日志当前每秒报告 Overload Error。

## 风险与回滚

**风险**：
- 直接把 enable_thinking 改为 true 可能增加响应延迟，并改变流式事件行为。
- 若未来替换 TTS SDK，应保留零输入不建会话的回归测试。
- 左天线持续过载可能造成电机发热或进一步机械损伤。

**回滚**：
- 应用修复前备份 llm_client.py、engine.py 和 tts_client.py；测试失败时恢复备份并重启服务。
- 联网路径可临时回退到 chat/completions 或将 search_policy 设为 off，以恢复基础对话。
- 天线检查前可停止运动相关功能并让机器人进入安全姿态。

## 引用资源

- **software-patterns**：`references/sw-diagnostic-patterns.md`
- **jetpack-knowledge**：`references/jetpack-knowledge.md`

---

_本报告由 jetson-troubleshooter skill 生成。_