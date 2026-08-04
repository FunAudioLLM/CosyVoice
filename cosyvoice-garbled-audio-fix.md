# CosyVoice 杂音/杂英语音故障复盘

## 结论

LiveKit 房间中听到的杂音或类似杂英语音，根因不在 LiveKit 编解码、zero-shot 参照音频或 prompt 文本，而在 CosyVoice HTTP 服务使用了原生 PyTorch 推理路径。将服务切换为已验证的 `vLLM + TensorRT` 路径后，项目 HTTP 输出与正常对照音频完全一致，实际房间测试的语音质量满足使用要求。

修复提交：

- `29eb6b3 feat: add CosyVoice TTS and Moss fallback`（部署配置）
- `1630825 feat: add CosyVoice TTS backend diagnostics`（Agent 音频边界诊断）

## 问题现象

- 浏览器访问手工 CosyVoice WebSocket 服务时，声音正常。
- LiveKit Agent 调用项目中的 HTTP TTS 服务后，房间中语音无法辨认。
- 将 HTTP 响应保存为 WAV 后，未进入 LiveKit 前的 `02-http-short-segment.wav` 已经异常，说明问题发生在 TTS 推理或 HTTP 封装之前，而不是房间传输阶段。

## 排查过程

### 1. 对齐 zero-shot 输入

手工 WebSocket 服务与项目 HTTP 服务使用完全相同的：

- 模型目录：`/models/Fun-CosyVoice3-0___5B-2512`
- 参照音频：`zero_shot_prompt.wav`
- prompt 文本：

  ```text
  You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。
  ```

WebSocket 服务的 `ready` 事件会回显完整 prompt，确认 `<|endofprompt|>` 与标点没有丢失。因此参照音频、prompt 文本及其特殊分隔符不是故障来源。

### 2. 在各组件边界保存音频

Agent 新增了由 `TTS_DEBUG_DUMP_DIR` 控制的诊断落盘。每个分句都会保存：

- `*-text.txt`：送入 TTS 的文本。
- `*-http.wav`：HTTP TTS 原始响应。
- `*-decoded-pcm.wav`：解析 WAV 后的 PCM16。
- `*-pre-livekit.wav`：推送到 LiveKit 前的 PCM16。

对同一个分句，HTTP WAV、解析后的 PCM 和送入 LiveKit 前的 PCM 内容一致。由此排除了 WAV 解码、声道处理和 Agent 到 LiveKit 的 PCM 封装是首个失真点的可能。

### 3. 单变量对照

使用文本 `后退、左转。` 生成以下样本：

| 样本 | 推理路径 | 人工试听 |
| --- | --- | --- |
| `02-http-short-segment.wav` | 原生 PyTorch，`load_vllm=false`、`load_trt=false`、`fp16=true` | 异常 |
| `03-http-vllm-trt-short-segment.wav` | HTTP，`load_vllm=true`、`load_trt=true`、`fp16=false` | 正常 |
| `04-websocket-vllm-trt-streaming-short-segment.wav` | WebSocket 流式 vLLM + TensorRT | 正常 |
| `05-http-project-vllm-trt-short-segment.wav` | 修复后的项目 HTTP 服务 | 正常 |

`03` 与 `05` 的 SHA-256 完全相同，证明项目 HTTP 服务切换后与已确认正常的对照路径输出了相同音频字节。

## 根因

原始 Compose 配置将 CosyVoice 固定在：

```text
COSYVOICE_LOAD_VLLM=false
COSYVOICE_LOAD_TRT=false
COSYVOICE_FP16=true
```

该原生 PyTorch 路径在当前模型、驱动和 RTX 2080 Ti 环境下产生了异常语音。正常对照使用 vLLM 生成语音 token，并使用 TensorRT 执行 flow decoder。

模型目录只有 FP32 TensorRT 引擎：

```text
flow.decoder.estimator.fp32.mygpu.plan
```

因此不能只把 `COSYVOICE_FP16` 打开。CosyVoice 会在 FP16 模式查找或构建不同的 TensorRT 引擎；CosyVoice3 源码也对 DiT FP16 TensorRT 引擎提示存在性能问题。

## 最终修复配置

默认 Compose 配置位于 `docker-compose.yml`：

```text
COSYVOICE_LOAD_VLLM=true
COSYVOICE_LOAD_TRT=true
COSYVOICE_FP16=false
COSYVOICE_SAMPLE_RATE=24000
VLLM_USE_FLASHINFER_SAMPLER=0
FLASHINFER_ENABLED=0
VLLM_USE_V1=1
VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

最后四项是 RTX 2080 Ti（Compute Capability 7.5）的兼容配置：禁用 FlashInfer，并强制 vLLM 使用 Triton attention。vLLM 的语言模型会因该显卡不支持 BF16 而自动回退到 FP16；这里的 `COSYVOICE_FP16=false` 仅保留已验证的 FP32 TensorRT 声学解码器。

## 启动与切换

设定环境文件：

```bash
ENV_FILE=envs/simulation_remote_livekit.env
```

默认使用 CosyVoice：

```bash
docker compose --env-file "$ENV_FILE" \
  up -d --force-recreate --no-build --remove-orphans \
  sensevoice-asr cosyvoice-tts agentic-platform livekit-agent
```

使用 Moss 作为回退：

```bash
docker compose --env-file "$ENV_FILE" \
  -f docker-compose.yml -f docker-compose.moss-fallback.yml \
  up -d --force-recreate --no-build --remove-orphans \
  sensevoice-asr moss-tts agentic-platform livekit-agent
```

Moss 覆盖会将 Agent 指向 `moss-tts:9100`，把 `cosyvoice-tts` 标记为非必需依赖，并通过 Compose 的 `!reset` 清空 CosyVoice 的端口映射，避免两个服务同时绑定 `9100`。

## 验证方法

### 配置验证

```bash
bash scripts/test-production-compose.sh
```

该脚本验证 CosyVoice 默认配置、GPU 保留、Moss 覆盖的 TTS URL、依赖关系与端口清空行为。

### 直接生成 WAV

```bash
python3 scripts/cosyvoice-tts-test.py "机器人已收到指令。"
```

输出 WAV 可作为 HTTP 层快速试听样本。

### 房间问题复现时的取证顺序

1. 在 `artifacts/tts-debug/` 找到最新同一 utterance 的四类文件。
2. 先听 `*-http.wav`：异常则排查 CosyVoice 推理配置或输入。
3. 若 HTTP WAV 正常，再对比 `*-decoded-pcm.wav` 与 `*-pre-livekit.wav`：两者应保持一致。
4. 仅当前三级都正常、房间仍异常时，再检查 LiveKit 发布/订阅端的采样率、播放设备和网络状态。

## 已知限制

- RTX 2080 Ti 不支持 FP8 Tensor Core，不能将本服务切换到 FP8 推理。
- 不建议在当前模型上启用 `COSYVOICE_FP16=true`：本地没有预构建 FP16 TensorRT 引擎，且 CosyVoice3 对该路径有性能风险提示。
- 诊断目录只用于排查，不应提交；`.gitignore` 已忽略 `artifacts/` 和 `test-output/`。
