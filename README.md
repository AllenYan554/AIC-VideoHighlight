# AIC-VideoHighlight

全球校园人工智能算法精英大赛“基于视频大模型的通用视频高光剪辑”项目。本仓库当前只实现**高光候选召回**的 Zero-shot baseline 框架。

## 当前范围

高光候选召回负责：读取视频元数据、按重叠时间窗临时切分长视频、调用由 vLLM 托管的 Qwen3.5-4B 视频理解接口、解析秒级局部候选、映射到全局时间、按 Temporal IoU 合并重复候选，并提供粗召回评估工具。

高光候选召回不负责最终成片决策，也不实现边界精修、SAM、Tracking、逐帧 bbox、构图优化、时序框平滑、SFT、LoRA 或 GRPO/RL。`HighlightRetrievalResult` 是内部中间产物，不是比赛最终 `predictions.jsonl` 格式。

> **Highlight retrieval metric != Official Competition Metric**
>
> 当前仅计算秒级区间的 duration-based temporal precision、recall、F1 与 Temporal IoU。官方最终指标仍需 frame exact match、bbox IoU 与 IoU-weighted F 等空间/逐帧信息。

## 固定目录

- 本地开发目录：`D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight`
- AutoDL 运行目录：`/root/autodl-tmp/AIC-VideoHighlight`
- AutoDL 数据目录：`/root/autodl-tmp/datasets`
- Hugging Face / Qwen 缓存：`/root/autodl-tmp/hf-cache`
- 模型目录规划：`/root/autodl-tmp/models`

模型、训练/测试视频、压缩数据集、缓存与大量实验输出不得进入 Git。

## 方案 B 工作流

```text
Local Windows development
  -> git commit
  -> git push（配置远程后）
  -> AutoDL git pull
  -> AutoDL GPU run
```

本仓库没有预设远程地址。本轮只创建本地提交，不 push。

## 本地纯逻辑测试

要求 Python 3.12：

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

这些测试不需要 GPU、vLLM、FFmpeg 或模型权重。FFprobe/FFmpeg 是真实 Pipeline 的运行时边界。

## AutoDL 启动顺序

把仓库拉取到 `/root/autodl-tmp/AIC-VideoHighlight` 后：

```bash
cd /root/autodl-tmp/AIC-VideoHighlight
bash remote/bootstrap_autodl.sh
source .venv/bin/activate
bash remote/serve_qwen_vllm.sh
```

另开一个终端执行：

```bash
source .venv/bin/activate
python scripts/smoke_text.py
python scripts/smoke_video.py --video /root/autodl-tmp/datasets/sample.mp4
python scripts/run_highlight_retrieval.py \
  --video /root/autodl-tmp/datasets/sample.mp4 \
  --config configs/highlight_retrieval.yaml \
  --output outputs/highlight_retrieval_result.json \
  --base-url http://127.0.0.1:8000/v1
```

长视频只会在源视频目录下生成自动清理的临时 MP4 chunk，不会永久拆帧或产生大量 JPG/PNG。源视频应位于 `/root/autodl-tmp/datasets`，以匹配 vLLM 的本地媒体白名单。

## 配置与兼容性说明

默认参数位于 `configs/highlight_retrieval.yaml`：30 秒窗口、5 秒重叠、2 FPS 粗采样、每块最多 5 个候选、256 个输出 token、temperature 0、合并阈值 0.5。

客户端按当前 vLLM OpenAI-compatible 多模态格式发送 `video_url`，本地文件使用 `file://` URL；服务端通过 `--allowed-local-media-path` 限定读取目录。`coarse_fps` 通过 `media_io_kwargs.video.fps` 传入。vLLM 尚未锁版本：必须先在 AutoDL 实机核对 Qwen3.5-4B、`video_url`、`--allowed-local-media-path`、`--media-io-kwargs` 和显存参数，Smoke Test 成功后再固定版本。当前没有声称 GPU 推理已验证。

## 当前下一步

在 AutoDL RTX 4090D 环境依次完成：Qwen3.5-4B 与 vLLM 首次兼容性验证、文本 Smoke Test、视频 Smoke Test，然后对单个样例运行高光候选召回，并在确认 `train.jsonl` 的真实标注语义后才接入评估。
