# 高光候选召回设计

## 目标与边界

高光候选召回不是最终剪辑器。优化方向是 Recall 优先，允许把不确定但可能有价值的事件交给后续边界精修和过滤。

## 数据流

```text
Video
  -> FFprobe VideoMeta
  -> overlapping VideoChunk list
  -> temporary MP4 chunk (only for long videos)
  -> QwenVLLMClient / video_url
  -> strict JSON parser
  -> chunk-local seconds + chunk.start_sec
  -> global HighlightSegment list
  -> Temporal-IoU deduplication
  -> internal HighlightRetrievalResult
```

模型必须返回当前 chunk 内的相对秒数。Pipeline 独占局部到全局的转换：

```text
global_start = chunk.start_sec + local_start
global_end   = chunk.start_sec + local_end
```

模型不负责计算全局时间，也不输出帧号。

## 解析策略

解析器接受严格 JSON，以及完整包裹 JSON 的单个 Markdown JSON fence；不会从自然语言中抓取数字。负起点、逆序区间、无穷数和越界 score 会拒绝。合法起点但终点超出 chunk 时会裁剪到 chunk 末端，并把裁剪信息追加到 `reason`。

## 合并策略

候选先按起止时间与 score 稳定排序。相邻候选 Temporal IoU 大于等于配置阈值时：起止取时间并集，score 取最大值，原因按稳定顺序去重拼接；跨 chunk 合并后 `source_chunk` 置空。默认阈值为 0.5。

## 指标声明

高光候选召回仅提供区间集合去重后的持续时间交集、并集、Temporal IoU、duration-based temporal precision、recall 与 F1。这些不是官方最终比赛分数。官方指标仍需逐帧精确匹配、bbox IoU 与 IoU-weighted F。

## 尚待 AutoDL 实测

- Qwen/Qwen3.5-4B 在选定 vLLM 版本上的视频输入支持。
- `file://` + `--allowed-local-media-path` 的实际行为。
- `media_io_kwargs.video.fps` 与 2 FPS 粗采样行为。
- RTX 4090D 24GB 的显存占用、最大上下文与稳定参数。
- FFmpeg stream-copy 临时 chunk 的关键帧边界误差是否影响粗召回；若影响，再小步替换为受控重编码。
