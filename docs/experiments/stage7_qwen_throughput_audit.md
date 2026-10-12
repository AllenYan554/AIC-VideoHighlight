# Stage 7 / PHD² Qwen Throughput Optimization Audit (Round 2)

```text
COMPUTE_STATUS       : COMPLETE (AutoDL RTX 4090D, real vLLM 0.28.0 service)
LOCAL_ARCHIVE_STATUS : COMPLETE
AUDIT_DATE           : 2026-09-22
FULL1500_EXECUTION   : NOT STARTED
RTDETR_RERUN         : NO (round-1 one-pass decode kept as-is)
SCIENCE_SCOPE        : PURE_EXECUTION / SYSTEM_OPTIMIZATION (no numerical change)
GIT_COMMIT / PUSH    : NO
```

## 1. Identity and frozen semantics

Nothing in this round changed: Qwen model identity (`Qwen/Qwen3.5-4B`,
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), prompt
(`high_recall_retrieval_v0`), sampling fps (2.0), chunk duration (30 s), overlap
(5 s), input frames/timestamps, temperature (0), thinking (false), max token
limits (512 primary), parser, merge semantics, retry parameters, output artifact
schema, or the frozen manifests. The only promoted change is where the CPU-side
ffmpeg render runs relative to an in-flight request.

Server configuration (unchanged, read from the live process log):

```text
vllm 0.28.0 | dtype bfloat16 | max_model_len 32768 | gpu_memory_utilization 0.80
--allowed-local-media-path /root/autodl-tmp/datasets/phd2_benchmark32
KV cache 227,001 tokens (7.72 GiB) | decoder/encoder cache 16,384 tokens
enable_chunked_prefill True (default) | enable_prefix_caching True (default)
max_num_batched_tokens 2048 (default) | max_num_seqs 128 (default)
enforce_eager False (CUDA graphs on) | mm_processor_cache_gb 4 (LRU)
FlashAttention 2 | FlashInfer sampler | GDN prefill Triton / decode CUDA
```

## 2. Method: frozen audit subset

- Manifest: `benchmark_manifests/qwen_determinism8.json`
  SHA256 `31ce2961a4d6c7201b589bff39c9213883a1ae37a157a4cecd407956a48bf5d8`
- 8 videos / 114 chunks, covering 0-1 / 1-3 / 3-10 / 10-30 min bins, 360p and
  720p arms, AV1 and VP9/H.264 codecs, 1-chunk and 42-chunk videos
- Every arm started a **fresh vLLM process** (cold MM/prefix cache) so wall-clock
  comparisons are not biased by warm caches
- All audit runs write to fresh roots; no cached artifact was reused

## 3. Qwen c=1 self-determinism (A/B/C)

| Comparison | Chunks | Raw response exact match | Chunk semantic match | Video candidate match |
|---|---:|---:|---:|---:|
| c1_a vs c1_b | 114 | 114/114 = 1.000 | 1.000 | 8/8 = 1.000 |
| c1_a vs c1_c | 114 | 114/114 = 1.000 | 1.000 | 8/8 = 1.000 |
| c1_b vs c1_c | 114 | 114/114 = 1.000 | 1.000 | 8/8 = 1.000 |

Token accounting is identical in all three runs (prompt 909,514; completion
30,341) and all three artifact sets equal the promoted Baseline32 artifacts
(8/8 videos) produced by the untouched production code path in a different
session. `C1_SELF_DETERMINISM = PASS`.

Wall-time variance between identical c=1 runs is CPU-side only: summed render
473.0 / 373.2 / 387.8 s vs stable request totals 609.4 / 625.1 / 607.3 s
(+/-1.5%).

## 4. Concurrency interpretation

Because c=1 is byte-stable across sessions, the previous round's
`c=2/4/8 != c=1` rejections cannot be explained by run-to-run noise: the only
remaining variable is batch composition. Examples from the round-1 sweep show the
same chunk returning different scores and reason text at different concurrency
levels, so concurrency changes model numerics, not just scheduling.
`temperature = 0` decoding is greedy, and vLLM's `seed` (`SamplingParams.seed`,
present in 0.28.0) only affects stochastic sampling; a fixed seed therefore
cannot restore equivalence. **Concurrency stays closed at 1.**

## 5. Qwen wall-time breakdown

Per chunk, Benchmark32 promoted arm (426 chunks, 5.3247 s/chunk mean wall) and
the serial c=1 arm (114 chunks, 5.483 s/chunk request + 3.274 s/chunk render):

| Bucket | Serial c=1 (s/chunk) | Promoted (s/chunk) | Measurement |
|---|---:|---:|---|
| A/B chunk extraction / ffmpeg render | 3.274 | 3.228 (overlapped) | client timer around the frozen renderer |
| C/D server input processing (MM decode, sample, resize, tokenize, detokenize) | 2.024 | 1.638 | server `e2e` minus engine inference |
| E/F/J client serialization + HTTP + receive | 0.032 | 0.026 | client request minus server `e2e` |
| G queue wait | 0.00006 | 0.00006 | `vllm:request_queue_time_seconds` |
| H prefill | 0.782 | 0.703 | `vllm:request_prefill_time_seconds` |
| I decode / generation | 2.645 | 2.693 | `vllm:request_decode_time_seconds` |
| K JSON parse | 0.000004 | 0.0001 | client timer |
| L candidate merge | 0.000001 | 0.0000 | client timer |
| M retry overhead | 0 (subset) | 1 extra request (e6UI4KJNEOs) | audit sidecar |

Derived: engine (GPU) share of the request wall 62.5% -> 67.2%; prompt tokens
7,978/chunk, completion 266/chunk; no queueing and no preemption in any arm.

## 6. Root cause of the "36 retries"

Only one Baseline32 video needed recovery: `e6UI4KJNEOs` (862.6 s, 35 chunks).

- Chunk 33 deterministically returns non-JSON at temperature 0
  (`finish_reason=stop`, no truncation); the documented temperature-0.3 retry
  recovers it (1 extra request).
- The round-1 recovery path re-ran the **entire 35-chunk video** rather than the
  failed chunk: the recovery log contains 35 `primary` attempts + 1
  `parse_retry_temp`, i.e. the "36 retries" statistic is one whole-video re-run.
- Cost: first (failed) pass 241.2 s + recovery pass 231.0 s = 472.2 s for a video
  whose chunk work is 231 s. Avoidable waste ~227 s = 6.6% of the 3418.96 s
  Baseline32 Qwen wall.
- The audited execution path retries the failed chunk in place. Re-running
  `e6UI4KJNEOs` with it reproduced the promoted artifact byte-for-byte with
  exactly 1 extra request.

## 7. Promoted optimization: one-chunk render look-ahead (concurrency = 1)

Implementation: `src/aic_video_highlight/ftnet/qwen_execution.py::execute_chunks`.
Chunk *N+1* is rendered in a single background worker while the request for chunk
*N* is in flight. Requests stay strictly serial and in chunk order, on the same
frozen chunk files (the renderer is byte-deterministic; two renders of the same
chunk produce identical SHA256).

| Arm | Execution | Chunks | Summed video wall | s/chunk | Speedup |
|---|---|---:|---:|---:|---:|
| c1_a / c1_b / c1_c | serial c=1 | 114 | 1083.8 / 999.7 / 996.3 | 9.507 / 8.769 / 8.740 | 1.000x |
| c1_p | render look-ahead | 114 | 655.6 | 5.751 | **1.653x vs c1_a, 1.566x vs the c=1 mean** |
| bench32_p | render look-ahead | 426 | 2268.3 | 5.325 | **1.507x vs Baseline32 (3418.96 s)** |

Correctness gates:

- 114/114 chunks byte-identical vs c1_a, c1_b and c1_c; 8/8 videos identical
- 426/426 chunks / 32/32 videos semantically identical vs the promoted
  Baseline32 artifacts (including the recovery video)
- no OOM, no error, no new retry; VRAM 21.1 GiB peak, power mean ~163 W,
  GPU util mean 36.7% (serial) -> higher under the overlapped arm

The render is no longer additive: total render-wait across Benchmark32 is 8.4 s
out of 2268 s, i.e. 99.6% of the ffmpeg work is hidden.

Host telemetry (2 s sampler, arm windows):

| Arm | GPU util mean | samples GPU>80% | GPU power mean | VRAM peak | CPU util mean |
|---|---:|---:|---:|---:|---:|
| c1_a / c1_b / c1_c serial | 36.7 / 35.7 / 35.5% | 35 / 35 / 34% | 163 / 161 / 163 W | 21,135 MiB | 25.2 / 37.9 / 35.4% |
| c1_p look-ahead (8 videos) | 51.5% | 50% | 209 W | 21,135 MiB | 22.8% |
| bench32_p look-ahead (32 videos) | 61.6% | 61% | 239 W | 20,913 MiB | 34.7% |

Promotion record

| Gate | Result |
|---|---|
| semantic equivalence | PASS (114/114 chunks, 8/8 videos; 426/426 chunks, 32/32 videos vs Baseline32) |
| new errors | none (no exception, no OOM, no new retry beyond the pre-existing e6UI4KJNEOs chunk) |
| OOM | none (peak VRAM 20.9-21.1 GiB of 24 GiB) |
| reproducible | yes (deterministic outputs; wall variance only in the CPU render bucket) |
| wall-time gain | 1.653x (8-video subset vs slowest control), 1.507x (full Benchmark32, official basis) |

## 8. Full1500 Qwen ETA (extrapolation, Full1500 not run)

Basis: 20,627 chunks from the frozen 1500-record manifest, scaled by the measured
Benchmark32 rate (same 30 s / 5 s formula).

| Configuration | s/chunk | Qwen Full1500 | Speedup |
|---|---:|---:|---:|
| serial c=1 (round-1 basis 3404.337 s / 426 chunks) | 7.991 | 45.79 h | 1.000x |
| serial c=1 (artifact-wall basis 3418.961 s) | 8.026 | 45.99 h | 1.000x |
| promoted render look-ahead (2268.303 s / 426 chunks) | 5.325 | **30.51 h** | **1.507x** |

Promoted Full1500 Qwen composition (extrapolated): client request 29.04 h
(server decode 15.47 h, server input processing 9.39 h, prefill 4.04 h) plus
~1.47 h of probe/parse/merge and residual render overflow. The request path is
the floor for the c=1 discipline; the promoted arm reaches 95.2% of it.

## 9. Corrected speedup nomenclature

| Scope | Baseline | Optimized | Ratio | Status |
|---|---:|---:|---:|---|
| RT-DETR / detection sub-chain (decode+detect+write) | 2932.430 s | 491.151 s | **5.971x** | measured single run, detection only |
| Detection run wall incl. model load/orchestration | 2937.754 s | 496.491 s | 5.917x | measured single run, detection only |
| RT-DETR decode only | 2614.537 s | 227.456 s | 11.496x | measured, fixed subset + full Benchmark32 |
| Whole Qwen -> RT-DETR upstream | 6351.391 s | 2759.454 s | **2.302x** | `NOT_DIRECTLY_MEASURED_COMPONENT_WISE_SUM` |

The previously reported "5.917x overall end-to-end speedup" was the
detection-only run-wall ratio (both benchmark roots served Qwen from cache,
`retrieval.wall_seconds` 0.036 s / 0.032 s); the regenerated report labels it
accordingly and adds the component-wise whole-upstream figure.

## 10. Non-findings (measured, no change needed)

- **Repeated chunk decode:** none. ffmpeg decodes exactly the chunk window
  (mean decode amplification 0.999844 over 63 chunks of 3 videos, 360p/720p,
  VP9/AV1/H.264); render cost is 3.0 s per 30 s chunk at 360p and 7.0-9.0 s at
  720p.
- **AV1 compatibility transcode:** one video, once per run, 1.3 s (0.04% of the
  Qwen wall); multi-chunk AV1 videos never use it. A transcode cache is not
  justified.
- **HTTP/client overhead:** the client is persistent per video (keep-alive
  connection reuse); client-observed request time exceeds server `e2e` by
  26-32 ms/chunk (0.6% of the request wall). A shared pooled client would not
  measure here.
- **Server-side configuration:** queue wait is 0.00006 s/chunk with zero
  preemptions, so no scheduler setting can raise c=1 throughput. Decode already
  runs at ~100 tokens/s = ~92% of the RTX 4090D memory-bandwidth roofline
  (9.25 GB bf16 weights / 1008 GB/s = 109 tokens/s), so engine parameters and
  CUDA-graph settings cannot speed up generation. Prefix caching and the MM
  processor cache record 0 hits because every chunk is unique content; disabling
  them is numerics-neutral but expected to gain ~0, so no server parameter was
  promoted. Server-side A/B tests were intentionally not run to avoid spending
  GPU time on a measured non-bottleneck.

## 11. Remaining safe work / forbidden directions

- Safe but unmetered: nothing material. The c=1 floor is the request wall
  (29.04 h of the 30.51 h estimate); closing the last 1.47 h would require
  overlapping API-side input processing, which is the concurrency route.
- `FUTURE_NUMERICAL_OR_SCIENTIFIC_OPTIMIZATION` (not implemented, would change
  model inputs or outputs): GPU/NVDEC video loader or a different media backend,
  `mm_processor_kwargs` device moves, quantization/FP8, speculative decoding,
  reduced sampling fps or resolution, smaller chunk/overlap, prompt or token
  limit changes, and any concurrency > 1.

## 12. Full1500 gate

Full1500 execution is **not** started. There is no outstanding engineering
blocker for the Qwen stage: the promoted execution is deterministic, reproducible,
gated and measured at 30.51 h. The single blocker for starting Full1500 is
operational rather than technical: a ~37 h (Qwen 30.5 h + RT-DETR 6.4 h +
postprocess 0.2 h) uninterrupted 4090D budget plus local-only handling of the
official test set.
