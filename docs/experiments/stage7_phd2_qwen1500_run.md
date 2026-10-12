# Stage 7 / PHD² Full1500 Qwen Retrieval Production Run

```text
RUN_ID               : phd2_qwen1500_202609222118
STATUS               : RUNNING (launched 2026-09-22 21:17:48 Asia/Shanghai)
STAGE SCOPE          : QWEN RETRIEVAL ONLY (no RT-DETR, no detection, no materialization)
GIT_COMMIT / PUSH    : NO / NO
```

## 1. Cloud disk cleanup (before upload)

| Item | Value |
|---|---|
| CLEANUP_BEFORE | 24 GiB used / 27 GiB available on `/dev/md0` (50 GiB) |
| deleted | `/root/autodl-tmp/tmp/yt154_raw.tar` (4.1 GiB, stale Sep-18 staging tar; extracted raw verified 154/154 identical) |
| deleted | `/root/autodl-tmp/outputs/phd2_upstream_preexperiment/` (145 MiB, round-1 benchmark/sweep; report/tables/figures/profiles/correctness/machine_outputs already archived on Windows) |
| deleted | `/root/autodl-tmp/outputs/phd2_qwen_acceleration/` (5.4 MiB, round-2 audit; summary/arms/report archived on Windows) |
| deleted | leaked temp render dir from the aborted 21:08 launch (17 MiB, mtime verified pre-run) |
| CLEANUP_AFTER | 20 GiB used / 31 GiB available |
| kept | `models/Qwen3.5-4B`, `models/rtdetr_r50vd`, conda/vLLM env, repo, `datasets/youtube_highlights`, stage3-6 outputs, all HF caches |

Not deleted because they are not part of the completed pre-experiment: stage3/4/5/6
outputs, `outputs/vhicraft`, `derived/`, `datasets/youtube_highlights` (needed for
FTNet later). The 30 GiB safety margin requested is not achievable on a 50 GiB
volume; after the full raw upload the volume holds 40 GiB used / ~11 GiB free,
which covers the retrieval artifacts (~0.3 GiB) plus transient chunk renders.

## 2. Frozen raw upload and identity

| Item | Value |
|---|---|
| local source | `E:\ResearchData\datasets\phd2` (frozen; never modified) |
| manifest | `manifests/phd2_raw_1500_202609220403.json`, SHA256 `ef0a07a3e437840a5a61b4b48e4524651015f70d9f3a91a3174eac58a3aa9c63` (identical locally and remotely) |
| remote target | `/root/autodl-tmp/datasets/phd2_raw_1500/` |
| transfer | 6 parallel tar-over-ssh streams, resumable, size-verified per file |
| result | `UPLOAD_COMPLETE_ALL_SIZES_MATCH`: 1500/1500 videos + manifest, 0 missing, 0 mismatched |
| raw identity | 1500 records, unique ids 1500, 21,105,628,182 bytes = 19.6561 GiB |
| arms | 360p 1270 / 720p 230 |
| codecs | vp9 902 / av1 354 / h264 244 |
| expected chunks | 20,627 (frozen 30 s chunk / 5 s overlap formula) |

The full raw is retained on the volume for tomorrow's detection stage; it is not
deleted after the Qwen pass.

## 3. Production integration of the promoted execution mode

- `real_provider.run_retrieval_for_entry(..., overlap_render: bool = False, audit_path=None)`
  - default `False` keeps the untouched frozen path;
  - when enabled it routes to the already-validated
    `ftnet.qwen_execution.run_retrieval_audited` (no second scheduler was written);
  - chunk-level in-place recovery; a failing chunk never re-runs the whole video.
- `ftnet/phd2_qwen1500.py`: frozen-manifest validation, retrieval-only index entries,
  artifact validation, run counters/progress, failure classification.
- `scripts/experiments/stage7/run_phd2_qwen1500.py`: `verify-raw` / `plan` / `run` /
  `status` / `verify`, resume-safe, failure-isolated, failed-only retry rounds.

Code identity recorded inside the run manifest:

```text
qwen_execution.py    79c88ebae5e3af0363d4b67ed82ac954102bc335d054e247f1aa000f09facebf
qwen_vllm_client.py  d0b68c612929c49e9637ebf01c094df0473e60cc6890aa6e271fce21b1eb7158
configs/highlight_retrieval.yaml 119a4c17d310b9e37f064b4f5748b574a36e2daca03cc6ee46c015adf9a0854a
```

## 4. Validation before launch

| Gate | Result |
|---|---|
| local targeted tests | `tests/test_phd2_qwen1500.py` + `tests/test_qwen_execution.py` 13 passed |
| remote targeted tests | 18 passed (`test_phd2_qwen1500`, `test_qwen_execution`, `test_phd2_upstream_benchmark`) |
| production A/B smoke (real service) | serial vs overlap on 3 videos: **40/40 non-retried chunks byte-identical**; the single mismatch is chunk 33 of `e6UI4KJNEOs`, whose frozen temp-0.3 recovery retry is stochastic by design |
| runner end-to-end smoke | 3 real + 1 broken record: correct order, failure recorded and isolated, retry round executed, honest `INCOMPLETE` summary, `verify` reported 1 missing |
| raw verification | `verify-raw` PASS: 1500 records, all sizes match, expected chunks 20,627 |

## 5. Frozen production configuration

```text
model            Qwen/Qwen3.5-4B (revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
prompt           high_recall_retrieval_v0
chunk / overlap  30 s / 5 s          sampling fps 2.0
temperature 0.0  enable_thinking False   max_new_tokens 512 (primary)
merge            tiou 0.5            parser semantics unchanged
request concurrency 1                render look-ahead: one chunk, strictly ordered
server           vllm 0.28.0, gpu-memory-utilization 0.80, max-model-len 32768,
                 allowed-local-media-path = the raw root
```

## 6. Run locations and monitoring

```text
run root   /root/autodl-tmp/outputs/phd2_full1500_qwen_retrieval/phd2_qwen1500_202609222118
progress   <run root>/progress.json  and  progress.jsonl
main log   <run root>/logs/main.log   (human-readable [n/1500] lines with ETA)
artifacts  <run root>/retrieval/<video_id>.json
responses  <run root>/raw_responses/<video_id>.chunks.json
failures   <run root>/failures/<video_id>.json
summary    <run root>/summary/run_summary.json (written at the end)
```

One-command status:

```bash
ssh autodl-stage1 'bash /root/autodl-tmp/phd2_qwen1500_status.sh'
```

Resume after any interruption (skips valid artifacts, re-runs only what is missing):

```bash
ssh autodl-stage1 'cd /root/autodl-tmp/AIC-VideoHighlight-run && setsid nohup /root/miniconda3/envs/aic-video-highlight/bin/python \
  scripts/experiments/stage7/run_phd2_qwen1500.py run \
  --raw-manifest /root/autodl-tmp/datasets/phd2_raw_1500/manifests/phd2_raw_1500_202609220403.json \
  --raw-manifest-sha256 ef0a07a3e437840a5a61b4b48e4524651015f70d9f3a91a3174eac58a3aa9c63 \
  --dataset-root /root/autodl-tmp/datasets/phd2_raw_1500 --expected-records 1500 \
  --run-root /root/autodl-tmp/outputs/phd2_full1500_qwen_retrieval/phd2_qwen1500_202609222118 \
  --run-id phd2_qwen1500_202609222118 --environment configs/environments/autodl.json \
  --overlap-render --retry-failed-rounds 2 > /dev/null 2>&1 &'
```

Expected wall time: 20,627 chunks at the measured promoted rate (0.1878 chunks/s)
= about 30.5 h, i.e. completion around 2026-09-24 early morning, plus a vLLM
start-up and the failed-only retry rounds.

## 7. Boundaries and next stage

- This run performs **Qwen retrieval only**. No RT-DETR, detection, Visual256,
  LOC, CMP, TS, Native16, supervision or FTNet code is invoked anywhere in the
  runner or the module it calls.
- Tomorrow's detection stage additionally needs the probed frozen index
  (`frame_count`, `fps`, `timestamp_mode`, PTS table) which the retrieval stage
  deliberately does not build; that preparation is CPU-only and is not started
  here.
- The official competition test set stays local; nothing is uploaded from it.
