# Baseline 20 样本集

## 1. 目的

本样本集用于首轮 Qwen3.5-4B Zero-shot Highlight Retrieval 小规模基线。它只包含 `dataset_split == "train"` 的 20 条固定记录；`segments` 在 manifest 中统一命名为 `weak_reference_segments`，不表示赛事官方 Ground Truth。

## 2. 选择规则

在 887 条 train 记录上计算 clip 时长、segment 数量与时长、segment 间隔、`mean_seed_score`、源时间档和空间标签状态。选择结果固定，不依赖文件遍历顺序；并以种子 `20260904` 作为同等候选时的固定约定。

- Group A（5 条）：单区间样本，覆盖短、中、长 clip 以及低、中、高 weak score。
- Group B（5 条）：双区间样本，覆盖 0.167–23.649 秒的不同区间间隔和全部五种源时间档。
- Group C（5 条）：复杂多区间样本，覆盖 3、4、5、7 个 segment；7 是 train 数据中的最大值。
- Group D（5 条）：边界/困难样本，覆盖两条零时长 segment、0.633 ms 舍入越界、clip 从 0 开始、clip 到 150 秒、长 segment 及不同空间标签状态。

20 个 `source_group` 和 YouTube ID 均唯一。文件名严格由 `provenance.source_group + ".mp4"` 生成，远端路径严格由 MP4 文件名首字符与完整文件名组成。

## 3. 样本总表

| # | Group | video_id | source_group | segment数 | clip时长(s) | 空间标签状态 | 选择原因 |
|---:|---|---|---|---:|---:|---|---|
| 1 | A | `qvh_000023_9x16` | `67yNlWjxkNc_60.0_210.0` | 1 | 5.238 | seed | 最短 clip 桶；低 weak score；完整 crop 标签 |
| 2 | A | `qvh_000337_9x16` | `h4SpzDYV50Q_60.0_210.0` | 1 | 13.334 | dropped_center_default | 中等 clip；高 weak score |
| 3 | A | `qvh_000757_9x16` | `pV1SU8XyXr4_210.0_360.0` | 1 | 32.134 | dropped_center_default | 最长 clip 桶；中等 weak score |
| 4 | A | `qvh_000602_9x16` | `5uuG5Z0-rYU_360.0_510.0` | 1 | 32.133 | dropped_center_default | 长 clip；较高 weak score |
| 5 | A | `qvh_000949_9x16` | `d5mpNbgVU5s_510.0_660.0` | 1 | 32.134 | dropped_center_default | 长 clip；中等 weak score |
| 6 | B | `qvh_000531_9x16` | `f0S6MWNcJjY_60.0_210.0` | 2 | 9.400 | dropped_center_default | 双区间；0.634 秒短间隔；替代原 #6 |
| 7 | B | `qvh_000612_9x16` | `nuZ_0pN8F-U_210.0_360.0` | 2 | 13.500 | seed | 0.167 秒极短间隔 |
| 8 | B | `qvh_000067_9x16` | `vqsojDj6j_s_360.0_510.0` | 2 | 32.133 | seed | 长 clip；23.649 秒大间隔 |
| 9 | B | `qvh_000781_9x16` | `cF9EiA0DRrE_510.0_660.0` | 2 | 19.266 | seed | 16.450 秒大间隔 |
| 10 | B | `qvh_000629_9x16` | `tYKDJDlWRgY_660.0_810.0` | 2 | 22.200 | seed | 较低 weak score；12.767 秒间隔 |
| 11 | C | `qvh_000008_9x16` | `9vyrO1Y_T1M_210.0_360.0` | 3 | 21.560 | seed | 3 区间；含两个短区间；替代原 #11 |
| 12 | C | `qvh_000134_9x16` | `nHvGP413OUU_60.0_210.0` | 4 | 15.766 | dropped_center_default | 全选样本最低 weak score |
| 13 | C | `qvh_000089_9x16` | `qt_z-TPv8zQ_660.0_810.0` | 4 | 16.520 | seed | 4 区间；覆盖 660–810 源档 |
| 14 | C | `qvh_000829_9x16` | `-d3Oru5Mj_A_360.0_510.0` | 5 | 32.133 | seed | 5 区间；完整 crop 标签；验证 `-` 首字符路径 |
| 15 | C | `qvh_000358_9x16` | `aXgE_cVxJi0_60.0_210.0` | 5 | 25.000 | seed | 高复杂度；由 7 降为 5 区间；替代原 #15 |
| 16 | D | `qvh_000553_9x16` | `mh9Gm5UOMpI_60.0_210.0` | 2 | 14.666 | seed | 包含零时长 weak reference segment |
| 17 | D | `qvh_000721_9x16` | `fQL4I1-5D4k_60.0_210.0` | 4 | 15.760 | seed | 含 0.12 秒短边界区间；替代原 #17 |
| 18 | D | `qvh_000363_9x16` | `5ypSTZYixSc_210.0_360.0` | 2 | 25.440 | seed | 末段距 clip 尾部 0.44 秒；替代原 #18 |
| 19 | D | `qvh_000574_9x16` | `1G5bSIisZSA_210.0_360.0` | 1 | 31.400 | seed | clip 从 0 开始；31.031 秒长区间 |
| 20 | D | `qvh_000915_9x16` | `hgVHo7fv8cg_60.0_210.0` | 1 | 9.567 | dropped_center_default | clip 到 150 秒；替代原 #20 |

## 4. 分布统计

- 分组：A/B/C/D 各 5 条。
- segment count：1×7、2×7、3×1、4×3、5×2。原唯一可选 7 区间视频在社区镜像中缺失，复杂度上限降为 5。
- clip 时长：最小 5.238 秒，平均 20.964 秒，最大 32.134 秒。
- 源时间档：60–210×8、210–360×5、360–510×3、510–660×2、660–810×2。
- 空间标签：`seed`×13、`dropped_center_default`×7；两类均被覆盖。
- 样本级 segment `mean_seed_score` 均值：最小 0.1453，平均 0.7144，最大 0.8989。
- 唯一性：20 个 `source_group`、文件名和 YouTube ID 均唯一。

## 5. 已知边界样本

- #16 包含一个零时长 `weak_reference_segments` 元素。全数据中的另一个零时长样本原位于 #17，但其视频在社区镜像中返回 404；该原始标注继续作为 annotation-only edge case 记录。
- #17 含 0.12 秒短边界区间，用于替代不可运行的第二个零时长视频，但不宣称二者等价。
- 原 #18 是唯一已知的约 0.000633 秒时间基/舍入越界案例，其视频在社区镜像中返回 404，现转为 annotation-only edge case；新 #18 是末段距 clip 尾部 0.44 秒的正常边界样本。
- #19 的 clip 从源视频局部 0 秒开始，并包含约 31.031 秒的长区间。
- #20 的 `clip.end_sec` 为 150 秒且空间标签为 `dropped_center_default`。
- #14 的文件名以 `-` 开头，可验证首字符目录规则没有错误限定为字母。

## Replacement History

首版 manifest 中有 6 个视频在社区镜像 `ayushsdev/qvhighlights-videos` 上返回 HTTP 404。所有替代候选均先通过镜像 metadata/HEAD 确认可取得，再写入 manifest；原失败 ID、文件名与原因保留在各条目的 `replacement` 对象中。

| 样本位 | 原 video_id / 文件 | 替代 video_id / 文件 | 保留的实验角色 | 变化与风险 |
|---:|---|---|---|---|
| #6 | `qvh_000885_9x16` / `VSrS_p3h8jI_60.0_210.0.mp4` | `qvh_000531_9x16` / `f0S6MWNcJjY_60.0_210.0.mp4` | 双区间、60–210 源档、短间隔 | 空间标签由 seed 变为 dropped_center_default |
| #11 | `qvh_000762_9x16` / `Hi4qIbImolM_210.0_360.0.mp4` | `qvh_000008_9x16` / `9vyrO1Y_T1M_210.0_360.0.mp4` | 3 区间、210–360 源档 | 长 clip 改为中长 clip，并包含两个短区间 |
| #15 | `qvh_000917_9x16` / `XFg-PaelogA_60.0_210.0.mp4` | `qvh_000358_9x16` / `aXgE_cVxJi0_60.0_210.0.mp4` | 复杂多区间、60–210 源档 | 复杂度从全数据唯一的 7 区间降为 5 区间 |
| #17 | `qvh_000961_9x16` / `XSi9PFacbgA_210.0_360.0.mp4` | `qvh_000721_9x16` / `fQL4I1-5D4k_60.0_210.0.mp4` | 困难/短边界区间 | 原零时长标注转为 annotation-only；新样本最短区间为 0.12 秒；另一个真实零时长案例仍由 #16 覆盖 |
| #18 | `qvh_000871_9x16` / `FpNQqtWd9Eo_660.0_810.0.mp4` | `qvh_000363_9x16` / `5ypSTZYixSc_210.0_360.0.mp4` | clip 尾部边界 | 唯一 0.633 ms 越界不能等价替代，原标注明确保留为 annotation-only edge case |
| #20 | `qvh_000181_9x16` / `T2t1_b9qPyQ_660.0_810.0.mp4` | `qvh_000915_9x16` / `hgVHo7fv8cg_60.0_210.0.mp4` | `clip.end_sec=150` 且 dropped_center_default | 源时间档由 660–810 变为 60–210，整体仍保留五档覆盖 |

## 6. 数据来源说明

视频来自公开 QVHighlights 社区镜像 `ayushsdev/qvhighlights-videos`。赛事 `train.jsonl` 的 `provenance.source` 在 987/987 条记录中均为 `QVHighlights`，命名规则和源片段边界也高度一致。

该公开镜像目前作为高可信开发来源使用。本样本集应称为 **baseline development candidate data**；尚未证明镜像 MP4 与赛事百度网盘版本逐字节完全一致。后续若取得赛方同名文件，应抽样比较 SHA256、ffprobe 元数据以及关键帧或感知内容。
