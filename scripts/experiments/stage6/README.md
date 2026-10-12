# Stage6.2B 本地原生信号审计

本目录的脚本只服务本次冻结Dev control。输入只读、无网络、无模型调用、无视频解码。所有产物只写入指定的 `03_Stage6_2B_Task_Native_Signal_Audit` 实验记录目录；任何mask都标记为 `POSTHOC_DIAGNOSTIC_MASK`，不写新predictions。

`verify_stage6_2b_inputs.py` 独立验证恢复清单的701个文件、协议身份、上游hash链、candidate投影、LOC/CMP/TS逐帧连接和最终bbox。原有DeepSeek verification会按内容hash保存到 `audit/history/`，原缺件manifest不覆盖。

`analyze_stage6_2b_native_signals.py` 在闭合通过后冻结 `audit/analysis_protocol.json`，然后派生77个注册项、Y/Q/Z，输出完整信号统计、video/source bootstrap、固定5/10/20/30/50%预算和六种严格预算对照。空输出视频仍留在macro scope。source按 `source_ytid` 聚类，比分窗 `source_group` 更保守。连续score缺失不填0；预算只在其OBSERVED层，所有对照共用同一层。

`plot_stage6_2b_native_signals.py` 从完整CSV生成八张图、图表来源manifest及中文图注。统计使用Codex bundled Python 3.12.14及其已有NumPy/Pandas；绘图使用本机Python 3.10已有Matplotlib。没有安装或更新依赖。

`summarize_stage6_2b_consistency.py` 补齐全部已注册feature的逐video/source五分位与contribution enrichment、source-equal估计。空输出视频和source仍保留在bootstrap总体；不可计算值为NA。`--source-only` 仅重算source汇总，避免重复已完成的分位表。

该汇总脚本另有 `--event-risk-only`（全部原生逐帧mask逐event风险）、`--block-strata-only`（特征变化/常数分层的paired对照与Recall tail）和 `--reference-geometry-only`（标签居中几何混杂，非部署feature）三个独立补表命令。完整复现时依次执行这三个选项，无需重新运行全量mask计算。

`finalize_stage6_2b_native_audit.py` 只核验已经生成的产物：全部701个输入hash、Y/Q/Z语义、全预算匹配、全部aggregate与逐视频结果的恒等式、baseline和必要表/图。随后写最终QA与产物manifest。Gate由完整报告中的证据审查给定；不在脚本中搜索最佳feature或budget。

本轮是用户明确要求的本地离线事后审计，不是Smoke/Formal/pipeline Validation。以下直接命令不启动实验launcher、不打开交互窗口、不检查SSH。

```powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight\scripts\experiments\stage6\verify_stage6_2b_inputs.py'
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight\scripts\experiments\stage6\analyze_stage6_2b_native_signals.py'
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight\scripts\experiments\stage6\summarize_stage6_2b_consistency.py'
& 'C:\Users\lenovo\AppData\Local\Programs\Python\Python310\python.exe' 'D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight\scripts\experiments\stage6\plot_stage6_2b_native_signals.py'
```

最小测试（在本目录运行）：

```powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest -v test_stage6_2b_audit
```

在完整报告、H1-H7审查表、中文handoff及八张图的人工视觉审查记录就绪后，运行最终核验：

```powershell
& 'C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight\scripts\experiments\stage6\finalize_stage6_2b_native_audit.py'
```

重新绘图会将figure_manifest的review_status恢复为PENDING_VISUAL_REVIEW，必须实际逐图检查后才能记录VISUALLY_REVIEWED；不要仅为通过最终核验修改该标志。最终Gate是完整证据的人工审查结论，不能靠单个指标自动放行。已完成审计后不要为重复确认而重跑全量统计。

定义、方向和窗口不可在看结果后更改。协议不一致时脚本停止。pooled AP/Q/Z为描述点估计，CI明确对应video-equal均值；pooled AUROC的CI使用精确cluster pair kernel。随机对照200次均值的CI条件于这些Monte Carlo样本，不是frame iid区间。报告需完整保留负结果，并将弱参考hash/provenance限制与信号有效性分开。
