# AIC-VideoHighlight Agent Rules

1. 进入仓库后只读取当前项目根目录的 `AGENTS.md`；不要扫描父目录或其他项目。
2. Git 操作必须安全；禁止 `reset`、`clean`、`stash`、`rebase`、force push，未经授权不要 push 或猜测 remote。
3. 不允许把 `models/`、`datasets/`、`hf-cache/`、视频、模型权重、压缩数据集或大量输出提交到 Git。
4. 不允许把训练集或测试集上传到外部服务；数据只能在用户指定的本地/AutoDL 环境处理。
5. 修改应小步进行，优先测试公开行为；展示 diff 并运行最小相关测试。
6. Stage 1 当前优先建立高 Recall 的 Zero-shot baseline，内部结果不得冒充官方最终提交格式或官方指标。
7. 未经明确授权，不要开始 Stage C、SAM、Tracking、逐帧 bbox、构图优化、时序平滑、SFT、LoRA 或 RL/GRPO。
8. Qwen 模型由 AutoDL 上的 vLLM 服务管理；不要在 Windows 本地下载模型、安装 vLLM 或尝试 CUDA 推理。
