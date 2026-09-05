# AIC-VideoHighlight Agent Rules

1. 进入仓库后只读取当前项目根目录的 `AGENTS.md`；不要扫描父目录或其他项目。
2. Git 操作必须安全；禁止 `reset`、`clean`、`stash`、`rebase`、force push，未经授权不要 push 或猜测 remote。
3. 不允许把 `models/`、`datasets/`、`hf-cache/`、视频、模型权重、压缩数据集或大量输出提交到 Git。
4. 不允许把训练集或测试集上传到外部服务；数据只能在用户指定的本地/AutoDL 环境处理。
5. 修改应小步进行，优先测试公开行为；展示 diff 并运行最小相关测试。
6. 高光候选召回当前优先建立高 Recall 的 Zero-shot baseline，内部结果不得冒充官方最终提交格式或官方指标。
7. 未经明确授权，不要开始边界精修、SAM、Tracking、逐帧 bbox、构图优化、时序平滑、SFT、LoRA 或 RL/GRPO。
8. Qwen 模型由 AutoDL 上的 vLLM 服务管理；不要在 Windows 本地下载模型、安装 vLLM 或尝试 CUDA 推理。

## Local / Cloud Storage Policy

### Windows Local Repository

唯一正式本地项目：

`D:\CDUT\硕士\2026-09 AIC\AIC-VideoHighlight`

### AutoDL Repository

唯一正式云端项目：

`/root/autodl-tmp/AIC-VideoHighlight-run`

### Experiment Records

所有本地实验、训练、测试和验证报告统一归档到：

`D:\CDUT\硕士\2026-09 AIC\实验记录`

### Forbidden

- 禁止创建 `.codex_remote_aic` 或其他云端项目本地镜像。
- 禁止整体复制云端 checkout 到 Windows。
- 禁止整体复制云端 outputs 到 Windows。
- 禁止在 Windows 保留第二套项目源码、测试或脚本目录。

### Sync Strategy

- 代码通过 GitHub SSH 同步。
- 实验只同步轻量报告、CSV、JSON、必要图表和配置摘要。
- 模型、数据集、HF/vLLM cache 保留在云端或专门数据位置。
- Git 仓库中的 `docs/experiments` 用于版本追踪；Windows 的 `实验记录` 是完整本地主归档。

### Official Competition Test

正式赛方 test 只保存在本地，禁止上传到 AutoDL。
