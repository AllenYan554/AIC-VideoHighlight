# AutoDL 运行说明

本目录只提供下一轮在 AutoDL Linux / RTX 4090D 上执行的脚本。本轮没有 SSH、没有下载模型，也没有验证 GPU 推理。

1. 通过 Git 把仓库放到 `/root/autodl-tmp/AIC-VideoHighlight`。
2. 执行 `bash remote/bootstrap_autodl.sh` 创建 `.venv`、安装项目依赖和未锁版本的 vLLM。
3. 执行 `bash remote/serve_qwen.sh`，模型缓存写入 `/root/autodl-tmp/hf-cache`。
4. 依次运行文本与视频 Smoke Test。
5. Smoke Test 成功后记录 Python、PyTorch、CUDA、vLLM 与模型 revision，再固定兼容版本。

脚本不会修改 NVIDIA Driver、删除系统 CUDA 或安装随机 CUDA Toolkit。服务只监听 `127.0.0.1:8000`，单 GPU，初始 `--gpu-memory-utilization` 为 0.85。本地视频白名单固定为 `/root/autodl-tmp/datasets`；不要扩大为 `/`。

vLLM 的 CLI、多模态媒体加载与 Qwen 模型适配可能随版本变化。首次实机需要验证 `--allowed-local-media-path`、`--media-io-kwargs`、`video_url` 和 Qwen3.5-4B 兼容性；此处不编造成功结果。
