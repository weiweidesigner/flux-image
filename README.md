# FLUX Image

调用外部本地 FLUX 运行器生成或编辑图片，并为新 PNG 创建 HTML 展示页。Python 封装只使用标准库；模型、MFLUX/MLX 环境与运行器由本地部署提供，不包含在仓库中。

## 配置与使用

建议 Python 3.11+。设置项目根目录与实际运行器：

```bash
export AICHART_PROJECT_ROOT=/path/to/host-project
export AICHART_FLUX2_RUNNER=/path/to/flux2/run.sh
export AICHART_FLUX_RUNNER="$AICHART_FLUX2_RUNNER"
python app/skills/flux-image/scripts/generate_image.py --prompt 'A red fox in snow' --output "$AICHART_PROJECT_ROOT/outputs/generated/fox.html"
```

项目根目录默认当前工作目录，FLUX.2 运行器默认 `<project-root>/runners/flux2/run.sh`。设置 `AICHART_FLUX_RUNNER` 为不同运行器时沿用 FLUX.1 兼容模式，因此 FLUX.2 应同时配置两项路径。

运行器接收首个位置参数为英文 prompt，并支持 `--size`、`--steps`、`--seed`、`--output`。参考图编辑还需支持 `--input-image`；FLUX.1 可选 `--strength`，FLUX.2 不接受该参数。运行器必须生成新的非空 PNG，成功后模块输出 PNG/HTML 路径。

非英文提示词会按配置调用兼容 OpenAI 的翻译服务或本地 Ollama。模型凭据通过本地环境提供；配置示例 `.env.example` 的所有凭据字段为空，真实 `.env` 不提交。

完整使用规则见 `app/skills/flux-image/SKILL.md`。图片和输出 HTML 位于项目的 `outputs/generated/` 中，不上传模型权重或生成内容。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

测试使用模拟运行器与翻译响应，不执行真实模型推理。
