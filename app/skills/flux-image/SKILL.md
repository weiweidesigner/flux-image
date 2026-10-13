---
name: flux-image
description: Use the locally deployed FLUX.2 [klein] 4B model for text-to-image generation or reference-image edits of one uploaded image in Aime Chart. Not for data-accurate financial charts or factual image analysis.
---

# FLUX image generation

Use `scripts/generate_image.py` when the user asks for a newly imagined picture or asks to redraw one uploaded image with a prompt. It calls the local Apple Silicon MLX deployment of `FLUX.2 [klein] 4B` and returns a PNG plus a standalone HTML page that Aime Chart can display. This is generative artwork, not a source of factual or numerical evidence; use chart skills for data visualization. Do not route ordinary questions about an uploaded image to this Skill.

The 4-bit quantized weights come from `Runpod/FLUX.2-klein-4B-mflux-4bit`, compatible with MFLUX. Generation uses model files managed by the configured local runner; no Hugging Face login or remote service is needed at inference time. If weights are missing, report that plainly; never substitute a remote image service or fabricate an output.

## Run

```bash
python3 app/skills/flux-image/scripts/generate_image.py \
  --prompt "A cinematic photograph of a red fox in a snowy forest" \
  --output outputs/generated/flux-red-fox.html

# Edit one local image using it as a reference:
python3 app/skills/flux-image/scripts/generate_image.py \
  --prompt "Turn this scene into a watercolor painting" \
  --input-image /absolute/path/to/source.png \
  --output outputs/generated/flux-watercolor.html
```

Optional: `--size 512x512`, `--steps 1..4`, `--seed 42`. Dimensions are height × width, multiples of 16 and at least 256. The script defaults to 512×512, 4 steps, and a random seed. `--output` must be an HTML path inside `outputs/generated/`; the adjacent PNG has the same stem. The script prints a JSON object with absolute `html`, `image`, `seed`, `size`, and `model` fields after success. The local runner is configured through `AICHART_FLUX_RUNNER`; its default is `<project-root>/runners/flux2/run.sh`, configurable with `AICHART_FLUX2_RUNNER`. The previous FLUX.1-schnell runner remains available by setting `AICHART_FLUX_RUNNER=/path/to/legacy/run.sh`.

For image editing, pass one local `--input-image`. FLUX.2 treats it as a reference image; it does not support the former FLUX.1 `--strength` parameter. Describe what to preserve and what to change in the prompt. The uploaded image remains an input, not a public generated artifact. Multiple uploaded images are not supported by this workflow.

The script translates non-ASCII prompts to English before inference. In the default `auto` mode it first uses the currently selected remote model (OpenAI-compatible `/chat/completions`, authenticated by `DEEPSEEK_API_KEY`) and falls back to local Ollama (`AICHART_FLUX_TRANSLATOR_MODEL`, default `glm4:9b`). Configure the order with `AICHART_FLUX_TRANSLATOR_PROVIDER=auto|deepseek|ollama`; `AICHART_FLUX_TRANSLATOR_BASE_URL` and `AICHART_FLUX_REMOTE_TRANSLATOR_MODEL` can override the remote endpoint and model. The original prompt is retained only as image alt text, not a visible caption. If every configured translator fails, report the combined errors rather than claiming Metal is unavailable.

Check the printed paths exist and are non-empty. In the final Aime Chart answer, place `[[AIME_CHART:/absolute/path/to/output.html]]` on its own line near the description, and include `flux-image` in `[[AIME_SKILLS:...]]`. Do not cite the generated picture as evidence for market or company claims.

## Standalone Runtime

Set `AICHART_PROJECT_ROOT` to the host project root (defaults to the working directory). Set `AICHART_FLUX2_RUNNER` to the local FLUX.2 runner and `AICHART_FLUX_RUNNER` to the runner to execute. Model weights and local credentials are not bundled.
