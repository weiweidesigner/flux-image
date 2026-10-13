#!/usr/bin/env python3
"""Run local FLUX.2 Klein and wrap the PNG for Aime Chart's HTML viewer."""

from __future__ import annotations

import argparse
import html
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib.parse import quote
from urllib import request as urllib_request


PROJECT_ROOT = Path(os.environ.get("AICHART_PROJECT_ROOT", str(Path.cwd()))).expanduser().resolve()
GENERATED_DIR = PROJECT_ROOT / "outputs" / "generated"
FLUX2_RUNNER = Path(os.environ.get("AICHART_FLUX2_RUNNER", str(PROJECT_ROOT / "runners/flux2/run.sh"))).expanduser()
DEFAULT_RUNNER = FLUX2_RUNNER


def _translation_instruction(prompt: str) -> str:
    return (
        "Translate the following image-generation prompt into natural English. "
        "Preserve every subject, action, visual detail, relationship, composition, "
        "lighting cue, and style. Use ASCII characters only. Output only the English "
        "prompt, with no explanation.\n\n" + prompt
    )


def _clean_translation(value: object) -> str:
    translated = str(value or "").strip().strip('`').strip().strip('"').strip()
    if not translated or not translated.isascii():
        raise ValueError("翻译模型未返回纯英文内容")
    return translated[:2000]


def _translate_with_deepseek(prompt: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    base_url = (
        os.environ.get("AICHART_FLUX_TRANSLATOR_BASE_URL", "").strip()
        or os.environ.get("MODEL_BASE_URL", "").strip()
        or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com"
    ).rstrip("/")
    model = (
        os.environ.get("AICHART_FLUX_SELECTED_MODEL", "").strip()
        or os.environ.get("AICHART_FLUX_REMOTE_TRANSLATOR_MODEL", "").strip()
        or os.environ.get("AICHART_DEEPSEEK_MODEL", "").strip()
        or os.environ.get("DEEPSEEK_MODEL", "").strip()
        or "deepseek-chat"
    )
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You translate prompts for an image generator."},
            {"role": "user", "content": _translation_instruction(prompt)},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 1000,
    }).encode("utf-8")
    request = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("远程翻译模型没有返回候选结果")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("远程翻译模型返回结果缺少 message")
    return _clean_translation(message.get("content"))


def _translate_with_ollama(prompt: str) -> str:
    if prompt.isascii():
        return prompt
    model = os.environ.get("AICHART_FLUX_TRANSLATOR_MODEL", "glm4:9b").strip() or "glm4:9b"
    base_url = (os.environ.get("AICHART_OLLAMA_URL", "http://127.0.0.1:11434").strip()
                or "http://127.0.0.1:11434").rstrip("/")
    body = json.dumps({
        "model": model,
        "prompt": _translation_instruction(prompt),
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0},
    }).encode("utf-8")
    request = urllib_request.Request(
        f"{base_url}/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib_request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _clean_translation(payload.get("response") if isinstance(payload, dict) else "")


def model_prompt(prompt: str) -> str:
    """Translate non-ASCII prompts without making Ollama a single point of failure."""
    if prompt.isascii():
        return prompt
    provider = os.environ.get("AICHART_FLUX_TRANSLATOR_PROVIDER", "auto").strip().lower() or "auto"
    translators = []
    if provider in {"auto", "deepseek", "remote"}:
        translators.append(("当前远程模型", _translate_with_deepseek))
    if provider in {"auto", "ollama", "local"}:
        translators.append(("本地 Ollama", _translate_with_ollama))
    if not translators:
        raise SystemExit(f"不支持的提示词翻译服务：{provider}")
    errors: list[str] = []
    for label, translator in translators:
        try:
            return translator(prompt)
        except (OSError, urllib_error.URLError, json.JSONDecodeError, KeyError, IndexError,
                TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{label}：{exc}")
    raise SystemExit("中文提示词翻译失败（已尝试远程模型和本地 Ollama）：" + "；".join(errors))


def parse_size(value: str) -> str:
    try:
        height, width = (int(part) for part in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be HEIGHTxWIDTH") from exc
    if min(height, width) < 256 or height % 16 or width % 16:
        raise argparse.ArgumentTypeError("dimensions must be >=256 and divisible by 16")
    if height * width > 1024 * 1024:
        raise argparse.ArgumentTypeError("image area must not exceed 1024x1024")
    return f"{height}x{width}"


def output_path(value: str) -> Path:
    supplied = Path(value).expanduser()
    path = (supplied if supplied.is_absolute() else PROJECT_ROOT / supplied).resolve()
    if path.suffix.lower() != ".html" or GENERATED_DIR.resolve() not in path.parents:
        raise argparse.ArgumentTypeError("output must be an HTML file in outputs/generated")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=output_path)
    parser.add_argument("--size", default="512x512", type=parse_size)
    parser.add_argument("--steps", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input-image", type=Path, help="Existing image for image-to-image generation")
    parser.add_argument("--strength", type=float)
    return parser


def render_page(prompt: str, image_name: str, size: str, seed: int,
                model_name: str = "FLUX.2 [klein] 4B") -> str:
    safe_prompt = html.escape(prompt, quote=True)
    safe_image = quote(image_name)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model_name)} 生图</title>
<style>
  :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; background: transparent; color: #172033; }}
  main {{ max-width: 560px; margin: 0; }}
  img {{ display: block; width: auto; height: auto; max-width: 100%; max-height: 72vh; object-fit: contain; border: 0; background: transparent; }}
</style>
</head>
<body>
<main>
  <img src="{safe_image}" alt="{safe_prompt}">
</main>
<script>
  function reportHeight() {{ parent.postMessage({{type: 'analysis-planner-report-height', height: document.documentElement.scrollHeight}}, '*'); }}
  addEventListener('load', reportHeight);
  document.querySelector('img').addEventListener('load', reportHeight);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = args.prompt.strip()
    if not prompt:
        raise SystemExit("prompt must not be empty")
    runner = Path(os.environ.get("AICHART_FLUX_RUNNER", str(DEFAULT_RUNNER))).expanduser()
    is_flux2 = runner.resolve() == FLUX2_RUNNER.resolve()
    model_name = "FLUX.2 [klein] 4B" if is_flux2 else "FLUX.1-schnell"
    if not runner.is_file():
        raise SystemExit(f"Local FLUX runner not found: {runner}")
    if not os.access(runner, os.X_OK):
        raise SystemExit(f"Local FLUX runner is not executable: {runner}")

    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    if args.input_image and not args.input_image.is_file():
        raise SystemExit(f"原图不存在：{args.input_image}")
    if args.strength is not None and not 0 < args.strength <= 1:
        raise SystemExit("--strength 必须大于 0 且不超过 1")
    if is_flux2 and args.strength is not None:
        raise SystemExit("FLUX.2 参考图编辑不支持 --strength，请仅使用 --input-image")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = args.output.with_suffix(".png")
    started_ns = time.time_ns()
    command = [
        str(runner), model_prompt(prompt), "--size", args.size, "--steps", str(args.steps),
        "--seed", str(seed), "--output", str(image),
    ]
    if args.input_image:
        command.extend(["--input-image", str(args.input_image.resolve())])
        if args.strength is not None:
            command.extend(["--strength", str(args.strength)])
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("Local FLUX generation timed out after 30 minutes") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise SystemExit(f"Local FLUX generation failed: {detail}")
    if not image.is_file() or image.stat().st_size == 0 or image.stat().st_mtime_ns < started_ns:
        raise SystemExit("Local FLUX runner did not produce a fresh, non-empty PNG")
    args.output.write_text(render_page(prompt, image.name, args.size, seed, model_name), encoding="utf-8")
    print(json.dumps({
        "html": str(args.output), "image": str(image), "seed": seed,
        "size": args.size, "model": model_name,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
