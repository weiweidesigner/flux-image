import importlib.util
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest



SCRIPT = Path(__file__).resolve().parents[1] / "app/skills/flux-image/scripts/generate_image.py"
SPEC = importlib.util.spec_from_file_location("flux_image_skill", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_size_and_output_validation(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    generated.mkdir()
    monkeypatch.setattr(MODULE, "GENERATED_DIR", generated)
    assert MODULE.parse_size("512x768") == "512x768"
    with pytest.raises(Exception):
        MODULE.parse_size("513x512")
    assert MODULE.output_path(str(generated / "picture.html")) == generated / "picture.html"
    with pytest.raises(Exception):
        MODULE.output_path(str(tmp_path / "elsewhere.html"))


def test_generates_html_wrapper_for_fresh_png(tmp_path, monkeypatch, capsys):
    generated = tmp_path / "generated"
    generated.mkdir()
    runner = tmp_path / "run.sh"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setenv("AICHART_FLUX_RUNNER", str(runner))
    monkeypatch.setattr(MODULE, "GENERATED_DIR", generated)

    def fake_run(command, **_kwargs):
        Path(command[command.index("--output") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    output = generated / "fox.html"
    assert MODULE.main(["--prompt", '<fox & snow>', "--output", str(output), "--seed", "7"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["html"] == str(output)
    assert payload["image"] == str(output.with_suffix(".png"))
    html = output.read_text(encoding="utf-8")
    assert 'src="fox.png"' in html
    assert "&lt;fox &amp; snow&gt;" in html
    assert "<fox & snow>" not in html
    assert 'class="frame"' not in html
    assert "max-width: 560px" in html
    assert "background: transparent" in html
    assert "<p>" not in html
    assert "seed 7" not in html


def test_non_ascii_prompt_uses_selected_remote_model(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"A pelican riding a bicycle."}}]}'

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 90
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("AICHART_FLUX_SELECTED_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(MODULE.urllib_request, "urlopen", fake_urlopen)
    assert MODULE.model_prompt("画一个鹈鹕骑车") == "A pelican riding a bicycle."
    assert MODULE.model_prompt("A red apple") == "A red apple"
    assert len(requests) == 1
    body = json.loads(requests[0].data)
    assert body["model"] == "deepseek-v4-flash"
    assert "画一个鹈鹕骑车" in body["messages"][1]["content"]
    assert requests[0].get_header("Authorization") == "Bearer test-key"


def test_non_ascii_prompt_falls_back_to_ollama(monkeypatch):
    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/chat/completions"):
            raise MODULE.urllib_error.URLError("remote unavailable")
        return FakeResponse(b'{"response":"A pelican riding a bicycle."}')

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(MODULE.urllib_request, "urlopen", fake_urlopen)
    assert MODULE.model_prompt("画一个鹈鹕骑车") == "A pelican riding a bicycle."
    assert [request.full_url.rsplit("/", 2)[-1] for request in requests] == ["completions", "generate"]


def test_flux2_runner_uses_reference_edit_without_legacy_strength(tmp_path, monkeypatch, capsys):
    generated = tmp_path / "generated"
    generated.mkdir()
    runner = tmp_path / "run.sh"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    runner.chmod(0o755)
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    monkeypatch.setenv("AICHART_FLUX_RUNNER", str(runner))
    monkeypatch.setattr(MODULE, "FLUX2_RUNNER", runner)
    monkeypatch.setattr(MODULE, "GENERATED_DIR", generated)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    output = generated / "edited.html"
    assert MODULE.main(["--prompt", "Watercolor cat", "--input-image", str(source),
                        "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "FLUX.2 [klein] 4B"
    assert "--input-image" in commands[0]
    assert "--strength" not in commands[0]

