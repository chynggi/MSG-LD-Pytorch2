import importlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utilities import model as vocoder_module  # type: ignore[import-not-found]


def reload_module():
    """Reload the utilities.model module to reset patched state when needed."""
    return importlib.reload(vocoder_module)


def test_get_vocoder_routes_to_discoder(monkeypatch):
    module = reload_module()

    captured = {}

    def fake_discoder(device, mel_bins, config):
        captured["device"] = device
        captured["mel_bins"] = mel_bins
        captured["config"] = config
        return "discoder-object"

    monkeypatch.setattr(module, "_load_discoder_vocoder", fake_discoder)
    monkeypatch.setattr(
        module,
        "_load_hifigan_vocoder",
        lambda *args, **kwargs: pytest.fail("HiFi-GAN loader should not be used"),
    )

    result = module.get_vocoder(
        config=None,
        device="cuda:0",
        mel_bins=128,
        vocoder_config={"type": "discoder", "target_sample_rate": 44100},
    )

    assert result == "discoder-object"
    assert captured["device"] == "cuda:0"
    assert captured["mel_bins"] == 128
    # ensure we pass through the original configuration without mutation
    assert captured["config"]["target_sample_rate"] == 44100


def test_get_vocoder_defaults_to_hifigan_and_infers_sample_rate(monkeypatch):
    module = reload_module()

    captured_cfg = {}

    def fake_hifigan(device, mel_bins, ckpt_path, cfg, root):
        captured_cfg.update(cfg)
        return "hifigan-object"

    monkeypatch.setattr(module, "_load_hifigan_vocoder", fake_hifigan)
    monkeypatch.setattr(
        module,
        "_load_discoder_vocoder",
        lambda *args, **kwargs: pytest.fail("DISCoder loader should not be invoked"),
    )

    config = {"preprocessing": {"audio": {"sampling_rate": 22050}}}

    result = module.get_vocoder(
        config=config,
        device="cpu",
        mel_bins=64,
        ckpt_path="checkpoint.pt",
        vocoder_config={"type": "HiFi-GAN"},
    )

    assert result == "hifigan-object"
    assert captured_cfg["target_sample_rate"] == 22050
    assert captured_cfg.get("checkpoint_path") == "checkpoint.pt"