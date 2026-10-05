from pathlib import Path

from vulgar_en.config import load_config
from vulgar_en.prepare_data import prepare

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.model.use_qlora is True
    assert config.lora.r == 16
    assert "q_proj" in config.lora.target_modules


def test_prepare_writes_splits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    counts = prepare(str(config_path))
    assert counts["data/processed/train.jsonl"] > 0
    assert counts["data/processed/eval.jsonl"] > 0
    assert counts["data/processed/test.jsonl"] > 0
    assert (tmp_path / "data/processed/train.jsonl").exists()
