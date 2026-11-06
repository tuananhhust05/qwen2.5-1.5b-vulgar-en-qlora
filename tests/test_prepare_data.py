import json
from pathlib import Path

from vulgar_en.config import load_config
from vulgar_en.prepare_data import SEED_SOURCE, dedupe_key, prepare

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return config_path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_load_default_config() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.model.use_qlora is True
    assert config.lora.r == 16
    assert "q_proj" in config.lora.target_modules


def test_prepare_seed_only_writes_splits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    summary = prepare(str(_copy_config(tmp_path)))
    for name in ("train", "eval", "test"):
        assert (tmp_path / f"data/processed/{name}.jsonl").exists()
        assert summary["splits"][f"data/processed/{name}.jsonl"]["rows"] > 0


def test_seed_rows_are_train_only_and_no_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data/raw/labeled.jsonl"
    raw.parent.mkdir(parents=True)
    rows = []
    for index in range(300):
        vulgar = index % 2 == 0
        label = (
            {"is_vulgar": True, "severity": "strong", "vulgar_terms": ["shit"]}
            if vulgar
            else {"is_vulgar": False, "severity": "none", "vulgar_terms": []}
        )
        rows.append({"input_text": f"post number {index} {'shit' if vulgar else 'ok'}", "label": label, "source": "toxic_spans"})
    # case/punctuation variant of row 0 must be collapsed, not leaked across splits
    rows.append({"input_text": "POST number 0, shit!", "label": rows[0]["label"], "source": "toxic_spans"})
    raw.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    prepare(str(_copy_config(tmp_path)))
    splits = {name: _read(tmp_path / f"data/processed/{name}.jsonl") for name in ("train", "eval", "test")}
    assert all(row["source"] != SEED_SOURCE for row in splits["eval"] + splits["test"])
    assert any(row["source"] == SEED_SOURCE for row in splits["train"])
    keys = {name: {dedupe_key(row["input_text"]) for row in split} for name, split in splits.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["eval"]
    assert not keys["eval"] & keys["test"]
    public_total = sum(1 for split in splits.values() for row in split if row["source"] != SEED_SOURCE)
    assert public_total == 300
