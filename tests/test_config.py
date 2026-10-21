import json
from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, load_config


@pytest.mark.parametrize("dataset,language", [
    ("multi_news", "en"), ("multi_xscience", "en"), ("vn_mds", "vi"), ("vims", "vi"),
])
def test_release_configs_resolve(dataset, language, tmp_path):
    config = load_config(overlay_configs=["configs/agents/default.yaml", f"configs/evaluations/{dataset}.yaml"])
    assert config.require("evaluation.language") == language
    assert config.require("run.dataset") == dataset
    assert config.require("outputs.root_dir") == config.require("storage.output_root")
    assert "${" not in json.dumps(config.to_dict())
    assert config.require("agents.execution_order") == ["extractor", "kgsum", "abstractor", "amf"]
    snapshot = config.snapshot(tmp_path / "config.yaml")
    assert yaml.safe_load(snapshot.read_text()) == config.to_dict()


def test_inheritance_and_overlay_precedence(tmp_path):
    (tmp_path / "storage.yaml").write_text("root: /tmp/synthetic\n")
    (tmp_path / "parent.yaml").write_text("model: {name: parent, tokens: 10}\nseed: 42\n")
    (tmp_path / "base.yaml").write_text("inherits: parent.yaml\nmodel: {name: base}\noutput: '${storage.root}/out'\n")
    (tmp_path / "overlay.yaml").write_text("model: {tokens: 20}\n")
    config = Config.load(str(tmp_path / "base.yaml"), str(tmp_path / "storage.yaml"),
                         overlay_config_paths=[str(tmp_path / "overlay.yaml")])
    assert config.require("model") == {"name": "base", "tokens": 20}
    assert config.require("output") == "/tmp/synthetic/out"
    assert config.require("seed") == 42
    with pytest.raises(KeyError):
        config.require("missing.key")


def test_all_yaml_files_parse():
    for path in Path("configs").rglob("*.yaml"):
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict), path
