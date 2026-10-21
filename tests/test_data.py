import json

import pytest

from src.data.main import DataConfig, get_dataset_spec, load_dataset, prepare_dataset
from src.data.processors.base import BaseDataset, normalize_text


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture(params=["multi_news", "multi_xscience", "vn_mds", "vims"])
def prepared_dataset(request, tmp_path):
    name = request.param
    raw = tmp_path / name
    if name == "multi_news":
        write(raw / "raw/test.src.cleaned", " First  report ||||| Second report\nThird report\n")
        write(raw / "raw/test.tgt", "First summary\nSecond summary\n")
        expected_documents = ["First report", "Second report"]
    elif name == "multi_xscience":
        rows = [
            {"aid": "paper-1", "abstract": "First abstract", "ref_abstract":
             {"ref-1": {"abstract": "Prior abstract"}}, "related_work": "Related @cite_1 work"},
            {"aid": "paper-2", "abstract": "Another abstract", "related_work": "Another summary"},
        ]
        write(raw / "raw/test.json", json.dumps(rows))
        expected_documents = ["First abstract", "Prior abstract"]
    elif name == "vn_mds":
        for number in [10, 2]:
            cluster = raw / f"raw/extracted/clusters/cluster_{number}"
            write(cluster / "doc_1.body.txt", "Tin  tức Hà Nội")
            write(cluster / f"cluster_{number}.ref1.txt", "Tóm tắt Hà Nội")
        expected_documents = ["Tin tức Hà Nội"]
    else:
        for number in [10, 2]:
            root = raw / "raw/extracted/ViMs"
            write(root / f"original/Cluster_{number}/original/doc_1.txt",
                  "Title: Bản tin\nURL: https://example.org\nContent:\nTin  tức Hà Nội")
            write(root / f"summary/Cluster_{number}/ref_1.gold.txt", "Tóm tắt Hà Nội")
            write(root / f"summary/Cluster_{number}/ref_2.gold.txt", "Tóm tắt khác")
        expected_documents = ["Tin tức Hà Nội"]

    spec = get_dataset_spec(name)
    raw_key = spec.raw_config_key.split(".")[-1]
    processed_key = spec.processed_config_key.split(".")[-1]
    config = DataConfig({
        "global": {raw_key: str(raw)},
        "workspace": {processed_key: str(tmp_path / "processed" / name)},
        "datasets": {name: {"test_samples": 2}},
    })
    metadata = prepare_dataset(name, config=config)
    return name, config, metadata, expected_documents


def test_prepare_load_and_select_all_datasets(prepared_dataset):
    name, config, metadata, expected_documents = prepared_dataset
    dataset = load_dataset(name, config=config, prepare=False)
    assert metadata["sample_count"] == len(dataset) == 2
    assert metadata["count_matches_expected"] is True
    first = dataset[0]
    assert first["dataset_name"] == name
    assert first["language"] == ("en" if name.startswith("multi_") else "vi")
    assert first["documents"] == expected_documents
    assert first["reference_summary"]
    assert len({sample["sample_id"] for sample in dataset}) == 2
    if name in {"vn_mds", "vims"}:
        assert first["sample_id"].endswith("002")  # Natural, not lexicographic order.
    if name == "multi_xscience":
        assert first["reference_summary"] == "Related work"
    if name == "vims":
        assert first["metadata"]["reference_summaries"] == ["Tóm tắt Hà Nội", "Tóm tắt khác"]

    selected = load_dataset(name, sample_ids=[dataset[1]["sample_id"]], config=config, prepare=False)
    assert len(selected) == 1
    assert selected[0] == dataset[1]
    assert len(load_dataset(name, num_samples=1, config=config, prepare=False)) == 1


def test_normalization_preserves_vietnamese():
    assert normalize_text("\ufeff  Hà  Nội\r\nNEWLINE_CHAR\n\nTin\t mới  ") == "Hà Nội\n\nTin mới"


def test_invalid_samples_are_rejected():
    with pytest.raises(ValueError, match="missing required fields"):
        BaseDataset.validate_sample({"documents": []})
    sample = dict(sample_id="x", dataset_name="vims", split="test", language="vi",
                  documents=[], reference_summary="test", metadata={})
    with pytest.raises(ValueError, match="must not be empty"):
        BaseDataset.validate_sample(sample)
    sample["documents"] = [123]
    with pytest.raises(TypeError):
        BaseDataset.validate_sample(sample)


def test_unknown_dataset_is_actionable():
    with pytest.raises(ValueError, match="Available:"):
        get_dataset_spec("missing")
