from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests
from tqdm import tqdm


DATASET_SPECS = {
    "multinews": {
        "kind": "raw_files",
        "target_dir": "multinews",
        "files": {
            "raw/train.src.cleaned": "https://huggingface.co/datasets/multi_news/resolve/main/data/train.src.cleaned",
            "raw/train.tgt": "https://huggingface.co/datasets/multi_news/resolve/main/data/train.tgt",
            "raw/val.src.cleaned": "https://huggingface.co/datasets/multi_news/resolve/main/data/val.src.cleaned",
            "raw/val.tgt": "https://huggingface.co/datasets/multi_news/resolve/main/data/val.tgt",
            "raw/test.src.cleaned": "https://huggingface.co/datasets/multi_news/resolve/main/data/test.src.cleaned",
            "raw/test.tgt": "https://huggingface.co/datasets/multi_news/resolve/main/data/test.tgt",
        },
    },

    "multixscience": {
        "kind": "raw_files",
        "target_dir": "multixscience",
        "files": {
            "raw/train.json.gz": "https://github.com/yaolu/Multi-XScience/raw/master/data/train.json.gz",
            "raw/val.json.gz": "https://github.com/yaolu/Multi-XScience/raw/master/data/val.json.gz",
            "raw/test.json.gz": "https://github.com/yaolu/Multi-XScience/raw/master/data/test.json.gz",
        },
        "gunzip": True,
    },

    "vnmds": {
        "kind": "zip",
        "target_dir": "vnmds",
        "filename": "KTLAB-200document-clusters.zip",
        "url": "https://raw.githubusercontent.com/lupanh/VietnameseMDS/master/KTLAB-200document-clusters.zip",
    },

    "vims": {
        "kind": "zip",
        "target_dir": "vims",
        "filename": "ViMs.zip",
        "url": "https://raw.githubusercontent.com/CLC-HCMUS/ViMs-Dataset/master/ViMs.zip",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download_file(url: str, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"[SKIP] File already exists: {output_path}")
        return

    ensure_dir(output_path.parent)

    print(f"[DOWNLOAD] {url}")
    print(f"[TO]       {output_path}")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

def download_raw_files_dataset(spec: dict, root_dir: Path, force: bool = False) -> None:
    dataset_dir = root_dir / spec["target_dir"]
    ensure_dir(dataset_dir)

    downloaded_files = []

    for relative_path, url in spec["files"].items():
        output_path = dataset_dir / relative_path
        download_file(url, output_path, force=force)
        downloaded_files.append(str(output_path))

        if spec.get("gunzip") and output_path.suffix == ".gz":
            decompressed_path = output_path.with_suffix("")
            decompress_gz(output_path, decompressed_path, force=force)
            downloaded_files.append(str(decompressed_path))

    write_json(
        dataset_dir / "download_metadata.json",
        {
            "source": "raw_files",
            "files": spec["files"],
            "downloaded_files": downloaded_files,
        },
    )

def safe_extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> None:
    marker = extract_dir / ".extract_complete"

    if marker.exists() and not force:
        print(f"[SKIP] Already extracted: {extract_dir}")
        return

    ensure_dir(extract_dir)

    print(f"[UNZIP] {zip_path}")
    print(f"[TO]    {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target_path = extract_dir / member.filename
            resolved_target = target_path.resolve()
            resolved_root = extract_dir.resolve()

            # Prevent zip-slip path traversal.
            if not str(resolved_target).startswith(str(resolved_root)):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")

        zf.extractall(extract_dir)

    marker.write_text("ok", encoding="utf-8")


def decompress_gz(gz_path: Path, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"[SKIP] Decompressed file already exists: {output_path}")
        return

    ensure_dir(output_path.parent)

    print(f"[GUNZIP] {gz_path}")
    print(f"[TO]     {output_path}")

    with gzip.open(gz_path, "rb") as src, open(output_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def export_hf_dataset_to_jsonl(dataset, output_dir: Path) -> None:
    """
    Export HF DatasetDict splits to JSONL files for transparent raw inspection.
    """
    ensure_dir(output_dir)

    for split_name, split_data in dataset.items():
        out_path = output_dir / f"{split_name}.jsonl"

        print(f"[EXPORT] HF split '{split_name}' -> {out_path}")

        with out_path.open("w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_hf_dataset(
    hf_id: str,
    target_dir: Path,
    hf_config: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
):
    """
    Download a Hugging Face dataset and save it to disk.
    """
    from datasets import load_dataset

    hf_dataset_dir = target_dir / "hf_dataset"
    raw_jsonl_dir = target_dir / "raw_jsonl"

    if hf_dataset_dir.exists() and not force:
        print(f"[SKIP] HF dataset already exists: {hf_dataset_dir}")
        return

    ensure_dir(target_dir)

    print(f"[HF] Loading dataset: {hf_id}")
    if hf_config:
        print(f"[HF] Config: {hf_config}")

    load_kwargs = {
        "path": hf_id,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "trust_remote_code": True,
    }

    # Remove None values because datasets.load_dataset may not expect them.
    load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

    if hf_config:
        dataset = load_dataset(hf_id, hf_config, **{k: v for k, v in load_kwargs.items() if k != "path"})
    else:
        dataset = load_dataset(**load_kwargs)

    print(f"[HF] Saving dataset to: {hf_dataset_dir}")
    dataset.save_to_disk(str(hf_dataset_dir))

    export_hf_dataset_to_jsonl(dataset, raw_jsonl_dir)

    write_json(
        target_dir / "download_metadata.json",
        {
            "source": "huggingface",
            "hf_id": hf_id,
            "hf_config": hf_config,
            "hf_dataset_dir": str(hf_dataset_dir),
            "raw_jsonl_dir": str(raw_jsonl_dir),
        },
    )


def download_hf_with_candidates(
    candidates: Iterable[Dict[str, Optional[str]]],
    target_dir: Path,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> None:
    last_error = None

    for candidate in candidates:
        hf_id = candidate["hf_id"]
        hf_config = candidate.get("hf_config")

        try:
            download_hf_dataset(
                hf_id=hf_id,
                hf_config=hf_config,
                target_dir=target_dir,
                cache_dir=cache_dir,
                force=force,
            )
            return
        except Exception as exc:
            print(f"[WARN] Failed to download HF dataset {hf_id}: {exc}")
            last_error = exc

    raise RuntimeError(f"All HF candidates failed. Last error: {last_error}")


def download_zip_dataset(spec: dict, root_dir: Path, force: bool = False) -> None:
    dataset_dir = root_dir / spec["target_dir"]
    raw_dir = dataset_dir / "raw"
    archive_dir = dataset_dir / "archives"
    extract_dir = raw_dir / "extracted"

    ensure_dir(dataset_dir)
    ensure_dir(raw_dir)
    ensure_dir(archive_dir)

    archive_path = archive_dir / spec["filename"]

    download_file(spec["url"], archive_path, force=force)
    safe_extract_zip(archive_path, extract_dir, force=force)

    write_json(
        dataset_dir / "download_metadata.json",
        {
            "source": "github_raw_zip",
            "url": spec["url"],
            "archive_path": str(archive_path),
            "extract_dir": str(extract_dir),
        },
    )


def download_multixscience_raw_gz(spec: dict, dataset_dir: Path, force: bool = False) -> None:
    """
    Keep original raw .json.gz files from the Multi-XScience repository.
    This is useful even if the HF loader succeeds, because it preserves
    the paper-released raw files directly.
    """
    raw_gz_dir = dataset_dir / "original_gz"
    raw_json_dir = dataset_dir / "original_json"

    ensure_dir(raw_gz_dir)
    ensure_dir(raw_json_dir)

    for filename, url in spec["raw_gz_files"].items():
        gz_path = raw_gz_dir / filename
        json_path = raw_json_dir / filename.replace(".json.gz", ".json")

        download_file(url, gz_path, force=force)
        decompress_gz(gz_path, json_path, force=force)


def download_dataset(name: str, root_dir: Path, cache_dir: Optional[Path], force: bool = False) -> None:
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_SPECS)}")

    spec = DATASET_SPECS[name]
    dataset_dir = root_dir / spec["target_dir"]
    ensure_dir(dataset_dir)

    print("=" * 80)
    print(f"[DATASET] {name}")
    print(f"[DIR]     {dataset_dir}")
    print("=" * 80)

    kind = spec["kind"]

    if kind == "hf":
        download_hf_dataset(
            hf_id=spec["hf_id"],
            hf_config=spec.get("hf_config"),
            target_dir=dataset_dir,
            cache_dir=cache_dir,
            force=force,
        )
        

    elif kind == "hf_plus_raw_gz":
        download_hf_with_candidates(
            candidates=spec["hf_candidates"],
            target_dir=dataset_dir,
            cache_dir=cache_dir,
            force=force,
        )

        # Also preserve original raw .json.gz files from the paper repo.
        download_multixscience_raw_gz(spec, dataset_dir, force=force)

    elif kind == "zip":
        download_zip_dataset(spec, root_dir=root_dir, force=force)
        
    elif kind == "raw_files":
        download_raw_files_dataset(spec, root_dir=root_dir, force=force)

    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    print(f"[DONE] {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw datasets for MoA-MDS experiments."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/data/mds",
        help="Root directory for raw datasets. Default: /data/mds",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Datasets to download: multinews multixscience vnmds vims all. "
            "Default: all"
        ),
    )

    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extract.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    cache_dir = Path(args.hf_cache_dir).expanduser().resolve() if args.hf_cache_dir else None

    ensure_dir(root_dir)

    selected = args.datasets
    if "all" in selected:
        selected = list(DATASET_SPECS.keys())

    for dataset_name in selected:
        download_dataset(
            name=dataset_name,
            root_dir=root_dir,
            cache_dir=cache_dir,
            force=args.force,
        )

if __name__ == "__main__":
    main()