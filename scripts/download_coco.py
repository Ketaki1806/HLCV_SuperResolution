"""Download COCO image splits. Intended for the HPC cluster, not local machines."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

COCO_IMAGE_URLS = {
    "train2017": "https://images.cocodataset.org/zips/train2017.zip",
    "val2017": "https://images.cocodataset.org/zips/val2017.zip",
}


class _DownloadProgress:
    def __init__(self) -> None:
        self._bar: tqdm | None = None

    def __call__(self, block_count: int, block_size: int, total_size: int) -> None:
        if self._bar is None:
            self._bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
        self._bar.update(block_count * block_size)


def download_split(split: str, root: Path) -> Path:
    if split not in COCO_IMAGE_URLS:
        raise ValueError(f"Unknown split {split!r}. Choose from {list(COCO_IMAGE_URLS)}")

    root.mkdir(parents=True, exist_ok=True)
    split_dir = root / split
    if split_dir.is_dir() and any(split_dir.glob("*.jpg")):
        print(f"{split} already present at {split_dir}")
        return split_dir

    url = COCO_IMAGE_URLS[split]
    zip_path = root / f"{split}.zip"

    print(f"Downloading {split} from {url}")
    urlretrieve(url, zip_path, reporthook=_DownloadProgress())

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root)

    zip_path.unlink(missing_ok=True)
    print(f"COCO {split} ready at {split_dir}")
    return split_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO image splits")
    parser.add_argument(
        "--split",
        choices=list(COCO_IMAGE_URLS),
        default="val2017",
        help="COCO split to download (default: val2017)",
    )
    default_root = Path(os.environ.get("COCO_ROOT", Path.home() / "data" / "coco"))
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Directory where COCO data will be stored (default: $COCO_ROOT or ~/data/coco)",
    )
    args = parser.parse_args()
    download_split(args.split, args.root)


if __name__ == "__main__":
    main()
