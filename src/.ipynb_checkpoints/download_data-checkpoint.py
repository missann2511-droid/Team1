"""Download the raw dataset BEFORE any processing (Checkpoint criterion W6.1).

Idempotent: skips download if the file already exists. The dataset URL is
configurable via the DATA_URL environment variable.
"""
from __future__ import annotations

import sys
import urllib.request

from .config import DATA_URL, RAW_PATH


def download(force: bool = False) -> None:
    if RAW_PATH.exists() and not force:
        print(f"[download] Raw data already present: {RAW_PATH} "
              f"({RAW_PATH.stat().st_size / 1e6:.1f} MB) — skipping.")
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching dataset from {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    except Exception as exc:  # pragma: no cover
        print(f"[download] ERROR: could not download data: {exc}", file=sys.stderr)
        raise
    print(f"[download] Saved to {RAW_PATH} "
          f"({RAW_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
