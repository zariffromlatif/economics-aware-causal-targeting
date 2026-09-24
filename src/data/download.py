"""Dataset download and validation utility.

Downloads the Hillstrom MineThatData E-Mail Analytics challenge dataset
and validates its integrity (schema, row counts, and no missing values).
"""

import os
import sys
import hashlib
from pathlib import Path
import urllib.request
import pandas as pd
import yaml

EXPECTED_ROWS = 64000
EXPECTED_COLS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]

DEFAULT_URLS = [
    "https://raw.githubusercontent.com/maks-sh/scikit-uplift/master/sklift/datasets/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    "https://raw.githubusercontent.com/uber/causalml/master/causalml/datasets/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
]


def load_config(config_path: str = "configs/experiment_hillstrom.yaml") -> dict:
    """Load experiment configuration YAML."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def download_hillstrom(
    target_path: str = "data/raw/hillstrom.csv",
    urls: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Download the Hillstrom MineThatData CSV file with fallback URLs.

    Args:
        target_path: Destination path on disk.
        urls: List of mirror URLs to try.
        force: If True, redownload even if file exists.

    Returns:
        Path to downloaded and verified CSV file.
    """
    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[INFO] Dataset already exists at {dest}. Verifying...")
        df = pd.read_csv(dest)
        _verify_dataframe(df)
        return dest

    candidate_urls = urls or DEFAULT_URLS
    downloaded = False
    last_error = None

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in candidate_urls:
        print(f"[INFO] Attempting download from: {url}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(dest, "wb") as out_file:
                out_file.write(response.read())
            print(f"[SUCCESS] Download completed from {url}")
            downloaded = True
            break
        except Exception as e:
            print(f"[WARNING] Failed from {url}: {e}")
            last_error = e

    if not downloaded:
        raise RuntimeError(f"Failed to download Hillstrom dataset from all mirrors: {last_error}")

    # Validate downloaded dataset
    df = pd.read_csv(dest)
    _verify_dataframe(df)

    # Compute sha256 checksum
    with open(dest, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[INFO] SHA256 Checksum: {sha256}")
    print(f"[INFO] Dataset verified: {len(df):,} rows, {len(df.columns)} columns.")

    return dest


def _verify_dataframe(df: pd.DataFrame) -> None:
    """Verify that dataframe conforms exactly to Hillstrom schema."""
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
    missing_cols = set(EXPECTED_COLS) - set(df.columns)
    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    null_counts = df.isnull().sum().sum()
    assert null_counts == 0, f"Found {null_counts} unexpected null values in dataset"


if __name__ == "__main__":
    cfg = load_config()
    target = cfg.get("data", {}).get("raw_path", "data/raw/hillstrom.csv")
    urls = cfg.get("data", {}).get("remote_urls", DEFAULT_URLS)
    download_hillstrom(target_path=target, urls=urls)
