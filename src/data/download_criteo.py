"""Criteo Uplift Prediction Dataset Ingestion Utility.

Supports streaming download of the 13.98M observation Criteo benchmark dataset.
Provides options for:
1. Full dataset download (for Power PC RTX 4090 / 64GB DDR5 workstation).
2. Subsampled representative debug benchmark (for local laptop prototyping).
"""

from pathlib import Path
import urllib.request
import gzip
import shutil
import pandas as pd

CRITEO_URL = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
DEFAULT_RAW_DEST = "data/raw/criteo-uplift-v2.1.csv.gz"


def download_criteo(
    dest_path: str = DEFAULT_RAW_DEST,
    url: str = CRITEO_URL,
    force: bool = False,
) -> Path:
    """Download compressed Criteo uplift benchmark dataset."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[INFO] Criteo dataset already exists at {dest}.")
        return dest

    print(f"[INFO] Downloading Criteo Uplift dataset (~300MB compressed) from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    print(f"[SUCCESS] Criteo dataset downloaded to {dest}")
    return dest


def extract_criteo_subsample(
    source_gz: str = DEFAULT_RAW_DEST,
    output_csv: str = "data/processed/criteo_sample_500k.parquet",
    sample_size: int = 500_000,
    seed: int = 42,
) -> Path:
    """Extract a representative stratified sample for local development before Power PC execution."""
    src = Path(source_gz)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"[INFO] Subsample already exists at {out}.")
        return out

    print(f"[INFO] Reading Criteo compressed stream and sampling {sample_size:,} rows...")
    # Read chunkwise to preserve memory
    chunks = []
    total_read = 0
    chunk_size = 100_000

    for chunk in pd.read_csv(src, compression="gzip", chunksize=chunk_size):
        chunks.append(chunk)
        total_read += len(chunk)
        if total_read >= sample_size:
            break

    df_sample = pd.concat(chunks, ignore_index=True).iloc[:sample_size]
    df_sample.to_parquet(out, index=False)
    print(f"[SUCCESS] Sample saved to {out} ({len(df_sample):,} rows).")
    return out


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "info"
    if action == "download":
        download_criteo()
    elif action == "sample":
        extract_criteo_subsample()
    else:
        print("Usage: python -m src.data.download_criteo [download|sample]")
