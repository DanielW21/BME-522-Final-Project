#!/usr/bin/env python3
"""Download and extract CalMS21 Task 1 using only the Python standard library."""

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://data.caltech.edu/records/s0vdx-0k302/files"
# Checksums published by CaltechDATA for this dataset version.
DOWNLOADS = {
    "task1_classic_classification.zip": "8a02654fddae28614ee24a6a082261b8",
    "readme.md": "ad3faeae6835747fd30fcc0c8e83ad2e",
    "calms21_convert_to_npy.py": "068f0362ea9879a4051b66bd60656c29",
}
CHUNK_SIZE = 1024 * 1024


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, expected_md5):
    """Keep verified downloads; replace missing or damaged files atomically."""
    if destination.is_file() and md5(destination) == expected_md5:
        print(f"Verified, skipping download: {destination.name}", flush=True)
        return

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, 4):
        try:
            print(f"Downloading {destination.name} (attempt {attempt}/3)...", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "BME522-data-setup/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                received = 0
                next_update = 32 * CHUNK_SIZE
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    output.write(chunk)
                    received += len(chunk)
                    if received >= next_update:
                        print(f"  {received / CHUNK_SIZE:.0f} MiB downloaded", flush=True)
                        next_update = received + 32 * CHUNK_SIZE
            if md5(partial) != expected_md5:
                raise ValueError(f"Checksum mismatch for {destination.name}")
            partial.replace(destination)
            print(f"Checksum verified: {destination.name}", flush=True)
            return
        except (OSError, urllib.error.URLError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"Could not download {destination.name}: {error}") from error
            print(f"  Download failed: {error}; retrying...", flush=True)
            time.sleep(attempt)
        finally:
            partial.unlink(missing_ok=True)


def matches_archive(path, member):
    """Check size and CRC, including same-size corruption in extracted files."""
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == member.CRC


def extract(archive, destination):
    """Extract missing or damaged members without replacing valid files."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        targets = []
        for member in source.infolist():
            target = (root / member.filename).resolve()
            # Validate all paths before writing any archive member.
            target.relative_to(root)
            targets.append((member, target))
        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if matches_archive(target, member):
                print(f"Verified, skipping extraction: {member.filename}", flush=True)
                continue
            print(f"Extracting {member.filename}...", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            try:
                with source.open(member) as incoming, partial.open("wb") as output:
                    # ZipFile checks each member's CRC while reading.
                    shutil.copyfileobj(incoming, output, length=CHUNK_SIZE)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data",
        help="Dataset directory (default: this project's data/ directory).",
    )
    args = parser.parse_args()
    raw = args.data_dir.resolve() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "processed").mkdir(parents=True, exist_ok=True)
    for name, checksum in DOWNLOADS.items():
        download(f"{SOURCE}/{name}?download=1", raw / name, checksum)
    extract(raw / "task1_classic_classification.zip", raw)
    print(f"\nCalMS21 Task 1 is ready in {raw / 'task1_classic_classification'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSetup interrupted. Rerun the same command to finish.", file=sys.stderr)
        sys.exit(130)
