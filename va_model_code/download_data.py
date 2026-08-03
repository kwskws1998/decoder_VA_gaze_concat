"""Download and validate one authorized English VA ZIP from Google Drive."""

from __future__ import annotations

import argparse
from typing import Sequence

if __package__:
    from .decoder_va.downloads import (
        DEFAULT_BUNDLE_FILE_ID,
        DEFAULT_BUNDLE_SHA256,
        download_gdrive_zip,
        sha256_file,
    )
else:
    from decoder_va.downloads import (
        DEFAULT_BUNDLE_FILE_ID,
        DEFAULT_BUNDLE_SHA256,
        download_gdrive_zip,
        sha256_file,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url")
    source.add_argument("--file-id")
    parser.add_argument(
        "--output", default="data/external/english_va_bundle.zip"
    )
    parser.add_argument("--sha256", default=DEFAULT_BUNDLE_SHA256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = download_gdrive_zip(
        args.output,
        url=args.url,
        file_id=args.file_id or (None if args.url else DEFAULT_BUNDLE_FILE_ID),
        expected_sha256=args.sha256,
        force=args.force,
    )
    print(f"Downloaded ZIP: {path}")
    print(f"SHA256: {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
