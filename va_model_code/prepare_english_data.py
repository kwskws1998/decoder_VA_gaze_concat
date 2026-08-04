"""Command-line wrapper for deterministic English VA preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__:
    from .decoder_va.downloads import (
        DEFAULT_BUNDLE_FILE_ID,
        DEFAULT_BUNDLE_SHA256,
        download_gdrive_zip,
    )
    from .decoder_va.preprocessing import NORMALIZATION_CHOICES, build_english_dataset
else:
    from decoder_va.downloads import (
        DEFAULT_BUNDLE_FILE_ID,
        DEFAULT_BUNDLE_SHA256,
        download_gdrive_zip,
    )
    from decoder_va.preprocessing import NORMALIZATION_CHOICES, build_english_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or load the seven-source English VA bundle and build two folds."
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--gdrive-url", default=None)
    parser.add_argument("--gdrive-file-id", default=None)
    parser.add_argument(
        "--download-path",
        default="data/external/english_va_bundle.zip",
        help="Local ZIP destination when a Google Drive source is supplied.",
    )
    parser.add_argument(
        "--sha256",
        "--gdrive-sha256",
        dest="sha256",
        default=DEFAULT_BUNDLE_SHA256,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_CHOICES,
        default=None,
        help=(
            "Score normalization. Defaults to observed for the legacy protocol "
            "and source-scale for --paper-protocol."
        ),
    )
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help=(
            "Preserve valid source rows and whitespace, disable deduplication, "
            "force source-scale normalization, and split each source independently "
            "in half before combining the two folds."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Ignore a valid build cache.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--allow-partial-sources",
        action="store_true",
        help="Permit a known-source subset; the standard build requires all seven sources.",
    )
    parser.add_argument(
        "--download-default",
        action="store_true",
        help="Download the configured authorized seven-source bundle before preprocessing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.paper_protocol and args.normalization not in (None, "source-scale"):
        parser.error("--paper-protocol requires --normalization source-scale.")
    normalization = args.normalization or (
        "source-scale" if args.paper_protocol else "observed"
    )
    has_download_source = bool(
        args.gdrive_url or args.gdrive_file_id or args.download_default
    )
    choices = int(bool(args.source_dir)) + int(bool(args.archive)) + int(has_download_source)
    if choices > 1:
        raise ValueError(
            "Choose one input mode: --source-dir, --archive, or a Google Drive URL/file ID."
        )

    source_dir = args.source_dir
    archive = args.archive
    if has_download_source:
        archive = download_gdrive_zip(
            args.download_path,
            url=args.gdrive_url,
            file_id=(
                args.gdrive_file_id
                or (DEFAULT_BUNDLE_FILE_ID if args.download_default else None)
            ),
            expected_sha256=args.sha256,
            force=args.force_download,
        )
    elif choices == 0:
        script_root = Path(__file__).resolve().parent
        source_candidates = (
            Path("data/external/source_tsv"),
            script_root / "data/external/source_tsv",
        )
        archive_candidates = (
            Path("data/external/english_va_bundle.zip"),
            script_root / "data/external/english_va_bundle.zip",
        )
        default_sources = next((path for path in source_candidates if path.is_dir()), None)
        default_archive = next((path for path in archive_candidates if path.is_file()), None)
        if default_sources is not None:
            source_dir = default_sources
        elif default_archive is not None:
            archive = default_archive
        else:
            raise FileNotFoundError(
                "No input was supplied and neither data/external/source_tsv nor "
                "data/external/english_va_bundle.zip exists."
            )

    result = build_english_dataset(
        args.output_dir,
        source_dir=source_dir,
        archive_path=archive,
        expected_sha256=args.sha256 if archive is not None else None,
        seed=args.seed,
        normalization=normalization,
        paper_protocol=args.paper_protocol,
        force=args.force,
        require_all_sources=not args.allow_partial_sources,
    )
    protocol = "paper" if args.paper_protocol else "legacy"
    print(
        f"Prepared {result.total_rows} rows with seed {args.seed} "
        f"({protocol} protocol, {normalization} normalization)."
    )
    for name, count in result.dataset_counts.items():
        print(f"  {name}: {count}")
    print(f"Fold 1: {result.fold1_path}")
    print(f"Fold 2: {result.fold2_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
