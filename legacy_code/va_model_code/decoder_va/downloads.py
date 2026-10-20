"""Validated Google Drive ZIP downloads and safe TSV extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Callable, Iterable
import uuid
import zipfile


DEFAULT_BUNDLE_FILE_ID = "1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj"
DEFAULT_BUNDLE_SHA256 = (
    "5db750ededfd9717dcca465b34fd7e6c348e50e563ad2c0814c458b04441e81d"
)
DOWNLOAD_METADATA_VERSION = 1
MAX_ARCHIVE_MEMBERS = 1024
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200.0


@dataclass(frozen=True)
class GDriveSource:
    """A single Google Drive file, identified by URL or file ID."""

    file_id: str | None = None
    url: str | None = None

    def resolved_file_id(self) -> str | None:
        explicit = self.file_id.strip() if self.file_id else None
        parsed = google_drive_file_id(self.url) if self.url else None
        if explicit and parsed and explicit != parsed:
            raise ValueError("The Google Drive URL and --file-id refer to different files.")
        if explicit and not _is_drive_file_id(explicit):
            raise ValueError(f"Invalid Google Drive file ID: {explicit!r}")
        return explicit or parsed

    def validate(self) -> None:
        if not self.file_id and not self.url:
            raise ValueError("Provide a Google Drive file ID or a direct file share URL.")
        if self.url:
            stripped = self.url.strip()
            if not stripped.startswith(("https://", "http://")):
                raise ValueError("Google Drive URL must start with http:// or https://.")
            if "drive.google.com" in stripped and google_drive_file_id(stripped) is None:
                raise ValueError(
                    "The URL does not identify one Google Drive file. Open the ZIP's "
                    "share dialog and use a /file/d/<id>/... link or pass --file-id."
                )

    def identity(self) -> dict[str, str]:
        self.validate()
        file_id = self.resolved_file_id()
        if file_id:
            return {"kind": "google_drive_file_id", "value": file_id}
        return {"kind": "url", "value": self.url.strip()}


def _is_drive_file_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", value))


def google_drive_file_id(url_or_id: str | None) -> str | None:
    """Return a Drive file ID from a bare ID or supported file-share URL."""

    if url_or_id is None:
        return None
    value = str(url_or_id).strip()
    if not value:
        return None
    if _is_drive_file_id(value) and "/" not in value:
        return value
    patterns = (
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/uc\?(?:[^#]*&)?id=([A-Za-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def validate_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("Expected SHA256 must contain exactly 64 hexadecimal characters.")
    return normalized


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_safe_member(member: zipfile.ZipInfo) -> bool:
    name = member.filename
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    unix_mode = (member.external_attr >> 16) & 0o170000
    return unix_mode != stat.S_IFLNK


def validate_tsv_zip(
    path: str | os.PathLike[str],
    expected_sha256: str | None = None,
) -> tuple[str, ...]:
    """Validate checksum, CRC, paths, sizes, and unique TSV basenames."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"ZIP archive not found: {archive_path}")
    expected = validate_sha256(expected_sha256)
    if expected is not None:
        actual = sha256_file(archive_path)
        if actual != expected:
            raise ValueError(
                f"SHA256 mismatch for {archive_path}: expected {expected}, got {actual}"
            )

    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {archive_path}") from exc

    with archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"ZIP has {len(members)} files; limit is {MAX_ARCHIVE_MEMBERS}."
            )

        total_size = 0
        selected: list[str] = []
        basenames: set[str] = set()
        for member in members:
            if not _is_safe_member(member):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"ZIP member is too large: {member.filename}")
            total_size += member.file_size
            if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP exceeds the total uncompressed-size limit.")
            if member.file_size:
                ratio = member.file_size / max(member.compress_size, 1)
                if ratio > MAX_COMPRESSION_RATIO:
                    raise ValueError(
                        f"ZIP member has an excessive compression ratio: {member.filename}"
                    )

            base_name = PurePosixPath(member.filename).name
            if member.filename.startswith("__MACOSX/") or base_name.startswith("._"):
                continue
            if not base_name.casefold().endswith(".tsv"):
                continue
            key = base_name.casefold()
            if key in basenames:
                raise ValueError(
                    "ZIP contains duplicate TSV basenames that would collide: "
                    f"{base_name}"
                )
            basenames.add(key)
            selected.append(member.filename)

        if not selected:
            raise ValueError(f"ZIP contains no TSV source files: {archive_path}")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"ZIP CRC check failed for member: {corrupt_member}")

    return tuple(sorted(selected, key=lambda value: value.casefold()))


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".download.json")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_matches(
    destination: Path,
    source: GDriveSource,
    expected_sha256: str | None,
) -> bool:
    try:
        validate_tsv_zip(destination, expected_sha256)
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    if expected_sha256 is not None:
        return True
    metadata_path = _metadata_path(destination)
    if not metadata_path.is_file():
        return False
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == DOWNLOAD_METADATA_VERSION
        and metadata.get("source") == source.identity()
        and metadata.get("archive_sha256") == sha256_file(destination)
        and metadata.get("size_bytes") == destination.stat().st_size
    )


def download_gdrive_zip(
    destination: str | os.PathLike[str],
    *,
    url: str | None = None,
    file_id: str | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
    downloader: Callable[..., str | None] | None = None,
) -> Path:
    """Download to a temporary path, validate, then atomically publish the ZIP."""

    source = GDriveSource(file_id=file_id, url=url)
    source.validate()
    expected = validate_sha256(expected_sha256)
    output_path = Path(destination)
    if output_path.name in {"", ".", ".."}:
        raise ValueError("Download destination must be a ZIP filename.")
    if output_path.suffix.casefold() != ".zip":
        raise ValueError("Download destination must end in .zip.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file() and not force and _cache_matches(output_path, source, expected):
        return output_path

    if downloader is None:
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "gdown is required for Google Drive downloads. Install it with "
                "`python -m pip install gdown`."
            ) from exc
        downloader = gdown.download

    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.part")
    try:
        resolved_id = source.resolved_file_id()
        kwargs: dict[str, object] = {
            "output": str(temporary),
            "quiet": False,
        }
        if resolved_id:
            kwargs["id"] = resolved_id
        else:
            kwargs["url"] = source.url.strip()
            kwargs["fuzzy"] = True
        result = downloader(**kwargs)
        if not result or not temporary.is_file():
            raise RuntimeError("gdown did not create the requested ZIP archive.")
        validate_tsv_zip(temporary, expected)
        archive_sha256 = sha256_file(temporary)
        size_bytes = temporary.stat().st_size
        os.replace(temporary, output_path)
        _atomic_json(
            _metadata_path(output_path),
            {
                "schema_version": DOWNLOAD_METADATA_VERSION,
                "source": source.identity(),
                "expected_sha256": expected,
                "archive_sha256": archive_sha256,
                "size_bytes": size_bytes,
            },
        )
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def extract_tsv_zip(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    expected_sha256: str | None = None,
) -> tuple[Path, ...]:
    """Safely extract validated TSV members into a new directory."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    selected = validate_tsv_zip(archive_path, expected_sha256)
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    published = False
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member_name in selected:
                base_name = PurePosixPath(member_name).name
                target = staging / base_name
                with archive.open(member_name, "r") as input_file, open(target, "xb") as output_file:
                    for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                        output_file.write(chunk)
        os.replace(staging, destination)
        published = True
    finally:
        if not published and staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
    return tuple(sorted(destination.glob("*.tsv"), key=lambda path: path.name.casefold()))


def file_hash_records(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, object]]:
    return [
        {
            "name": Path(path).name,
            "sha256": sha256_file(path),
            "size_bytes": Path(path).stat().st_size,
        }
        for path in sorted((Path(path) for path in paths), key=lambda item: item.name.casefold())
    ]
