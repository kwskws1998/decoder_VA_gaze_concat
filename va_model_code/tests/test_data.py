from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import pytest
import torch

from va_model_code.decoder_va.dataset import TokenizedVADataset, VABatchCollator
from va_model_code.decoder_va.downloads import (
    GDriveSource,
    download_gdrive_zip,
    google_drive_file_id,
    sha256_file,
    validate_tsv_zip,
)
from va_model_code.decoder_va.filters import (
    apply_dataset_filters,
    collect_exclude_patterns,
    filter_fold_frames,
    read_fold,
    resolve_excluded_datasets,
)
from va_model_code.decoder_va.preprocessing import (
    FOLD_FILENAMES,
    LEGACY_DEDUP_POLICY,
    LEGACY_PROTOCOL,
    LEGACY_SPLIT_STRATEGY,
    LEGACY_TEXT_POLICY,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    MERGED_FILENAME,
    OUTPUT_COLUMNS,
    PAPER_DEDUP_POLICY,
    PAPER_PROTOCOL,
    PAPER_SPLIT_STRATEGY,
    PAPER_TEXT_POLICY,
    SOURCE_NAME_MAP,
    build_english_dataset,
)
from va_model_code.prepare_english_data import main as prepare_english_data


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _make_small_bundle(source_dir: Path) -> None:
    source_dir.mkdir()
    _write_source(
        source_dir / "emobank.tsv",
        [
            {"text": "  hello   world  ", "valence": 0.2, "arousal": 0.3},
            {"text": "hello world", "valence": 0.9, "arousal": 0.8},
            {"text": "hello world", "valence": 0.7, "arousal": 0.6},
            {"text": "   ", "valence": 0.4, "arousal": 0.5},
            {"text": None, "valence": 0.6, "arousal": 0.7},
            {"text": "invalid", "valence": "bad", "arousal": 0.1},
        ],
    )
    _write_source(
        source_dir / "facebook_va.tsv",
        [
            {"text": "low", "valence": 1.0, "arousal": 1.0},
            {"text": "middle", "valence": 5.0, "arousal": 5.0},
            {"text": "high", "valence": 9.0, "arousal": 9.0},
        ],
    )
    remaining = {
        "emotales": (0.11, 0.12),
        "iemocap": (0.21, 0.22),
        "nrc_vad": (0.31, 0.32),
        "scott_et_al": (0.41, 0.42),
        "warriner_et_al": (0.51, 0.52),
    }
    for stem, (valence, arousal) in remaining.items():
        text = (
            "cross-source duplicate"
            if stem in {"emotales", "iemocap"}
            else stem
        )
        _write_source(
            source_dir / f"{stem}.tsv",
            [{"text": text, "valence": valence, "arousal": arousal}],
        )


def test_google_drive_source_parsing_rejects_my_drive_page() -> None:
    file_id = "1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj"
    assert google_drive_file_id(file_id) == file_id
    assert (
        google_drive_file_id(f"https://drive.google.com/file/d/{file_id}/view?usp=sharing")
        == file_id
    )
    with pytest.raises(ValueError, match="does not identify one Google Drive file"):
        GDriveSource(url="https://drive.google.com/drive/my-drive").validate()


def test_download_is_checksum_validated_and_atomic(tmp_path: Path) -> None:
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("emobank.tsv", "text\tvalence\tarousal\nok\t.5\t.5\n")
    digest = sha256_file(source_zip)
    destination = tmp_path / "downloaded.zip"

    def fake_download(**kwargs):
        shutil.copyfile(source_zip, kwargs["output"])
        return kwargs["output"]

    result = download_gdrive_zip(
        destination,
        file_id="test_file_id",
        expected_sha256=digest,
        downloader=fake_download,
    )
    assert result == destination
    assert validate_tsv_zip(result, digest) == ("emobank.tsv",)
    original = destination.read_bytes()

    def corrupt_download(**kwargs):
        Path(kwargs["output"]).write_bytes(b"not a zip")
        return kwargs["output"]

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_gdrive_zip(
            destination,
            file_id="test_file_id",
            expected_sha256=digest,
            force=True,
            downloader=corrupt_download,
        )
    assert destination.read_bytes() == original


def test_zip_validation_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../emobank.tsv", "text\tvalence\tarousal\n")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        validate_tsv_zip(archive_path)


def test_preprocessing_preserves_legacy_semantics(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_tsv"
    _make_small_bundle(source_dir)
    output_a = tmp_path / "data_a"
    result_a = build_english_dataset(output_a, source_dir=source_dir, seed=42)
    output_b = tmp_path / "data_b"
    result_b = build_english_dataset(output_b, source_dir=source_dir, seed=42)

    assert result_a.total_rows == 10
    assert len(pd.read_csv(result_a.fold1_path, sep="\t", keep_default_na=False)) == 5
    assert len(pd.read_csv(result_a.fold2_path, sep="\t", keep_default_na=False)) == 5
    assert sha256_file(result_a.fold1_path) == sha256_file(result_b.fold1_path)
    assert sha256_file(result_a.fold2_path) == sha256_file(result_b.fold2_path)

    merged = pd.read_csv(result_a.merged_path, sep="\t", keep_default_na=False)
    assert tuple(merged.columns) == OUTPUT_COLUMNS
    assert merged["index"].tolist() == list(range(10))
    assert (merged["text"] == "").sum() == 1
    emobank = merged[merged["dataset_of_origin"] == "Emobank"]
    assert sorted(emobank["text"].tolist()) == ["", "hello world"]
    assert emobank.loc[emobank["text"] == "hello world", "valence"].item() == pytest.approx(0.2)
    facebook = merged[merged["dataset_of_origin"] == "fb"].set_index("text")
    assert facebook.loc["low", "valence"] == pytest.approx(0.0)
    assert facebook.loc["middle", "valence"] == pytest.approx(0.5)
    assert facebook.loc["high", "valence"] == pytest.approx(1.0)

    manifest = json.loads(result_a.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_VERSION
    assert manifest["protocol"] == LEGACY_PROTOCOL
    assert manifest["paper_protocol"] is False
    assert manifest["normalization"] == "observed"
    assert manifest["text_policy"] == LEGACY_TEXT_POLICY
    assert manifest["dedup_policy"] == LEGACY_DEDUP_POLICY
    assert manifest["split"]["strategy"] == LEGACY_SPLIT_STRATEGY
    assert manifest["build"]["protocol"] == LEGACY_PROTOCOL
    assert manifest["build"]["text_policy"] == LEGACY_TEXT_POLICY
    assert manifest["build"]["dedup_policy"] == LEGACY_DEDUP_POLICY
    assert [item["filename"] for item in manifest["sources"]] == sorted(
        (f"{stem}.tsv" for stem in SOURCE_NAME_MAP if stem != "fb"),
        key=str.casefold,
    )
    assert manifest["blank_text_rows_retained"] == 1
    assert manifest["sources"][0]["invalid_va_rows_dropped"] == 1
    for filename in (*FOLD_FILENAMES, MERGED_FILENAME):
        path = output_a / filename
        assert manifest["outputs"][filename]["sha256"] == sha256_file(path)


def test_paper_protocol_cli_preserves_rows_and_splits_each_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source_tsv"
    _make_small_bundle(source_dir)
    output_a = tmp_path / "paper_a"
    assert (
        prepare_english_data(
            [
                "--source-dir",
                str(source_dir),
                "--paper-protocol",
                "--output-dir",
                str(output_a),
                "--seed",
                "42",
            ]
        )
        == 0
    )
    output_b = tmp_path / "paper_b"
    result_b = build_english_dataset(
        output_b,
        source_dir=source_dir,
        seed=42,
        paper_protocol=True,
    )

    merged = pd.read_csv(
        output_a / MERGED_FILENAME,
        sep="\t",
        keep_default_na=False,
    )
    assert len(merged) == 13
    assert result_b.total_rows == 13
    assert merged["index"].tolist() == list(range(13))
    assert "  hello   world  " in merged["text"].tolist()
    assert "   " in merged["text"].tolist()
    assert "" in merged["text"].tolist()
    assert (merged["text"] == "hello world").sum() == 2
    assert (merged["text"] == "cross-source duplicate").sum() == 2

    facebook = merged[merged["dataset_of_origin"] == "fb"].set_index("text")
    assert facebook.loc["low", "valence"] == pytest.approx(0.0)
    assert facebook.loc["middle", "valence"] == pytest.approx(0.5)
    assert facebook.loc["high", "valence"] == pytest.approx(1.0)

    manifest = json.loads(
        (output_a / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == MANIFEST_VERSION
    assert manifest["protocol"] == PAPER_PROTOCOL
    assert manifest["paper_protocol"] is True
    assert manifest["normalization"] == "source-scale"
    assert manifest["text_policy"] == PAPER_TEXT_POLICY
    assert manifest["dedup_policy"] == PAPER_DEDUP_POLICY
    assert manifest["split"]["strategy"] == PAPER_SPLIT_STRATEGY
    assert manifest["build"]["protocol"] == PAPER_PROTOCOL
    assert manifest["build"]["text_policy"] == PAPER_TEXT_POLICY
    assert manifest["build"]["dedup_policy"] == PAPER_DEDUP_POLICY
    assert all(
        source["duplicate_text_rows_dropped"] == 0
        for source in manifest["sources"]
    )

    source_counts = manifest["split"]["per_source_fold_counts"]
    for counts in source_counts.values():
        assert counts["fold1_rows"] + counts["fold2_rows"] == counts["total_rows"]
        assert abs(counts["fold1_rows"] - counts["fold2_rows"]) <= 1

    assert sha256_file(output_a / FOLD_FILENAMES[0]) == sha256_file(
        result_b.fold1_path
    )
    assert sha256_file(output_a / FOLD_FILENAMES[1]) == sha256_file(
        result_b.fold2_path
    )
    fold_indices = set()
    for filename in FOLD_FILENAMES:
        frame = read_fold(output_a / filename)
        assert fold_indices.isdisjoint(frame["index"])
        fold_indices.update(frame["index"])
    assert fold_indices == set(range(13))


def test_paper_protocol_rejects_observed_normalization(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_tsv"
    _make_small_bundle(source_dir)
    with pytest.raises(ValueError, match="requires normalization='source-scale'"):
        build_english_dataset(
            tmp_path / "paper",
            source_dir=source_dir,
            paper_protocol=True,
            normalization="observed",
        )


def test_actual_bundle_row_counts_and_no_iemocap_regression(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    archive = project_root / "data/external/english_va_bundle.zip"
    expected = "5db750ededfd9717dcca465b34fd7e6c348e50e563ad2c0814c458b04441e81d"
    result = build_english_dataset(
        tmp_path / "data",
        archive_path=archive,
        expected_sha256=expected,
        seed=42,
    )
    assert result.total_rows == 61_614
    assert result.dataset_counts == {
        "EmoTales sentences": 1_369,
        "Emobank": 9_906,
        "GlasgowNorms": 5_553,
        "IEMOCAP sentences": 8_013,
        "fb": 2_887,
        "nrc-vad": 19_971,
        "word ratings ENG": 13_915,
    }
    assert sha256_file(result.fold1_path) == (
        "ba9a833084064ab5807dc600aa834d2c21c8d3bb9019ba40147196995d6cc714"
    )
    assert sha256_file(result.fold2_path) == (
        "1a972b4ed29692de47887991340bfe1656b85a6f8a9e84dacfc88553f6f22f2f"
    )
    assert sha256_file(result.merged_path) == (
        "957c20b155a7ad1069a0ec168dacc00fbc494ce4b5adeae403c0d1c0468f3a01"
    )
    folds = {
        filename: read_fold(tmp_path / "data" / filename)
        for filename in FOLD_FILENAMES
    }
    filtered = filter_fold_frames(folds, no_ieomcap=True)
    assert filtered.excluded_names == ("IEMOCAP sentences",)
    assert sum(len(frame) for frame in filtered.folds.values()) == 53_601
    manifest = json.loads((tmp_path / "data" / MANIFEST_FILENAME).read_text())
    assert manifest["blank_text_rows_retained"] == 3


def test_actual_bundle_paper_protocol_counts_and_source_balance(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    archive = project_root / "data/external/english_va_bundle.zip"
    expected = "5db750ededfd9717dcca465b34fd7e6c348e50e563ad2c0814c458b04441e81d"
    result = build_english_dataset(
        tmp_path / "data_paper",
        archive_path=archive,
        expected_sha256=expected,
        seed=42,
        paper_protocol=True,
    )
    assert result.total_rows == 63_823
    assert result.dataset_counts == {
        "EmoTales sentences": 1_395,
        "Emobank": 10_062,
        "GlasgowNorms": 5_553,
        "IEMOCAP sentences": 10_032,
        "fb": 2_895,
        "nrc-vad": 19_971,
        "word ratings ENG": 13_915,
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["protocol"] == PAPER_PROTOCOL
    assert manifest["split"]["fold1_rows"] == 31_909
    assert manifest["split"]["fold2_rows"] == 31_914
    expected_fold_counts = {
        "EmoTales sentences": (697, 698),
        "Emobank": (5_031, 5_031),
        "GlasgowNorms": (2_776, 2_777),
        "IEMOCAP sentences": (5_016, 5_016),
        "fb": (1_447, 1_448),
        "nrc-vad": (9_985, 9_986),
        "word ratings ENG": (6_957, 6_958),
    }
    source_counts = manifest["split"]["per_source_fold_counts"]
    for name, (fold1_rows, fold2_rows) in expected_fold_counts.items():
        assert source_counts[name] == {
            "fold1_rows": fold1_rows,
            "fold2_rows": fold2_rows,
            "total_rows": fold1_rows + fold2_rows,
        }
        assert abs(fold1_rows - fold2_rows) <= 1
    assert sum(
        source["invalid_va_rows_dropped"] for source in manifest["sources"]
    ) == 3
    assert all(
        source["duplicate_text_rows_dropped"] == 0
        for source in manifest["sources"]
    )


def test_missing_fold_error_shows_both_generation_commands(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        read_fold(tmp_path / "missing" / FOLD_FILENAMES[0])
    message = str(error.value)
    assert "--download-default --output-dir <dir>" in message
    assert "--download-default --paper-protocol --output-dir <dir>" in message
    assert "--data-dir <dir>" in message


def test_repeat_comma_filters_apply_to_both_folds_and_fail_unresolved() -> None:
    columns = ["index", "text", "dataset_of_origin", "valence", "arousal"]
    fold1 = pd.DataFrame(
        [
            [0, "a", "IEMOCAP sentences", 0.1, 0.2],
            [1, "b", "fb", 0.3, 0.4],
            [2, "c", "Emobank", 0.5, 0.6],
        ],
        columns=columns,
    )
    fold2 = pd.DataFrame(
        [
            [3, "d", "fb", 0.2, 0.3],
            [4, "e", "IEMOCAP sentences", 0.4, 0.5],
            [5, "f", "nrc-vad", 0.6, 0.7],
        ],
        columns=columns,
    )
    assert collect_exclude_patterns(["fb, Emobank"], no_ieomcap=True) == (
        "fb",
        "Emobank",
        "IEMOCAP",
    )
    filtered1, filtered2, excluded = apply_dataset_filters(
        fold1,
        fold2,
        exclude_dataset=["fb,Emobank"],
        no_iemocap=True,
    )
    assert excluded == ("Emobank", "fb", "IEMOCAP sentences")
    assert filtered1["dataset_of_origin"].tolist() == []
    assert filtered2["dataset_of_origin"].tolist() == ["nrc-vad"]
    with pytest.raises(ValueError, match="Available dataset_of_origin values"):
        resolve_excluded_datasets(
            ["IEMOCAP sentences", "fb"], ["does-not-exist"]
        )


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def __call__(self, text, **kwargs):
        ids = [2] if text == "" else [ord(char) % 17 + 3 for char in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_tokenized_dataset_and_collator_need_no_model_download() -> None:
    frame = pd.DataFrame(
        {
            "index": [0, 1],
            "text": ["", "abc"],
            "dataset_of_origin": ["Emobank", "fb"],
            "valence": [0.1, 0.2],
            "arousal": [0.3, 0.4],
        }
    )
    tokenizer = _TinyTokenizer()
    dataset = TokenizedVADataset(frame, tokenizer, max_length=8)
    batch = VABatchCollator(tokenizer)([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][0].tolist() == [0, 0, 2]
    assert batch["attention_mask"][0].tolist() == [0, 0, 1]
    assert batch["labels"].dtype == torch.float32
    assert batch["labels"].shape == (2, 2)
