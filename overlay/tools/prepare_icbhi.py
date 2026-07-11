from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


REQUIRED_METADATA = (
    "metadata.txt",
    "official_split.txt",
    "patient_list_foldwise.txt",
)
OPTIONAL_METADATA = ("patient_diagnosis.txt",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare ICBHI for the PAFA repository")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--source_dir", type=Path)
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_audio_root(root: Path) -> Path:
    candidates = []
    for directory in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        wav_count = sum(1 for _ in directory.glob("*.wav"))
        txt_count = sum(1 for _ in directory.glob("*.txt"))
        if wav_count:
            candidates.append((wav_count, txt_count, directory))
    if not candidates:
        raise FileNotFoundError(f"No WAV files were found below {root}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def locate_metadata(repo_root: Path, filename: str) -> Path | None:
    preferred = repo_root / "data" / filename
    if preferred.exists():
        return preferred
    matches = [
        path
        for path in (repo_root / "data").rglob(filename)
        if "icbhi_dataset" not in path.parts
    ]
    return matches[0] if matches else None


def copy_file(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not (repo_root / "main.py").exists():
        raise FileNotFoundError(
            f"{repo_root} does not look like the PAFA repository: main.py is missing"
        )

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.archive is not None:
            archive = args.archive.resolve()
            if not archive.exists():
                raise FileNotFoundError(archive)
            temporary = tempfile.TemporaryDirectory(prefix="icbhi_extract_")
            extraction_root = Path(temporary.name)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extraction_root)
            source_root = find_audio_root(extraction_root)
        else:
            source_root = find_audio_root(args.source_dir.resolve())

        dataset_root = repo_root / "data" / "icbhi_dataset"
        audio_root = dataset_root / "audio_test_data"
        audio_root.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(source_root.glob("*.wav"))
        annotation_files = [
            path
            for path in source_root.glob("*.txt")
            if path.with_suffix(".wav").exists()
        ]
        for source in wav_files + annotation_files:
            copy_file(source, audio_root / source.name, args.force)

        for filename in REQUIRED_METADATA + OPTIONAL_METADATA:
            source = locate_metadata(repo_root, filename)
            if source is None:
                if filename in REQUIRED_METADATA:
                    raise FileNotFoundError(
                        f"Missing upstream PAFA metadata file: data/{filename}"
                    )
                continue
            copy_file(source, dataset_root / filename, args.force)

        for cache in (repo_root / "data" / "training.pt", repo_root / "data" / "test.pt"):
            if cache.exists():
                cache.unlink()

        copied_wav = len(list(audio_root.glob("*.wav")))
        copied_txt = sum(
            1
            for path in audio_root.glob("*.txt")
            if path.with_suffix(".wav").exists()
        )
        print(f"ICBHI audio directory: {audio_root}")
        print(f"WAV recordings: {copied_wav}")
        print(f"Annotation files: {copied_txt}")
        if copied_wav < 900 or copied_txt < 900:
            raise RuntimeError(
                "The prepared directory contains fewer than 900 recording/annotation pairs. "
                "The complete ICBHI archive normally contains 920 recordings."
            )
        print("ICBHI preparation completed")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
