from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def status(ok: bool, message: str) -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {message}")


def main() -> None:
    parser = argparse.ArgumentParser("Verify PAFA-SPA installation")
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()

    checks = []
    for relative in (
        "main.py",
        "main_spa.py",
        "eval_pafa_checkpoint.py",
        "method/pafa.py",
        "models/__init__.py",
        "util/icbhi_dataset.py",
        "spa/head.py",
    ):
        present = (root / relative).exists()
        status(present, relative)
        checks.append(present)

    dataset_root = root / "data" / "icbhi_dataset"
    audio_root = dataset_root / "audio_test_data"
    wav_count = len(list(audio_root.glob("*.wav"))) if audio_root.exists() else 0
    annotation_count = (
        sum(1 for path in audio_root.glob("*.txt") if path.with_suffix(".wav").exists())
        if audio_root.exists()
        else 0
    )
    dataset_ok = wav_count >= 900 and annotation_count >= 900
    status(dataset_ok, f"ICBHI pairs: wav={wav_count}, annotations={annotation_count}")
    checks.append(dataset_ok)

    for filename in ("metadata.txt", "official_split.txt", "patient_list_foldwise.txt"):
        present = (dataset_root / filename).exists()
        status(present, f"data/icbhi_dataset/{filename}")
        checks.append(present)

    weight = root / "pretrained_models" / "BEATs_iter3_plus_AS2M.pt"
    weight_ok = weight.exists()
    status(weight_ok, str(weight.relative_to(root)))
    checks.append(weight_ok)

    torch_found = importlib.util.find_spec("torch") is not None
    torchaudio_found = importlib.util.find_spec("torchaudio") is not None
    status(torch_found, "Python package torch")
    status(torchaudio_found, "Python package torchaudio")
    checks.extend([torch_found, torchaudio_found])

    if all(checks):
        print("Installation verification passed")
    else:
        raise SystemExit("Installation verification failed; resolve the failed checks above")


if __name__ == "__main__":
    main()
