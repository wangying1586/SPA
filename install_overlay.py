from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Install PAFA-SPA files into a PAFA clone")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    overlay = package_root / "overlay"
    target = args.target.resolve()
    required = (target / "main.py", target / "method" / "pafa.py", target / "util" / "icbhi_dataset.py")
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(
            "Target is not a compatible PAFA clone. Missing: " + ", ".join(missing)
        )

    installed = []
    for source in overlay.rglob("*"):
        if source.is_dir() or "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        relative = source.relative_to(overlay)
        destination = target / relative
        if destination.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite {destination}. Re-run with --force after reviewing it."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.suffix == ".sh":
            destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        installed.append(relative)

    shutil.copy2(package_root / "requirements-spa.txt", target / "requirements-spa.txt")
    shutil.copy2(package_root / "environment.yml", target / "environment.yml")
    shutil.copy2(package_root / "README.md", target / "README_SPA.md")
    shutil.copy2(package_root / "UPSTREAM.md", target / "UPSTREAM.md")
    shutil.copy2(package_root / "VALIDATION.md", target / "VALIDATION_SPA.md")
    print(f"Installed {len(installed)} files into {target}")
    print("Next: install dependencies, prepare ICBHI, then run tools/verify_install.py")


if __name__ == "__main__":
    main()
