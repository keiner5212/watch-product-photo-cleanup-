#!/usr/bin/env python3
"""Approve the last script's output: replace matching input files.

Reads OUTPUT_DIR/<BrandFolder>/<sku>.png and moves each file to
INPUT_DIR/<BrandFolder>/<sku>.png (deleting the old input version first).
This is the "I liked the output, push it back to the inputs" step.

Use after any of the standalone scripts (process_images.py,
remove_magenta.py, normalize_watches.py) to promote the output.

Does NOT delete input files that have no matching output (those stay
untouched -- you might want to keep old variants around).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get(
    "WATCH_INPUT_DIR", Path(__file__).parent,
)).resolve()
OUTPUT_DIR = Path(os.environ.get(
    "WATCH_OUTPUT_DIR", Path(__file__).parent / "output",
)).resolve()

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collect_output(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXT:
            continue
        # skip hidden subdirs (e.g. .cache/ for resume state)
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(
        description="Approve the last script's output by replacing matching input files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be moved without touching any files.",
    )
    parser.add_argument(
        "--clear-after", action="store_true",
        help="After moving, empty OUTPUT_DIR (preserves hidden subdirs).",
    )
    args = parser.parse_args()

    if INPUT_DIR == OUTPUT_DIR:
        sys.exit(f"input and output dirs must differ: {INPUT_DIR}")
    if not OUTPUT_DIR.exists():
        sys.exit(f"output dir missing: {OUTPUT_DIR}")

    outputs = collect_output(OUTPUT_DIR)
    if not outputs:
        sys.exit(f"no approved files in {OUTPUT_DIR}")

    print(f"approving {len(outputs)} files from {OUTPUT_DIR} -> {INPUT_DIR}")
    if args.dry_run:
        print("  (dry run: nothing will be written)")
    moved = 0
    replaced = 0
    for out_path in outputs:
        rel = out_path.relative_to(OUTPUT_DIR)
        target = INPUT_DIR / rel
        existed = target.is_file()
        if args.dry_run:
            tag = "replace" if existed else "add"
            print(f"  {tag:8s} {rel}")
            moved += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if existed:
            target.unlink()
            replaced += 1
        shutil.move(str(out_path), str(target))
        moved += 1
        tag = "replaced" if existed else "added"
        print(f"  {tag:9s} {rel}", flush=True)

    if args.clear_after and not args.dry_run:
        cleared = 0
        for entry in OUTPUT_DIR.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            else:
                shutil.rmtree(entry)
            cleared += 1
        if cleared:
            print(f"  cleared {cleared} remaining entries in {OUTPUT_DIR}")

    print(
        f"done. moved={moved} replaced={replaced} "
        f"(INPUT_DIR = {INPUT_DIR})"
    )


if __name__ == "__main__":
    main()
