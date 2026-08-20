#!/usr/bin/env python3
"""Geometric normalization for watch product photos.

Pure PIL. No AI. Reads watches from INPUT_DIR and writes
consistent-size, consistently-margined square canvases to OUTPUT_DIR.
Empties OUTPUT_DIR at startup (preserving hidden subdirs like .cache/).
Standalone: run it on raw originals OR on AI-processed cutouts from
process_images.py; each script starts and ends in its own state.
"""
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

# ============================================================================
# CONFIG - tweak these at the top, no need to touch the rest of the file
# ============================================================================

# Source folder. Same default as process_images.py so both scripts can be
# pointed at the same root; override via WATCH_INPUT_DIR otherwise.
INPUT_DIR = Path(os.environ.get(
    "WATCH_INPUT_DIR", Path(__file__).parent,
)).resolve()

# Destination for the normalized square canvases. Same default as
# process_images.py so all scripts share one output folder. Each script
# empties this on startup; approve with approve_output.py to push back.
OUTPUT_DIR = Path(os.environ.get(
    "WATCH_OUTPUT_DIR", Path(__file__).parent / "output",
)).resolve()

# Final canvas edge length (square). Bigger = heavier files.
TARGET_SIZE = 1200

# Pixels of empty space between the watch bbox and the canvas edge.
# Same on all 4 sides by construction (symmetric crop + centered paste).
PADDING = 80

# Background fill. None = transparent PNG. RGB tuple/hex string = opaque fill.
# Examples: None, (255, 255, 255), "#ffffff", "#0a0a0a"
BACKGROUND = None

# "PNG" (transparent-safe) or "JPEG" (smaller, requires BACKGROUND opaque).
OUTPUT_FORMAT = "PNG"
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0

# Refine the cutout alpha before measuring the watch bbox.
ALPHA_THRESHOLD = 128
ALPHA_DILATE_SIZE = 5
ALPHA_SMOOTH_RADIUS = 1.2

# Set to 0 to disable the unsharp mask. Default off -> watch pixels untouched.
SHARPEN_PERCENT = 0
SHARPEN_RADIUS = 1.5
SHARPEN_THRESHOLD = 2

# When upscaling the watch to fit, use a smooth filter (LANCZOS).
# When downscaling a lot (>2x), do a two-pass BOX downscale to avoid aliasing.
SCALE_UP_FILTER = Image.LANCZOS
SCALE_DOWN_FILTER = Image.BOX
TWO_PASS_THRESHOLD = 0.5
PRESCALE_SHARPEN_PERCENT = 0  # 0 = off

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
IMAGE_LIMIT = None

# If False, skip files that already exist in OUTPUT_DIR (resume-safe).
OVERWRITE = False

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"
SKIP_DIR_NAMES = frozenset({"output", ".cache"})

# ============================================================================
# END CONFIG
# ============================================================================


@dataclass
class ImageJob:
    src_path: Path
    rel: Path


def _vlog(*args):
    if VERBOSE_LOG:
        print("    [v]", *args)


def _open_with_exif(path):
    raw = Image.open(path)
    try:
        raw = ImageOps.exif_transpose(raw)
    except Exception:
        pass
    return raw


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return a.resolve().samefile(b.resolve())
    except OSError:
        return False


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _should_skip_input(path: Path, input_root: Path) -> bool:
    """Skip OUTPUT_DIR and SKIP_DIR_NAMES subtrees relative to INPUT_DIR.
    Tolerates INPUT_DIR living under .cache/ (i.e. process_images bg-cut cache)."""
    resolved = path.resolve()
    if _is_under(resolved, OUTPUT_DIR):
        return True
    try:
        rel = resolved.relative_to(input_root.resolve())
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] in SKIP_DIR_NAMES


def _clean_output_dir(out: Path) -> int:
    """Empty OUT, preserving hidden subdirs (e.g. .cache/ for resume state).
    Returns count of removed entries."""
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return 0
    removed = 0
    for entry in out.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file() or entry.is_symlink():
            entry.unlink()
            removed += 1
        else:
            shutil.rmtree(entry)
            removed += 1
    return removed


def _parse_background(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return tuple(int(c) for c in value) + (255,)
        if len(value) == 4:
            return tuple(int(c) for c in value)
        raise ValueError(f"BACKGROUND tuple must have 3 or 4 components: {value!r}")
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255)
        if len(s) == 8:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16))
    raise ValueError(f"BACKGROUND must be None, RGB(A) tuple, or hex string: {value!r}")


def _normalize_alpha(image):
    if image.mode != "RGBA":
        return image
    import numpy as np
    alpha = np.array(image.split()[-1])
    binary = (alpha >= ALPHA_THRESHOLD).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")
    if ALPHA_DILATE_SIZE > 1:
        mask = mask.filter(ImageFilter.MaxFilter(ALPHA_DILATE_SIZE))
    if ALPHA_SMOOTH_RADIUS > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(ALPHA_SMOOTH_RADIUS))
    r, g, b, _ = image.split()
    return Image.merge("RGBA", (r, g, b, mask))


def symmetric_crop(cutout):
    """Square crop centered on the watch bbox. Returns None for empty alpha."""
    bbox = cutout.split()[-1].getbbox()
    if not bbox:
        return None
    left, top, right, bottom = bbox
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    side = max(right - left, bottom - top)
    half = side // 2
    crop_left = max(0, cx - half)
    crop_top = max(0, cy - half)
    crop_right = min(cutout.width, crop_left + side)
    crop_bottom = min(cutout.height, crop_top + side)
    if crop_right - crop_left < side:
        crop_left = max(0, crop_right - side)
    if crop_bottom - crop_top < side:
        crop_top = max(0, crop_bottom - side)
    return cutout.crop((crop_left, crop_top, crop_right, crop_bottom))


def scale_watch(watch, target_size, padding):
    avail = target_size - 2 * padding
    orig_w, orig_h = watch.width, watch.height
    side = max(orig_w, orig_h)
    scale = avail / side
    if scale == 1.0:
        return watch
    final_size = (
        max(1, int(orig_w * scale)),
        max(1, int(orig_h * scale)),
    )
    if scale > 1.0:
        return watch.resize(final_size, SCALE_UP_FILTER)
    if PRESCALE_SHARPEN_PERCENT > 0 and watch.mode == "RGBA":
        r, g, b, a = watch.split()
        rgb = Image.merge("RGB", (r, g, b)).filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=PRESCALE_SHARPEN_PERCENT, threshold=2)
        )
        watch = Image.merge("RGBA", (*rgb.split(), a))
    if scale < TWO_PASS_THRESHOLD:
        mid_size = (
            max(1, int(orig_w * TWO_PASS_THRESHOLD)),
            max(1, int(orig_h * TWO_PASS_THRESHOLD)),
        )
        watch = watch.resize(mid_size, SCALE_DOWN_FILTER)
    return watch.resize(final_size, SCALE_DOWN_FILTER)


def sharpen_watch(watch):
    if SHARPEN_PERCENT <= 0 or watch.mode != "RGBA":
        return watch
    r, g, b, a = watch.split()
    rgb = Image.merge("RGB", (r, g, b)).filter(
        ImageFilter.UnsharpMask(SHARPEN_RADIUS, SHARPEN_PERCENT, SHARPEN_THRESHOLD)
    )
    return Image.merge("RGBA", (*rgb.split(), a))


def compose_canvas(watch, target_size, background):
    bg = (0, 0, 0, 0) if background is None else background
    canvas = Image.new("RGBA", (target_size, target_size), bg)
    x = (target_size - watch.width) // 2
    y = (target_size - watch.height) // 2
    canvas.paste(watch, (x, y), watch)
    return canvas


def _save(canvas, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = OUTPUT_FORMAT.upper()
    if fmt == "JPEG":
        canvas.convert("RGB").save(
            out_path, "JPEG",
            quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING,
        )
    elif fmt == "PNG":
        canvas.save(out_path, "PNG", optimize=True)
    else:
        canvas.save(out_path, fmt)


def collect_images(root):
    root = root.resolve()
    images = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXT
        and not p.name.startswith(".")
        and not _should_skip_input(p, root)
    )
    if IMAGE_LIMIT is not None:
        images = images[:IMAGE_LIMIT]
    return images


def process_one(job, out_path):
    raw = _open_with_exif(job.src_path)
    if raw.mode != "RGBA":
        _vlog(f"{job.rel}: no alpha channel -> treating whole image as watch bbox")
        cutout = raw.convert("RGBA")
    else:
        cutout = _normalize_alpha(raw)

    watch = symmetric_crop(cutout)
    if watch is None:
        return "empty"

    watch = scale_watch(watch, TARGET_SIZE, PADDING)
    watch = sharpen_watch(watch)
    canvas = compose_canvas(watch, TARGET_SIZE, _parse_background(BACKGROUND))
    _save(canvas, out_path)
    return "saved"


def main():
    if not INPUT_DIR.exists():
        sys.exit(f"input dir missing: {INPUT_DIR}")
    if _same_dir(INPUT_DIR, OUTPUT_DIR):
        sys.exit(f"input and output dirs must differ: {INPUT_DIR}")

    n = _clean_output_dir(OUTPUT_DIR)
    if n:
        print(f"  cleared {n} stale entries in {OUTPUT_DIR} (preserved hidden dirs)", flush=True)

    paths = collect_images(INPUT_DIR)
    if not paths:
        sys.exit(f"no images found under {INPUT_DIR}")
    jobs = [ImageJob(src_path=p, rel=p.relative_to(INPUT_DIR)) for p in paths]

    bg_desc = "transparent" if BACKGROUND is None else f"{BACKGROUND}"
    print(f"normalizing {len(jobs)} images from {INPUT_DIR}")
    print(
        f"  target: {TARGET_SIZE}x{TARGET_SIZE}, padding: {PADDING}, "
        f"bg: {bg_desc}, format: {OUTPUT_FORMAT}, sharpen: {SHARPEN_PERCENT}%"
    )
    print(f"  output: {OUTPUT_DIR}  (overwrite={OVERWRITE})")

    counts = {"saved": 0, "skipped": 0, "empty": 0, "hit": 0}
    for job in jobs:
        out = OUTPUT_DIR / job.rel.with_suffix("." + OUTPUT_FORMAT.lower())
        if not OVERWRITE and out.is_file() and out.stat().st_size > 0:
            counts["hit"] += 1
            continue
        try:
            status = process_one(job, out)
        except Exception as exc:
            print(f"  ERR {job.rel}: {exc}", flush=True)
            counts["skipped"] += 1
            continue
        if status == "saved":
            counts["saved"] += 1
            print(f"  ok {job.rel}", flush=True)
        elif status == "empty":
            counts["empty"] += 1
            print(f"  skip {job.rel}: empty content", flush=True)

    print(
        f"done. saved={counts['saved']} hit={counts['hit']} "
        f"empty={counts['empty']} error={counts['skipped']} "
        f"-> {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
