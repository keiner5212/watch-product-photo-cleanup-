#!/usr/bin/env python3
"""Remove user-marked magenta (#ff00bf) regions from processed watch images.

Pipeline per image:
  1. Detect #ff00bf pixels with tolerance (catches anti-aliased edges).
  2. Build mask, clean it (close + area filter + dilate + soft feather).
  3. Composite onto white so LaMa reads a clean RGB background.
  4. Inpaint with Big-LaMa (same model as process_images.py).
  5. Restore original alpha on the inpainted RGB; save PNG to OUTPUT_DIR.

Input  : INPUT_DIR tree (PNG/JPG/WEBP, mirrors process_images.py).
Output : OUTPUT_DIR / <rel_path>.png  (same convention as process_images.py).

No upscale, no bg-cut, no OCR — only magenta detection + LaMa inpainting.
Reuses LaMa from simple_lama_inpainting (pre-baked in the docker image).
"""
import gc
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# basicsr pins old tqdm that ignores TQDM_DISABLE; force silent bars.
# Mirror process_images.py so the silent-tqdm shim is consistent.
try:
    import tqdm as _tqdm_mod

    class _SilentTqdm:
        def __init__(self, *a, **k):
            self.n = 0
            self.total = k.get("total") or (a[0] if a and isinstance(a[0], int) else 0)

        def __iter__(self):
            return iter(())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def update(self, *a, **k):
            pass

        def close(self):
            pass

        def set_description(self, *a, **k):
            pass

        def set_postfix(self, *a, **k):
            pass

        def write(self, *a, **k):
            pass

        def refresh(self):
            pass

    _tqdm_mod.tqdm = _SilentTqdm
    _tqdm_mod.trange = lambda *a, **k: _SilentTqdm(*a, **k)
except ImportError:
    pass

INPUT_DIR = Path(os.environ.get("WATCH_INPUT_DIR", Path(__file__).parent)).resolve()
OUTPUT_DIR = Path(os.environ.get("WATCH_OUTPUT_DIR", Path(__file__).parent / "output")).resolve()
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SKIP_DIR_NAMES = frozenset({"output", ".cache"})

# Marker color the user paints to mark regions for removal.
MAGENTA_TARGET = (255, 0, 191)
# Per-channel max diff to count a pixel as "magenta".
# 60 catches the AA fringe of a #ff00bf brush stroke without
# bleeding into watch dial textures (next band starts ~150+).
MAGENTA_TOLERANCE = 60

# Mask cleanup — keep tight, with a soft edge for LaMa to blend.
MASK_CLOSE_PX = 3       # bridge 1-px gaps inside the painted area
MASK_DILATE_PX = 3      # expand ~1 px so the painted color never bleeds through
MASK_FEATHER_PX = 5     # Gaussian feather (kernel 2*px+1) on the mask edge
MASK_MIN_AREA = 30      # ignore isolated noise smaller than this
MASK_MAX_AREA_FRAC = 0.20  # safety: skip if user marked >20% of the image

# LaMa — same limits as process_images.py.
LAMA_MAX_DIM = 1280

_INPAINT_MODEL = None

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"


def _vlog(*args):
    if VERBOSE_LOG:
        print("    [v]", *args)


def _free_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _drop_inpaint():
    global _INPAINT_MODEL
    _INPAINT_MODEL = None
    _free_vram()


def _open_with_exif(path):
    raw = Image.open(path)
    try:
        raw = ImageOps.exif_transpose(raw)
    except Exception:
        pass
    return raw


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return a.resolve().samefile(b.resolve())
    except OSError:
        return False


def _should_skip_input(path: Path, input_root: Path) -> bool:
    """Skip output/.cache even when docker mounts project root as /data/in."""
    resolved = path.resolve()
    if _is_under(resolved, OUTPUT_DIR):
        return True
    input_output = input_root / "output"
    if input_output.exists() and _is_under(resolved, input_output):
        return True
    if any(part in SKIP_DIR_NAMES for part in resolved.parts):
        return True
    try:
        rel = resolved.relative_to(input_root.resolve())
    except ValueError:
        return True
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


def _get_lama():
    global _INPAINT_MODEL
    if _INPAINT_MODEL is None:
        print("loading inpaint model (Big-LaMa)...", flush=True)
        from simple_lama_inpainting import SimpleLama
        _INPAINT_MODEL = SimpleLama()
        print("inpaint model ready.", flush=True)
    return _INPAINT_MODEL


def _inpaint_lama(rgb: Image.Image, mask: Image.Image) -> Image.Image:
    """Run LaMa on an RGB image + L-mode mask. Matches process_images.py sizing logic."""
    lama = _get_lama()
    orig_size = rgb.size
    if max(orig_size) > LAMA_MAX_DIM:
        scale = LAMA_MAX_DIM / max(orig_size)
        small_size = (int(rgb.width * scale), int(rgb.height * scale))
        rgb = rgb.resize(small_size, Image.LANCZOS)
        mask = mask.resize(small_size, Image.LANCZOS)
        result = lama(rgb, mask)
        if result.size != small_size:
            result = result.crop((0, 0, small_size[0], small_size[1]))
        result = result.resize(orig_size, Image.LANCZOS)
    else:
        result = lama(rgb, mask)
        if result.size != orig_size:
            result = result.crop((0, 0, orig_size[0], orig_size[1]))
    return result


def detect_magenta_mask(rgba: Image.Image) -> Image.Image | None:
    """Return L-mode mask of magenta-marked regions, or None if none found.

    Steps: per-channel distance test -> close -> drop tiny noise -> dilate -> feather.
    Only opaque pixels (alpha > 0) are considered so transparent background
    RGB noise doesn't trigger a false positive.
    """
    arr = np.array(rgba.convert("RGBA"))
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    target = np.array(MAGENTA_TARGET, dtype=np.int16)
    diff = np.abs(rgb - target).max(axis=2)
    raw = ((diff <= MAGENTA_TOLERANCE) & (alpha > 0)).astype(np.uint8) * 255
    if raw.sum() == 0:
        return None

    import cv2

    # Close small holes inside the painted region.
    if MASK_CLOSE_PX > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_CLOSE_PX, MASK_CLOSE_PX))
        closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    else:
        closed = raw

    # Drop isolated noise components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MASK_MIN_AREA:
            cleaned[labels == i] = 255
    if cleaned.sum() == 0:
        return None

    # Dilate ~1 px so the painted color is fully inside the mask.
    if MASK_DILATE_PX > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX, MASK_DILATE_PX))
        cleaned = cv2.dilate(cleaned, k, iterations=1)

    coverage = (cleaned >= 128).sum() / cleaned.size
    if coverage > MASK_MAX_AREA_FRAC:
        print(
            f"  WARN: mask covers {coverage*100:.1f}% of image "
            f"(>{MASK_MAX_AREA_FRAC*100:.0f}%) — skipping inpaint for this image",
            flush=True,
        )
        return None

    # Soft Gaussian feather so LaMa blends seamlessly.
    if MASK_FEATHER_PX > 0:
        feather = MASK_FEATHER_PX
        cleaned = cv2.GaussianBlur(
            cleaned, (feather * 2 + 1, feather * 2 + 1), 0
        )

    if VERBOSE_LOG:
        try:
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            base = Image.alpha_composite(white, rgba).convert("RGB")
            mask_rgb = Image.fromarray(cleaned, mode="L")
            red = Image.new("RGB", base.size, (255, 0, 0))
            overlay = Image.composite(red, base, mask_rgb)
            overlay.save(OUTPUT_DIR / ".debug_mask_overlay.png")
            _vlog(f"saved .debug_mask_overlay.png ({rgba.size[0]}x{rgba.size[1]})")
        except Exception as e:
            _vlog(f"mask overlay failed: {e}")

    return Image.fromarray(cleaned, mode="L")


def collect_images(root: Path) -> list[Path]:
    root = root.resolve()
    images = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXT
        and not p.name.startswith(".")
        and not _should_skip_input(p, root)
    )
    return images


def process_one(src_path: Path, rel: Path) -> str:
    out_path = OUTPUT_DIR / rel.with_suffix(".png")
    if out_path.is_file() and out_path.stat().st_size > 0:
        return "cached"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgba = _open_with_exif(src_path).convert("RGBA")
    mask_l = detect_magenta_mask(rgba)
    if mask_l is None:
        # No magenta detected — copy original unchanged so the output tree mirrors input.
        rgba.save(out_path, "PNG", optimize=True)
        return "none"

    # Composite onto white so LaMa never reads stray RGB values from transparent pixels.
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(white, rgba).convert("RGB")
    result_rgb = _inpaint_lama(rgb, mask_l)

    # Restore the original alpha on the inpainted RGB.
    orig_alpha = rgba.split()[3]
    r, g, b = result_rgb.split()
    final = Image.merge("RGBA", (r, g, b, orig_alpha))
    final.save(out_path, "PNG", optimize=True)
    return "inpainted"


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

    print(f"scanning {len(paths)} images for magenta marks (#ff00bf ± {MAGENTA_TOLERANCE})")
    print(f"  input : {INPUT_DIR}")
    print(f"  output: {OUTPUT_DIR}")
    print(
        f"  mask cleanup: close={MASK_CLOSE_PX}px dilate={MASK_DILATE_PX}px "
        f"feather={MASK_FEATHER_PX}px min_area={MASK_MIN_AREA}px"
    )

    counts = {"inpainted": 0, "none": 0, "cached": 0, "errors": 0}
    try:
        for i, src in enumerate(paths, 1):
            rel = src.relative_to(INPUT_DIR)
            try:
                tag = process_one(src, rel)
            except Exception as e:
                print(
                    f"  [{i}/{len(paths)}] ERR {rel}: {type(e).__name__}: {e}",
                    flush=True,
                )
                counts["errors"] += 1
                continue
            counts[tag] += 1
            print(f"  [{i}/{len(paths)}] {tag:9s} {rel}", flush=True)
    finally:
        _drop_inpaint()

    print(
        f"done. inpainted={counts['inpainted']} "
        f"unchanged={counts['none']} cached={counts['cached']} "
        f"errors={counts['errors']}  ->  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()