#!/usr/bin/env python3
"""Watch image cleanup pipeline for Shopify (AI stages only).

Pipeline (per batch, one model at a time):
  upscale -> inpaint (OCR + LaMa) -> bg-cut (BiRefNet)
Reads watches from INPUT_DIR, writes transparent cutouts to OUTPUT_DIR.
The intermediate upscale + inpaint caches live under OUTPUT_DIR/.cache/
for resume-safe re-runs. Geometric normalization (crop/scale/compose) is
handled by the standalone normalize_watches.py script.
"""
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# basicsr pins old tqdm that ignores TQDM_DISABLE; force silent bars.
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
IMAGE_LIMIT = None

ALPHA_THRESHOLD = 128
ALPHA_DILATE_SIZE = 5
ALPHA_SMOOTH_RADIUS = 1.2

NO_BG_THRESHOLD = 0.05

STAGE_BG_ENABLED = True
STAGE_BG_MODEL = "birefnet-general"

STAGE_UPSCALE_ENABLED = True
STAGE_UPSCALE_MODEL = "RealESRGAN_x2plus.pth"
STAGE_UPSCALE_OUTSCALE = 2
STAGE_UPSCALE_TILE = 512

STAGE_INPAINT_ENABLED = True
STAGE_OCR_KEYWORDS = ["SEIKO", "YATCH-MASTER", "DATEJUST", "DAYDATE", "OYSTER PERPETUAL", "SUBMARINE", "DAYTONA"]
STAGE_OCR_LANG = "en"
STAGE_OCR_CONFIDENCE = 0.6
EXCEPTIONS_PATH = Path(__file__).parent / "exceptions.txt"
STAGE_MASK_FEATHER_PX = 11
STAGE_MASK_MAX_AREA_FRAC = 0.05
STAGE_INPAINT_MODEL = "lama"
LAMA_MAX_DIM = 1280
STAGE_SD_PROMPT = "smooth watch dial face texture, no text, no logo, plain surface"
STAGE_SD_NEGATIVE = "text, letters, words, logo, brand name, hands, indices, numbers, blurry, artifacts"
STAGE_SD_STEPS = 20
STAGE_SD_GUIDANCE = 7.5
STAGE_SD_SEED = 42

CACHE_UPSCALE = OUTPUT_DIR / ".cache" / "02_upscale"
CACHE_INPAINT = OUTPUT_DIR / ".cache" / "03_inpaint"
SKIP_DIR_NAMES = frozenset({"output", ".cache"})


@dataclass
class ImageJob:
    src_path: Path
    rel: Path
    current: Path


_OCR_ENGINE = None
_INPAINT_MODEL = None
_BG_SESSION = None
_EXCEPTIONS_CACHE: dict | None = None

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"


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


def _free_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _drop_ocr():
    global _OCR_ENGINE
    _OCR_ENGINE = None
    _free_vram()


def _drop_inpaint():
    global _INPAINT_MODEL
    _INPAINT_MODEL = None
    _free_vram()


def _drop_bg():
    global _BG_SESSION
    _BG_SESSION = None
    _free_vram()


def _drop_all_models():
    _drop_ocr()
    _drop_inpaint()
    _drop_bg()


def _cache_path(cache_root: Path, rel: Path) -> Path:
    return cache_root / rel.with_suffix(".png")


def _use_cache_hit(job: ImageJob, out_path: Path) -> bool:
    if out_path.is_file() and out_path.stat().st_size > 0:
        job.current = out_path
        return True
    return False


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
            import shutil as _sh
            _sh.rmtree(entry)
            removed += 1
    return removed


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


def refine_cutout(image):
    if image.mode != "RGBA":
        return image
    alpha = np.array(image.split()[-1])
    binary = np.where(alpha >= ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    mask = Image.fromarray(binary, mode="L")
    if ALPHA_DILATE_SIZE > 1:
        mask = mask.filter(ImageFilter.MaxFilter(ALPHA_DILATE_SIZE))
    if ALPHA_SMOOTH_RADIUS > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(ALPHA_SMOOTH_RADIUS))
    r, g, b, _ = image.split()
    return Image.merge("RGBA", (r, g, b, mask))


def has_no_background(image, threshold=NO_BG_THRESHOLD):
    if image.mode != "RGBA":
        return False
    alpha = np.array(image.split()[-1])
    return (alpha < 10).sum() / alpha.size > threshold


_REALESRGAN_URL_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1"


def _download_realesrgan_weight(filename):
    cache_dir = Path.home() / ".cache" / "realesrgan"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / filename
    if local.exists():
        return str(local)
    url = f"{_REALESRGAN_URL_BASE}/{filename}"
    print(f"  downloading {filename}...", flush=True)
    import urllib.request
    urllib.request.urlretrieve(url, str(local))
    return str(local)


def _stage_upscale(jobs):
    """Real-ESRGAN only. Cache -> .cache/02_upscale. Unload before return."""
    if not STAGE_UPSCALE_ENABLED:
        return 0
    CACHE_UPSCALE.mkdir(parents=True, exist_ok=True)

    pending = []
    for job in jobs:
        out_path = _cache_path(CACHE_UPSCALE, job.rel)
        if _use_cache_hit(job, out_path):
            continue
        pending.append((job, out_path))
    if not pending:
        print(f"  upscale: all {len(jobs)} cached", flush=True)
        return 0

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    print(f"loading upscale model ({STAGE_UPSCALE_MODEL})...", flush=True)
    model_path = _download_realesrgan_weight(STAGE_UPSCALE_MODEL)
    lname = STAGE_UPSCALE_MODEL.lower()
    if "x2" in lname:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
    else:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        scale = 4

    half = torch.cuda.is_available()
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=net,
        tile=STAGE_UPSCALE_TILE,
        tile_pad=10,
        pre_pad=0,
        half=half,
        gpu_id=0 if torch.cuda.is_available() else None,
    )
    print("upscale model ready.", flush=True)
    done = 0
    try:
        for job, out_path in pending:
            img = _open_with_exif(job.current).convert("RGBA")
            if has_no_background(img):
                white = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white, img)
            rgb = np.array(img.convert("RGB"))
            bgr = rgb[..., ::-1].copy()
            out_bgr, _ = upsampler.enhance(bgr, outscale=STAGE_UPSCALE_OUTSCALE)
            out_rgb = out_bgr[..., ::-1]
            h, w = out_rgb.shape[:2]
            r = Image.fromarray(out_rgb[..., 0], mode="L")
            g = Image.fromarray(out_rgb[..., 1], mode="L")
            b = Image.fromarray(out_rgb[..., 2], mode="L")
            alpha = img.split()[3].resize((w, h), Image.LANCZOS)
            out = Image.merge("RGBA", (r, g, b, alpha))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.save(out_path)
            job.current = out_path
            done += 1
            print(f"  upscale {done}/{len(pending)}: {job.rel}", flush=True)
    finally:
        del upsampler
        _free_vram()
    return done


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        print("loading ocr (EasyOCR)...", flush=True)
        import easyocr
        _OCR_ENGINE = easyocr.Reader(
            [STAGE_OCR_LANG],
            gpu=True,
            model_storage_directory="/root/.cache/EasyOCR",
            user_network_directory="/root/.cache/EasyOCR",
        )
        print("ocr ready.", flush=True)
    return _OCR_ENGINE


def _load_exceptions(path):
    out: dict[str, list[tuple[str, float]]] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            sku, rest = line.split(":", 1)
            text_part, conf_part = rest.rsplit(":", 1)
            text = text_part.strip().strip('"').strip("'")
            threshold = float(conf_part.strip())
            out.setdefault(sku.strip(), []).append((text, threshold))
        except (ValueError, AttributeError):
            print(f"  WARN: malformed exception line ignored: {line!r}")
    return out


def _get_exceptions():
    global _EXCEPTIONS_CACHE
    if _EXCEPTIONS_CACHE is None:
        _EXCEPTIONS_CACHE = _load_exceptions(EXCEPTIONS_PATH)
        if _EXCEPTIONS_CACHE:
            print(
                f"loaded {sum(len(v) for v in _EXCEPTIONS_CACHE.values())} "
                f"exception(s) from {EXCEPTIONS_PATH}",
                flush=True,
            )
    return _EXCEPTIONS_CACHE


def _fuzzy_keyword_hit(ocr_text, keyword, max_edits=2):
    text = "".join(c for c in ocr_text.lower() if c not in " -")
    kw = "".join(c for c in keyword.lower() if c not in " -")
    if kw in text:
        return True
    n = len(kw)
    if len(text) < max(1, n - 2):
        return False
    for n_window in range(max(1, n - 1), n + 2):
        for i in range(len(text) - n_window + 1):
            sub = text[i:i + n_window]
            best = 0
            for offset in range(0, n_window - n + 2):
                hits = sum(
                    1 for a, b in zip(sub[offset:offset + n], kw)
                    if a == b
                )
                if hits > best:
                    best = hits
            if best >= n - max_edits:
                return True
    return False


def _detect_mask(image, sku=""):
    import cv2

    reader = _get_ocr()
    rgb = np.array(image.convert("RGB"))
    results = reader.readtext(rgb)

    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    keywords_lower = [k.lower() for k in STAGE_OCR_KEYWORDS]
    matched = 0
    for det in results:
        try:
            bbox, text, conf = det
        except ValueError:
            continue
        if not isinstance(text, str):
            continue
        if conf < STAGE_OCR_CONFIDENCE:
            _vlog(f'  skip: "{text}" conf {conf:.3f} < threshold {STAGE_OCR_CONFIDENCE}')
            continue
        if not any(_fuzzy_keyword_hit(text, k) for k in keywords_lower):
            _vlog(f'  skip: "{text}" no keyword match')
            continue
        poly = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        _vlog(f'  MASK: "{text}" -> polygon filled, {len(poly)} vertices')
        matched += 1

    if sku and matched == 0:
        exc_for_sku = _get_exceptions().get(sku, [])
        if exc_for_sku:
            for det in results:
                try:
                    bbox, text, conf = det
                except ValueError:
                    continue
                if not isinstance(text, str):
                    continue
                for exc_text, exc_threshold in exc_for_sku:
                    if conf >= exc_threshold and _fuzzy_keyword_hit(text, exc_text):
                        poly = np.array(bbox, dtype=np.int32)
                        cv2.fillPoly(mask, [poly], 255)
                        _vlog(
                            f'  MASK (exception): "{text}" matches {exc_text!r} '
                            f"@ conf {conf:.3f} >= {exc_threshold}"
                        )
                        matched += 1
                        break

    if sku and matched == 0:
        try:
            exc_path = OUTPUT_DIR / "exceptions.txt"
            with open(exc_path, "a", encoding="utf-8") as fh:
                for det in results:
                    try:
                        _, t, c = det
                    except ValueError:
                        continue
                    if not isinstance(t, str):
                        continue
                    fh.write(f'{sku}:"{t}":{c:.3f}\n')
        except Exception as e:
            _vlog(f"exception log failed: {e}")

    if matched == 0:
        return None

    coverage = mask.sum() / 255 / mask.size
    if coverage > STAGE_MASK_MAX_AREA_FRAC:
        print(
            f"  WARN: inpaint mask covers {coverage*100:.1f}% of image "
            f"(>{STAGE_MASK_MAX_AREA_FRAC*100:.0f}%) — skipping inpaint for this image"
        )
        return None

    feather = STAGE_MASK_FEATHER_PX
    mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)

    if VERBOSE_LOG:
        try:
            overlay = Image.fromarray(rgb).convert("RGBA")
            mask_rgb = Image.fromarray(mask).convert("L")
            red = Image.new("RGBA", overlay.size, (255, 0, 0, 110))
            overlay.paste(red, mask=mask_rgb)
            overlay.save(OUTPUT_DIR / ".debug_mask_overlay.png")
            _vlog(f"saved .debug_mask_overlay.png ({w}x{h})")
        except Exception as e:
            _vlog(f"mask overlay failed: {e}")

    return Image.fromarray(mask, mode="L")


def _get_lama():
    global _INPAINT_MODEL
    if _INPAINT_MODEL is None:
        print("loading inpaint model (Big-LaMa)...", flush=True)
        from simple_lama_inpainting import SimpleLama
        _INPAINT_MODEL = SimpleLama()
        print("inpaint model ready.", flush=True)
    return _INPAINT_MODEL


def _inpaint_lama(image, mask):
    lama = _get_lama()
    rgb = image.convert("RGB")
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


def _inpaint_sd(image, mask):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            sys.exit(
                f"SD inpaint needs ~5 GB free VRAM; "
                f"only {free/(1024**3):.1f}/{total/(1024**3):.1f} GB free."
            )

    global _INPAINT_MODEL
    if _INPAINT_MODEL is None:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        _INPAINT_MODEL = pipe

    pipe = _INPAINT_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(STAGE_SD_SEED)
    rgb = image.convert("RGB")
    with torch.inference_mode():
        result = pipe(
            prompt=STAGE_SD_PROMPT,
            negative_prompt=STAGE_SD_NEGATIVE,
            image=rgb,
            mask_image=mask,
            num_inference_steps=STAGE_SD_STEPS,
            guidance_scale=STAGE_SD_GUIDANCE,
            generator=generator,
        ).images[0]
    return result


def _inpaint(image, mask):
    if STAGE_INPAINT_MODEL == "lama":
        return _inpaint_lama(image, mask)
    if STAGE_INPAINT_MODEL == "sd":
        return _inpaint_sd(image, mask)
    raise ValueError(f"unknown STAGE_INPAINT_MODEL: {STAGE_INPAINT_MODEL}")


def _stage_inpaint(jobs):
    """OCR + LaMa only. Cache -> .cache/03_inpaint. Unload before return."""
    CACHE_INPAINT.mkdir(parents=True, exist_ok=True)
    if not STAGE_INPAINT_ENABLED:
        for job in jobs:
            out_path = _cache_path(CACHE_INPAINT, job.rel)
            if _use_cache_hit(job, out_path):
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = _open_with_exif(job.current).convert("RGB")
            img.save(out_path)
            job.current = out_path
        return 0

    pending = []
    for job in jobs:
        out_path = _cache_path(CACHE_INPAINT, job.rel)
        if _use_cache_hit(job, out_path):
            continue
        pending.append((job, out_path))
    if not pending:
        print(f"  inpaint: all {len(jobs)} cached", flush=True)
        return 0

    done = 0
    inpainted = 0
    try:
        for job, out_path in pending:
            rgb = _open_with_exif(job.current).convert("RGB")
            mask = _detect_mask(rgb, sku=job.rel.stem)
            if mask is not None:
                rgb = _inpaint(rgb, mask)
                inpainted += 1
                tag = "inp"
            else:
                tag = "ok "
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(out_path)
            job.current = out_path
            done += 1
            print(f"  inpaint {tag} {done}/{len(pending)}: {job.rel}", flush=True)
    finally:
        _drop_ocr()
        _drop_inpaint()
    return inpainted


def _get_bg_session():
    global _BG_SESSION
    if _BG_SESSION is None:
        print(f"loading bg model ({STAGE_BG_MODEL})...", flush=True)
        from rembg import new_session
        _BG_SESSION = new_session(STAGE_BG_MODEL)
        print("bg model ready.", flush=True)
    return _BG_SESSION


def _stage_bg(jobs):
    """BiRefNet. Writes transparent watch cutouts straight to OUTPUT_DIR.
    Skips when output exists with non-zero size. Unloads model before return."""
    pending = []
    for job in jobs:
        out_path = OUTPUT_DIR / job.rel
        if out_path.is_file() and out_path.stat().st_size > 0:
            job.current = out_path
            continue
        pending.append((job, out_path))
    if not pending:
        print(f"  bg: all {len(jobs)} already done", flush=True)
        return 0

    if not STAGE_BG_ENABLED:
        for job, out_path in pending:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = _open_with_exif(job.current).convert("RGBA")
            img.save(out_path)
            job.current = out_path
        return len(pending)

    done = 0
    try:
        from rembg import remove
        session = _get_bg_session()
        for job, out_path in pending:
            rgb = _open_with_exif(job.current).convert("RGB")
            cutout = refine_cutout(remove(rgb, session=session))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cutout.save(out_path)
            job.current = out_path
            done += 1
            print(f"  bg {done}/{len(pending)}: {job.rel}", flush=True)
    finally:
        _drop_bg()
    return done


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
    jobs = [
        ImageJob(src_path=p, rel=p.relative_to(INPUT_DIR), current=p)
        for p in paths
    ]

    print(f"processing {len(jobs)} images from {INPUT_DIR}")
    print(
        f"  pipeline (one model at a time, cached):\n"
        f"    1 upscale  : {STAGE_UPSCALE_ENABLED} ({STAGE_UPSCALE_MODEL} x{STAGE_UPSCALE_OUTSCALE})\n"
        f"    2 inpaint  : {STAGE_INPAINT_ENABLED} "
        f"({STAGE_INPAINT_MODEL}; OCR '{STAGE_OCR_LANG}' conf>={STAGE_OCR_CONFIDENCE})\n"
        f"    3 bg-cut   : {STAGE_BG_ENABLED} ({STAGE_BG_MODEL})\n"
        f"  geometric normalization (crop/scale/compose) lives in normalize_watches.py"
    )

    try:
        n = _stage_upscale(jobs)
        print(f"  upscale done ({n} new) -> {CACHE_UPSCALE}", flush=True)

        n = _stage_inpaint(jobs)
        print(f"  inpaint done ({n} inpainted) -> {CACHE_INPAINT}", flush=True)

        n = _stage_bg(jobs)
        print(f"  bg done ({n} new) -> {OUTPUT_DIR}", flush=True)
    finally:
        _drop_all_models()

    print(
        f"done. {n} transparent watches saved to {OUTPUT_DIR}  |  "
        f"chain: run normalize_watches.py for the final canvas"
    )


if __name__ == "__main__":
    main()
