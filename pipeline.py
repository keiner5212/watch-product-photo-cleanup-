#!/usr/bin/env python3
"""One command watch image pipeline for Shopify.

    docker compose run --rm pipeline

Stages
------
0.5  preflight   count Shopify products against local images, stop on any gap
1    upscale     Real-ESRGAN x2plus on the GPU, tiled
2    matte       BiRefNet background removal, transparent cutout
3    normalize   square canvas, uniform margin, WEBP with alpha
4    publish     upload to Shopify, replacing the media of the matching SKU

Stages 1 and 2 run in two passes so the two GPU models never compete for the
6 GB of VRAM: the matte is computed on the source pixels (BiRefNet always
works on a 1024x1024 copy of the frame anyway) and cached as an alpha PNG,
then the cutout is upscaled and placed on the canvas.

Framing rule (stage 3) is what keeps the gallery steady: every watch is
measured on its own alpha mask, cropped to a square centred on that mask,
scaled so its longest side always spans the same usable area, and centred on
the canvas. The margin is therefore identical on all four sides of every
image, and swapping one photo for another does not move the subject.
"""
import argparse
import gc
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps

import shopify_api
from shopify_api import ShopifyError

INPUT_DIR = Path(os.environ.get("WATCH_INPUT_DIR", Path(__file__).parent)).resolve()
OUTPUT_DIR = Path(os.environ.get("WATCH_OUTPUT_DIR", Path(__file__).parent / "output")).resolve()

CANVAS_SIZE = int(os.environ.get("CANVAS_SIZE", 1200))
PADDING_RATIO = float(os.environ.get("PADDING_RATIO", 0.08))
PADDING = round(CANVAS_SIZE * PADDING_RATIO)

UPSCALE_SCALE = int(os.environ.get("UPSCALE_SCALE", 2))
UPSCALE_TILE = int(os.environ.get("UPSCALE_TILE", 256))
UPSCALE_TILE_MIN = 128
UPSCALE_WEIGHTS = Path(
    os.environ.get("UPSCALE_WEIGHTS", "/root/.cache/realesrgan/RealESRGAN_x2plus.pth")
)

MATTE_MODEL = os.environ.get("STAGE_BG_MODEL", "birefnet-general")
ALPHA_LOW = int(os.environ.get("ALPHA_LOW", 64))
ALPHA_HIGH = int(os.environ.get("ALPHA_HIGH", 192))

ALPHA_THRESHOLD = int(os.environ.get("ALPHA_THRESHOLD", 128))
COMPONENT_RATIO = float(os.environ.get("COMPONENT_RATIO", 0.15))

WEBP_QUALITY = int(os.environ.get("WEBP_QUALITY", 90))
WEBP_METHOD = int(os.environ.get("WEBP_METHOD", 6))

SKU_PREFIX = "iTrade-"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {"output", ".venv", ".tmp", "node_modules", "__pycache__"}


class PipelineError(RuntimeError):
    """The pipeline cannot continue."""


def safe(text):
    """Strip control characters from store and file names before printing."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "?", str(text))


def gpu_available():
    """True when the container can see an NVIDIA GPU."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass
class LocalImage:
    sku: str
    path: Path
    relative: Path

    @property
    def output(self) -> Path:
        return OUTPUT_DIR / self.relative.with_suffix(".webp")

    @property
    def matte(self) -> Path:
        return OUTPUT_DIR / ".cache" / "matte" / self.relative.with_suffix(".png")


def open_image(path: Path) -> Image.Image:
    raw = Image.open(path)
    try:
        raw = ImageOps.exif_transpose(raw)
    except Exception:
        pass
    return raw.convert("RGBA")


def flatten_on_white(image: Image.Image) -> Image.Image:
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, image.convert("RGBA")).convert("RGB")


def collect_local_images(input_dir: Path = INPUT_DIR) -> dict:
    """Map every iTrade-<SKU> file under input_dir to its SKU."""
    images = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXT:
            continue
        relative = path.relative_to(input_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if relative.parts[0] in SKIP_DIRS:
            continue
        if not path.stem.startswith(SKU_PREFIX):
            continue
        sku = path.stem[len(SKU_PREFIX):]
        if not sku:
            continue
        if sku in images:
            raise PipelineError(
                f"duplicate SKU {sku}: {images[sku].relative} and {relative}"
            )
        images[sku] = LocalImage(sku=sku, path=path, relative=relative)
    return images


class Upscaler:
    """Real-ESRGAN x2plus on the GPU, one tile at a time."""

    def __init__(self, scale=UPSCALE_SCALE, tile=UPSCALE_TILE):
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if not torch.cuda.is_available():
            raise PipelineError("upscale needs a CUDA GPU")
        if not UPSCALE_WEIGHTS.is_file():
            raise PipelineError(f"missing upscale weights: {UPSCALE_WEIGHTS}")

        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        self._model = RealESRGANer(
            scale=scale,
            model_path=str(UPSCALE_WEIGHTS),
            model=net,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=torch.device("cuda"),
        )
        self.device = "cuda"

    @property
    def tile(self):
        return self._model.tile_size

    def upscale(self, image: Image.Image) -> Image.Image:
        import torch

        rgb = np.asarray(image.convert("RGB"))
        while True:
            try:
                out = self._upscale_tiles(rgb)
                break
            except RuntimeError as error:
                if not self._is_oom(error):
                    raise
                if not self._shrink_tile():
                    raise PipelineError(
                        f"upscale out of GPU memory at tile {self.tile}"
                    ) from error
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"  upscale short on VRAM, tile -> {self.tile}",
                    flush=True,
                )
        return Image.fromarray(out, mode="RGB")

    def _shrink_tile(self):
        current = self.tile
        if 0 < current <= UPSCALE_TILE_MIN:
            return False
        self._model.tile_size = (
            256 if current == 0 else max(UPSCALE_TILE_MIN, current // 2)
        )
        return True

    @staticmethod
    def _is_oom(error):
        return (
            type(error).__name__ == "OutOfMemoryError"
            or "out of memory" in str(error).lower()
        )

    def _upscale_tiles(self, rgb):
        # realesrgan catches OOM per tile and keeps going, so the tiles run here.
        import torch
        import torch.nn.functional as F

        scale = self._model.scale
        tile = self.tile
        overlap = self._model.tile_pad
        frame = torch.tensor(
            np.transpose(rgb, (2, 0, 1)), device="cuda", dtype=torch.float16
        ).unsqueeze(0).div_(255)
        _, _, height, width = frame.shape
        pad_h = (scale - height % scale) % scale
        pad_w = (scale - width % scale) % scale
        if pad_h or pad_w:
            frame = F.pad(frame, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, padded_h, padded_w = frame.shape
        canvas = np.empty((padded_h * scale, padded_w * scale, 3), dtype=np.uint8)
        windows = self._windows(padded_h, padded_w, tile, overlap)
        for top, left, bottom, right, y1, x1, y2, x2 in windows:
            piece = frame[:, :, y1:y2, x1:x2].contiguous()
            with torch.inference_mode():
                pred = self._model.model(piece)
            pred = pred[
                :,
                :,
                (top - y1) * scale:(bottom - y1) * scale,
                (left - x1) * scale:(right - x1) * scale,
            ]
            patch = (
                pred.squeeze(0).float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            )
            canvas[top * scale:bottom * scale, left * scale:right * scale] = (
                patch.permute(1, 2, 0).cpu().numpy()
            )
            del pred, patch, piece
        return np.ascontiguousarray(canvas[: height * scale, : width * scale])

    @staticmethod
    def _windows(height, width, tile, overlap):
        if tile <= 0:
            return [(0, 0, height, width, 0, 0, height, width)]
        windows = []
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                bottom = min(top + tile, height)
                right = min(left + tile, width)
                windows.append((
                    top,
                    left,
                    bottom,
                    right,
                    max(top - overlap, 0),
                    max(left - overlap, 0),
                    min(bottom + overlap, height),
                    min(right + overlap, width),
                ))
        return windows

    def release(self):
        del self._model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class MatteRemover:
    """BiRefNet background removal."""

    gpu_usable = True

    def __init__(self, model=MATTE_MODEL):
        from rembg import new_session

        self._model = model
        self._new_session = new_session
        self._session = None
        self.device = "none"
        providers = ["CPUExecutionProvider"]
        if gpu_available() and self.gpu_usable:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._open(providers)

    def _open(self, providers):
        self._session = self._new_session(self._model, providers=providers)
        inner = getattr(self._session, "inner_session", None)
        active = list(inner.get_providers()) if inner is not None else []
        self.device = active[0] if active else "unknown"

    def _run(self, image):
        from rembg import remove

        mask = remove(
            flatten_on_white(image), session=self._session, only_mask=True
        ).convert("L")
        values = np.asarray(mask, dtype=np.int32)
        span = max(ALPHA_HIGH - ALPHA_LOW, 1)
        tightened = np.clip((values - ALPHA_LOW) * 255 // span, 0, 255).astype(np.uint8)
        return Image.fromarray(tightened, mode="L")

    def alpha(self, image: Image.Image) -> Image.Image:
        """Return the alpha matte of the subject, tightened for crisp edges."""
        try:
            return self._run(image)
        except Exception as error:
            if not self.device.startswith("CUDA"):
                raise
            print(
                f"  GPU matte failed ({type(error).__name__}), retrying on CPU",
                flush=True,
            )
            MatteRemover.gpu_usable = False
            self._open(["CPUExecutionProvider"])
            return self._run(image)

    def release(self):
        del self._session
        gc.collect()


def subject_mask(cutout: Image.Image) -> Image.Image:
    """Binary mask of the watch: the dominant shapes above the alpha threshold."""
    alpha = np.asarray(cutout.split()[3])
    binary = (alpha >= ALPHA_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return Image.fromarray(binary * 255, mode="L")
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = float(areas.max()) * COMPONENT_RATIO
    keep = np.zeros(binary.shape, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= threshold:
            keep |= labels == label
    return Image.fromarray(keep.astype(np.uint8) * 255, mode="L")


def resize_by_scale(image: Image.Image, scale: float) -> Image.Image:
    """Resize by a factor, keeping the aspect ratio and the edges clean."""
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if scale == 1.0:
        return image
    if scale > 1.0:
        return image.resize(size, Image.LANCZOS)
    if scale < 0.5:
        half = (max(1, round(image.width * 0.5)), max(1, round(image.height * 0.5)))
        image = image.resize(half, Image.BOX)
    return image.resize(size, Image.BOX)


def normalize_canvas(cutout: Image.Image) -> Image.Image:
    """Place the watch centred on a square transparent canvas.

    The subject itself is measured, scaled so its longest side always spans
    the same usable area, and centred. Every photo therefore shows the watch
    at the same size and in the same place, and the margin on the long axis
    is identical from one image to the next.
    """
    bbox = subject_mask(cutout).getbbox()
    if not bbox:
        raise PipelineError("empty cutout: nothing above the alpha threshold")
    left, top, right, bottom = bbox
    subject_width = right - left
    subject_height = bottom - top
    scale = (CANVAS_SIZE - 2 * PADDING) / max(subject_width, subject_height)

    # Keep a rim around the subject so the soft edge of the matte survives.
    rim = max(2, round(max(subject_width, subject_height) * 0.02))
    crop_box = (
        max(0, left - rim),
        max(0, top - rim),
        min(cutout.width, right + rim),
        min(cutout.height, bottom + rim),
    )
    scaled = resize_by_scale(cutout.crop(crop_box), scale)

    subject_left = (left - crop_box[0]) * scale
    subject_top = (top - crop_box[1]) * scale
    offset_x = round((CANVAS_SIZE - subject_width * scale) / 2 - subject_left)
    offset_y = round((CANVAS_SIZE - subject_height * scale) / 2 - subject_top)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    canvas.paste(scaled, (offset_x, offset_y), scaled)
    return canvas


def preflight(products, local_images) -> dict:
    """Compare the store against the local files. Raises when they disagree."""
    online = {}
    variants_without_sku = 0
    for product in products:
        for variant in product["variants"]:
            sku = variant.get("sku")
            if not sku:
                variants_without_sku += 1
                continue
            online.setdefault(sku, []).append((product, variant))

    missing = sorted(set(online) - set(local_images))
    unused = sorted(set(local_images) - set(online))

    print("step 0.5  preflight")
    print(f"  shopify products        {len(products)}")
    print(f"  shopify variants        {sum(len(p['variants']) for p in products)}")
    print(f"  variants with a SKU     {len(online)}")
    print(f"  local images detected   {len(local_images)}")
    if variants_without_sku:
        print(f"  variants without a SKU  {variants_without_sku}")

    if missing or unused:
        if missing:
            print(f"  ERROR {len(missing)} SKU(s) in Shopify have no local image:")
            for sku in missing:
                print(f"    {safe(SKU_PREFIX + sku)}")
        if unused:
            print(f"  ERROR {len(unused)} local image(s) match no Shopify variant:")
            for sku in unused:
                print(f"    {safe(SKU_PREFIX + sku)}  ({safe(local_images[sku].relative)})")
        raise PipelineError("preflight failed: the store and the local images disagree")

    print("  match                   ok")
    return {"online": online, "products": products}


def build_mattes(local_images, pending, force):
    """Stage 2: background removal, cached as an alpha PNG per image."""
    missing = [image for image in pending if force or not image.matte.is_file()]
    if not missing:
        print(f"step 2  background removal: {len(pending)} matte(s) cached")
        return
    matter = MatteRemover()
    print(f"step 2  background removal ({MATTE_MODEL}, {matter.device})")
    done = 0
    for image in missing:
        started = time.monotonic()
        try:
            alpha = matter.alpha(open_image(image.path))
            image.matte.parent.mkdir(parents=True, exist_ok=True)
            alpha.save(image.matte, "PNG", optimize=True)
        except Exception as error:
            print(f"  FAIL {safe(image.relative)}: {error}", flush=True)
            continue
        done += 1
        print(
            f"  [{done:>4}/{len(missing)}] {safe(image.relative)}  "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
    matter.release()


def build_images(local_images, force=False, limit=0):
    """Stages 1 to 3: upscale, remove the background, normalize."""
    pending = []
    for sku in sorted(local_images):
        image = local_images[sku]
        if not force and image.output.is_file() and image.output.stat().st_size > 0:
            continue
        pending.append(image)
        if limit and len(pending) >= limit:
            break

    if not pending:
        print("step 1-3  upscale, matte, normalize")
        print(f"  all {len(local_images)} image(s) already normalized, nothing to do")
        return 0

    print("step 1-3  upscale, matte, normalize")
    print(
        f"  canvas {CANVAS_SIZE}x{CANVAS_SIZE}, margin {PADDING}px, "
        f"upscale x{UPSCALE_SCALE}"
    )
    build_mattes(local_images, pending, force)

    upscaler = Upscaler()
    tile_label = upscaler.tile if upscaler.tile else "full"
    print(
        f"step 1  upscale (Real-ESRGAN x{UPSCALE_SCALE}, {upscaler.device}, "
        f"tile {tile_label})"
    )
    done = 0
    for image in pending:
        started = time.monotonic()
        try:
            source = open_image(image.path)
            with Image.open(image.matte) as cached:
                matte = cached.convert("L")
            alpha = ImageChops.multiply(source.split()[3], matte)
            scaled = upscaler.upscale(flatten_on_white(source))
            scaled_alpha = alpha.resize(scaled.size, Image.LANCZOS)
            cutout = Image.merge("RGBA", (*scaled.split(), scaled_alpha))
            canvas = normalize_canvas(cutout)
            image.output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(
                image.output,
                "WEBP",
                quality=WEBP_QUALITY,
                method=WEBP_METHOD,
            )
        except Exception as error:
            print(f"  FAIL {safe(image.relative)}: {error}", flush=True)
            continue
        done += 1
        size_kb = image.output.stat().st_size / 1024
        print(
            f"  [{done:>4}/{len(pending)}] {safe(image.relative)} -> "
            f"{safe(image.output.relative_to(OUTPUT_DIR))}  {size_kb:.0f} KB  "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
    upscaler.release()
    print(f"  {done} image(s) normalized")
    return done


def publish_product(token, product, matches):
    """Upload the images of one product and replace the media of its SKUs.

    The new image is uploaded and attached before the old one is removed, so
    a failure never leaves the variant without a picture.
    """
    missing_outputs = [
        local.relative for _, local in matches if not local.output.is_file()
    ]
    if missing_outputs:
        raise PipelineError(
            f"cannot publish {safe(product['title'])}: missing output for "
            + ", ".join(safe(item) for item in missing_outputs)
        )

    media_items = shopify_api.fetch_product_media(token, product["id"])
    old_by_sku = {}
    for media in media_items:
        sku = shopify_api.media_sku(media)
        if sku:
            old_by_sku.setdefault(sku, []).append(media)

    new_media = {}
    for variant, local in matches:
        alt = f"{SKU_PREFIX}{local.sku} - {product['title']}"
        media_id = shopify_api.attach_image_to_product(
            token, product["id"], local.output, alt
        )
        new_media[local.sku] = media_id

    detach_pairs = []
    detached_ids = set()
    for variant, local in matches:
        referenced = {
            node["id"] for node in variant.get("media", {}).get("nodes", [])
        }
        for media in old_by_sku.get(local.sku, []):
            if media["id"] in referenced:
                detach_pairs.append((variant["id"], media["id"]))
                detached_ids.add(media["id"])
    if detach_pairs:
        shopify_api.detach_variant_media(token, product["id"], detach_pairs)

    assignments = [(variant["id"], new_media[local.sku]) for variant, local in matches]
    shopify_api.set_variant_media(token, product["id"], assignments)

    still_used = set()
    for variant in product["variants"]:
        for node in variant.get("media", {}).get("nodes", []):
            still_used.add(node["id"])
    still_used -= detached_ids
    deletable = [
        media["id"]
        for media in media_items
        if media["id"] not in still_used and shopify_api.media_sku(media) in new_media
    ]
    if deletable:
        shopify_api.delete_product_media(token, product["id"], deletable)

    print(
        f"  {safe(product['title'])}: {len(matches)} variant(s), "
        f"{len(deletable)} media replaced",
        flush=True,
    )
    return len(new_media), len(deletable)


def publish(token, products, local_images):
    """Stage 4: upload every image and replace the media of the matching SKU."""
    print("step 4  publish to Shopify")
    matched_products = 0
    uploaded = 0
    replaced = 0

    for product in products:
        matches = [
            (variant, local_images[variant["sku"]])
            for variant in product["variants"]
            if variant.get("sku") in local_images
        ]
        if not matches:
            continue
        matched_products += 1
        sent, gone = publish_product(token, product, matches)
        uploaded += sent
        replaced += gone

    print(
        f"  {matched_products} product(s) published, {uploaded} image(s) uploaded, "
        f"{replaced} media replaced"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare watch images and publish them to Shopify"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the preflight counts only, touch nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reprocess images that already have an output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N images (debug aid)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not INPUT_DIR.is_dir():
        raise PipelineError(f"input dir missing: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    token = shopify_api.load_token()
    local_images = collect_local_images()
    if not local_images:
        raise PipelineError(f"no {SKU_PREFIX}<SKU> images found under {INPUT_DIR}")

    products = shopify_api.fetch_products(token)
    preflight(products, local_images)

    if args.dry_run:
        print("dry run: nothing processed, nothing uploaded")
        return

    build_images(local_images, force=args.force, limit=args.limit)
    if args.limit:
        print(f"limited to {args.limit} image(s), nothing published")
        return
    publish(token, products, local_images)
    print("done")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, PipelineError, ShopifyError) as error:
        sys.exit(str(error))
