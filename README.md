# watch-product-photo-cleanup

Automated pipeline that prepares watch product photos for Shopify: removes the
brand text from the dial, removes the background, normalises size, and uploads
to the matching product variant. Currently tuned for the SEIKO catalog (brand
text removed via OCR + inpainting) but works for any brand — just edit
`STAGE_OCR_KEYWORDS` in `process_images.py`.

## Pipeline

The cleanup is split into two passes; the second one is optional and only runs
when the first pass left marks behind that the OCR missed.

### Pass 1 — `process_images.py`

For every image under `./<BrandFolder>/`:

1. **Upscale** (`RealESRGAN_x2plus`) — 2× resolution.
2. **OCR + inpaint** (`EasyOCR` + `big-LaMa`) — finds brand text on the dial via
   CRAFT detection, inpaint it with surrounding dial texture.
3. **Background cut** (`BiRefNet-general`) — transparent background.
4. **Finalize** — symmetric square crop, scale to 1200×1200 with 80 px
   padding, sharpen, compose onto a 1200×1200 canvas.

Output lands at `./output/<BrandFolder>/iTrade-<SKU>.png`.

### Pass 2 — `remove_magenta.py` (manual touch-up)

Pass 1's OCR can miss small marks (logos, model numbers, faint text) or marks
that fall outside `STAGE_MASK_MAX_AREA_FRAC`. For those, paint the unwanted
area in the final PNG with the exact colour **`#ff00bf`** (R=255, G=0, B=191)
and run `remove_magenta.py`. It scans the tree, builds a tight mask from the
magenta pixels (tolerance 60 + close + dilate + feather) and inpaints each
masked region with `big-LaMa` so the dial texture is recreated.

Output lands at `./output/<BrandFolder>/iTrade-<SKU>.png` (same path as
pass 1 — re-running pass 2 on an already-clean file is a cache hit).

## Two-pass workflow

1. **Run pass 1** to upscale, centre, censor brand text, and cut the
   background for the whole batch.
   ```bash
   docker compose run --rm pipeline process_images.py
   ```
2. **Replace originals with the pass-1 output** — delete every file under
   `./<BrandFolder>/` and move/rename the matching files from
   `./output/<BrandFolder>/` back into `./<BrandFolder>/`. The processed
   PNGs are now your new "originals". Doing this once means the rest of the
   batch never gets re-upscaled / re-cut, and the next pass only has to
   touch the spots you mark.
3. **Manually mark leftovers** — open each processed PNG and fill any
   remaining brand text / logo / artifact with the exact colour
   **`#ff00bf`** (pure magenta, no anti-alias on the fill). Any image
   selector works; GIMP, Photoshop, paint.net, or a quick script.
4. **Run pass 2** on the now-magenta-tagged tree.
   ```bash
   docker compose run --rm pipeline remove_magenta.py
   ```
   Images with no magenta are copied through unchanged; images with magenta
   are inpainted and written to `./output/<BrandFolder>/`.
5. **Upload to Shopify** from the final `./output/`.
   ```bash
   docker compose run --rm pipeline shopify_auth.py --dry-run
   docker compose run --rm pipeline shopify_auth.py
   ```

Pass 2 dramatically cuts the workload because pass 1 has already done the
heavy lifting (upscale, centring, background cut, brand-text OCR). What
remains is a handful of targeted magenta strokes per image.

## Hardware

- NVIDIA GPU with at least **6 GB VRAM** (RTX 3050 6 GB works).
- ~10 GB disk for the Docker image (PyTorch + models pre-baked).
- ~20 GB more for build cache if you rebuild from scratch.

## Setup

```bash
# Build the image (downloads AI weights at build time)
docker compose build

# Pass 1: process a batch of images from ./<BrandFolder>/
docker compose run --rm pipeline process_images.py

# Verbose trace + mask overlay (saves ./output/.debug_mask_overlay.png)
docker compose run --rm -e VERBOSE_LOG=1 pipeline process_images.py

# Pass 2: remove manually-marked magenta regions from the processed tree
docker compose run --rm pipeline remove_magenta.py
```

The pipeline is fully self-contained: scripts are bind-mounted into the
container (`process_images.py`, `remove_magenta.py`, `shopify_auth.py`) so
edits take effect on the next run without rebuilding the image.

## Image requirements

- **Naming**: `iTrade-<SKU>.<ext>` — SKU becomes the filename stem.
- **Format**: JPEG, PNG, or WebP (any size; 1080×1200+ recommended).
- **Subject**: watch centred on plain background. The OCR relies on the
  bezel + dial being visible at the top of the image.
- **Output**: 1200×1200 PNG with transparent background, well under
  Shopify's 20 MB per-image limit (typical output is ~700 KB).
- **Pass-2 marker colour**: paint leftovers with exact `#ff00bf`
  (R=255, G=0, B=191). Don't anti-alias the fill — `remove_magenta.py`
  tolerates a small fringe but a clean fill produces the tightest mask.

## Uploading to Shopify

`shopify_auth.py` reads SKUs from `output/<BrandFolder>/iTrade-<SKU>.png`,
matches them to Shopify product variants, uploads the missing images via the
staged upload API, and associates each one with its variant.

```bash
# Prerequisites (one-time setup)
shopify store auth --store axuzmh-jz.myshopify.com \
    --scopes "read_products,write_products,read_inventory,write_inventory,read_locations"

# Dry run: print matches only
docker compose run --rm pipeline shopify_auth.py --dry-run

# Upload and associate
docker compose run --rm pipeline shopify_auth.py
```

The Shopify access token is hardcoded in `shopify_auth.py` — rotate it
before publishing and paste the new value from `shopify store auth`.

## Configuration knobs

`process_images.py` exposes these (top of file):

| Constant | Default | What it controls |
|---|---|---|
| `STAGE_BG_ENABLED` | `True` | Run BiRefNet background removal |
| `STAGE_BG_MODEL` | `birefnet-general` | rembg session name |
| `STAGE_UPSCALE_ENABLED` | `True` | Run Real-ESRGAN upscale |
| `STAGE_UPSCALE_MODEL` | `RealESRGAN_x2plus.pth` | Real-ESRGAN weight file |
| `STAGE_UPSCALE_OUTSCALE` | `2` | Output = input × this |
| `STAGE_UPSCALE_TILE` | `512` | Tile size for VRAM control (0 = no tiling, OOM-prone) |
| `STAGE_INPAINT_ENABLED` | `True` | Run OCR + LaMa inpaint |
| `STAGE_OCR_KEYWORDS` | `["SEIKO"]` | Brand text to remove (fuzzy-matched) |
| `STAGE_OCR_CONFIDENCE` | `0.6` | OCR confidence floor |
| `STAGE_MASK_FEATHER_PX` | `11` | Gaussian blur on the inpaint mask edges |
| `LAMA_MAX_DIM` | `1280` | Downscale before LaMa on big images (6 GB VRAM safety) |
| `TARGET_SIZE` | `1200` | Final square canvas |
| `PADDING` | `80` | Pixels of padding around the watch in the final canvas |
| `OUTPUT_FORMAT` | `PNG` | `PNG` or `JPEG` |
| `IMAGE_LIMIT` | `None` | Stop after this many images (debug aid) |
| `VERBOSE_LOG` | `0` | Set to `1` for per-image trace + mask overlay |

## Project layout

```
.
├── Dockerfile              # Pytorch base + rembg/easyocr/simplelama/realesrgan
├── docker-compose.yml      # GPU passthrough, bind-mounts, env (incl. PYTHONUNBUFFERED)
├── LICENSE                  # MIT
├── README.md                # you are here
├── process_images.py        # pass 1: upscale + OCR-inpaint + bg-cut + finalize
├── remove_magenta.py        # pass 2: detect #ff00bf marks + LaMa inpaint
├── shopify_auth.py          # Shopify uploader
├── .dockerignore            # excludes cache/output/etc from build context
├── .gitignore               # excludes cache/output/credentials/etc
├── <BrandFolder>/           # input images (e.g. GMT/, Royal Oak/, ...)
└── output/                  # pipeline output (.cache/ + final PNGs)
```

## License

MIT — see `LICENSE`.
