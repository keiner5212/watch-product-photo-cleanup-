# watch-product-photo-cleanup

Automated pipeline that prepares watch product photos for Shopify: removes the
brand text from the dial, removes the background, normalises size, and uploads
to the matching product variant. Currently tuned for the SEIKO catalog (brand
text removed via OCR + inpainting) but works for any brand -- just edit
`STAGE_OCR_KEYWORDS` in `process_images.py`.

## Pipeline

Four standalone scripts, one shared output folder (`./output/`). Each script
**empties `./output/` at startup** (preserving hidden subdirs like `.cache/`
for resume state) so every run starts from a clean slate. After each run,
review `./output/` and either approve the result or discard it.

```
+------------------+        +------------------+        +------------------+
|  BrandFolder/    |  -->   |  process_images  |  -->   |  ./output/       |
|  (raw inputs)    |        |  (AI stages)     |        |  (transparent)   |
+------------------+        +------------------+        +------------------+
                                                              |
                                                  approve_output.py
                                                              v
                                                   BrandFolder/ (replaced)
                                                              |
+------------------+        +------------------+        +------------------+
|  BrandFolder/    |  -->   |  remove_magenta  |  -->   |  ./output/       |
|  (AI-cleaned,    |        |  (LaMa touch-up) |        |  (touched-up)    |
|   marked #ff00bf)|        +------------------+        +------------------+
+------------------+                                               |
                                                          approve_output.py
                                                                     v
                                                      BrandFolder/ (replaced)
                                                                     |
+------------------+        +------------------+        +------------------+
|  BrandFolder/    |  -->   | normalize_watches|  -->   |  ./output/       |
|  (clean inputs)  |        |  (geometric,     |        |  (1200x1200      |
|                  |        |   no AI)         |        |   canvases)      |
+------------------+        +------------------+        +------------------+
                                                                     |
                                                       shopify_auth.py
                                                                     v
                                                            Shopify (uploaded)
```

### Pass 1 -- `process_images.py` (AI stages)

For every image under `./<BrandFolder>/`:

1. **Upscale** (`RealESRGAN_x2plus`) -- 2x resolution.
2. **OCR + inpaint** (`EasyOCR` + `big-LaMa`) -- finds brand text on the dial via
   CRAFT detection, inpaint it with surrounding dial texture.
3. **Background cut** (`BiRefNet-general`) -- transparent background.

Output lands at `./output/<BrandFolder>/iTrade-<SKU>.png` (transparent watch,
variable size). The upscale and inpaint steps cache to
`./output/.cache/02_upscale/` and `./output/.cache/03_inpaint/` (preserved
across runs for resume).

### Pass 2 -- `remove_magenta.py` (manual touch-up, optional)

Pass 1's OCR can miss small marks (logos, model numbers, faint text) or marks
that fall outside `STAGE_MASK_MAX_AREA_FRAC`. For those, paint the unwanted
area in the final PNG with the exact colour **`#ff00bf`** (R=255, G=0, B=191)
and run `remove_magenta.py`. It scans the tree, builds a tight mask from the
magenta pixels (tolerance 60 + close + dilate + feather) and inpaints each
masked region with `big-LaMa` so the dial texture is recreated.

Output lands at `./output/<BrandFolder>/iTrade-<SKU>.png`.

### Pass 3 -- `normalize_watches.py` (geometric normalization)

Pure PIL. No AI. Reads watches from `WATCH_INPUT_DIR` and writes
consistent-size, consistently-margined 1200x1200 square canvases to
`./output/<BrandFolder>/iTrade-<SKU>.png`.

For each image: alpha-threshold + dilate + blur to tighten the cutout,
symmetric square crop around the watch bbox, scale to fit
`TARGET_SIZE - 2*PADDING` (LANCZOS up, two-pass BOX down for >2x), compose
centered on the canvas. Sharpening is off by default -- the watch pixels
are not modified unless you set `SHARPEN_PERCENT > 0` at the top of the
file.

### `approve_output.py` (promote output to input)

Moves every file in `./output/<BrandFolder>/<sku>.png` to
`./<BrandFolder>/<sku>.png` (deleting the old input version first). This is
the "I liked the output, push it back to the inputs" step. Run it after any
of the passes above if you want to keep the result and feed it into the
next pass.

```bash
docker compose run --rm pipeline approve_output.py             # promote
docker compose run --rm pipeline approve_output.py --dry-run   # show what would move
docker compose run --rm pipeline approve_output.py --clear-after  # also clear ./output/
```

Does NOT delete input files that have no matching output (orphans stay).

## Workflow

Typical end-to-end:

1. **Run pass 1** to upscale, censor brand text, and cut the background.
   ```bash
   docker compose run --rm pipeline process_images.py
   ```
2. **Review `./output/`** -- open the PNGs and check the result.
3. **Approve** (push the output back to the brand folders as the new originals).
   ```bash
   docker compose run --rm pipeline approve_output.py --clear-after
   ```
4. **(Optional) Mark leftovers** with magenta `#ff00bf` in any image that
   still has OCR-missed marks, then run pass 2.
   ```bash
   docker compose run --rm pipeline remove_magenta.py
   docker compose run --rm pipeline approve_output.py --clear-after
   ```
5. **Run pass 3** to compose the final 1200x1200 canvases.
   ```bash
   docker compose run --rm pipeline normalize_watches.py
   ```
6. **Upload to Shopify** (reads from `./output/` directly) or approve first
   to archive in the brand folders.
   ```bash
   docker compose run --rm pipeline shopify_auth.py --dry-run
   docker compose run --rm pipeline shopify_auth.py
   ```

If you don't like a pass's output, just re-run the same script -- it will
empty `./output/` and start fresh.

## Hardware

- NVIDIA GPU with at least **6 GB VRAM** (RTX 3050 6 GB works).
- ~10 GB disk for the Docker image (PyTorch + models pre-baked).
- ~20 GB more for build cache if you rebuild from scratch.

## Setup

```bash
# Build the image (downloads AI weights at build time)
docker compose build

# Pass 1: AI stages (upscale + inpaint + bg-cut)
docker compose run --rm pipeline process_images.py

# Verbose trace + mask overlay (saves ./output/.debug_mask_overlay.png)
docker compose run --rm -e VERBOSE_LOG=1 pipeline process_images.py

# Pass 2: remove manually-marked magenta regions (optional)
docker compose run --rm pipeline remove_magenta.py

# Pass 3: geometric normalization (size + margins, no AI)
docker compose run --rm pipeline normalize_watches.py

# Promote output -> input when satisfied
docker compose run --rm pipeline approve_output.py --clear-after

# Shopify upload (reads from ./output/)
docker compose run --rm pipeline shopify_auth.py --dry-run
docker compose run --rm pipeline shopify_auth.py
```

The pipeline is fully self-contained: scripts are bind-mounted into the
container (`process_images.py`, `normalize_watches.py`, `remove_magenta.py`,
`approve_output.py`, `shopify_auth.py`) so edits take effect on the next run
without rebuilding the image.

## Image requirements

- **Naming**: `iTrade-<SKU>.<ext>` -- SKU becomes the filename stem.
- **Format**: JPEG, PNG, or WebP (any size; 1080x1200+ recommended).
- **Subject**: watch centred on plain background. The OCR relies on the
  bezel + dial being visible at the top of the image.
- **Output**: `./output/<BrandFolder>/` -- whatever the last script wrote
  (transparent watches post-AI, or 1200x1200 composed canvases after
  `normalize_watches.py`). Well under Shopify's 20 MB per-image limit
  (typical 1200x1200 PNG is ~700 KB).
- **Pass-2 marker colour**: paint leftovers with exact `#ff00bf`
  (R=255, G=0, B=191). Don't anti-alias the fill -- `remove_magenta.py`
  tolerates a small fringe but a clean fill produces the tightest mask.

## Uploading to Shopify

`shopify_auth.py` reads SKUs from `./output/<BrandFolder>/iTrade-<SKU>.png`
(the result of `normalize_watches.py`), matches them to Shopify product
variants, uploads the missing images via the staged upload API, and
associates each one with its variant.

```bash
# Prerequisites (one-time setup)
shopify store auth --store axuzmh-jz.myshopify.com \
    --scopes "read_products,write_products,read_inventory,write_inventory,read_locations"

# After auth, the CLI saves the access token to a per-store config file
# (Linux: ~/.config/shopify/<store>.json). Read it with cat:
cat ~/.config/shopify/axuzmh-jz.myshopify.com.json

# Dry run: print matches only
docker compose run --rm pipeline shopify_auth.py --dry-run

# Upload and associate
docker compose run --rm pipeline shopify_auth.py
```

The Shopify access token is hardcoded in `shopify_auth.py` -- rotate it
before publishing and paste the new value from `shopify store auth`.

## Configuration knobs

### `process_images.py` (AI stages)

| Constant | Default | What it controls |
|---|---|---|
| `STAGE_BG_ENABLED` | `True` | Run BiRefNet background removal |
| `STAGE_BG_MODEL` | `birefnet-general` | rembg session name |
| `STAGE_UPSCALE_ENABLED` | `True` | Run Real-ESRGAN upscale |
| `STAGE_UPSCALE_MODEL` | `RealESRGAN_x2plus.pth` | Real-ESRGAN weight file |
| `STAGE_UPSCALE_OUTSCALE` | `2` | Output = input x this |
| `STAGE_UPSCALE_TILE` | `512` | Tile size for VRAM control (0 = no tiling, OOM-prone) |
| `STAGE_INPAINT_ENABLED` | `True` | Run OCR + LaMa inpaint |
| `STAGE_OCR_KEYWORDS` | `["SEIKO", ...]` | Brand text to remove (fuzzy-matched) |
| `STAGE_OCR_CONFIDENCE` | `0.6` | OCR confidence floor |
| `STAGE_MASK_FEATHER_PX` | `11` | Gaussian blur on the inpaint mask edges |
| `LAMA_MAX_DIM` | `1280` | Downscale before LaMa on big images (6 GB VRAM safety) |
| `IMAGE_LIMIT` | `None` | Stop after this many images (debug aid) |
| `VERBOSE_LOG` | `0` | Set to `1` for per-image trace + mask overlay |

### `normalize_watches.py` (geometric normalization)

| Constant | Default | What it controls |
|---|---|---|
| `TARGET_SIZE` | `1200` | Final square canvas edge length |
| `PADDING` | `80` | Pixels of empty space around the watch bbox |
| `BACKGROUND` | `None` | `None` = transparent PNG; tuple/`"#hex"` = opaque fill |
| `OUTPUT_FORMAT` | `PNG` | `PNG` (transparent-safe) or `JPEG` (needs opaque bg) |
| `JPEG_QUALITY` | `95` | JPEG quality when `OUTPUT_FORMAT = "JPEG"` |
| `ALPHA_THRESHOLD` | `128` | Cutout alpha threshold for bbox detection |
| `ALPHA_DILATE_SIZE` | `5` | Max-filter on alpha before bbox (tightens edge) |
| `ALPHA_SMOOTH_RADIUS` | `1.2` | Gaussian blur on alpha before bbox |
| `SHARPEN_PERCENT` | `0` | Unsharp mask strength; `0` = off (watch pixels untouched) |
| `OVERWRITE` | `False` | (legacy -- OUTPUT_DIR is cleared at start anyway) |
| `IMAGE_LIMIT` | `None` | Stop after this many images (debug aid) |
| `VERBOSE_LOG` | `0` | Set to `1` for per-image trace |

## Project layout

```
.
├── Dockerfile                # Pytorch base + rembg/easyocr/simplelama/realesrgan
├── docker-compose.yml        # GPU passthrough, bind-mounts, env
├── LICENSE                    # MIT
├── README.md                  # you are here
├── process_images.py          # pass 1: upscale + OCR-inpaint + bg-cut (AI only)
├── remove_magenta.py          # pass 2: detect #ff00bf marks + LaMa inpaint
├── normalize_watches.py       # pass 3: symmetric crop + scale + compose (no AI)
├── approve_output.py          # promote ./output/ back to ./<BrandFolder>/
├── shopify_auth.py            # Shopify uploader (reads from ./output/)
├── .dockerignore              # excludes cache/output/etc from build context
├── .gitignore                 # excludes cache/output/credentials/etc
├── <BrandFolder>/             # input images (e.g. GMT/, Royal Oak/, ...)
└── output/                    # shared output folder (each script clears it on start;
                               # .cache/ survives for resume)
```

## License

MIT -- see `LICENSE`.
