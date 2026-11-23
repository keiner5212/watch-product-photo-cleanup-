# watch-product-photo-cleanup

One command, all inside Docker: prepares watch product photos for Shopify and
publishes them to the matching product variants.

```bash
docker compose run --rm pipeline
```

That single run counts what the store expects against what is on disk, upscales
every photo, removes the background, normalizes the framing, and uploads the
result to Shopify, replacing the image of every variant it matches.

## Flow

```
+---------------------------+     +--------------------+     +------------------+
|  <BrandFolder>/           | --> |  1 upscale         | --> |  2 background     |
|  iTrade-<SKU>.<ext>       |     |  Real-ESRGAN x2    |     |  BiRefNet matte   |
|  (raw inputs)             |     |  (GPU)             |     |  (transparent)    |
+---------------------------+     +--------------------+     +------------------+
                                                                     |
                                                                     v
+---------------------------+     +--------------------+     +------------------+
|  4 Shopify                | <-- |  3 normalize       | <-- |  2 background     |
|  product variant images   |     |  1200x1200 WEBP    |     |  BiRefNet matte   |
|  (replaced per SKU)       |     |  transparent, even |     |                  |
+---------------------------+     |  margin            |     +------------------+
                                  +--------------------+

  0.5 preflight runs before anything else and stops the run when the store
  and the local images disagree.
```

### 0.5 Preflight

Counts the Shopify products and variants against the local images and refuses
to continue on any gap:

- a variant SKU with no `iTrade-<SKU>` file -> **stop**
- a local file whose SKU is not in the store -> **stop**

Both lists are printed before the run ends, so nothing is uploaded half done.

### 1 Upscale

Real-ESRGAN x2plus on the GPU. The frame is cut into 256 px tiles with a
10 px overlap so the model fits in 6 GB. If a tile still runs out of VRAM
the tile is halved down to 128 px and that image is retried on the GPU.
The extra resolution is what the final canvas is built on.

### 2 Background removal

BiRefNet (`birefnet-general`) writes a soft alpha matte. The matte is what
keeps the edges clean: no halo, no fringe, no hard cut. The transition band
of the matte is then tightened so the edge stays crisp at 1200 px.

BiRefNet needs about 6 GB of VRAM, which is more than a 6 GB card has left
once the upscaler is loaded. The pipeline therefore runs the matte on the GPU
when there is room and falls back to the CPU automatically when the GPU
cannot hold it (about 20 s per image on the CPU).

Internally the matte is computed first and cached as an alpha PNG under
`./output/.cache/matte/`, then the cutout is upscaled. That keeps the two GPU
models from competing for VRAM and lets a re-run skip the work already done.

### 3 Normalization

Pure geometry, no AI. The framing rule is the same for every image, which is
what keeps the gallery steady when the customer clicks through it:

1. measure the watch on its own alpha mask
2. crop a square centred on that mask
3. scale it so its longest side always spans the same usable area
4. centre it on a 1200x1200 transparent canvas

The margin is therefore identical on all four sides of every image, and the
subject keeps the same size and the same position from one photo to the next.
Output is WEBP with alpha, quality 90, which Shopify accepts as-is and serves
with its own responsive sizes.

### 4 Publish

For every product that matches a local SKU:

1. upload the new image (staged upload) and wait until Shopify reports it
   `READY`
2. detach the media the variant was using
3. point the variant at the new media
4. delete the replaced media

Media that does not belong to a SKU is never touched.

## Setup

Requirements: Docker with the NVIDIA container toolkit, an NVIDIA GPU with
6 GB VRAM, and the Shopify CLI on the host for the one-time login.

```bash
# one time, on the host: opens the browser and stores the session
shopify store auth --store axuzmh-jz.myshopify.com \
    --scopes "read_products,write_products,read_inventory,write_inventory,read_locations"

# build the image (model weights are baked in at build time)
docker compose build

# run the whole flow
docker compose run --rm pipeline
```

The Shopify CLI writes the session to
`~/.config/shopify-cli-store-nodejs/config.json`. The compose file mounts that
directory read-only into the container, and the pipeline reads the token from
it. Nothing is written to the host except the images in `./output/`, and
nothing is left installed on the host: no Python packages, no virtualenv, no
model weights.

If the token is missing or expired the run stops immediately and prints the
command to fix it.

To hand the container less than the whole CLI folder, point it at a folder
that holds only the session you want to use:

```bash
mkdir -p ~/.config/shopify-session
jq 'to_entries[0].value.sessionsByUserId | to_entries[0].value
    | {store, accessToken, scopes, expiresAt}' \
    ~/.config/shopify-cli-store-nodejs/config.json > ~/.config/shopify-session/config.json
SHOPIFY_CLI_CONFIG_DIR=~/.config/shopify-session docker compose run --rm pipeline
```

## Useful flags

```bash
docker compose run --rm pipeline --dry-run   # preflight counts only, touches nothing
docker compose run --rm pipeline --force     # reprocess images that already have an output
docker compose run --rm pipeline --limit 5   # process the first 5 images only
```

## Utilities

Two helpers that do not belong to the image flow live next to it:

```bash
docker compose run --rm --entrypoint python pipeline shopify_metafields.py --dry-run
docker compose run --rm --entrypoint python pipeline shopify_strip_tags.py --dry-run
```

## Configuration

Every knob is an environment variable set in `docker-compose.yml`.

| Variable | Default | What it controls |
|---|---|---|
| `CANVAS_SIZE` | `1200` | Final square canvas edge length |
| `PADDING_RATIO` | `0.08` | Empty margin as a share of the canvas |
| `UPSCALE_SCALE` | `2` | Real-ESRGAN factor |
| `UPSCALE_TILE` | `256` | Input tile in px. Halves down to 128 on GPU OOM. 0 tries the whole frame first |
| `STAGE_BG_MODEL` | `birefnet-general` | rembg session name (`birefnet-general-lite` runs on the GPU) |
| `ALPHA_LOW` | `64` | Alpha below this counts as background |
| `ALPHA_HIGH` | `192` | Alpha above this counts as subject, the band in between is the soft edge |
| `ALPHA_THRESHOLD` | `128` | Alpha level that counts as subject |
| `COMPONENT_RATIO` | `0.15` | Smaller shapes kept next to the main one |
| `WEBP_QUALITY` | `90` | WEBP quality |
| `WEBP_METHOD` | `6` | WEBP encoder effort (0 fastest, 6 smallest) |
| `SHOPIFY_STORE` | `axuzmh-jz.myshopify.com` | Target store |
| `SHOPIFY_CLI_CONFIG_DIR` | `~/.config/shopify-cli-store-nodejs` | Where the CLI session lives |

## Image requirements

- **Naming**: `iTrade-<SKU>.<ext>` -- the SKU becomes the filename stem and
  is what the pipeline matches against the store.
- **Format**: JPEG, PNG or WebP. Sources are usually around 1200 px.
- **Subject**: the watch centred on a plain background.

## Project layout

```
.
├── Dockerfile                # torch + rembg + realesrgan, weights baked in
├── docker-compose.yml        # GPU passthrough, bind mounts, env
├── pipeline.py               # the one command: preflight, upscale, matte, normalize, publish
├── shopify_api.py            # Shopify client: token, queries, staged upload, media replace
├── shopify_metafields.py     # utility: assigns custom.sale_count
├── shopify_strip_tags.py     # utility: strips `coleccion:` tags
├── <BrandFolder>/            # input images (GMT/, Royal Oak/, ...)
└── output/                   # results, one WEBP per input image
```

## License

MIT -- see `LICENSE`.
