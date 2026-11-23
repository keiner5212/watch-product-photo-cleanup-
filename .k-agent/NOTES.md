# Notes

- Preflight is strict in both directions: the run stops when a Shopify SKU has
  no local image or a local image has no Shopify variant.
- BiRefNet needs about 6 GB of VRAM, more than the 6 GB card has free once
  the upscaler is loaded, so the matte runs on the CPU (about 20 s per image).
  The pipeline falls back on its own, no flag needed.
- As of 2026-11-23 the store has four SKUs without a local image (00722,
  00723, 00806, 00818, all SeikoJust Day-Date) and one local image without a
  variant (Nautilius/iTrade-00408.png), so the preflight blocks the run until
  those are sorted.
- Old pipeline leftovers live in ./output (PNG files per brand and
  ./output/others). The new flow writes WEBP files with the same names.
