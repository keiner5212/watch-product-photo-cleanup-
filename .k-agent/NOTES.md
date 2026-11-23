# Notes

- Preflight is strict in both directions: the run stops when a Shopify SKU has
  no local image or a local image has no Shopify variant.
- Matte is PyTorch BiRefNet general, fp16, CUDA only. No CPU device and no
  CPU retry. The ONNX export OOMs on the 6 GB card (784 MB decoder buffer
  with ~600 MB free). Revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4.
  The matte runs before the upscaler is loaded.
- As of 2026-11-23 the store has four SKUs without a local image (00722,
  00723, 00806, 00818, all SeikoJust Day-Date) and one local image without a
  variant (Nautilius/iTrade-00408.png), so the preflight blocks the run until
  those are sorted.
- Old pipeline leftovers live in ./output (PNG files per brand and
  ./output/others). The new flow writes WEBP files with the same names.
