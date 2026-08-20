# syntax=docker/dockerfile:1.7
# Watch image pipeline.
#
# Why pytorch/pytorch:2.7.0-cuda12.4-cudnn-runtime as base, not
# nvidia/cuda or python:3.11-slim-bookworm:
#   - torch + torchvision already installed (PyTorch's own build, ~8 GB
#     base image). No `pip install torch==2.5.0+cu124` at build time, so
#     no ENOSPC error on /tmp extracting 2 GB wheels.
#   - The image is regenerated periodically by PyTorch team, so apt keys
#     are usually fresh and no GPG signature mismatch on apt update.
#   - Smaller build steps above the base: just ~500 MB of AI deps.
# Host prereqs:
#   - NVIDIA driver (535.x or newer recommended for CUDA 12.4+)
#   - nvidia-container-toolkit

FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# pytorch/pytorch base images use Ubuntu 22.04 as underlying. apt keys
# typically valid, but use --allow-unauthenticated as defense for any
# rotated repos.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Pre-baked AI weights land in image filesystem.
    XDG_CACHE_HOME=/root/.cache \
    HF_HOME=/root/.cache/huggingface \
    HOME=/root

# OS deps in one layer. pytorch/pytorch:2.7.0 minimal base has no
# wget/curl, so install curl first.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        curl ca-certificates \
        git \
        libgl1 libglib2.0-0 \
        libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# All AI deps with NORMAL dep resolution (no --no-deps).
# pip handles transitive deps automatically — no more manual ModuleNotFoundError.
#
# rembg installs with cpu onnxruntime (not [gpu]).  Stage 1 bg-cut
# runs on CPU (~20 s/image).  Stages 2-4 stay GPU via torch.
# Dropped: paddlepaddle / paddleocr (CRAFT detection is more
# accurate than PaddleOCR's DB for thin serif on dark dials),
# onnxruntime-gpu (cuBLAS11 version chaos), nvidia-cublas-cu11
# pip wheels (LD_LIBRARY_PATH never resolved correctly).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
        "rembg[cpu]==2.0.77" \
        "easyocr>=1.7" \
        "simple-lama-inpainting==0.1.0" \
        "opencv-python-headless<5.0,>=4.10" \
        "facexlib" \
        "gfpgan" \
        "realesrgan==0.3.0"

# basicsr's PyPI sdist is broken on Python 3.11+. Patch get_version().
RUN cat > /tmp/patch_basicsr.py <<'PATCH_EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()
start = next(i for i, l in enumerate(lines) if l.startswith('def get_version'))
end = len(lines)
for i in range(start + 1, len(lines)):
    s = lines[i].lstrip()
    if s and s[0] not in (' ', chr(9)) and (s.startswith('def ') or s.startswith('class ')):
        end = i
        break
new = lines[:start] + [
    'def get_version():\n',
    '    return "1.4.2"\n',
    '\n',
] + lines[end:]
with open(path, 'w') as f:
    f.writelines(new)
print(path, 'patched')
PATCH_EOF
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip download "basicsr==1.4.2" --no-binary :all: --no-deps -d /tmp/bsr && \
    cd /tmp && tar xzf bsr/basicsr-1.4.2.tar.gz && \
    python3 /tmp/patch_basicsr.py /tmp/basicsr-1.4.2/setup.py && \
    pip install /tmp/basicsr-1.4.2 --no-deps --no-build-isolation && \
    rm -rf /tmp/bsr /tmp/basicsr-1.4.2 /tmp/patch_basicsr.py

# torchvision path-moved: rgb_to_grayscale is in functional, not functional_tensor.
RUN SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    sed -i "s|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|" \
        "$SITE/basicsr/data/degradations.py"

# Pre-bake AI weights into image filesystem. No GPU exposed at build, so
# model init might fail; downloads still get cached for first run.
RUN mkdir -p /root/.cache/realesrgan && \
    curl -fsSL -o /root/.cache/realesrgan/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    cat > /tmp/prebake.py <<'PREBAKE_EOF'
def step(name, fn):
    try:
        print('pre-baking ' + name + '...')
        fn()
        print('  ok')
    except BaseException as e:
        print('  WARN ' + name + ' failed: ' + type(e).__name__ + ': ' + str(e)[:200])

step('rembg BiRefNet', lambda: __import__('rembg').new_session('birefnet-general'))
step('EasyOCR English', lambda: __import__('easyocr').Reader(['en'], gpu=False, model_storage_directory='/root/.cache/EasyOCR', user_network_directory='/root/.cache/EasyOCR', download_enabled=True, detector=True, recognizer=True))
step('Big-LaMa', lambda: __import__('simple_lama_inpainting').SimpleLama())
print('pre-bake complete (warnings are non-fatal; downloads retry on first docker run)')
PREBAKE_EOF
RUN python3 /tmp/prebake.py && rm /tmp/prebake.py

# Bundle ALL Python scripts so any can be run inside the container.
COPY *.py ./

RUN mkdir -p /data/in /data/out

ENV WATCH_INPUT_DIR=/data/in \
    WATCH_OUTPUT_DIR=/data/out \
    STAGE_BG_MODEL=birefnet-general

# Default runs process_images.py. Override for any other python:
#   docker compose run --rm pipeline shopify_auth.py
#   docker compose run --rm pipeline update_stock.py
#   docker compose run --rm pipeline bash
ENTRYPOINT ["python"]
CMD ["process_images.py"]
