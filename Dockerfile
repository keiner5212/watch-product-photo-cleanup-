# syntax=docker/dockerfile:1.7
# Watch image pipeline: one command, GPU upscale + background removal + Shopify upload.
#
# Base: pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
#   torch + CUDA are already installed (no torch wheel download at build time).
#   The nvidia CUDA runtime libs live in site-packages/nvidia/<lib>/lib, so they
#   are registered with ldconfig for onnxruntime-gpu (BiRefNet) to load its
#   CUDA execution provider.
#
# Host prereqs:
#   - NVIDIA driver (535.x or newer for CUDA 12.6)
#   - nvidia-container-toolkit

FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime@sha256:27c3135420bc184e86977170b6158c6133be3c7cc5c35e9e4fa87bdda629dc2b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/root/.cache \
    HF_HOME=/root/.cache/huggingface \
    HOME=/root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    curl ca-certificates \
    libgl1 libglib2.0-0 \
    libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# rembg pulls the CPU-only onnxruntime wheel, so it is removed before the GPU
# wheel is installed to keep one single onnxruntime package on the image.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
    "rembg==2.0.77" \
    "realesrgan==0.3.0" \
    "opencv-python-headless>=4.10,<5.0" && \
    pip uninstall -y onnxruntime || true
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
    flatbuffers protobuf coloredlogs psutil && \
    pip install --no-cache-dir --no-deps "onnxruntime-gpu==1.20.0" && \
    python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__, ort.get_available_providers())"

# Make the pip-installed CUDA libs discoverable by the dynamic linker.
# onnxruntime-gpu loads libcudart, libcudnn and libcublas at runtime, and this
# image keeps them under site-packages/nvidia/<lib>/lib.
RUN SITE=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    for dir in "$SITE"/nvidia/*/lib; do echo "$dir"; done \
    > /etc/ld.so.conf.d/zz-nvidia-pip.conf && \
    ldconfig && \
    ldconfig -p | grep -q libcudart.so.12

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

# Pre-bake model weights into the image filesystem, checksummed.
RUN mkdir -p /root/.cache/realesrgan && \
    curl -fsSL -o /root/.cache/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    echo "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb  /root/.cache/realesrgan/RealESRGAN_x2plus.pth" \
    | sha256sum -c -

# PyTorch BiRefNet general, fp16, CUDA only at runtime. The ONNX export does
# not fit this 6 GB card. Revision must match MATTE_REVISION in pipeline.py.
ENV MATTE_REVISION=e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
    "transformers==4.46.3" \
    "huggingface_hub==0.36.2" \
    "timm==1.0.31" \
    "kornia==0.9.2" \
    "einops==0.8.2" && \
    python3 -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__"
RUN python3 -c "import os; from huggingface_hub import snapshot_download; from transformers import AutoModelForImageSegmentation; revision=os.environ['MATTE_REVISION']; snapshot_download('ZhengPeng7/BiRefNet', revision=revision); AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', revision=revision, trust_remote_code=True)"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY *.py ./

RUN mkdir -p /data/in /data/out

ENV WATCH_INPUT_DIR=/data/in \
    WATCH_OUTPUT_DIR=/data/out \
    STAGE_BG_MODEL=birefnet-general

ENTRYPOINT ["python", "-m", "pipeline"]
