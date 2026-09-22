FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# uv はバージョン固定で公式イメージからバイナリをコピーする（issue #10）
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOME=/root
ENV HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    iproute2 \
    libgl1 \
    libglib2.0-0 \
    less \
    libice6 \
    libsm6 \
    libxi6 \
    libxkbcommon0 \
    libxrender1 \
    ninja-build \
    pkg-config \
    procps \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ripgrep \
    tini \
    unzip \
    vim-tiny \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        "huggingface_hub[cli]" \
        ipywidgets \
        jupyterlab \
        notebook \
        pyyaml

RUN mkdir -p /app/models /app/.cache/huggingface /notebooks

WORKDIR /notebooks

EXPOSE 8188 8888

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"]
