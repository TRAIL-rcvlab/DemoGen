# ============================================================
# DemoGen Dockerfile - Multi-stage build with proper layering
# Supports training and inference of experiments on GPU servers
# ============================================================

# ------ Stage 1: Base image with system dependencies ------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglfw3 \
        libglew-dev \
        libosmesa6-dev \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ------ Stage 2: Install Python dependencies (cached layer) ------
FROM base AS deps

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

# Install PyTorch with CUDA 11.8 support first (large, rarely changes)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies (changes less often than source code)
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ------ Stage 3: Install project packages ------
FROM deps AS app

WORKDIR /workspace/DemoGen

# Copy full source code
COPY . .

# Install project packages in editable mode
RUN cd demo_generation && pip install --no-cache-dir -e . && cd .. && \
    cd diffusion_policies && pip install --no-cache-dir -e . && cd .. && \
    cd pcd_visualizer && pip install --no-cache-dir -e . && cd ..

ENV HYDRA_FULL_ERROR=1
ENV PYTHONPATH=/workspace/DemoGen:/workspace/DemoGen/diffusion_policies

CMD ["bash"]
