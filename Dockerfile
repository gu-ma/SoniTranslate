# Base image used as starter
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install necessary dependencies
RUN apt update && \
    apt install -y \
    python3-pip \
    git \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libsm6 \
    libxext6 \
    wget \
    curl \
    iproute2 \
    iputils-ping \
    pkg-config \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install -U --no-cache-dir pip==24.0 setuptools wheel

# Install 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy content to image
COPY . /src

# Install 
RUN pip install -r /src/requirements_base.txt
RUN pip install -r /src/requirements_extra.txt

# Install ONNX runtime Cuda 12.0 (https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x)
# RUN pip install onnxruntime-gpu

# Install ONNX runtime Cuda 11.X (https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-11x)
RUN pip install flatbuffers numpy packaging protobuf sympy
RUN pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

# Install optionnal
RUN pip install -q piper-tts==1.2.0
RUN pip install -q -r /src/requirements_xtts.txt
RUN pip install -q TTS==0.21.1  --no-deps

# Install additional
RUN pip install deepl elevenlabs python-dotenv

# TCMAlloc
RUN apt-get update && apt-get install -y --no-install-recommends libgoogle-perftools-dev

# Symlink python
RUN ln -n /usr/bin/python3 /usr/bin/python