# Base image used as starter
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    curl \
    iproute2 \
    iputils-ping \
    && apt-get autoremove -y \
    && apt-get clean \
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
RUN pip install onnxruntime-gpu

# Install optionnal
RUN pip install -q piper-tts==1.2.0
RUN pip install -q -r /src/requirements_xtts.txt
RUN pip install -q TTS==0.21.1  --no-deps

# TCMAlloc
RUN apt-get update && apt-get install -y --no-install-recommends libgoogle-perftools-dev

# Symlink python
RUN ln -n /usr/bin/python3 /usr/bin/python

RUN pip install python-dotenv