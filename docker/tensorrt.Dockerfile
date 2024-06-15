FROM nvcr.io/nvidia/tensorrt:24.05-py3

ARG CMAKE_VERSION=3.29.3
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        wget \
        git && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install cuda-python==12.3.0 \
                numpy==1.26.3 \
                onnx==1.15.0
RUN pip install --extra-index-url https://pypi.ngc.nvidia.com onnx_graphsurgeon==0.3.27
RUN pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121