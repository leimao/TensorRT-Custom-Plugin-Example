FROM nvcr.io/nvidia/tensorrt:25.02-py3

ARG CMAKE_VERSION=4.0.2
ARG NUM_JOBS=8
ARG PYTHON_VENV_PATH=/python/venv

ENV DEBIAN_FRONTEND=noninteractive

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
        python3-full \
        wget \
        git && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN mkdir -p ${PYTHON_VENV_PATH} && \
    python3 -m venv ${PYTHON_VENV_PATH}

ENV PATH=${PYTHON_VENV_PATH}/bin:$PATH

RUN cd ${PYTHON_VENV_PATH}/bin && \
    pip install --upgrade pip setuptools wheel

RUN cd ${PYTHON_VENV_PATH}/bin && \
    pip install cuda-python==12.6.2.post1 \
                numpy==2.2.6 \
                onnx==1.18.0 \
                onnx_graphsurgeon==0.5.8

RUN cd ${PYTHON_VENV_PATH}/bin && \
    pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126