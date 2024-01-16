# TensorRT Plugin Example

## Introduction

Quick and self-contained TensorRT quick example.

## Usages

### Build Docker Images

To build the custom Docker image, please run the following command.

```bash
$ docker build -f docker/tensorrt.Dockerfile --no-cache --tag=tensorrt:23.12 .
```

### Run Docker Container

To run the custom Docker container, please run the following command.

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt tensorrt:23.12
```

### Build

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
$ cmake --install build
```

```
./build/src/build_engine
./build/src/run_engine
```
