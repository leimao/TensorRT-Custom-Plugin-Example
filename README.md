# TensorRT Custom Plugin Example

## Introduction

This is a quick and self-contained TensorRT example. It demonstrates how to build a TensorRT custom plugin and how to use it in a TensorRT engine without complicated dependencies and too much abstraction.

The ONNX model we created is a simple identity neural network that consists of three `Conv` nodes whose weights and attributes are orchestrated so that the convolution operation is a simple identity operation. The second `Conv` node in the ONNX is replaced with a ONNX custom node `IdentityConv` that is not defined in the ONNX operator set. The TensorRT custom plugin is implemented to perform the `IdentityConv` operation using CUDA memory copy in the TensorRT engine.

## Usages

### Build Docker Images

To build the custom Docker image, please run the following command.

```bash
$ docker build -f docker/tensorrt.Dockerfile --no-cache --tag=tensorrt:25.02 .
```

### Run Docker Container

To run the custom Docker container, please run the following command.

```bash
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt tensorrt:25.02
```

### Build Application

To build the application, please run the following command.

```bash
$ cmake -B build
$ cmake --build build --config Release --parallel
```

Under the `build/src/plugins` directory, the custom plugin library will be saved as `libidentity_conv_iplugin_v2_io_ext.so` for `IPluginV2Ext` and `libidentity_conv_iplugin_v3.so` for `IPluginV3`, respectively. The `IPluginV2Ext` plugin interface has been deprecated since TensorRT 10.0.0 and will be removed in the future. The `IPluginV3` plugin interface is the only recommended interface for custom plugin development.

Under the `build/src/apps` directory, the engine builder will be saved as `build_engine`, and the engine runner will be saved as `run_engine`.

### Build ONNX Model

To build the ONNX model, please run the following command.

```bash
$ python scripts/create_identity_neural_network.py
```

The ONNX model will be saved as `identity_neural_network.onnx` under the `data` directory.

### Export ONNX Model

Alternatively, the ONNX model can be exported from a PyTorch model.

To export an ONNX model with ONNX Opset 15 or above, please run the following command.

```bash
$ python scripts/export_identity_neural_network_new_opset.py
```

To export an ONNX model with ONNX Opset 14 or below, please run the following command.

```bash
$ python scripts/export_identity_neural_network_old_opset.py
```

The ONNX model will be saved as `identity_neural_network.onnx` under the `data` directory.

### Build TensorRT Engine

To build the TensorRT engine from the ONNX model, please run the following command.

#### Build Engine with IPluginV2IOExt

We could build TensorRT engine using the builder application.

```bash
$ ./build/src/apps/build_engine data/identity_neural_network.onnx build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so data/identity_neural_network_iplugin_v2_io_ext.engine
```

Alternatively, we could use `trtexec` to build the TensorRT engine.

```bash
$ trtexec --onnx=data/identity_neural_network.onnx --saveEngine=data/identity_neural_network_iplugin_v2_io_ext.engine --staticPlugins=build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so
```

#### Build Engine with IPluginV3

We could build TensorRT engine using the builder application.

```bash
$ ./build/src/apps/build_engine data/identity_neural_network.onnx build/src/plugins/IdentityConvIPluginV3/libidentity_conv_iplugin_v3.so data/identity_neural_network_iplugin_v3.engine
```

Alternatively, we could use `trtexec` to build the TensorRT engine.

```bash
$ trtexec --onnx=data/identity_neural_network.onnx --saveEngine=data/identity_neural_network_iplugin_v3.engine --staticPlugins=build/src/plugins/IdentityConvIPluginV3/libidentity_conv_iplugin_v3.so
```

### Run TensorRT Engine

To run the TensorRT engine, please run the following command.

#### Run Engine with IPluginV2IOExt

```bash
$ ./build/src/apps/run_engine build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so data/identity_neural_network_iplugin_v2_io_ext.engine
```

#### Run Engine with IPluginV3

```bash
$ ./build/src/apps/run_engine build/src/plugins/IdentityConvIPluginV3/libidentity_conv_iplugin_v3.so data/identity_neural_network_iplugin_v3.engine
```

If the custom plugin implementation and integration are correct, the output of the TensorRT engine should be the same as the input.

## References

- [TensorRT Custom Plugin Example](https://leimao.github.io/blog/TensorRT-Custom-Plugin-Example/)
