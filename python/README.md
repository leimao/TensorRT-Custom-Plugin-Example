# TensorRT Python Inference

## Unit Test

Assuming the `IPluginV2IOExt` and `IPluginV3` plugins have been built, the engine that uses each of the plugins have been built, the unit tests can be run.

To run the unit test, please run the following command.

```bash
$ python -m unittest test_plugin
$ python -m unittest test_engine
```

## Run TensorRT Engine

To run the TensorRT engine, please run the following command.

### IPluginV2IOExt

```bash
$ python main.py --engine_file_path ../data/identity_neural_network_iplugin_v2_io_ext.engine --plugin_lib_file_path ../build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so
```

### IPluginV3

```bash
$ python main.py --engine_file_path ../data/identity_neural_network_iplugin_v3.engine --plugin_lib_file_path ../build/src/plugins/IdentityConvIPluginV3/libidentity_conv_iplugin_v3.so
```
