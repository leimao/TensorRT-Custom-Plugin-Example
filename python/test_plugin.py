import unittest
import numpy as np

import common
import common_runtime

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# Build engine from a plugin.
def build_engine_from_plugin(plugin_lib_file_path):

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    # TensorRT runtime usually has to be initialized before `plugin_creator.create_plugin` is called.
    # Because the plugin creator may need to access the some functions, such as `getLogger`, from `NvInferRuntime.h`.
    # Otherwise, segmentation fault will occur because those functions are not accessible.
    # However, if the plugin creator does not need to access the functions from `NvInferRuntime.h`, the runtime can be initialized later.
    runtime = trt.Runtime(TRT_LOGGER)

    common_runtime.load_plugin_lib(plugin_lib_file_path)
    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_plugin_creator("IdentityConv", "1")
    assert plugin_creator is not None
    attribute_kernel_shape = trt.PluginField("kernel_shape",
                                             np.array([1, 1], dtype=np.int32),
                                             trt.PluginFieldType.INT32)
    attribute_strides = trt.PluginField("strides",
                                        np.array([1, 1], dtype=np.int32),
                                        trt.PluginFieldType.INT32)
    attribute_pads = trt.PluginField("pads",
                                     np.array([0, 0, 0, 0], dtype=np.int32),
                                     trt.PluginFieldType.INT32)
    attribute_group = trt.PluginField("group", np.array([3], dtype=np.int32),
                                      trt.PluginFieldType.INT32)
    field_collection = trt.PluginFieldCollection([
        attribute_kernel_shape, attribute_strides, attribute_pads,
        attribute_group
    ])
    plugin = plugin_creator.create_plugin(name="IdentityConv",
                                          field_collection=field_collection)
    input_layer = network.add_input(name="input_layer",
                                    dtype=trt.float32,
                                    shape=(1, 3, 480, 960))
    constant_weights = trt.Weights(np.ones((3, 1, 1, 1), dtype=np.float32))
    constant_layer = network.add_constant((3, 1, 1, 1), constant_weights)
    constant_layer_output = constant_layer.get_output(0)
    plugin_layer = network.add_plugin_v2(
        inputs=[input_layer, constant_layer_output], plugin=plugin)
    network.mark_output(plugin_layer.get_output(0))
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)

    return engine


class TestMain(unittest.TestCase):

    def test_plugin(self):

        plugin_lib_file_path = "../build/src/libidentity_conv.so"

        engine = build_engine_from_plugin(
            plugin_lib_file_path=plugin_lib_file_path)

        inputs, outputs, bindings, stream = common.allocate_buffers(
            engine=engine, profile_idx=None)

        for host_device_buffer in inputs:
            data = np.random.uniform(low=-10.0,
                                     high=10.0,
                                     size=host_device_buffer.shape).astype(
                                         host_device_buffer.dtype).flatten()
            np.copyto(host_device_buffer.host, data)

        context = engine.create_execution_context()
        common.do_inference_v2(context,
                               bindings=bindings,
                               inputs=inputs,
                               outputs=outputs,
                               stream=stream)

        for input_host_device_buffer, output_host_device_buffer in zip(
                inputs, outputs):
            np.testing.assert_equal(input_host_device_buffer.host,
                                    output_host_device_buffer.host)

        common.free_buffers(inputs=inputs, outputs=outputs, stream=stream)


if __name__ == "__main__":

    unittest.main()
