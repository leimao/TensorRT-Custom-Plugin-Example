import unittest
import numpy as np

import common
import common_runtime


def test_engine(engine_file_path: str, plugin_lib_file_path: str):

    common_runtime.load_plugin_lib(plugin_lib_file_path=plugin_lib_file_path)
    engine = common_runtime.load_engine(engine_file_path=engine_file_path)

    inputs, outputs, bindings, stream = common.allocate_buffers(
        engine=engine, profile_idx=None)

    for host_device_buffer in inputs:
        data = np.random.uniform(low=-10.0,
                                 high=10.0,
                                 size=host_device_buffer.shape).astype(
                                     host_device_buffer.dtype).flatten()
        np.copyto(host_device_buffer.host, data)

    context = engine.create_execution_context()
    common.do_inference(
        context=context,
        engine=engine,
        inputs=inputs,
        outputs=outputs,
        bindings=bindings,
        stream=stream,
    )

    for input_host_device_buffer, output_host_device_buffer in zip(
            inputs, outputs):
        np.testing.assert_equal(input_host_device_buffer.host,
                                output_host_device_buffer.host)

    common.free_buffers(inputs=inputs, outputs=outputs, stream=stream)


class TestMain(unittest.TestCase):

    def test_engine_v2(self):

        engine_file_path = "../data/identity_neural_network_iplugin_v2_io_ext.engine"
        plugin_lib_file_path = "../build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so"
        test_engine(engine_file_path=engine_file_path,
                    plugin_lib_file_path=plugin_lib_file_path)

    def test_engine_v3(self):

        engine_file_path = "../data/identity_neural_network_iplugin_v3.engine"
        plugin_lib_file_path = "../build/src/plugins/IdentityConvIPluginV3/libidentity_conv_iplugin_v3.so"
        test_engine(engine_file_path=engine_file_path,
                    plugin_lib_file_path=plugin_lib_file_path)


if __name__ == "__main__":

    unittest.main()
