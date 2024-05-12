import unittest
import numpy as np

import common
import common_runtime


class TestMain(unittest.TestCase):

    def test_engine(self):

        engine_file_path = "../data/identity_neural_network.engine"
        plugin_lib_file_path = "../build/src/libidentity_conv.so"
        common_runtime.load_plugin_lib(
            plugin_lib_file_path=plugin_lib_file_path)
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
