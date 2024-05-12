import numpy as np

import common
import common_runtime


def main():

    engine_file_path = "../data/identity_neural_network.engine"
    plugin_lib_file_path = "../build/src/libidentity_conv.so"

    common_runtime.load_plugin_lib(plugin_lib_file_path)
    engine = common_runtime.load_engine(engine_file_path)

    # Profile index is only useful when the engine has dynamic shapes.
    inputs, outputs, bindings, stream = common.allocate_buffers(
        engine=engine, profile_idx=None)

    # Print input tensor information.
    print("Input Tensor:")
    for host_device_buffer in inputs:
        print(
            f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
            f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
        )
    # Print output tensor information.
    print("Output Tensor:")
    for host_device_buffer in outputs:
        print(
            f"Tensor Name: {host_device_buffer.name} Shape: {host_device_buffer.shape} "
            f"Data Type: {host_device_buffer.dtype} Data Format: {host_device_buffer.format}"
        )

    # Dummy example.
    # Fill each input with random values.
    for host_device_buffer in inputs:
        data = np.random.uniform(low=-10.0,
                                 high=10.0,
                                 size=host_device_buffer.shape).astype(
                                     host_device_buffer.dtype).flatten()
        # Print input tensor data.
        print(f"Input Tensor: {host_device_buffer.name}")
        print(data)
        # Copy data from numpy array to host buffer.
        np.copyto(host_device_buffer.host, data)

    # Execute the engine.
    context = engine.create_execution_context()
    common.do_inference_v2(context,
                           bindings=bindings,
                           inputs=inputs,
                           outputs=outputs,
                           stream=stream)

    # Print output tensor data.
    for host_device_buffer in outputs:
        print(f"Output Tensor: {host_device_buffer.name}")
        print(host_device_buffer.host)

    # In our case, the input and output tensor data should be exactly the same.
    for input_host_device_buffer, output_host_device_buffer in zip(
            inputs, outputs):
        np.testing.assert_equal(input_host_device_buffer.host,
                                output_host_device_buffer.host)

    # Clean up.
    common.free_buffers(inputs=inputs, outputs=outputs, stream=stream)


if __name__ == "__main__":

    main()
