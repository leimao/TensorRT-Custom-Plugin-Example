import argparse
import numpy as np

import common
import common_runtime


def main():

    # Add an argparser to specify the engine file path and plugin library file path.
    parser = argparse.ArgumentParser(
        description="Run an engine with Identity Plugin.")
    parser.add_argument(
        "--engine_file_path",
        type=str,
        default="../data/identity_neural_network_iplugin_v3.engine",
        help="Path to the engine file.",
    )
    parser.add_argument(
        "--plugin_lib_file_path",
        type=str,
        default=
        "../build/src/plugins/IdentityConvIPluginV3/libidentity_conv_iplugin_v3.so",
        help="Path to the plugin library file.",
    )

    args = parser.parse_args()
    engine_file_path = args.engine_file_path
    plugin_lib_file_path = args.plugin_lib_file_path

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
    common.do_inference(
        context=context,
        engine=engine,
        inputs=inputs,
        outputs=outputs,
        bindings=bindings,
        stream=stream,
    )

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
