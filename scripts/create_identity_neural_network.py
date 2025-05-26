# Create a neural network that consists of three identity convolutional layers.

import os

import numpy as np

import onnx
import onnx_graphsurgeon as gs


def main():

    opset_version = 15
    data_directory_path = "data"
    onnx_file_name = "identity_neural_network.onnx"
    onnx_file_path = os.path.join(data_directory_path, onnx_file_name)

    input_shape = (1, 3, 480, 960)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    input_channels = input_shape[1]

    # configure a dummy conv:
    weights_shape = (input_channels, 1, 1, 1)
    num_groups = input_channels
    weights_data = np.ones(weights_shape, dtype=np.float32)

    # generate ONNX model
    X0 = gs.Variable(name="X0", dtype=np.float32, shape=input_shape)
    W0 = gs.Constant(name="W0", values=weights_data)
    X1 = gs.Variable(name="X1", dtype=np.float32, shape=input_shape)
    W1 = gs.Constant(name="W1", values=weights_data)
    X2 = gs.Variable(name="X2", dtype=np.float32, shape=input_shape)
    W2 = gs.Constant(name="W2", values=weights_data)
    X3 = gs.Variable(name="X3", dtype=np.float32, shape=input_shape)

    node_1 = gs.Node(name="Conv-1",
                     op="Conv",
                     inputs=[X0, W0],
                     outputs=[X1],
                     attrs={
                         "kernel_shape": [1, 1],
                         "strides": [1, 1],
                         "pads": [0, 0, 0, 0],
                         "group": num_groups
                     })
    # Use an custom operator IdentityConv Instead.
    # This operator is not defined by ONNX and cannot be parsed by ONNX parser without custom plugin.
    # node_2 = gs.Node(name="Conv-2", op="Conv",
    #                inputs=[X1, W1],
    #                outputs=[X2],
    #                attrs={
    #                    "kernel_shape": [1, 1],
    #                    "strides": [1, 1],
    #                    "pads": [0, 0, 0, 0],
    #                    "group": num_groups
    #                })
    node_2 = gs.Node(
        name="Conv-2",
        op="IdentityConv",
        inputs=[X1, W1],
        outputs=[X2],
        attrs={
            "kernel_shape": [1, 1],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "group": num_groups,
            # The plugin needs to be found by the `nvinfer1::IPluginRegistry::getPluginCreator` function.
            # https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_plugin_registry.html#a7069e891f4c02cfb206d300f236bc697
            # Otherwise, the build will fail because the plugin is not found.
            # This means the plugin must also be registered with the same plugin namespace and version.
            # The version is implemented by the plugin and cannot be changed during the runtime.
            # The namespace is configured when the plugin creator is registered during the runtime.
            "plugin_version": "1",
            "plugin_namespace": "",
        })
    node_3 = gs.Node(name="Conv-3",
                     op="Conv",
                     inputs=[X2, W2],
                     outputs=[X3],
                     attrs={
                         "kernel_shape": [1, 1],
                         "strides": [1, 1],
                         "pads": [0, 0, 0, 0],
                         "group": num_groups
                     })

    graph = gs.Graph(nodes=[node_1, node_2, node_3],
                     inputs=[X0],
                     outputs=[X3],
                     opset=opset_version)
    model = gs.export_onnx(graph)
    # Shape inference does not quite work here because of the custom operator.
    # model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_file_path)


if __name__ == '__main__':

    main()
