# Define a PyTorch identity neural network and export it to ONNX format with custom ONNX operators.

# Example: https://github.com/microsoft/onnxruntime-extensions/blob/d47a3dd2ddf5d11139671f81a399ff43605f7e43/test/test_torch_ops.py
# Example: https://github.com/onnx/onnx/issues/4569

import os
from typing import List, ClassVar

import torch
import torch.nn as nn
from torch.onnx.symbolic_helper import _get_tensor_sizes

import onnx
import onnx_graphsurgeon as gs


class IdentityConvBase(nn.Module):

    def __init__(self, channels):
        super(IdentityConvBase, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              padding=(0, 0),
                              dilation=(1, 1),
                              groups=channels,
                              bias=False)
        # Fill the weight values with 1.0.
        self.conv.weight.data = torch.ones(channels, 1, 1, 1)
        # Turn off the gradient for the weights, making it a constant.
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class DummyIdentityConvOp(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, weight):
        output_type = input.type().with_sizes(_get_tensor_sizes(input))
        return g.op("CustomTorchOps::IdentityConv", input,
                    weight).setType(output_type)

    @staticmethod
    def forward(ctx, input, weight):
        # We don't have to actually implement the correct forward pass,
        # as long as the shape of the output is correct.
        return input


class DummyIdentityConv(nn.Module):

    def __init__(self, channels):
        super(DummyIdentityConv, self).__init__()
        self.kernel_shape = (1, 1)
        self.strides = (1, 1)
        self.pads = (0, 0, 0, 0)
        self.group = channels

        # Fill the weight values with 1.0.
        self.weight = torch.ones(channels, 1, 1, 1)
        # Turn off the gradient for the weights, making it a constant.
        self.weight.requires_grad = False

    def forward(self, x):
        x = DummyIdentityConvOp.apply(x, self.weight)
        return x


class IdentityConv(IdentityConvBase):

    __constants__ = ["kernel_shape", "strides", "pads", "group"]
    # Attributes to match the plugin requirements.
    # Must follow the type annotations via PEP 526-style.
    # https://peps.python.org/pep-0526/#class-and-instance-variable-annotations
    kernel_shape: ClassVar[List[int]]
    strides: ClassVar[List[int]]
    pads: ClassVar[List[int]]
    group: int

    def __init__(self, channels):
        super(IdentityConv, self).__init__(channels)
        self.kernel_shape = list(self.conv.kernel_size)
        self.strides = list(self.conv.stride)
        self.pads = list(self.conv.padding)
        # ONNX expects a list of 4 pad values whereas PyTorch uses a list of 2 pad values.
        self.pads = self.pads + self.pads
        self.group = self.conv.groups

    def forward(self, x):
        # Call the parent class method.
        x = super(IdentityConv, self).forward(x)
        # Apply the identity operation.
        return x


class IdentityNeuralNetwork(nn.Module):

    def __init__(self, channels):
        super(IdentityNeuralNetwork, self).__init__()
        self.conv1 = IdentityConvBase(channels)
        self.conv2 = IdentityConv(channels=channels)
        self.conv3 = IdentityConvBase(channels)

        self.conv2_export = DummyIdentityConv(channels=channels)

    def forward(self, x):
        x = self.conv1(x)
        if torch.onnx.is_in_onnx_export():
            x = self.conv2_export(x)
        else:
            x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == "__main__":

    opset_version = 15
    data_directory_path = "data"
    onnx_file_name = "identity_neural_network.onnx"
    onnx_file_path = os.path.join(data_directory_path, onnx_file_name)

    input_shape = (1, 3, 480, 960)
    input_data = torch.rand(*input_shape).float()
    input_channels = input_shape[1]

    # Create an instance of the identity neural network.
    identity_neural_network = IdentityNeuralNetwork(input_channels)
    # Set the model to evaluation mode.
    identity_neural_network.eval()
    output = identity_neural_network(input_data)
    # Export the model to ONNX format with the custom operator as ONNX custom functions.
    torch.onnx.export(model=identity_neural_network,
                      args=(input_data, ),
                      f=onnx_file_path,
                      input_names=["X0"],
                      output_names=["X3"],
                      opset_version=opset_version)

    # Unfortunately, the attributes of the custom operator are not exported to the custom ONNX operator.
    # We will use ONNX GraphSurgeon to add the attributes to the custom ONNX operator.

    onnx_model = onnx.load(onnx_file_path)
    graph = gs.import_onnx(onnx_model)
    # Iterate the ONNX graph and get the custom ONNX operator node by name.
    for node in graph.nodes:
        if node.op == "IdentityConv":
            # Add the attributes to the custom ONNX operator node.
            node.attrs["kernel_shape"] = [1, 1]
            node.attrs["strides"] = [1, 1]
            node.attrs["pads"] = [0, 0, 0, 0]
            node.attrs["group"] = input_channels
    # Export the ONNX graph to ONNX model.
    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, onnx_file_path)

    print(
        f"Exported the identity neural network to ONNX format: {onnx_file_path}"
    )
