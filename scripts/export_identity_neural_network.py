# Define a PyTorch identity neural network and export it to ONNX format with custom ONNX operators.
# This only works for ONNX Opset 15 and above.

import os
from typing import List, Tuple, ClassVar
import torch
import torch.nn as nn


class IdentityConvBase(nn.Module):

    def __init__(self, channels):
        super(IdentityConvBase, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=(1, 1),
                              groups=channels,
                              bias=False)
        # Fill the weight values with 1.0.
        self.conv.weight.data = torch.ones(channels, 1, 1, 1)
        # Turn off the gradient for the weights, making it a constant.
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


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
        self.conv2 = IdentityConv(channels)
        self.conv3 = IdentityConvBase(channels)

    def forward(self, x):
        x = self.conv1(x)
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
    # Export the model to ONNX format with the custom operator as ONNX custom functions.
    # TensorRT ONNX parser can parse the ONNX custom functions to TensorRT plugins.
    # References: https://github.com/pytorch/pytorch/issues/65199
    torch.onnx.export(model=identity_neural_network,
                      args=(input_data, ),
                      f=onnx_file_path,
                      input_names=["X0"],
                      output_names=["X3"],
                      opset_version=opset_version,
                      export_modules_as_functions={IdentityConv})
    print(
        f"Exported the identity neural network to ONNX format: {onnx_file_path}"
    )
