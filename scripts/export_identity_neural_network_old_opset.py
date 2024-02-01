# Define a PyTorch identity neural network and export it to ONNX format with custom ONNX operators.

import os

import torch
import torch.nn as nn
from torch.onnx.symbolic_helper import _get_tensor_sizes


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
    def symbolic(g, input, weight, kernel_shape, strides, pads, group):
        args = [input, weight]
        # These become the operator attributes.
        kwargs = {
            "kernel_shape_i": kernel_shape,
            "strides_i": strides,
            "pads_i": pads,
            "group_i": group
        }
        output_type = input.type().with_sizes(_get_tensor_sizes(input))
        return g.op("CustomTorchOps::IdentityConv", *args,
                    **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, weight, kernel_shape, strides, pads, group):
        # We don't have to actually implement the correct forward pass,
        # if the downstream graph is not data dependent,
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
        x = DummyIdentityConvOp.apply(x, self.weight, self.kernel_shape,
                                      self.strides, self.pads, self.group)
        return x


class IdentityConv(IdentityConvBase):

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
        # Create a dummy identity convolution only used for ONNX export.
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

    opset_version = 13
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

    print(
        f"Exported the identity neural network to ONNX format: {onnx_file_path}"
    )
