import os
import ctypes

import tensorrt as trt

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_plugin_lib(plugin_lib_file_path):

    if os.path.isfile(plugin_lib_file_path):
        try:
            # Python specifies that winmode is 0 by default, but some implementations
            # incorrectly default to None instead. See:
            # https://docs.python.org/3.8/library/ctypes.html
            # https://github.com/python/cpython/blob/3.10/Lib/ctypes/__init__.py#L343
            ctypes.CDLL(plugin_lib_file_path, winmode=0)
        except TypeError:
            # winmode only introduced in python 3.8
            ctypes.CDLL(plugin_lib_file_path)
        return

    raise IOError(f"Failed to load plugin library: {plugin_lib_file_path}")


def load_engine(engine_file_path):

    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())