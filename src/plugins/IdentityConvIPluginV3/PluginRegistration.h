#ifndef TENSORRT_PLUGIN_REGISTRATION_H
#define TENSORRT_PLUGIN_REGISTRATION_H

#include <NvInferRuntime.h>

// These are the functions that TensorRT library will call at the runtime.

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" nvinfer1::IPluginCreatorInterface* const*
getPluginCreators(int32_t& nbCreators);

#endif // TENSORRT_PLUGIN_REGISTRATION_H
