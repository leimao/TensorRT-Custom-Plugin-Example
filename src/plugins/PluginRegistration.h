#ifndef TENSORRT_PLUGIN_REGISTRATION_H
#define TENSORRT_PLUGIN_REGISTRATION_H

#include <NvInferRuntime.h>

// namespace nvinfer1
// {
// namespace plugin
// {

// These are the functions that TensorRT library will call at the runtime.

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" nvinfer1::IPluginCreator* const*
getPluginCreators(int32_t& nbCreators);

// } // namespace plugin
// } // namespace nvinfer1

#endif // TENSORRT_PLUGIN_REGISTRATION_H
