#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include "IdentityConvPlugin.h"
#include "PluginUtils.h"

IdentityConv::IdentityConv(IdentityConvParameters params) : mParams{params} {}

IdentityConv::IdentityConv(void const* data, size_t length)
{
    deserialize(static_cast<uint8_t const*>(data), length);
}

void IdentityConv::deserialize(uint8_t const* data, size_t length)
{
    // In our simple use case, even though there is no parameter used for this
    // plugin, we deserialize and serialize some attributes for demonstration
    // purposes.
    uint8_t const* d{data};
    mParams.group = read<int32_t>(d);
    PLUGIN_ASSERT(d == data + length);
}

int32_t IdentityConv::getNbOutputs() const noexcept { return 1; }

void IdentityConv::configurePlugin(nvinfer1::PluginTensorDesc const* in,
                                   int32_t nbInput,
                                   nvinfer1::PluginTensorDesc const* out,
                                   int32_t nbOutput) noexcept
{
    // Communicates the number of inputs and outputs, dimensions, and datatypes
    // of all inputs and outputs, broadcast information for all inputs and
    // outputs, the chosen plugin format, and maximum batch size. At this point,
    // the plugin sets up its internal state and selects the most appropriate
    // algorithm and data structures for the given configuration. Note: Resource
    // allocation is not allowed in this API because it causes a resource leak.

    // Validate input arguments.
    PLUGIN_ASSERT(nbInput == 1);
    PLUGIN_ASSERT(nbOutput == 1);
    PLUGIN_ASSERT(in[0].dims.nbDims == 3);
    PLUGIN_ASSERT(out[0].dims.nbDims == 3);
    PLUGIN_ASSERT(in[0].dims.d[0] == out[0].dims.d[0]);
    PLUGIN_ASSERT(in[0].dims.d[1] == out[0].dims.d[1]);
    PLUGIN_ASSERT(in[0].dims.d[2] == out[0].dims.d[2]);
    PLUGIN_ASSERT(in[0].type == out[0].type);

    mDtype = in[0].type;
    mChannelSize = in[0].dims.d[0];
    mHeight = in[0].dims.d[1];
    mWidth = in[0].dims.d[2];

    if (mDtype == nvinfer1::DataType::kINT8)
    {
        mDtypeBytes = 1;
    }
    else if (mDtype == nvinfer1::DataType::kHALF)
    {
        mDtypeBytes = 2;
    }
    else if (mDtype == nvinfer1::DataType::kFLOAT)
    {
        mDtypeBytes = 4;
    }
    else
    {
        PLUGIN_ASSERT(false);
    }
}

int32_t IdentityConv::initialize() noexcept
{
    // The configuration is known at this time, and the inference engine is
    // being created, so the plugin can set up its internal data structures and
    // prepare for execution. Such setup might include initializing libraries,
    // allocating memory, etc. In our case, we don't need to prepare anything.
    return 0;
}

void IdentityConv::terminate() noexcept
{
    // The engine context is destroyed, and all the resources held by the plugin
    // must be released.
}

nvinfer1::Dims IdentityConv::getOutputDimensions(int32_t index,
                                                 nvinfer1::Dims const* inputs,
                                                 int32_t nbInputDims) noexcept
{
    // Even though non-IPluginV2DynamicExt plugins are compatible with explicit
    // batch mode networks, their implementation must be independent of the type
    // of network (implicit/explicit batch mode) in which it is expected to be
    // used. As such, when using such plugins in explicit batch mode networks:
    // * The leading dimension of the first input (before being passed to the
    // plugin) is inferred to be the batch dimension.
    // * TensorRT pops this first dimension identified above before inputs are
    // passed to the plugin, and pushes it to the front of any outputs emitted
    // by the plugin. This means that the batch dimension must not be specified
    // in getOutputDimensions.
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputDims == 1);
    PLUGIN_ASSERT(inputs != nullptr);
    // CHW
    nvinfer1::Dims dimsOutput;
    // Don't trigger null dereference since we check if inputs is nullptr above.
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    PLUGIN_ASSERT(inputs[0].nbDims == 3);
    // Identity operation.
    // Just copy the dimensions from the input tensor.
    dimsOutput.nbDims = inputs[0].nbDims;
    dimsOutput.d[0] = inputs[0].d[0];
    dimsOutput.d[1] = inputs[0].d[1];
    dimsOutput.d[2] = inputs[0].d[2];

    return dimsOutput;
}

size_t IdentityConv::getWorkspaceSize(int32_t maxBatchSize) const noexcept
{
    // No scratch space is required for this plugin.
    return 0;
}

size_t IdentityConv::getSerializationSize() const noexcept
{
    return sizeof(IdentityConvParameters);
}

void IdentityConv::serialize(void* buffer) const noexcept
{
    char* d{reinterpret_cast<char*>(buffer)};
    char* const a{d};
    // Be cautious, the order has to match deserialization.
    write(d, mParams.group);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool IdentityConv::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) const noexcept
{
    PLUGIN_ASSERT(nbInputs == 1 && nbOutputs == 1 &&
                  pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // Suppose we support only a limited number of format configurations.
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].type == nvinfer1::DataType::kFLOAT);
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].type == nvinfer1::DataType::kHALF);
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kCHW32 &&
         inOut[pos].type == nvinfer1::DataType::kINT8);

    return isValidCombination;
}

char const* IdentityConv::getPluginType() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

char const* IdentityConv::getPluginVersion() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

void IdentityConv::destroy() noexcept { delete this; }

nvinfer1::IPluginV2IOExt* IdentityConv::clone() const noexcept
{
    // It's possible to encounter errors during cloning.
    // For example, if the memory to allocate is insufficient, exceptions can be
    // thrown.
    try
    {
        IPluginV2IOExt* const plugin{new IdentityConv{mParams}};
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void IdentityConv::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

char const* IdentityConv::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

nvinfer1::DataType
IdentityConv::getOutputDataType(int32_t index,
                                nvinfer1::DataType const* inputTypes,
                                int32_t nbInputs) const noexcept
{
    // One output.
    PLUGIN_ASSERT(index == 0);
    PLUGIN_ASSERT(nbInputs == 1);
    // The output type is the same as the input type.
    return inputTypes[0];
}

bool IdentityConv::isOutputBroadcastAcrossBatch(int32_t outputIndex,
                                                bool const* inputIsBroadcasted,
                                                int32_t nbInputs) const noexcept
{
    return false;
}

bool IdentityConv::canBroadcastInputAcrossBatch(
    int32_t inputIndex) const noexcept
{
    return false;
}

int32_t IdentityConv::enqueue(int32_t batchSize, void const* const* inputs,
                              void* const* outputs, void* workspace,
                              cudaStream_t stream) noexcept
{
    size_t const inputSize{
        static_cast<size_t>(batchSize * mChannelSize * mHeight * mWidth)};
    size_t const inputSizeBytes{inputSize * mDtypeBytes};
    cudaError_t const status{cudaMemcpyAsync(outputs[0], inputs[0],
                                             inputSizeBytes,
                                             cudaMemcpyDeviceToDevice, stream)};
    return status;
}