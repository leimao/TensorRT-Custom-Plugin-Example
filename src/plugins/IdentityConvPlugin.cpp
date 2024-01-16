#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include "IdentityConvPlugin.h"
#include "PluginUtils.h"

namespace nvinfer1
{
namespace plugin
{

// Write values into buffer
template <typename Type, typename BufferType>
void write(BufferType*& buffer, Type const& val)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    std::memcpy(buffer, &val, sizeof(Type));
    buffer += sizeof(Type);
}

// Read values from buffer
template <typename OutType, typename BufferType>
OutType read(BufferType const*& buffer)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    OutType val{};
    std::memcpy(&val, static_cast<void const*>(buffer), sizeof(OutType));
    buffer += sizeof(OutType);
    return val;
}

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
    mParams.dtype = read<nvinfer1::DataType>(d);
    mParams.channelSize = read<int32_t>(d);
    mParams.height = read<int32_t>(d);
    mParams.width = read<int32_t>(d);
    mParams.dtypeBytes = read<size_t>(d);
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

    // This member function will only be called during engine build time.

    // Validate input arguments.
    PLUGIN_ASSERT(nbInput == 2);
    PLUGIN_ASSERT(nbOutput == 1);
    PLUGIN_ASSERT(in[0].dims.nbDims == 3);
    PLUGIN_ASSERT(out[0].dims.nbDims == 3);
    PLUGIN_ASSERT(in[0].dims.d[0] == out[0].dims.d[0]);
    PLUGIN_ASSERT(in[0].dims.d[1] == out[0].dims.d[1]);
    PLUGIN_ASSERT(in[0].dims.d[2] == out[0].dims.d[2]);
    PLUGIN_ASSERT(in[0].type == out[0].type);

    mParams.dtype = in[0].type;
    mParams.channelSize = in[0].dims.d[0];
    mParams.height = in[0].dims.d[1];
    mParams.width = in[0].dims.d[2];

    if (mParams.dtype == nvinfer1::DataType::kINT8)
    {
        mParams.dtypeBytes = 1;
    }
    else if (mParams.dtype == nvinfer1::DataType::kHALF)
    {
        mParams.dtypeBytes = 2;
    }
    else if (mParams.dtype == nvinfer1::DataType::kFLOAT)
    {
        mParams.dtypeBytes = 4;
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
    // PLUGIN_ASSERT(nbInputDims == 1);
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
    // return sizeof(IdentityConvParameters);
    return sizeof(int32_t) * 4 + sizeof(nvinfer1::DataType) + sizeof(size_t);
}

void IdentityConv::serialize(void* buffer) const noexcept
{
    char* d{reinterpret_cast<char*>(buffer)};
    char* const a{d};
    // Be cautious, the order has to match deserialization.
    write(d, mParams.group);
    write(d, mParams.dtype);
    write(d, mParams.channelSize);
    write(d, mParams.height);
    write(d, mParams.width);
    write(d, mParams.dtypeBytes);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

bool IdentityConv::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) const noexcept
{
    // For this method inputs are numbered 0..(nbInputs-1) and outputs are
    // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
    // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
    PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1 &&
                  pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // Suppose we support only a limited number of format configurations.
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].type == nvinfer1::DataType::kFLOAT);
    isValidCombination |=
        (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].type == nvinfer1::DataType::kHALF);
    // Make sure the input tensor and output tensor types and formats are same.
    isValidCombination &=
        (pos < nbInputs || (inOut[pos].format == inOut[0].format &&
                            inOut[pos].type == inOut[0].type));

    return isValidCombination;
}

char const* IdentityConv::getPluginType() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

char const* IdentityConv::getPluginVersion() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_VERSION;
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
    PLUGIN_ASSERT(nbInputs == 2);
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
    size_t const inputSize{static_cast<size_t>(batchSize * mParams.channelSize *
                                               mParams.height * mParams.width)};
    size_t const inputSizeBytes{inputSize * mParams.dtypeBytes};
    cudaError_t const status{cudaMemcpyAsync(outputs[0], inputs[0],
                                             inputSizeBytes,
                                             cudaMemcpyDeviceToDevice, stream)};
    return status;
}

} // namespace plugin
} // namespace nvinfer1