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

IdentityConv::IdentityConv(IdentityConvParameters const& params)
    : mParams{params}
{
    initFieldsToSerialize();
}

void IdentityConv::initFieldsToSerialize()
{
    // Serialize IdentityConvParameters.
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField("parameters", &mParams, PluginFieldType::kUNKNOWN,
                              sizeof(IdentityConvParameters)));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability*
IdentityConv::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV3* IdentityConv::clone() noexcept
{
    // It's possible to encounter errors during cloning.
    // For example, if the memory to allocate is insufficient, exceptions can be
    // thrown.
    try
    {
        IPluginV3* const plugin{new IdentityConv{mParams}};
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* IdentityConv::getPluginName() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

char const* IdentityConv::getPluginVersion() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_VERSION;
}

char const* IdentityConv::getPluginNamespace() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAMESPACE;
}

int32_t IdentityConv::getNbOutputs() const noexcept { return 1; }

int32_t IdentityConv::configurePlugin(DynamicPluginTensorDesc const* in,
                                      int32_t nbInputs,
                                      DynamicPluginTensorDesc const* out,
                                      int32_t nbOutputs) noexcept
{
    // Communicates the number of inputs and outputs, dimensions, and datatypes
    // of all inputs and outputs, broadcast information for all inputs and
    // outputs, the chosen plugin format, and maximum batch size. At this point,
    // the plugin sets up its internal state and selects the most appropriate
    // algorithm and data structures for the given configuration. Note: Resource
    // allocation is not allowed in this API because it causes a resource leak.

    // This member function will only be called during engine build time.

    // Validate input arguments.
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(out[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(in[0].desc.dims.d[0] == out[0].desc.dims.d[0]);
    PLUGIN_ASSERT(in[0].desc.dims.d[1] == out[0].desc.dims.d[1]);
    PLUGIN_ASSERT(in[0].desc.dims.d[2] == out[0].desc.dims.d[2]);
    PLUGIN_ASSERT(in[0].desc.dims.d[3] == out[0].desc.dims.d[3]);
    PLUGIN_ASSERT(in[0].desc.type == out[0].desc.type);

    mParams.dtype = in[0].desc.type;
    mParams.channelSize = in[0].desc.dims.d[0];
    mParams.height = in[0].desc.dims.d[1];
    mParams.width = in[0].desc.dims.d[2];

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

    return 0;
}

bool IdentityConv::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    // For this method inputs are numbered 0..(nbInputs-1) and outputs are
    // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
    // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
    PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1 &&
                  pos < nbInputs + nbOutputs);
    bool isValidCombination = false;

    // Suppose we support only a limited number of format configurations.
    isValidCombination |=
        (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].desc.type == nvinfer1::DataType::kFLOAT);
    isValidCombination |=
        (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         inOut[pos].desc.type == nvinfer1::DataType::kHALF);
    // Make sure the input tensor and output tensor types and formats are same.
    isValidCombination &=
        (pos < nbInputs || (inOut[pos].desc.format == inOut[0].desc.format &&
                            inOut[pos].desc.type == inOut[0].desc.type));

    return isValidCombination;
}

int32_t IdentityConv::getOutputDataTypes(DataType* outputTypes,
                                         int32_t nbOutputs,
                                         DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept
{
    // One output.
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    // The output type is the same as the input type.
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t IdentityConv::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                                      DimsExprs const* shapeInputs,
                                      int32_t nbShapeInputs, DimsExprs* outputs,
                                      int32_t nbOutputs,
                                      IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(inputs[0].nbDims == 4);

    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t i{0}; i < inputs[0].nbDims; ++i)
    {
        outputs[0].d[i] = inputs[0].d[i];
    }

    return 0;
}

int32_t IdentityConv::enqueue(PluginTensorDesc const* inputDesc,
                              PluginTensorDesc const* outputDesc,
                              void const* const* inputs, void* const* outputs,
                              void* workspace, cudaStream_t stream) noexcept
{
    size_t const inputSize{
        static_cast<size_t>(inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] *
                            inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3])};
    size_t const inputSizeBytes{inputSize * mParams.dtypeBytes};
    cudaError_t const status{cudaMemcpyAsync(outputs[0], inputs[0],
                                             inputSizeBytes,
                                             cudaMemcpyDeviceToDevice, stream)};
    return status;
}

int32_t IdentityConv::onShapeChange(PluginTensorDesc const* in,
                                    int32_t nbInputs,
                                    PluginTensorDesc const* out,
                                    int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3*
IdentityConv::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* IdentityConv::getFieldsToSerialize() noexcept
{
    return &mFCToSerialize;
}

size_t IdentityConv::getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
                                      int32_t nbInputs,
                                      DynamicPluginTensorDesc const* outputs,
                                      int32_t nbOutputs) const noexcept
{
    return 0;
}

} // namespace plugin
} // namespace nvinfer1