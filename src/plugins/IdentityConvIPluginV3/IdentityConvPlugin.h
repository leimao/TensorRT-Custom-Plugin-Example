#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_H

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const* const kIDENTITY_CONV_PLUGIN_NAME{"IdentityConv"};
constexpr char const* const kIDENTITY_CONV_PLUGIN_VERSION{"1"};
constexpr char const* const kIDENTITY_CONV_PLUGIN_NAMESPACE{""};

namespace nvinfer1
{
namespace plugin
{

struct IdentityConvParameters
{
    int32_t group;
    nvinfer1::DataType dtype;
    int32_t channelSize;
    int32_t height;
    int32_t width;
    size_t dtypeBytes;
};

class IdentityConv : public IPluginV3,
                     public IPluginV3OneCore,
                     public IPluginV3OneBuild,
                     public IPluginV3OneRuntime
{
public:
    IdentityConv(IdentityConvParameters const& params);

    ~IdentityConv() override = default;

    // IPluginV3 Methods

    IPluginCapability*
    getCapabilityInterface(PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;

    // IPluginV3OneCore Methods

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild Methods

    int32_t getNbOutputs() const noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            DynamicPluginTensorDesc const* out,
                            int32_t nbOutputs) noexcept override;

    bool supportsFormatCombination(int32_t pos,
                                   DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes,
                               int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                            DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override;

    // IPluginV3OneRuntime Methods

    int32_t enqueue(PluginTensorDesc const* inputDesc,
                    PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
                          PluginTensorDesc const* out,
                          int32_t nbOutputs) noexcept override;

    IPluginV3*
    attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
                            int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;

private:
    // TensorRT plugin parameters.
    IdentityConvParameters mParams;

    void initFieldsToSerialize();

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_H
