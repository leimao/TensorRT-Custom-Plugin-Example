
#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H

#include <vector>

#include <NvInferRuntime.h>

#include "IdentityConvPlugin.h"

namespace nvinfer1
{
namespace plugin
{

class BaseCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    char const* getPluginNamespace() const noexcept override
    {
        return kIDENTITY_CONV_PLUGIN_NAMESPACE;
    }

    char const* getPluginName() const noexcept override
    {
        return kIDENTITY_CONV_PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kIDENTITY_CONV_PLUGIN_VERSION;
    }
};

// Plugin factory class.
class IdentityConvCreator : public BaseCreator
{
public:
    IdentityConvCreator();

    ~IdentityConvCreator() override = default;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc,
                            TensorRTPhase phase) noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
