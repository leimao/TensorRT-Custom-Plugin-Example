
#ifndef TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
#define TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H

#include <vector>

#include <NvInferRuntime.h>

namespace nvinfer1
{
namespace plugin
{

class BaseCreator : public nvinfer1::IPluginCreator
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

// Plugin factory class.
class IdentityConvCreator : public BaseCreator
{
public:
    IdentityConvCreator();

    ~IdentityConvCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2IOExt*
    createPlugin(char const* name,
                 nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2IOExt*
    deserializePlugin(char const* name, void const* serialData,
                      size_t serialLength) noexcept override;

private:
    // static nvinfer1::PluginFieldCollection mFC;
    // static std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" nvinfer1::IPluginCreator* const*
getPluginCreators(int32_t& nbCreators);

#endif // TENSORRT_IDENTITY_CONV_PLUGIN_CREATOR_H
