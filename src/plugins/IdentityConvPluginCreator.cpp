#include <exception>
#include <iostream>
#include <mutex>

#include <NvInferRuntimePlugin.h>

#include "IdentityConvPlugin.h"
#include "IdentityConvPluginCreator.h"
#include "PluginUtils.h"

namespace nvinfer1
{
namespace plugin
{

// REGISTER_TENSORRT_PLUGIN(IdentityConvCreator);

// Plugin creator
IdentityConvCreator::IdentityConvCreator() {}

char const* IdentityConvCreator::getPluginName() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_NAME;
}

char const* IdentityConvCreator::getPluginVersion() const noexcept
{
    return kIDENTITY_CONV_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const*
IdentityConvCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* IdentityConvCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    // The attributes from the ONNX node will be parsed and passed via fc.
    // In our dummy case,
    // attrs={
    //     "kernel_shape": [1, 1],
    //     "strides": [1, 1],
    //     "pads": [0, 0, 0, 0],
    //     "group": num_groups
    // }

    try
    {
        nvinfer1::PluginField const* fields{fc->fields};
        int32_t nbFields{fc->nbFields};

        PLUGIN_VALIDATE(nbFields == 4);

        std::vector<int32_t> kernelShape{};
        std::vector<int32_t> strides{};
        std::vector<int32_t> pads{};
        int32_t group{};

        for (int32_t i{0}; i < nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "kernel_shape"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const kernelShapeData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    kernelShape.push_back(kernelShapeData[j]);
                }
            }
            if (!strcmp(attrName, "strides"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const stridesData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    strides.push_back(stridesData[j]);
                }
            }
            if (!strcmp(attrName, "pads"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                int32_t const* const padsData{
                    static_cast<int32_t const*>(fields[i].data)};
                for (int32_t j{0}; j < fields[i].length; ++j)
                {
                    pads.push_back(padsData[j]);
                }
            }
            if (!strcmp(attrName, "group"))
            {
                PLUGIN_VALIDATE(fields[i].type ==
                                nvinfer1::PluginFieldType::kINT32);
                PLUGIN_VALIDATE(fields[i].length == 1);
                group = *(static_cast<int32_t const*>(fields[i].data));
            }
        }

        IdentityConvParameters const params{.group = group};

        IdentityConv* const plugin{new IdentityConv{params}};
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2IOExt*
IdentityConvCreator::deserializePlugin(char const* name, void const* serialData,
                                       size_t serialLength) noexcept
{
    try
    {
        IdentityConv* plugin = new IdentityConv{serialData, serialLength};
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

} // namespace plugin
} // namespace nvinfer1

class ThreadSafeLoggerFinder
{
public:
    ThreadSafeLoggerFinder() = default;

    // Set the logger finder.
    void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder == nullptr && finder != nullptr)
        {
            mLoggerFinder = finder;
        }
    }

    // Get the logger.
    nvinfer1::ILogger* getLogger() noexcept
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder != nullptr)
        {
            return mLoggerFinder->findLogger();
        }
        return nullptr;
    }

private:
    nvinfer1::ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;
};

ThreadSafeLoggerFinder gLoggerFinder;

// Not exposing this function to the user to get the plugin logger for the
// moment. Can switch the plugin logger to this in the future.

// ILogger* getPluginLogger()
// {
//     return gLoggerFinder.getLogger();
// }

// extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
// {
//     gLoggerFinder.setLoggerFinder(finder);
// }

// extern "C" nvinfer1::IPluginCreator* const*
// getPluginCreators(int32_t& nbCreators)
// {
//     nbCreators = 1;
//     static IdentityConvCreator identityConvCreator{};
//     static nvinfer1::IPluginCreator* const pluginCreatorList[] = {
//         &identityConvCreator};
//     return pluginCreatorList;
// }

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    gLoggerFinder.setLoggerFinder(finder);
}

extern "C" nvinfer1::IPluginCreator* const*
getPluginCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static nvinfer1::plugin::IdentityConvCreator identityConvCreator{};
    static nvinfer1::IPluginCreator* const pluginCreatorList[] = {
        &identityConvCreator};
    return pluginCreatorList;
}
