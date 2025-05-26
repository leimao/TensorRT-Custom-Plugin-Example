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

// Plugin static registration to namespace "".
// Static registration does not allow registering to a different namespace.
// If there is such a need, perform dynamic registration instead.
REGISTER_TENSORRT_PLUGIN(IdentityConvCreator);

// Plugin creator
IdentityConvCreator::IdentityConvCreator()
{
    // Declare the ONNX attributes that the ONNX parser will collect from the
    // ONNX model that contains the IdentityConv node.

    // In our dummy case,
    // attrs={
    //     "kernel_shape": [1, 1],
    //     "strides": [1, 1],
    //     "pads": [0, 0, 0, 0],
    //     "group": num_groups
    // }

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "kernel_shape", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("strides", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("pads", nullptr, PluginFieldType::kINT32, 4));
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("group", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

nvinfer1::PluginFieldCollection const*
IdentityConvCreator::getFieldNames() noexcept
{
    // This is only used in the build phase.
    return &mFC;
}

IPluginV3* IdentityConvCreator::createPlugin(char const* name,
                                             PluginFieldCollection const* fc,
                                             TensorRTPhase phase) noexcept
{
    // The build phase and the deserialization phase are handled differently.
    if (phase == TensorRTPhase::kBUILD)
    {
        // The attributes from the ONNX node will be parsed and passed via fc.
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

            // Log the attributes parsed from ONNX node.
            std::stringstream ss;
            ss << "Plugin Attributes:";
            logInfo(ss.str().c_str());

            ss.str("");
            ss << "kernel_shape: ";
            for (auto const& val : kernelShape)
            {
                ss << val << " ";
            }
            logInfo(ss.str().c_str());

            ss.str("");
            ss << "strides: ";
            for (auto const& val : strides)
            {
                ss << val << " ";
            }
            logInfo(ss.str().c_str());

            ss.str("");
            ss << "pads: ";
            for (auto const& val : pads)
            {
                ss << val << " ";
            }
            logInfo(ss.str().c_str());

            ss.str("");
            ss << "group: " << group;
            logInfo(ss.str().c_str());

            IdentityConvParameters const params{.group = group};

            IdentityConv* const plugin{new IdentityConv{params}};
            return plugin;
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }
    else if (phase == TensorRTPhase::kRUNTIME)
    {
        // The attributes from the serialized plugin will be passed via fc.
        try
        {
            nvinfer1::PluginField const* fields{fc->fields};
            int32_t nbFields{fc->nbFields};
            PLUGIN_VALIDATE(nbFields == 1);

            char const* attrName = fields[0].name;
            PLUGIN_VALIDATE(!strcmp(attrName, "parameters"));
            PLUGIN_VALIDATE(fields[0].type ==
                            nvinfer1::PluginFieldType::kUNKNOWN);
            PLUGIN_VALIDATE(fields[0].length == sizeof(IdentityConvParameters));
            IdentityConvParameters params{
                *(static_cast<IdentityConvParameters const*>(fields[0].data))};

            IdentityConv* const plugin{new IdentityConv{params}};
            return plugin;
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }
    else
    {
        return nullptr;
    }
}

} // namespace plugin
} // namespace nvinfer1
