// Create a TensorRT engine building program that builds an engine from an ONNX
// file and uses a custom plugin.

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <dlfcn.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "plugins/IdentityConvPluginCreator.h"
#include "plugins/PluginRegistration.h"

class Logger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kVERBOSE)
        {
            std::cout << msg << std::endl;
        }
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

int main(int argc, char** argv)
{
    Logger gLogger;

    std::string const data_dir_path{"data"};
    std::string const onnx_file_name{"identity_neural_network.onnx"};
    std::string const engine_file_name{"engine.trt"};
    std::string const onnx_file_path{data_dir_path + "/" + onnx_file_name};
    std::string const engine_file_path{data_dir_path + "/" + engine_file_name};
    std::string const plugin_library_name{"libidentity_conv.so"};
    std::string const plugin_library_dir_path{"build/src"};
    std::string const plugin_library_path{plugin_library_dir_path + "/" +
                                          plugin_library_name};
    char const* const plugin_library_path_c_str{plugin_library_path.c_str()};

    // This plugin creator initialization step is compulsory for plugin dynamic
    // registration. Otherwise loading the plugin library will complain the
    // plugin creator is an undefined symbol.
    std::unique_ptr<nvinfer1::plugin::IdentityConvCreator> pluginCreator{
        new nvinfer1::plugin::IdentityConvCreator{}};
    pluginCreator->setPluginNamespace("");

    // Create the builder.
    std::unique_ptr<nvinfer1::IBuilder, InferDeleter> builder{
        nvinfer1::createInferBuilder(gLogger)};
    if (builder == nullptr)
    {
        std::cerr << "Failed to create the builder." << std::endl;
        return EXIT_FAILURE;
    }
    void* const plugin_handle{
        builder->getPluginRegistry().loadLibrary(plugin_library_path.c_str())};
    if (plugin_handle == nullptr)
    {
        std::cerr << "Failed to load the plugin library." << std::endl;
        return EXIT_FAILURE;
    }

    // Create the network.
    uint32_t const flag{
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> network{
        builder->createNetworkV2(flag)};
    if (network == nullptr)
    {
        std::cerr << "Failed to create the network." << std::endl;
        return EXIT_FAILURE;
    }

    // Create the parser.
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser{
        nvonnxparser::createParser(*network, gLogger)};
    if (parser == nullptr)
    {
        std::cerr << "Failed to create the parser." << std::endl;
        return EXIT_FAILURE;
    }
    parser->parseFromFile(
        onnx_file_path.c_str(),
        static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // Set the allowed IO tensor formats.
    uint32_t const formats{
        1U << static_cast<uint32_t>(nvinfer1::TensorFormat::kLINEAR)};
    nvinfer1::DataType const dtype{nvinfer1::DataType::kFLOAT};
    network->getInput(0)->setAllowedFormats(formats);
    network->getInput(0)->setType(dtype);
    network->getOutput(0)->setAllowedFormats(formats);
    network->getOutput(0)->setType(dtype);

    // Build the engine.
    std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter> config{
        builder->createBuilderConfig()};
    if (config == nullptr)
    {
        std::cerr << "Failed to create the builder config." << std::endl;
        return EXIT_FAILURE;
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setPluginsToSerialize(&plugin_library_path_c_str, 1);

    std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> serializedModel{
        builder->buildSerializedNetwork(*network, *config)};

    // Write the serialized engine to a file.
    std::ofstream engineFile{engine_file_path.c_str(), std::ios::binary};
    if (!engineFile.is_open())
    {
        std::cerr << "Failed to open the engine file." << std::endl;
        return EXIT_FAILURE;
    }
    engineFile.write(static_cast<char const*>(serializedModel->data()),
                     serializedModel->size());
    engineFile.close();

    return EXIT_SUCCESS;
}