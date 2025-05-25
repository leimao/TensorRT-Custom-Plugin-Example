// Create a TensorRT engine building program that builds an engine from an ONNX
// file and uses a custom plugin.

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <dlfcn.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

class CustomLogger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kINFO)
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
    if (argc != 4)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <onnx_file_path> <plugin_library_path> <engine_file_path>"
            << std::endl;
        return EXIT_FAILURE;
    }

    std::string const onnx_file_path{argv[1]};
    std::string const plugin_library_path{argv[2]};
    std::string const engine_file_path{argv[3]};

    std::cout << "ONNX file path: " << onnx_file_path << std::endl;
    std::cout << "Plugin library path: " << plugin_library_path << std::endl;
    std::cout << "Engine file path: " << engine_file_path << std::endl;

    CustomLogger logger{};

    // Create the builder.
    std::unique_ptr<nvinfer1::IBuilder, InferDeleter> builder{
        nvinfer1::createInferBuilder(logger)};
    if (builder == nullptr)
    {
        std::cerr << "Failed to create the builder." << std::endl;
        return EXIT_FAILURE;
    }
    // dlopen the plugin library.
    // The plugin will be registered automatically when the library is loaded.
    void* const plugin_handle{dlopen(plugin_library_path.c_str(), RTLD_NOW)};
    if (plugin_handle == nullptr)
    {
        std::cerr << "Failed to load the plugin library: " << dlerror()
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Create the network.
    uint32_t flag{0U};
    // For TensorRT < 10.0, explicit dimension has to be specified to
    // distinguish from the implicit dimension. For TensorRT >= 10.0, explicit
    // dimension is the only choice and this flag has been deprecated.
    if (getInferLibVersion() < 100000)
    {
        flag |= 1U << static_cast<uint32_t>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    }
    std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> network{
        builder->createNetworkV2(flag)};
    if (network == nullptr)
    {
        std::cerr << "Failed to create the network." << std::endl;
        return EXIT_FAILURE;
    }

    // Create the parser.
    std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser{
        nvonnxparser::createParser(*network, logger)};
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

    std::cout << "Successfully serialized the engine to the file: "
              << engine_file_path << std::endl;

    return EXIT_SUCCESS;
}