#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include <dlfcn.h>

#include <NvInfer.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

class CustomLogger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override
    {
        // suppress info-level messages
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

void create_random_data(float* data, size_t const size, unsigned int seed = 1U)
{
    std::default_random_engine eng(seed);
    std::uniform_int_distribution<int32_t> dis(-16, 16);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t i{0U}; i < size; ++i)
    {
        data[i] = static_cast<float>(rand());
    }
}

bool all_close(float const* a, float const* b, size_t size, float rtol = 1e-5f,
               float atol = 1e-8f)
{
    for (size_t i{0U}; i < size; ++i)
    {
        float const diff{std::abs(a[i] - b[i])};
        if (diff > (atol + rtol * std::abs(b[i])))
        {
            std::cout << "a[" << i << "]: " << a[i] << std::endl;
            std::cout << "b[" << i << "]: " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <plugin_library_path> <engine_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string const plugin_library_path{argv[1]};
    std::string const engine_file_path{argv[2]};

    std::cout << "Plugin library path: " << plugin_library_path << std::endl;
    std::cout << "Engine file path: " << engine_file_path << std::endl;

    CustomLogger logger{};

    // dlopen the plugin library using RAII.
    // The plugin will be registered automatically when the library is loaded.
    // Library will be automatically closed when the unique_ptr goes out of
    // scope
    std::unique_ptr<void, decltype(&dlclose)> plugin_handle{
        dlopen(plugin_library_path.c_str(), RTLD_LAZY), &dlclose};
    if (plugin_handle == nullptr)
    {
        std::cerr << "Failed to load the plugin library: " << dlerror()
                  << std::endl;
        return EXIT_FAILURE;
    }
    // Plugin now loaded and will be automatically unloaded at the end of main()

    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // The engine we built is FP32 NCHW IO.
    nvinfer1::DataType const expected_dtype{nvinfer1::DataType::kFLOAT};
    size_t const expected_dtype_byte_size{4U};
    nvinfer1::TensorFormat const expected_format{
        nvinfer1::TensorFormat::kLINEAR};

    // IO tensor information and buffers.
    std::vector<nvinfer1::Dims> input_tensor_shapes{};
    std::vector<nvinfer1::Dims> output_tensor_shapes{};
    std::vector<size_t> input_tensor_sizes{};
    std::vector<size_t> output_tensor_sizes{};
    std::vector<char const*> input_tensor_names{};
    std::vector<char const*> output_tensor_names{};
    std::vector<void*> input_tensor_host_buffers{};
    std::vector<void*> input_tensor_device_buffers{};
    std::vector<void*> output_tensor_host_buffers{};
    std::vector<void*> output_tensor_device_buffers{};

    // Error tolerance for unit test.
    float const rtol{1e-5f};
    float const atol{1e-8f};

    // Deserialize the engine.
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime{
        nvinfer1::createInferRuntime(logger)};
    if (runtime == nullptr)
    {
        std::cerr << "Failed to create the runtime." << std::endl;
        return EXIT_FAILURE;
    }

    // Load the plugin library.
    runtime->getPluginRegistry().loadLibrary(plugin_library_path.c_str());

    std::ifstream engine_file{engine_file_path, std::ios::binary};
    if (!engine_file)
    {
        std::cerr << "Failed to open the engine file." << std::endl;
        return EXIT_FAILURE;
    }

    engine_file.seekg(0, std::ios::end);
    size_t const engine_file_size{static_cast<size_t>(engine_file.tellg())};
    engine_file.seekg(0, std::ios::beg);

    std::unique_ptr<char[]> engine_data{new char[engine_file_size]};
    engine_file.read(engine_data.get(), engine_file_size);

    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine{
        runtime->deserializeCudaEngine(engine_data.get(), engine_file_size)};
    if (engine == nullptr)
    {
        std::cerr << "Failed to deserialize the engine." << std::endl;
        return EXIT_FAILURE;
    }

    // Create the execution context.
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context{
        engine->createExecutionContext()};
    if (context == nullptr)
    {
        std::cerr << "Failed to create the execution context." << std::endl;
        return EXIT_FAILURE;
    }

    // Check the number of IO tensors.
    int32_t const num_io_tensors{engine->getNbIOTensors()};
    std::cout << "Number of IO Tensors: " << num_io_tensors << std::endl;
    for (int32_t i{0}; i < num_io_tensors; ++i)
    {
        char const* const tensor_name{engine->getIOTensorName(i)};
        std::cout << "Tensor name: " << tensor_name << std::endl;
        nvinfer1::TensorIOMode const io_mode{
            engine->getTensorIOMode(tensor_name)};
        nvinfer1::DataType const dtype{engine->getTensorDataType(tensor_name)};
        if (dtype != expected_dtype)
        {
            std::cerr << "Invalid data type." << std::endl;
            return EXIT_FAILURE;
        }
        nvinfer1::TensorFormat const format{
            engine->getTensorFormat(tensor_name)};
        if (format != expected_format)
        {
            std::cerr << "Invalid tensor format." << std::endl;
            return EXIT_FAILURE;
        }
        // Because the input and output shapes are static,
        // there is no need to set the IO tensor shapes.
        nvinfer1::Dims const shape{engine->getTensorShape(tensor_name)};
        // Print out dims.
        size_t tensor_size{1U};
        std::cout << "Tensor Dims: ";
        for (int32_t j{0}; j < shape.nbDims; ++j)
        {
            tensor_size *= shape.d[j];
            std::cout << shape.d[j] << " ";
        }
        std::cout << std::endl;

        // FP32 NCHW tensor format.
        size_t tensor_size_bytes{tensor_size * expected_dtype_byte_size};

        // Allocate host memory for the tensor.
        void* tensor_host_buffer{nullptr};
        CHECK_CUDA_ERROR(
            cudaMallocHost(&tensor_host_buffer, tensor_size_bytes));
        // Allocate device memory for the tensor.
        void* tensor_device_buffer{nullptr};
        CHECK_CUDA_ERROR(cudaMalloc(&tensor_device_buffer, tensor_size_bytes));

        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_tensor_host_buffers.push_back(tensor_host_buffer);
            input_tensor_device_buffers.push_back(tensor_device_buffer);
            input_tensor_shapes.push_back(shape);
            input_tensor_sizes.push_back(tensor_size);
            input_tensor_names.push_back(tensor_name);
        }
        else
        {
            output_tensor_host_buffers.push_back(tensor_host_buffer);
            output_tensor_device_buffers.push_back(tensor_device_buffer);
            output_tensor_shapes.push_back(shape);
            output_tensor_sizes.push_back(tensor_size);
            output_tensor_names.push_back(tensor_name);
        }
    }

    // Create random input values.
    for (size_t i{0U}; i < input_tensor_host_buffers.size(); ++i)
    {
        size_t const tensor_size{input_tensor_sizes.at(i)};
        create_random_data(static_cast<float*>(input_tensor_host_buffers.at(i)),
                           tensor_size);
    }

    // Copy input data from host to device.
    for (size_t i{0U}; i < input_tensor_host_buffers.size(); ++i)
    {
        size_t const tensor_size_bytes{input_tensor_sizes.at(i) *
                                       expected_dtype_byte_size};
        CHECK_CUDA_ERROR(cudaMemcpy(input_tensor_device_buffers.at(i),
                                    input_tensor_host_buffers.at(i),
                                    tensor_size_bytes, cudaMemcpyHostToDevice));
    }

    // Bind IO tensor buffers to the execution context.
    for (size_t i{0U}; i < input_tensor_device_buffers.size(); ++i)
    {
        char const* const tensor_name{input_tensor_names.at(i)};
        context->setTensorAddress(tensor_name,
                                  input_tensor_device_buffers.at(i));
    }
    for (size_t i{0U}; i < output_tensor_device_buffers.size(); ++i)
    {
        char const* const tensor_name{output_tensor_names.at(i)};
        context->setTensorAddress(tensor_name,
                                  output_tensor_device_buffers.at(i));
    }

    // Run inference a couple of times.
    size_t const num_iterations{8U};
    for (size_t i{0U}; i < num_iterations; ++i)
    {
        bool const status{context->enqueueV3(stream)};
        if (!status)
        {
            std::cerr << "Failed to run inference." << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Synchronize.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy output data from device to host.
    for (size_t i{0U}; i < output_tensor_host_buffers.size(); ++i)
    {
        size_t const tensor_size_bytes{output_tensor_sizes.at(i) *
                                       expected_dtype_byte_size};
        CHECK_CUDA_ERROR(cudaMemcpy(output_tensor_host_buffers.at(i),
                                    output_tensor_device_buffers.at(i),
                                    tensor_size_bytes, cudaMemcpyDeviceToHost));
    }

    // Verify the output given it's an identity neural network.
    for (size_t i{0U}; i < input_tensor_host_buffers.size(); ++i)
    {
        if (input_tensor_sizes.at(i) != output_tensor_sizes.at(i))
        {
            std::cerr << "Input and output tensor sizes do not match."
                      << std::endl;
            return EXIT_FAILURE;
        }
        if (!all_close(static_cast<float*>(input_tensor_host_buffers.at(i)),
                       static_cast<float*>(output_tensor_host_buffers.at(i)),
                       input_tensor_sizes.at(i), rtol, atol))
        {
            std::cerr << "Input and output tensor values do not match."
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Successfully verified the output." << std::endl;

    // Release resources.
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    for (size_t i{0U}; i < input_tensor_host_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(input_tensor_host_buffers.at(i)));
    }
    for (size_t i{0U}; i < input_tensor_device_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFree(input_tensor_device_buffers.at(i)));
    }
    for (size_t i{0U}; i < output_tensor_host_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(output_tensor_host_buffers.at(i)));
    }
    for (size_t i{0U}; i < output_tensor_device_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFree(output_tensor_device_buffers.at(i)));
    }
}