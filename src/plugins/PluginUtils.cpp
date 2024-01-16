#include <cstring>
#include <sstream>

#include <NvInferRuntime.h>

void caughtError(std::exception const& e)
{
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

void reportAssertion(bool success, char const* msg, char const* file,
                     int32_t line)
{
    if (!success)
    {
        std::ostringstream stream;
        stream << "Assertion failed: " << msg << std::endl
               << file << ':' << line << std::endl
               << "Aborting..." << std::endl;
        getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                         stream.str().c_str());
        std::abort();
    }
}

void reportValidation(bool success, char const* msg, char const* file,
                      int32_t line)
{
    if (!success)
    {
        std::ostringstream stream;
        stream << "Validation failed: " << msg << std::endl
               << file << ':' << line << std::endl;
        getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                         stream.str().c_str());
    }
}

// // Write values into buffer
// template <typename Type, typename BufferType>
// void write(BufferType*& buffer, Type const& val)
// {
//     static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte
//     type."); std::memcpy(buffer, &val, sizeof(Type)); buffer += sizeof(Type);
// }

// // Read values from buffer
// template <typename OutType, typename BufferType>
// OutType read(BufferType const*& buffer)
// {
//     static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte
//     type."); OutType val{}; std::memcpy(&val, static_cast<void
//     const*>(buffer), sizeof(OutType)); buffer += sizeof(OutType); return val;
// }

// // Explicit instantiations.
// template void write<int32_t, char>(char*& buffer, int32_t const& val);

// template int32_t read<int32_t, uint8_t>(uint8_t const*& buffer);
