#ifndef TENSORRT_PLUGIN_UTILS_H
#define TENSORRT_PLUGIN_UTILS_H

#include <cstring>
#include <sstream>

void caughtError(std::exception const& e);

void logInfo(char const* msg);

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)
void reportAssertion(bool success, char const* msg, char const* file,
                     int32_t line);

#define PLUGIN_VALIDATE(val) reportValidation((val), #val, __FILE__, __LINE__)
void reportValidation(bool success, char const* msg, char const* file,
                      int32_t line);

#endif // TENSORRT_PLUGIN_UTILS_H
