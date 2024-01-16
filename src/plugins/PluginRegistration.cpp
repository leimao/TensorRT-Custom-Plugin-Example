#include <iostream>
#include <mutex>

#include <NvInferRuntime.h>

#include "IdentityConvPluginCreator.h"

// namespace nvinfer1
// {
// namespace plugin
// {

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
    std::cout << "----getPluginCreators----" << std::endl;
    nbCreators = 1;
    static nvinfer1::plugin::IdentityConvCreator identityConvCreator{};
    static nvinfer1::IPluginCreator* const pluginCreatorList[] = {
        &identityConvCreator};
    return pluginCreatorList;
}

// } // namespace plugin
// } // namespace nvinfer1
