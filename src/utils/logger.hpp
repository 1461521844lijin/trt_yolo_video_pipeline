// #ifndef LOGGER_H
// #define LOGGER_H
 
// #include "spdlog/spdlog.h"
// #include "spdlog/async.h"
// #include "spdlog/sinks/basic_file_sink.h"
// #include "spdlog/sinks/stdout_color_sinks.h"

// #ifdef WIN32
// #define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1):__FILE__)
// #else
// #define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1):__FILE__)
// #endif

// inline std::shared_ptr<spdlog::logger> logger = spdlog::get("logger");

// if (!logger) {
//     auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
//     auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("./logs/run_log.txt");

//     consoleSink->set_level(spdlog::level::trace);
//     fileSink->set_level(spdlog::level::trace);
//     consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^---%L---%$] %v");
//     fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^---%L---%$] %v");
//     spdlog::logger logger("logger", {consoleSink, fileSink});
//     logger.flush_on(spdlog::level::trace);
// }
 
// #endif // LOGGER_H
// logger.h
#ifndef LOGGER_H
#define LOGGER_H
 
#include <spdlog/spdlog.h>
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef WIN32
#define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1):__FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1):__FILE__)
#endif

#ifndef LOGGER_NAME
#define LOGGER_NAME "default_logger"
#endif

inline auto GetLogger()
{
    //static auto logger = spdlog::stdout_logger_mt(LOGGER_NAME);
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("./logs/run_log.txt");

    consoleSink->set_level(spdlog::level::trace);
    fileSink->set_level(spdlog::level::trace);
    consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^---%L---%$] %v");
    fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^---%L---%$] %v");
    static spdlog::logger logger("logger", {consoleSink, fileSink});
    logger.flush_on(spdlog::level::trace);
    //std::shared_ptr<spdlog::logger> logger_ptr(logger);
    //std::shared_ptr<spdlog::logger> logger_ptr(logger);
    return logger;
}
 
#endif // LOGGER_H