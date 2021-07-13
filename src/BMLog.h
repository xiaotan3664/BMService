#ifndef BMLOG_H
#define BMLOG_H

#include <stdio.h>
#include <type_traits>
#include <stdexcept>

namespace bm {

typedef enum {
    DEBUG   = 0,
    INFO    = 1,
    WARNING = 2,
    WRONG   = 3,
    FATAL   = 4,
} LogLevel;

int get_log_level();
void set_log_level(LogLevel level);

template<int level, typename ... ArgTypes>
typename std::enable_if<level<FATAL , void>::type __bm_log(const char*fmt, ArgTypes ...args){
    if (level >= get_log_level()){
        printf(fmt, args...);
        fflush(stdout);
    }
}

template<int level, typename ... ArgTypes>
typename std::enable_if<level==FATAL , void>::type __bm_log(const char*fmt, ArgTypes ...args){
    char msg[1024];
    snprintf(msg, sizeof(msg)-1, fmt, args...);
    throw std::runtime_error(msg);
}

#define BMLOG(severity, fmt, ...) \
    __bm_log<LogLevel::severity>("[%s:%d] %s: " fmt "\n", __FILE__, __LINE__, #severity, ##__VA_ARGS__)

};

#endif // BMLOG_H
