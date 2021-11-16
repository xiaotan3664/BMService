#ifndef BMLOG_H
#define BMLOG_H

#include <stdio.h>
#include <type_traits>
#include <stdexcept>
#include <thread>

namespace bm {

typedef enum {
    DEBUG   = 0,
    INFO    = 1,
    WARNING = 2,
    ERROR   = 3,
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
void set_env_log_level(LogLevel level=LogLevel::INFO);

template<int level, typename ... ArgTypes>
typename std::enable_if<level==FATAL , void>::type __bm_log(const char* fmt, ArgTypes ...args){
    char msg[1024];
    snprintf(msg, sizeof(msg)-1, fmt, args...);
    printf("runtime_error, %s\n", msg);
    fflush(stdout);
    throw std::runtime_error(msg);
}

#define BMLOG(severity, fmt, ...) \
    __bm_log<LogLevel::severity>("[tid=%x] %s: " fmt "\n", std::this_thread::get_id(), #severity, ##__VA_ARGS__)

};

#define BM_ASSERT(cond, fmt, ...)                                  \
    do                                                             \
    {                                                              \
        if (!(cond))                                               \
        {                                                          \
            BMLOG(FATAL, "assert " #cond ":" #fmt, ##__VA_ARGS__); \
        }                                                          \
    } while (0)

#define BM_ASSERT_OP(v1, v2, OP)                                           \
    do                                                                     \
    {                                                                      \
        if (!((v1)OP(v2)))                                               \
        {                                                                  \
            BMLOG(FATAL, "assert " #v1 "(%d)" #OP #v2 "(%d)", (v1), (v2)); \
        }                                                                  \
    } while (0)

#define BM_ASSERT_EQ(v1, v2) BM_ASSERT_OP(v1, v2, ==)
#define BM_ASSERT_NE(v1, v2) BM_ASSERT_OP(v1, v2, !=)
#define BM_ASSERT_LT(v1, v2) BM_ASSERT_OP(v1, v2, <)
#define BM_ASSERT_GT(v1, v2) BM_ASSERT_OP(v1, v2, >)
#define BM_ASSERT_LE(v1, v2) BM_ASSERT_OP(v1, v2, <=)
#define BM_ASSERT_GE(v1, v2) BM_ASSERT_OP(v1, v2, >=)

#endif // BMLOG_H
