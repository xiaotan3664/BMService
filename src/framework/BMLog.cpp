#include "BMLog.h"
#include "BMEnv.h"

namespace bm {
static LogLevel BMRT_LOG_LEVEL_THRESHOLD;

int get_log_level()
{
    return BMRT_LOG_LEVEL_THRESHOLD;
}

void set_log_level(LogLevel level){
    BMRT_LOG_LEVEL_THRESHOLD = level;
}

struct __log_initializer{
    __log_initializer(){
        auto level = LogLevel::INFO;
        set_log_level(level);
        set_env_log_level();
    }
};
static __log_initializer __log_init();

void set_env_log_level()
{
    auto level_cstr = getenv(BM_LOG_LEVEL);
    if(level_cstr){
        LogLevel level = (LogLevel)atoi(level_cstr);
        set_log_level(level);
    }
}

}
