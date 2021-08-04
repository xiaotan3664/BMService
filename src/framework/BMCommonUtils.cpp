#include<iostream>
#include<string>
#include<ctime>
#include<iomanip>
#include<sstream>

#include "BMCommonUtils.h"
#include "BMLog.h"

namespace bm {

static std::time_t steady_to_time_t(const std::chrono::steady_clock::time_point& t){
    return std::chrono::system_clock::to_time_t(
                std::chrono::time_point_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now()
                + (t - std::chrono::steady_clock::now())));
}

std::string steadyToString(const std::chrono::steady_clock::time_point &tp){
    auto t = steady_to_time_t(tp);
    std::ostringstream oss;
    oss<<std::put_time(std::localtime(&t), "%F %T");
    return oss.str();
}

TimeRecorder::TimeRecorder(): start(TimerClock::now()) {
}

size_t TimeRecorder::getUS()
{
    auto end = TimerClock::now();
    return usBetween(start, end);
}

size_t usBetween(const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &stop){
    return std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
}

TimeRecorder::~TimeRecorder(){
}

};

std::string baseName(const std::string &fullPath){
    auto pos = fullPath.find_last_of('/');
    if(pos != std::string::npos){
        return fullPath.substr(pos+1);
    }
    return fullPath;
}
