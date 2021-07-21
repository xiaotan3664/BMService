#include<iostream>
#include<string>
#include<ctime>
#include<iomanip>
#include<sstream>

#include "BMCommonUtils.h"
#include "BMLog.h"

namespace bm {

static std::time_t steady_to_time_t(std::chrono::steady_clock::time_point& t){
    return std::chrono::system_clock::to_time_t(
                std::chrono::time_point_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now()
                + (t - std::chrono::steady_clock::now())));
}

std::string steadyToString(std::chrono::steady_clock::time_point& tp){
    auto t = steady_to_time_t(tp);
    std::ostringstream oss;
    oss<<std::put_time(std::localtime(&t), "%F %T");
    return oss.str();
}

TimeRecorder::TimeRecorder(const std::string &name): start(TimerClock::now()), name(name) {
}

size_t msBetween(std::chrono::steady_clock::time_point &start,
        std::chrono::steady_clock::time_point &stop){
    return std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
}

TimeRecorder::~TimeRecorder(){
    auto end = TimerClock::now();
    auto duration = msBetween(start, end);
    auto milliseconds = duration/1000.0;
    BMLOG(INFO,"%s costs %gms", name.c_str(), milliseconds);
}

};
