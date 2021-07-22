#ifndef BMCOMMONUTILS_H
#define BMCOMMONUTILS_H

#include <set>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <vector>

namespace bm {

class Uncopiable {
    public:
    Uncopiable() = default;
    Uncopiable(Uncopiable&) = delete;
    Uncopiable(const Uncopiable&) = delete;
    Uncopiable& operator = (const Uncopiable) = delete;
};

std::string steadyToString(const std::chrono::steady_clock::time_point& tp);

using TimerClock = std::chrono::steady_clock;
size_t usBetween(const std::chrono::steady_clock::time_point &start,
        const std::chrono::steady_clock::time_point &stop);

class TimeRecorder{
private:
    std::chrono::steady_clock::time_point start;
public:
    TimeRecorder();
    size_t getUS();
    ~TimeRecorder();
};

template <typename T>
std::set<T> stringToSet(const std::string &s)
{
    std::set<T> values;
    std::string new_str = s;
    for(auto& c: new_str){
       if(!isdigit(c) && c!='.') c= ' ';
    }
    std::istringstream is(new_str);
    T v= 0;
    while(is>>v){
        values.insert(v);
    }
    return values;
}

}

#endif // BMCOMMONUTILS_H
