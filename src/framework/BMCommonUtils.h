#ifndef BMCOMMONUTILS_H
#define BMCOMMONUTILS_H

#include <set>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <vector>
#include <algorithm>
#include <functional>

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
    std::vector<std::string> events;
    std::vector<size_t> timepoints;
public:
    void record(const std::string&event_name);
    void show() const;
    TimeRecorder();
    size_t getUS() const;
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

template<typename T>
std::vector<std::pair<size_t, T>> topk(const T* data, size_t len, size_t k) {
    std::vector<std::pair<size_t, T>> pair_data;
    for(size_t i = 0; i<len; i++){
        pair_data.emplace_back(i, data[i]);
    }
    std::partial_sort(pair_data.begin(), pair_data.begin()+k, pair_data.end(),
                      [](const std::pair<size_t, T>& a, const std::pair<size_t, T>&b){
                          return a.second>b.second;
                      });
    pair_data.resize(k);
    return pair_data;
}

std::size_t strReplaceAll(std::string& inout, const std::string& what, const std::string& with);

std::string baseName(const std::string& fullPath);

bool isWhitespace(char c);
std::string escape(const std::string& text);
std::vector<unsigned int> whitespaceSplitToUint(const std::string& text);

std::string strStrip(const std::string& text);
std::vector<std::string> strSplitByChar(const std::string& text, char splittor);
std::vector<std::string> whitespaceSplit(const std::string& text);
std::pair<std::string, std::string> splitStringPair(const std::string& line, char split=':');

void forEachFile(const std::string& path, std::function<void(const std::string& )> func);
void forEachBatch(const std::string& path, size_t batchSize, std::function<void(const std::vector<std::string>&)> func);

}
#endif // BMCOMMONUTILS_H
