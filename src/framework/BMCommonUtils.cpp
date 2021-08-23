#include<iostream>
#include<string>
#include<ctime>
#include<iomanip>
#include<sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <functional>

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

void TimeRecorder::record(const std::string &event_name)
{
    timepoints.push_back(getUS());
    events.push_back(event_name);
}

void TimeRecorder::show() const
{
    auto eventNum = events.size();
    auto endTime = getUS();
    if(eventNum == 0) return;
    for(size_t i=0; i<eventNum; i++){
        BMLOG(INFO, "%d: %s: %g ms", i, events[i].c_str(), timepoints[i]/1000.0);
        if(i!=eventNum-1){
            BMLOG(INFO, "  | %g ms", (timepoints[i+1]-timepoints[i])/1000.0);
        } else {
            BMLOG(INFO, "  | %g ms", (endTime-timepoints[i])/1000.0);
        }
    }
    BMLOG(INFO, "%d: %s: %g ms", eventNum, "==end==", endTime/1000.0);
}

TimeRecorder::TimeRecorder(): start(TimerClock::now()) {
}

size_t TimeRecorder::getUS() const
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

std::size_t strReplaceAll(std::string &inout, const std::string &what, const std::string &with)
{
    std::size_t count{};
    for (std::string::size_type pos{};
         inout.npos != (pos = inout.find(what.data(), pos, what.length()));
         pos += with.length(), ++count) {
        inout.replace(pos, what.length(), with.data(), with.length());
    }
    return count;
}

std::string baseName(const std::string &fullPath){
    auto pos = fullPath.find_last_of('/');
    if(pos != std::string::npos){
        return fullPath.substr(pos+1);
    }
    return fullPath;
}

bool isWhitespace(char c){
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string escape(const std::string &text){
    std::string encodeStr;
    size_t pos = 0;
    char buffer[16];
    while(pos<text.size()){
        unsigned char c = text[pos];
        unsigned int value;
        if((c&0x80) == 0) {
            if(c=='"') {
                encodeStr += '\\';
            }
            encodeStr+=c;
            pos++;
            continue;
        } else if((c>>5) == 6 && pos+1<text.size()) { // 110xxxxx 10xxxxxx
            value = (((c&0x1f))<<6) | (text[pos+1]&0x3f);
            pos += 2;
        } else if((c>>4) == 0xe && pos+2<text.size()) { // 1110xxxx 10xxxxxx 10xxxxxx
            value = (((c&0xF)<<12)) | ((text[pos+1]&0x3f)<<6) | (text[pos+2]&0x3f);
            pos += 3;
        } else if((c>>3) == 0x1e && pos+3<text.size()) { // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            value = (((c&0x7)<<18)) | ((text[pos+1]&0x3f)<<12) |
                    ((text[pos+2]&0x3f)<<6) | (text[pos+3]&0x3f);
            pos += 4;
        } else {
            BMLOG(WARNING, "'%s' cannot be encoded to utf8", text.c_str());
            return encodeStr;
        }
        sprintf(buffer, "\\u%04x", value);
        encodeStr += buffer;
    }
    return encodeStr;
}

std::string strStrip(const std::string &text){
    std::string result;
    size_t start=0;
    size_t end=text.size();
    while(start<end && isWhitespace(text[start])) start++;
    while(start<end && isWhitespace(text[end-1])) end--;
    return text.substr(start, end-start);
}

std::pair<std::string, std::string> splitStringPair(const std::string &line, char split){
    auto pos = line.find_first_of(split);
    std::string key = line.substr(0, pos);
    std::string value = "";
    if(pos != std::string::npos) {
        value = line.substr(pos+1);
    }
    return std::make_pair(strStrip(key), strStrip(value));
}

std::vector<std::string> whitespaceSplit(const std::string &text){
    std::vector<std::string> results;
    std::string slice;
    bool prevIsWS = true;
    for(const auto c: text){
        if(isWhitespace(c)){
            prevIsWS = true;
            if(!slice.empty()){
                results.push_back(slice);
                slice.clear();
            }
        } else {
            if(prevIsWS){
                slice = c;
            } else {
                slice += c;
            }
            prevIsWS = false;
        }
    }
    if(!slice.empty()){
        results.push_back(slice);
    }
    return results;
}

std::vector<unsigned int> whitespaceSplitToUint(const std::string &text){
    std::vector<unsigned int> results;
    std::string slice;
    bool prevIsWS = true;
    for(const auto c: text){
        if(isWhitespace(c)){
            prevIsWS = true;
            if(!slice.empty()){
                results.push_back(atoi(slice.c_str()));
                slice.clear();
            }
        } else {
            if(prevIsWS){
                slice = c;
            } else {
                slice += c;
            }
            prevIsWS = false;
        }
    }
    if(!slice.empty()){
        results.push_back(atoi(slice.c_str()));
    }
    return results;
}

std::vector<std::string> strSplitByChar(const std::string &text, char splittor)
{
    std::vector<std::string> results;
    std::string slice;
    bool prevIsWS = true;
    for(const auto c: text){
        if(c == splittor){
            prevIsWS = true;
            if(!slice.empty()){
                results.push_back(slice);
                slice.clear();
            }
        } else {
            if(prevIsWS){
                slice = c;
            } else {
                slice += c;
            }
            prevIsWS = false;
        }
    }
    if(!slice.empty()){
        results.push_back(slice);
    }
    return results;

}

void forEachFile(const std::string& path, std::function<bool (const std::string& )> func) {
    DIR *d = nullptr;
    struct dirent *dp = nullptr;
    struct stat st;
    std::string fullpath;
    if(!func) return;
    if(path.empty()) return;
    if(stat(path.c_str(), &st) < 0) return;
    if(S_ISDIR(st.st_mode)){
        if(!(d = opendir(path.c_str()))) return;
        while((dp = readdir(d)) != nullptr) {
            // if((!strncmp(dp->d_name, ".", 1)) || (!strncmp(dp->d_name, "..", 2)))continue;
            fullpath = path + "/" + dp->d_name;
            stat(fullpath.c_str(), &st);
            if(!S_ISDIR(st.st_mode)) {
                if(!func(fullpath)) break;
            }
        }
        closedir(d);
    } else {
        func(fullpath);
    }
}

void forEachBatch(const std::string &path, size_t batchSize, std::function<bool (const std::vector<std::string> &)> func)
{
    DIR *d = nullptr;
    struct dirent *dp = nullptr;
    struct stat st;
    std::string fullpath;
    std::vector<std::string> batchPaths;

    if(!func) return;
    if(path.empty()) return;
    if(stat(path.c_str(), &st) < 0) return;
    if(!S_ISDIR(st.st_mode)) return;
    if(!(d = opendir(path.c_str()))) return;

    while((dp = readdir(d)) != nullptr) {
        fullpath = path + "/" + dp->d_name;
        stat(fullpath.c_str(), &st);
        if(!S_ISDIR(st.st_mode)) {
            batchPaths.push_back(fullpath);
            if(batchSize == batchPaths.size()) {
                auto ok = func(std::move(batchPaths));
                batchPaths.clear();
                if(!ok) break;
            }
        }
    }

    if(!batchPaths.empty()){
        func(std::move(batchPaths));
        batchPaths.clear();
    }

    closedir(d);
}

float AUC(std::vector<std::pair<unsigned int, float>>& scores)
{
    std::sort(scores.begin(), scores.end(), [](
              const std::pair<unsigned int, float>& s1,
              const std::pair<unsigned int, float>& s2
              ){ return s1.second < s2.second; });
    size_t numPositive=0;
    float rankSum = 0;

    size_t sameCount = 0;
    size_t samePositive = 0;
    size_t sameSum = 0;
    float lastScore = 0;
    for(size_t i=0; i<scores.size(); i++){
        auto& s= scores[i];
        numPositive += (s.first!=0);
        if(s.second == lastScore || i==0){
            sameSum += i+1;
            sameCount++;
            samePositive += (s.first!=0);
        } else {
            rankSum = rankSum + (float) samePositive * sameSum/sameCount;
            sameSum = i+1;
            sameCount = 1;
            samePositive = (s.first!=0);
            lastScore = s.second;
        }
    }
    rankSum = rankSum + (float) samePositive * sameSum/sameCount;

    return (rankSum - (float)numPositive*(numPositive+1)/2)/(numPositive*(scores.size()-numPositive));
}


};
