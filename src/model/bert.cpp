#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "BMDevicePool.h"
#include "BMCommonUtils.h"
#include "BMLog.h"

using namespace bm;
bool isWhitespace(char c){
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::vector<unsigned int> whitespaceSplitToUint(const std::string& text){
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

std::vector<std::string> whitespaceSplit(const std::string& text){
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

std::pair<std::string, std::string> splitStringPair(const std::string& line, char split=':'){
    auto pos = line.find_first_of(split);
    std::string key = line.substr(0, pos);
    std::string value = "";
    if(pos != std::string::npos) {
        value = line.substr(pos+1);
    }
    return std::make_pair(key, value);
}

struct SquadData {
    unsigned long long id;
    std::vector<unsigned int> inputIds;
    std::vector<unsigned int> inputMask;
    std::vector<unsigned int> segmentIds;
    std::vector<std::string> docTokens;
    std::vector<std::string> tokens;
    std::map<size_t, size_t> tokenToOrigin;
    std::map<size_t, bool> tokenIsMaxContext;
    unsigned int refStart;
    unsigned int refEnd;
    std::string question;
    std::string answer;
    bool parse(const std::map<std::string, std::string>& dict){
        try{
            answer = dict.at("origin_answer");
            question = dict.at("question");
            refStart = atoi(dict.at("start_position").c_str());
            refEnd = atoi(dict.at("end_position").c_str());
            id = atoll(dict.at("unique_id").c_str());
            tokens = whitespaceSplit(dict.at("tokens"));
            docTokens = whitespaceSplit(dict.at("doc_tokens"));
            inputIds = whitespaceSplitToUint(dict.at("input_ids" ));
            inputMask = whitespaceSplitToUint(dict.at("input_mask" ));
            segmentIds = whitespaceSplitToUint(dict.at("segment_ids" ));
            inputMask.resize(inputIds.size());
            segmentIds.resize(inputIds.size());
            auto origMapItems = whitespaceSplit(dict.at("token_to_orig_map"));
            for(auto& s: origMapItems){
                auto p = splitStringPair(s);
                tokenToOrigin[atoi(p.first.c_str())] = atoi(p.second.c_str());
            }
            auto maxContexts = whitespaceSplit(dict.at("token_is_max_context"));
            for(auto& s: maxContexts){
                auto p = splitStringPair(s);
                tokenIsMaxContext[atoi(p.first.c_str())] = (p.second == "True");
            }

        } catch(...) {
            return false;
        }
        return true;
    }
};
std::ostream& operator << (std::ostream& s, const SquadData& data){
    s<<"id:"<<data.id<<std::endl;
    s<<"input_ids("<<data.inputIds.size()<<"):";
    std::copy(data.inputIds.begin(), data.inputIds.end(), std::ostream_iterator<unsigned int>{s, " "});
    s<<std::endl;
    s<<"input_mask:";
    std::copy(data.inputMask.begin(), data.inputMask.end(), std::ostream_iterator<unsigned int>{s, " "});
    s<<std::endl;
    s<<"segment_ids:";
    std::copy(data.segmentIds.begin(), data.segmentIds.end(), std::ostream_iterator<unsigned int>{s, " "});
    s<<std::endl;
//    s<<"input_tokens:"<<data.tokens.size();
//    std::copy(data.tokens.begin(), data.tokens.end(), std::ostream_iterator<std::string>{s, " "});
//    s<<std::endl;
}

template<typename Func>
void parseSquadFile(const std::string& filename, size_t batch, Func func) {
    std::ifstream ifs(filename);
    std::vector<std::shared_ptr<SquadData>> batchData;
    std::shared_ptr<SquadData> data;
    std::map<std::string, std::string> dict;
    std::string line;

    while(std::getline(ifs, line)){
        if(line == "---") {
            data = std::make_shared<SquadData>();
            if(data->parse(dict)){
                dict.clear();
                batchData.push_back(data);
                if(batchData.size() == batch){
                    if(!func(std::move(batchData))) return;
                    batchData.clear();
                }
            } else {
                BMLOG(WARNING, "parse failed, ignore");
            }
        } else {
            dict.insert(splitStringPair(line, ':'));
        }
    }
}

const size_t max_nbest = 20;
const size_t max_answer_length = 30;

using InType = std::vector<std::shared_ptr<SquadData>>;
struct SquadResult {
    std::pair<size_t, float> start;
    std::pair<size_t, float> end;
    std::string refAnswer;
    std::string answer;
    float prob;
    size_t originStart;
    size_t originEnd;
};

using NBestSquadResult = std::vector<SquadResult> ;

struct PostOutType {
    InType squadRecords;
    std::vector<NBestSquadResult> results;
};

using RunnerType = BMDevicePool<InType, PostOutType>;

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    BM_ASSERT_EQ(inTensors.size(),3);
    BM_ASSERT_LE(in.size(), inTensors[0]->shape(0)); //batch
    size_t offset = 0;
    for(auto data: in){
        BM_ASSERT_EQ(data->inputIds.size(), inTensors[0]->shape(1));
        inTensors[0]->fill_device_mem(data->inputIds.data(), data->inputIds.size()*sizeof(int), offset);
        inTensors[1]->fill_device_mem(data->inputMask.data(), data->inputMask.size()*sizeof(int), offset);
        inTensors[2]->fill_device_mem(data->segmentIds.data(), data->segmentIds.size()*sizeof(int), offset);
        offset += data->inputIds.size()*sizeof(int);
    }
    return true;
}

void softmaxResult(std::vector<SquadResult>& result){
    if(result.empty()) return;
    float maxProb = result[0].prob;
    for(size_t i=1; i<result.size(); i++){
        if(maxProb<result[i].prob){
            maxProb = result[i].prob;
        }
    }
    float sum = 0;
    for(size_t i=0; i<result.size(); i++){
        result[i].prob = exp(result[i].prob-maxProb);
        sum += result[i].prob;
    }
    for(size_t i=0; i<result.size(); i++){
        result[i].prob = result[i].prob/sum;
    }
}

static void stripSpaces(const std::string& text, std::string& chars, std::map<size_t, size_t>& posMap){

    for(size_t i=0; i<text.size(); i++){
        if(text[i]==' ') continue;
        posMap[chars.size()] = i;
        chars += text[i];
    }
}

static std::string getFinalText(const std::string& pred, const std::string& orig, bool uncased){

    std::string lowerOrigText = orig;
    for(auto&c : lowerOrigText){
        c = std::tolower(c);
    }

    size_t startPos = lowerOrigText.find_first_of(pred);
    if(startPos == std::string::npos){
        return orig;
    }
    size_t endPos = startPos + pred.size()-1;


    std::string lowerChars;
    std::map<size_t, size_t> lowerMap;
    stripSpaces(lowerOrigText, lowerChars, lowerMap);

    std::string origChars;
    std::map<size_t, size_t> origMap;
    stripSpaces(orig, origChars, origMap);
    if(origChars.size() != lowerChars.size()){
        return orig;
    }

    std::map<size_t, size_t> inversedLowerMap;
    for(auto& i: lowerMap){
        inversedLowerMap[i.second] = i.first;
    }

    size_t origStartPos = -1;
    if(inversedLowerMap.count(startPos)){
        size_t charsPos = inversedLowerMap[startPos];
        if(origMap.count(charsPos)){
            origStartPos = origMap[charsPos];
        }
    }
    if(origStartPos == -1) {
        return orig;
    }
    size_t origEndPos = -1;
    if(inversedLowerMap.count(endPos)){
        size_t charsPos = inversedLowerMap[endPos];
        if(origMap.count(charsPos)){
            origEndPos = origMap[charsPos];
        }
    }
    if(origEndPos == -1){
        return orig;
    }
    std::string output = orig.substr(origStartPos, origEndPos-origStartPos+1);
    return output;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    postOut.squadRecords = rawIn;
    postOut.results.resize(rawIn.size());

    size_t batch = rawIn.size();
    auto startTensor = outTensors[0];
    auto endTensor = outTensors[1];
    BM_ASSERT_EQ(startTensor->dims(), 2);
    BM_ASSERT_EQ(startTensor->shape(0), batch);
    BM_ASSERT_EQ(endTensor->dims(), 2);
    BM_ASSERT_EQ(endTensor->shape(0), batch);
    BM_ASSERT_EQ(startTensor->shape(1), endTensor->shape(1));
    auto seqLen = startTensor->shape(1);
    auto batchStart = startTensor->get_float_data();
    auto batchEnd = endTensor->get_float_data();
    for(size_t b=0; b<batch; b++){
        auto startData = batchStart + seqLen*b;
        auto endData = batchStart + seqLen*b;
        auto nbestStarts = topk(startData, seqLen, max_nbest);
        auto nbestEnds = topk(endData, seqLen, max_nbest);
        auto squad = postOut.squadRecords[b];
        auto& result = postOut.results[b];
        std::vector<SquadResult> candidates;
        for(auto& start: nbestStarts){
            for(auto& end: nbestEnds) {
                if(start.first >= squad->tokens.size()) continue;
                if(end.first >= squad->tokens.size()) continue;
                if(!squad->tokenToOrigin.count(start.first)) continue;
                if(!squad->tokenToOrigin.count(end.first)) continue;
                if(end.first<= start.first) continue;
                auto length = end.first - start.first + 1;
                if(length>max_answer_length) continue;
                SquadResult r;
                r.start = start;
                r.end = end;
                candidates.push_back(r);
            }
        }

        // decode the result
        float probSum = 0;
        std::set<std::string> knownAnswers;
        for(auto& c: candidates){
            c.answer = "";
            c.originStart = squad->tokenToOrigin[c.start.first];
            c.originEnd = squad->tokenToOrigin[c.end.first];
            if(c.start.first>0){
                c.prob = c.start.second + c.end.second;
                for(size_t i=c.start.first; i<c.end.first; i++){
                    c.answer += squad->tokens[i] + " ";
                }
                c.answer += squad->tokens[c.end.first];
                strReplaceAll(c.answer, " ##", "");
                strReplaceAll(c.answer, "##", "");
            }
            std::string origText;
            for(size_t i=c.originStart; i<c.originEnd; i++){
                origText += squad->docTokens[i]+" ";
            }
            origText += squad->docTokens[c.originEnd];
            // TODO: get_final_text()
            c.answer = getFinalText(c.answer, origText, true);

            if(knownAnswers.count(c.answer)){
                continue;
            } else {
                knownAnswers.insert(c.answer);
            }

            c.refAnswer = squad->answer;
            result.push_back(c);
            if(result.size()>=max_nbest) break;
        }
        softmaxResult(result);
    }
    return true;
}

bool resultProcess(const PostOutType& out){
    if(out.squadRecords.empty()) return false;
    auto batch = out.squadRecords.size();
    for(size_t b=0; b<batch; b++){
        auto squad = out.squadRecords[b];
        auto& result = out.results[b];
        BMLOG(INFO, "Q: %s", squad->question.c_str());
        BMLOG(INFO, "A: %s", squad->answer.c_str());
        for(auto& r: result){
            BMLOG(INFO, "  P(%.2f): '%s'", r.prob, r.answer.c_str(), r.prob);
        }
    }
    return true;
}

int main(int argc, char** argv){
    set_log_level(INFO);
    std::string squadPath = "../data/squad/squad_data.txt";
    std::string squadModel = "../models/bert_squad/fp32.bmodel";
    if(argc>1) squadPath = argv[1];
    if(argc>2) squadModel = argv[2];

    BMDevicePool<InType, PostOutType> runner(squadModel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info("squad-bert");
    std::thread dataThread([squadPath, batchSize, &runner](){
        parseSquadFile(squadPath, batchSize, [&runner](const std::vector<std::shared_ptr<SquadData>>& batchData){
            static int i=0;
            if(i>10) return false;
            runner.push(batchData);
            i++;
            return true;
        });
        while(!runner.allStopped()){
            if(runner.canPush()) {
                runner.push({});
            } else {
                std::this_thread::yield();
            }
        }
    });
    std::thread resultThread([&runner, &info](){
        PostOutType out;
        std::shared_ptr<ProcessStatus> status;
        bool stopped = false;
        while(true){
            while(!runner.pop(out, status)) {
                if(runner.allStopped()) {
                    stopped = true;
                    break;
                }
                std::this_thread::yield();
            }
            if(stopped) break;
            info.update(status);
            if(!resultProcess(out)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    info.show();
    return 0;
}
