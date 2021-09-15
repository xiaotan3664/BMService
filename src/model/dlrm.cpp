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
#include <cmath>
#include "BMDevicePool.h"
#include "BMCommonUtils.h"
#include "BMLog.h"

using namespace bm;

const int maxIndexRange = 10000000;
const size_t numDense = 13;
const size_t numCategory = 26;
std::vector<std::map<unsigned int, unsigned int>> globalCategoryDict;

void loadDict(const std::string& filePrefix, size_t num,
              std::vector<std::map<unsigned int, unsigned int>>& dicts) {
    dicts.resize(num);
    FILE* fp=nullptr;
    BMLOG(INFO, "Load %d dict files ...",  num);
    for(size_t i=0; i<num; i++){
        std::string filename = filePrefix + std::to_string(i);
        fp = fopen(filename.c_str(), "rb");
        BM_ASSERT_NE(fp, nullptr);
        unsigned int key = 0;
        unsigned int val = 0;
        auto& dict = dicts[i];
        while(!feof(fp)){
            if(fread(&key, sizeof(key), 1, fp)==1){
		    dict[key] = val++;
	    }
        }
        BMLOG(INFO, "  --> loading dict file '%s': count=%d", filename.c_str(), val);
        fclose(fp);
    }
}

struct DLRMRecord {
    std::vector<float> denseFeatures;
    std::vector<unsigned int> categoryFeatures;
    unsigned int target;
    bool parse(const std::string& line, int maxIndexRange){
        try{
            int currentIndex = 0;
            std::string slice;
            denseFeatures.assign(numDense, 0);
            categoryFeatures.assign(numCategory, 0);
            // 0: target
            // 1-13: 13 dense features, in dec format
            // 14-39: 26 category features in hex format
            for(auto c: line){
                if(c!='\t'){
                    slice += c;
                    continue;
                }
                if(currentIndex == 0){
                    target = atoi(slice.c_str());
                } else if(currentIndex>=1 && currentIndex<1+numDense){
                    int value = 0;
                    if(slice != "") {
                        value = strtol(slice.c_str(), nullptr, 10);
                        if (value<0) value = 0;
                    }
                    denseFeatures[currentIndex-1] = log(1.0+value);
                } else {
                    long feaValue = 0;
                    if (slice != ""){
                        feaValue = strtol(slice.c_str(), nullptr, 16);
                    }
                    if(maxIndexRange>0){
                        feaValue = feaValue % maxIndexRange;
                    }
                    size_t i = currentIndex - 1 -numDense;
                    categoryFeatures[i] = globalCategoryDict[i][feaValue];
                }
                currentIndex++;
                slice.clear();
            }
            if(slice != ""){
                long feaValue = strtol(slice.c_str(), nullptr, 16);
                if(maxIndexRange>0){
                    feaValue = feaValue % maxIndexRange;
                }
                size_t i = currentIndex - 1 - numDense;
                categoryFeatures[i] = globalCategoryDict[i][feaValue];
            }
            currentIndex++;
            BM_ASSERT_EQ(currentIndex, 1+numDense+numCategory);
        } catch(...) {
            BMLOG(INFO, "invalid sample: '%s'", line.c_str());
            return false;
        }
//        show();
        return true;
    }
    void show(){
        std::string s;
        s+= "dense [ ";
        for(auto d: denseFeatures){
            s += std::to_string(d) + " ";
        }
        s+= "] category [ ";
        for(auto c: categoryFeatures){
            s += std::to_string(c) + " ";
        }
        s+="]";
        BMLOG(INFO, "target=%d, %s", target, s.c_str());
    }
};

using DLRMInput = std::vector<DLRMRecord>;

struct DLRMOutput{
    DLRMInput inputs;
    std::vector<float> results;
};

using RunnerType = BMDevicePool<DLRMInput, DLRMOutput>;

bool preProcess(const DLRMInput& inputs, const TensorVec& inTensors, ContextPtr ctx){
    size_t batch = inputs.size();
    if(batch == 0) return false;
    size_t numDense = inputs[0].denseFeatures.size();
    size_t numCategory = inputs[0].categoryFeatures.size();
    size_t numInput = 2+numCategory; //dense offset numCategory

    BM_ASSERT_LE(inputs.size(), inTensors[0]->shape(0)); //batch
    BM_ASSERT_EQ(inTensors.size(), numInput);

    std::vector<float> batchDense(batch*numDense);
    for(size_t b=0; b<batch; b++){
        auto& dense = inputs[b].denseFeatures;
        std::copy(dense.begin(), dense.end(), batchDense.begin() + b*numDense);
    }


    auto offsetBatch = inTensors[0]->shape(0);
    std::vector<unsigned int> offsets(offsetBatch*numCategory);
    for(size_t c=0; c<numCategory; c++){
        for(size_t b=0; b<offsetBatch; b++){
            offsets[c*offsetBatch+b] = b;
        }
    }

    std::vector<std::vector<unsigned int>> batchCategories(numCategory);
    for(size_t c=0; c<numCategory; c++){
        auto& category = batchCategories[c];
        category.resize(batch);
        for(size_t b=0; b<batch; b++){
            category[b] = inputs[b].categoryFeatures[c];
        }
    }
    for(size_t i=0; i<numInput; i++){
        if(i==0){
            inTensors[i]->fill_device_mem(batchDense.data(), batchDense.size()*sizeof(float));
        } else if(i==1){
            inTensors[i]->fill_device_mem(offsets.data(), offsets.size()*sizeof(unsigned int));
        } else {
            inTensors[i]->fill_device_mem(batchCategories[i-2].data(), batchCategories[i-2].size()*sizeof(unsigned int));
        }
    }
    return true;
}

bool postProcess(const DLRMInput& inputs, const TensorVec& outTensors, DLRMOutput& postOut, ContextPtr ctx){
    if(inputs.empty()) return false;
    size_t batch = inputs.size();
    BM_ASSERT_EQ(outTensors.size(), 1);
    auto batchResultData = outTensors[0]->get_float_data();
    postOut.inputs = inputs;
    postOut.results.resize(batch);
    std::copy(batchResultData, batchResultData+batch, postOut.results.begin());
    return true;
}

bool resultProcess(const DLRMOutput& out, std::vector<std::pair<unsigned int, float>>& scores){
    if(out.inputs.empty()) return false;
    auto batch = out.inputs.size();
    static size_t sample_count=0;
    static size_t base_count=0;
    for(size_t i=0; i<batch; i++){
        scores.emplace_back(out.inputs[i].target, out.results[i]);
        sample_count++;
    }
    if(sample_count>10000){
        auto& score = scores.back();
        base_count+= sample_count;
        BMLOG(INFO, "%d: t=%d, i=%f", base_count, score.first, score.second);
        sample_count = 0;
    }
    return true;
}

int main(int argc, char** argv){
    set_env_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath =topDir + "data/criteo/data";
    std::string bmodel = topDir + "models/dlrm/fp32.bmodel";
    std::string dictPrefix = topDir + "data/criteo/dicts/day_fea_dict_";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) dictPrefix = argv[3];
//    globalCategoryDict.resize(26);
//            std::ifstream ifs("../data/criteo/data/simple_data");
//            std::string line;
//            while(std::getline(ifs, line)){
//                DLRMRecord input;
//                input.parse(line, maxIndexRange);
//            }

    loadDict(dictPrefix, numCategory, globalCategoryDict);

    RunnerType runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info(bmodel);
    info.start();
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachFile(dataPath, [&runner, batchSize](const std::string& filename){
            std::ifstream ifs(filename);
            BMLOG(INFO, "processing file %s", filename.c_str());
            std::string line;
            DLRMInput inputs;
            while(std::getline(ifs, line)){
                DLRMRecord input;
                if(!input.parse(line, maxIndexRange)) continue;
                inputs.push_back(std::move(input));
                if(inputs.size()==batchSize){
                    if(!runner.push(std::move(inputs))) return false;
                    inputs.clear();
                }
            }
            if(inputs.size()>0) return runner.push(std::move(inputs));
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
    std::vector<std::pair<unsigned int, float>> scores;
    std::thread resultThread([&runner, &info, &scores](){
        DLRMOutput out;
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
            info.update(status, out.inputs.size());
            if(!resultProcess(out, scores)){
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
    BMLOG(INFO, "--->AUC = %.2f%%", AUC(scores)*100);
    return 0;
}
