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

std::map<std::string, unsigned int> globalCategoryDict;

struct DLRMRecord {
    std::vector<float> denseFeatures;
    std::vector<unsigned int> categoryFeatures;
    unsigned int target;
    bool parse(const std::string& line){
        try{
            auto fields = strSplitByChar(line, '\t');
            // 0: target
            // 1-13: 13 dense features
            // 14-39: 26 category features
            const size_t numDense = 13;
            const size_t numCategory = 26;
            target = atoi(fields[0].c_str());
            denseFeatures.resize(numDense);
            for(size_t i=1; i<=numDense; i++){
                auto field = fields[i];
                if(field == ""){
                    denseFeatures[i-1] = 0;
                } else {
                    denseFeatures[i-1] = log(1.0+atoi(field.c_str()));
                }
            }
            categoryFeatures.resize(numCategory);
            for(size_t i=1+numDense; i<= numCategory+numDense; i++){
                categoryFeatures[i-numDense-1] = globalCategoryDict[fields[i]];
            }
        } catch(...) {
            return false;
        }
        return true;
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

    std::vector<unsigned int> offsets(batch*numCategory);
    for(size_t c=0; c<numCategory; c++){
        for(size_t b=0; b<batch; b++){
            offsets[c*batch+b] = b;
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
            inTensors[i]->fill_device_mem(batchCategories[i-2].data(), batchCategories[i-1].size()*sizeof(unsigned int));
        }
    }
    return true;
}

bool postProcess(const DLRMInput& inputs, const TensorVec& outTensors, DLRMOutput& postOut, ContextPtr ctx){
    if(inputs.empty()) return false;
    size_t batch = inputs.size();
    BM_ASSERT_EQ(outTensors.size(), 1);
    BM_ASSERT_EQ(outTensors[0]->shape(0), batch);
    auto batchResultData = outTensors[0]->get_float_data();
    postOut.inputs = inputs;
    postOut.results.resize(batch);
    std::copy(batchResultData, batchResultData+batch, postOut.results.begin());
    return true;
}

bool resultProcess(const DLRMOutput& out, std::vector<std::pair<unsigned int, float>>& scores){
    if(out.inputs.empty()) return false;
    auto batch = out.inputs.size();
    for(size_t i=0; i<batch; i++){
        scores.emplace_back(out.inputs[i].target, out.results[i]);
    }
    return true;
}

int main(int argc, char** argv){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath =topDir + "data/criteo/";
    std::string bmodel = topDir + "models/dlrm/fp32.bmodel";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];

    RunnerType runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info("squad-bert");
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachFile(dataPath, [&runner, batchSize](const std::string& filename){
            std::ifstream ifs(filename);
            std::string line;
            DLRMInput inputs;
            while(std::getline(ifs, line)){
                DLRMRecord input;
                if(!input.parse(line)) continue;
                inputs.push_back(input);
                if(inputs.size()==batchSize){
                    runner.push(inputs);
                    inputs.clear();
                }
            }
            if(inputs.size()>0) runner.push(inputs);
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

    BMLOG(INFO, "--->AUC = %.2f%%", AUC(scores)*100);
    dataThread.join();
    resultThread.join();
    info.show();
    return 0;
}
