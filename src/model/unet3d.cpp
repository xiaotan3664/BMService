#include <dirent.h>
#include<vector>
#include<thread>
#include<cstdio>
#include<sys/stat.h>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "bmcv_api.h"

using namespace bm;
using InType = std::vector<std::string>;
using ClassId = size_t;

#define OUTPUT_DIR "unet3d_out"

struct PostOutType {
    InType rawIns;
    std::vector<std::string> outFiles;
};

struct UNet3DConfig {
    bool initialized = false;
    size_t memSize;
    size_t netBatch;
    bm_image_data_format_ext netDtype;
    unsigned char* buffer;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        netBatch = inTensor->shape(0);
        memSize = inTensor->get_mem_size()/netBatch;
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        }
        // currently we just support fp32 input
        BM_ASSERT_EQ(netDtype, DATA_TYPE_EXT_FLOAT32);
        buffer = new unsigned char[memSize];
    }
    ~UNet3DConfig() {
        delete [] buffer;
    }
};

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    thread_local static UNet3DConfig cfg;
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 5);
    cfg.initialize(inTensor, ctx);

    for(size_t i=0; i<in.size(); i++){
        auto name = in[i];
        FILE* fp = fopen(name.c_str(), "rb");
        auto readLen = fread(cfg.buffer, cfg.memSize, 1, fp);
        BM_ASSERT_EQ(readLen, cfg.memSize);
        inTensor->fill_device_mem(cfg.buffer, cfg.memSize, i*cfg.memSize);
        fclose(fp);
    }

    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    BM_ASSERT_EQ(outTensors.size(), 1);
    postOut.rawIns = rawIn;
    auto outTensor = outTensors[0];
    size_t batch = rawIn.size();
    unsigned char* data = outTensor->get_raw_data();
    size_t len = outTensor->get_mem_size();
    size_t block = len/batch;
    BM_ASSERT_EQ(len%batch, 0);
    for(size_t i=0; i<rawIn.size(); i++){
        std::string outName = OUTPUT_DIR;
        outName += baseName(rawIn[i]) + ".out";
        FILE* fp = fopen(outName.c_str(), "wb");
        fwrite(data+i*block, block, 1, fp);
        fclose(fp);
        postOut.outFiles.push_back(outName);
    }
    return true;
}


bool resultProcess(const PostOutType& out){
    if(out.rawIns.empty()) return false;
    for(size_t i=0; i<out.rawIns.size(); i++){
        BMLOG(INFO, "processed %s -> %s", out.rawIns[i].c_str(), out.outFiles[i].c_str());
    }
    return true;
}


int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/preprocessed_data";
    std::string bmodel = topDir + "models/unet3d/fp32.bmodel";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    mkdir(OUTPUT_DIR, 0777);

    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info("unet3d");
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const InType& imageFiles){
            return runner.push(imageFiles);
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
            info.update(status, out.rawIns.size());

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

