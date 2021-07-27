#include <dirent.h>
#include<vector>
#include<thread>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "bmcv_api.h"

using namespace bm;
using InType = std::vector<std::string>;
using ClassId = size_t;

struct PostOutType {
    InType rawIns;
    std::vector<ClassId> classes;
    std::vector<float> scores;
};

struct InceptionConfig {
    bool initialized = false;
    bool isNCHW;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;
    // use static to cache resizedImage, to avoid allocating memory everytime
    std::vector<bm_image> resizedImages;
    // bmcv_image_convert_to do not support RGB_PACKED format directly
    // use grayImages as a RGB_PACKED wrapper
    std::vector<bm_image> grayImages;
    // used as a wrapper of input tensor
    std::vector<bm_image> preOutImages;
    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        netBatch = inTensor->shape(0);
        isNCHW = inTensor->shape(1) == 3;
        netHeight = inTensor->shape(2);
        netWidth = inTensor->shape(3);
        netFormat = FORMAT_RGB_PLANAR; // for NHWC input
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        }
        float input_scale = inTensor->get_scale();
        float scale = 1.0/255 * 2.0;
        float bias = -1;
        float real_scale = scale * input_scale;
        float real_bias = bias * input_scale;
        ConvertAttr.alpha_0 = real_scale;
        ConvertAttr.beta_0 = real_bias;
        ConvertAttr.alpha_1 = real_scale;
        ConvertAttr.beta_1 = real_bias;
        ConvertAttr.alpha_2 = real_scale;
        ConvertAttr.beta_2 = real_bias;
        resizedImages = ctx->allocAlignedImages(
                    netBatch, netHeight, netWidth, netFormat, DATA_TYPE_EXT_1N_BYTE);
        if(!isNCHW){

            grayImages = ctx->allocImagesWithoutMem(
                        netBatch, netHeight, netWidth*3, FORMAT_GRAY, DATA_TYPE_EXT_1N_BYTE, 64);
            bm_device_mem_t resizedMem;
            bm_image_get_device_mem(resizedImages[0], &resizedMem);
            bm_image_attach_contiguous_mem(grayImages.size(), grayImages.data(), resizedMem);
            preOutImages = ctx->allocImagesWithoutMem(
                        netBatch, netHeight, netWidth*3, FORMAT_GRAY, netDtype);
        } else {
            preOutImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth, netFormat, netDtype);
        }
    }
};

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    thread_local static InceptionConfig cfg;
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    cfg.initialize(inTensor, ctx);

    std::vector<bm_image> alignedInputs;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs.push_back(image);
    }

    centralCropAndResize(ctx->handle, alignedInputs, cfg.resizedImages);

    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);

    if(cfg.isNCHW){
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.resizedImages.data(), cfg.preOutImages.data());
    } else {
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.grayImages.data(), cfg.preOutImages.data());
    }

    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    postOut.rawIns = rawIn;
    auto outTensor = outTensors[0];
    float* data = outTensor->get_float_data();
    size_t batch = outTensor->shape(0);
    size_t len = outTensor->shape(1);
    postOut.classes.resize(batch);
    postOut.scores.resize(batch);
    //outTensor->dumpData("out.txt");
    for(size_t b=0; b<batch; b++){
        auto& score = postOut.scores[b];
        auto& cls = postOut.classes[b];
        float* allScores = data+b*len;
        score = allScores[0];
        cls = 0;
        for(size_t l=1; l<len; l++){
            if(score<allScores[l]) {
                score= allScores[l];
                cls=l;
            }
        }
    }
    return true;
}

bool resultProcess(const PostOutType& out, std::map<size_t, std::string>& labelMap){
    if(out.rawIns.empty()) return false;
    BM_ASSERT_EQ(out.rawIns.size(), out.classes.size());
    for(size_t i=0; i<out.rawIns.size(); i++){
        BMLOG(INFO, "%s: class_id=%d: score=%f: label=%s", out.rawIns[i].c_str(), out.classes[i], out.scores[i],
              labelMap[out.classes[i]].c_str());
    }
    return true;
}


int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string dataPath = "../dataset";
    std::string bmodel = "../inception_fp32/compilation.bmodel";
    std::string labelFile = "../inception_labels.txt";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) labelFile = argv[2];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    auto labelMap = loadLabels(labelFile);

    StatInfo info("inception");
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const InType& imageFiles){
            runner.push(imageFiles);
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
    std::thread resultThread([&runner, &labelMap, &info](){
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
//            status->show();
            info.update(status);
            if(!resultProcess(out, labelMap)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                info.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

