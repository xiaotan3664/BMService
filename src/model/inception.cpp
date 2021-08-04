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
    std::vector<std::vector<std::pair<ClassId, float>>> classAndScores;
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
            bm_image_get_contiguous_device_mem(resizedImages.size(), resizedImages.data(), &resizedMem);
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
    for(auto &image: alignedInputs) {
        bm_image_destroy(image);
    }
    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    const size_t K=5;
    postOut.rawIns = rawIn;
    auto outTensor = outTensors[0];
    float* data = outTensor->get_float_data();
    size_t batch = outTensor->shape(0);
    size_t len = outTensor->shape(1);

    postOut.classAndScores.resize(batch);
    for(size_t b=0; b<batch; b++){
        float* allScores = data+b*len;
        postOut.classAndScores[b] = topk(allScores, len, K);
    }
    return true;
}

struct Top5AccuracyStat {
    size_t samples=0;
    size_t top1=0;
    size_t top5=0;
    void show() {
        BMLOG(INFO, "Accuracy: top1=%g%%, top5=%g%%", 100.0*top1/samples, 100.0*top5/samples);
    }
};

bool resultProcess(const PostOutType& out, Top5AccuracyStat& stat,
                   std::map<std::string, size_t>& refMap,
                   std::map<size_t, std::string>& labelMap){
    if(out.rawIns.empty()) return false;
    BM_ASSERT_EQ(out.rawIns.size(), out.classAndScores.size());
    for(size_t i=0; i<out.rawIns.size(); i++){
        auto& inName = out.rawIns[i];
        auto realClass = refMap[out.rawIns[i]];
        auto& classAndScores = out.classAndScores[i];
        auto firstClass = classAndScores[0].first-1;
        auto firstScore = classAndScores[0].second;
        stat.samples++;
        stat.top1 += firstClass == realClass;
        for(auto& cs: classAndScores){
            if(cs.first-1 == realClass){
                stat.top5++;
                break;
            }
        }
        BMLOG(INFO, "%s: infer_class=%d: score=%f: real_class=%d: label=%s",
              out.rawIns[i].c_str(), firstClass, firstScore,
              realClass, labelMap[realClass].c_str());
    }
    return true;
}


int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/ILSVRC2012/images";
    std::string bmodel = topDir + "models/inception/fp32.bmodel";
    std::string refFile = topDir+ "data/ILSVRC2012/val.txt";
    std::string labelFile = topDir + "data/ILSVRC2012/labels.txt";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) refFile = argv[3];
    if(argc>4) labelFile = argv[4];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    std::string prefix = dataPath;
    if(prefix[prefix.size()-1] != '/'){
        prefix += "/";
    }
    auto refMap = loadClassRefs(refFile, prefix);
    auto labelMap = loadLabels(labelFile);
    ProcessStatInfo info("inception");
    Top5AccuracyStat topStat;
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
    std::thread resultThread([&runner, &refMap, &labelMap, &info](){
        PostOutType out;
        std::shared_ptr<ProcessStatus> status;
        bool stopped = false;
        Top5AccuracyStat stat;
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

            if(!resultProcess(out, stat, refMap, labelMap)){
                runner.stop(status->deviceId);
            }
            if(runner.allStopped()){
                info.show();
                stat.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

