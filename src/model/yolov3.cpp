#include <dirent.h>
#include<vector>
#include<map>
#include<thread>
#include<sys/stat.h>
#include <algorithm>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "BMDetectUtils.h"
#include "bmcv_api.h"

using namespace bm;
#define OUTPUT_DIR "out"
std::map<size_t, std::string> globalLabelMap;

struct YOLOv3Config {
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

    float probThreshold;
    float iouThreshold;
    const size_t classNum = 80;
    std::string savePath = OUTPUT_DIR;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        ctx->setConfigData(this);
        netBatch = inTensor->shape(0);
        isNCHW = inTensor->shape(1) == 3;
        netHeight = inTensor->shape(2);
        netWidth = inTensor->shape(3);
        netFormat = FORMAT_RGB_PLANAR; // for NHWC input
        float input_scale = 1.0;
        if(inTensor->get_dtype() == BM_FLOAT32){
            netDtype = DATA_TYPE_EXT_FLOAT32;
            probThreshold = 0.5;
            iouThreshold = 0.45;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            probThreshold = 0.2;
            iouThreshold = 0.45;
            input_scale = inTensor->get_scale();
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        }
        float scale = 1.0/255;
        float bias = 0;
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


using InType = std::vector<std::string>;

struct PostOutType {
    std::vector<std::string> rawIns;
    std::vector<std::vector<DetectBox>> results;
};

using RunnerType = BMDevicePool<InType, PostOutType>;

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);
    BM_ASSERT_EQ(inTensor->shape(3), 3);

    thread_local static YOLOv3Config cfg;
    cfg.initialize(inTensor, ctx);

    auto alignedInputs = new std::vector<bm_image>;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs->push_back(image);
    }
    bmcv_color_t color = {128, 128, 128};

    aspectScaleAndPad(ctx->handle, *alignedInputs, cfg.resizedImages, color);
    saveImage(cfg.resizedImages[0], "resize.jpg");

    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);

    if(cfg.isNCHW){
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.resizedImages.data(), cfg.preOutImages.data());
    } else {
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.grayImages.data(), cfg.preOutImages.data());
    }
    // pass input info to post process
    ctx->setPreExtra(alignedInputs);
    return true;
}

struct CoordConvertInfo {
    size_t inputWidth;
    size_t inputHeight;
    float ioRatio;
    float oiRatio;
    float hOffset;
    float wOffset;
};

bool yoloV3BoxParse(DetectBox& box, float* data, size_t len, float probThresh, CoordConvertInfo& ci){
    size_t classNum = len - 5;
    auto scores = &data[5];
    box.category = argmax(scores, classNum);
    box.confidence = data[4] * scores[box.category];
    if(box.confidence <= probThresh){
        return false;
    }
    // (cx,cy,w,h) -> (xmin, ymin, xmax, ymax)
    box.xmin = data[0] - data[2]*.5;
    box.xmax = data[0] + data[2]*.5;
    box.ymin = data[1] - data[3]*.5;
    box.ymax = data[1] + data[3]*.5;

    // restore original coordinates
    box.xmin = (box.xmin - ci.wOffset)* ci.ioRatio;
    box.xmax = (box.xmax - ci.wOffset)* ci.ioRatio;
    box.ymin = (box.ymin - ci.hOffset)* ci.ioRatio;
    box.ymax = (box.ymax - ci.hOffset)* ci.ioRatio;

    // filter some invalid boxes by confidence
    if(box.xmin >= box.xmax ||
            box.ymin >= box.ymax ||
            box.xmin<0 || box.xmax >= ci.inputWidth ||
            box.ymin<0 || box.ymax >= ci.inputHeight) {
        return false;
    }
    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    postOut.rawIns = rawIn;
    auto pCfg = (YOLOv3Config*)ctx->getConfigData();
    auto& cfg = *pCfg;

    auto pInputImages = reinterpret_cast<std::vector<bm_image>*>(ctx->getPostExtra());
    auto realBatch = rawIn.size();
    auto outTensor = outTensors[0];
    size_t batch = outTensor->shape(0);
    std::vector<size_t> boxNums(outTensors.size());
    size_t totalBoxNum=0;
    size_t classNum = outTensor->shape(4)-5;
    auto singleDataSize = outTensor->shape(4);
    BM_ASSERT_EQ(classNum, cfg.classNum);
    BM_ASSERT_EQ(batch, pInputImages->size());

    std::vector<CoordConvertInfo> coordInfos(batch);
    for(size_t b=0; b<batch; b++){
        auto& image = (*pInputImages)[b];
        auto& ci = coordInfos[b];

        ci.inputWidth = image.width;
        ci.inputHeight = image.height;
        ci.ioRatio = std::max((float)ci.inputWidth/cfg.netWidth,
                               (float)ci.inputHeight/cfg.netHeight);
        ci.oiRatio = 1/ci.ioRatio;
        ci.hOffset = (cfg.netHeight - ci.oiRatio * ci.inputHeight)/2;
        ci.wOffset = (cfg.netWidth - ci.oiRatio * ci.inputWidth)/2;
    }

    for(size_t i=0; i<outTensors.size(); i++){
        auto tensor = outTensors[i];
        BM_ASSERT_EQ(batch, tensor->shape(0));
        BM_ASSERT_EQ(classNum, tensor->shape(4)-5);
        boxNums[i] = tensor->partial_shape_count(1,3);
        totalBoxNum += boxNums[i];
    }
    std::vector<std::vector<DetectBox>> batchBoxInfos(batch, std::vector<DetectBox>(totalBoxNum));

    // fill batchBoxInfo
    std::vector<int> batchIndice(batch, 0);
    for(size_t i=0; i<outTensors.size(); i++){
        auto rawData = outTensors[i]->get_float_data();
        auto boxNum = boxNums[i];
        for(size_t b=0; b<batch; b++){
            auto& ci = coordInfos[b];
            for(size_t n=0; n<boxNum; n++){
                auto rawOffset = singleDataSize *(b*boxNum + n);
                auto rawBoxData = rawData + rawOffset;
                auto& boxInfo = batchBoxInfos[b][batchIndice[b]];
                auto scores = &rawBoxData[5];
                if(yoloV3BoxParse(boxInfo, rawBoxData, singleDataSize, cfg.probThreshold, ci)){
                        batchIndice[b]++;
                }
            }
        }
    }
    for(size_t b=0; b<batch; b++){
        batchBoxInfos[b].resize(batchIndice[b]);
    }

    // final results
    postOut.results = batchNMS(batchBoxInfos, cfg.iouThreshold);

    // draw rectangle
    for(size_t b=0; b<batch; b++){
        auto name = baseName(rawIn[b]);
        drawDetectBox(pInputImages->at(b), postOut.results[b], cfg.savePath+"/"+name, globalLabelMap);
    }

    // clear extra data
    for(size_t i=0; i<pInputImages->size(); i++) {
        bm_image_destroy(pInputImages->at(i));
    }
    delete pInputImages;
    return true;
}

bool resultProcess(const PostOutType& out){
    if(out.rawIns.empty()) return false;
    return true;
}

int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/coco/images";
    std::string bmodel = topDir + "models/yolov3/fp32.bmodel";
    std::string refFile = topDir+ "data/coco/val.txt";
    std::string labelFile = topDir + "data/coco/coco.names";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) refFile = argv[3];
    if(argc>4) labelFile = argv[4];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    mkdir(OUTPUT_DIR, 0777);
    globalLabelMap = loadLabels(labelFile);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info("yolov3");
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const std::vector<std::string> names){
            runner.push(names);
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
                info.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

