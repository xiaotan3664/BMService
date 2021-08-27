#include <dirent.h>
#include<vector>
#include<map>
#include<thread>
#include<sys/stat.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iterator>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "BMDetectUtils.h"
#include "bmcv_api.h"

using namespace bm;
#define OUTPUT_DIR "out"
#define OUTPUT_IMAGE_DIR  OUTPUT_DIR "/images"
#define OUTPUT_PREDICTION_DIR  OUTPUT_DIR "/prediction"
#define OUTPUT_GROUND_TRUTH_DIR OUTPUT_DIR "/groundtruth"
std::map<size_t, std::string> globalLabelMap;
std::map<std::string, std::vector<DetectBox>> globalGroundTruth;
std::set<std::string> globalLabelSet;

struct PriorBox {
    float cx;
    float cy;
    float w;
    float h;
};
std::ostream& operator<< (std::ostream& os, const PriorBox& box){
    os<<"["<<box.cx<<","<<box.cy<<","<<box.w<<","<<box.h<<"]";
    return os;
}

struct ImageShape {
ImageShape(float h, float w):h(h),w(w) {}
ImageShape():h(0),w(0) {}
    float h;
    float w;
};

std::vector<PriorBox> genPriorBox(const ImageShape& imageShape,
                                  const std::vector<ImageShape>& layerShapes,
                                  const std::vector<ImageShape>& layerSteps,
                                  const std::vector<std::vector<float>>& layerScales,
                                  const std::vector<std::vector<float>>& layerExtraScales,
                                  const std::vector<std::vector<float>>& layerRatios,
                                  float offset = 0.5) {
    std::vector<PriorBox> boxes;
    auto layerNum = layerShapes.size();
    for(size_t l=0; l<layerNum; l++){
        auto scales = layerScales[l];
        auto extraScales = layerExtraScales[l];
        auto ratios = layerRatios[l];
        std::vector<ImageShape> boxShapes;
        for(size_t s=0; s<scales.size(); s++){
            ImageShape shape;
            shape.h = scales[s];
            shape.w = scales[s];
            boxShapes.push_back(shape);
            auto meanScale = sqrt(scales[s]*extraScales[s]);
            shape.h = meanScale;
            shape.w = meanScale;
            boxShapes.push_back(shape);
        }
        for(auto s: scales){
            for(auto r: ratios){
                ImageShape shape;
                shape.h = s/sqrt(r);
                shape.w = s*sqrt(r);
                boxShapes.emplace_back(shape);
                std::swap(shape.h, shape.w);
                boxShapes.emplace_back(shape);
            }
        }
        auto& layerShape  = layerShapes[l];
        auto& layerStep = layerSteps[l];
        for(auto shape: boxShapes){
            for(size_t h=0; h<layerShape.h; h++){
                for(size_t w=0; w<layerShape.w; w++){
                    PriorBox box;
                    box.cx = (w+offset)*layerStep.w/imageShape.w;
                    box.cy = (h+offset)*layerStep.h/imageShape.h;
                    box.h = shape.h;
                    box.w = shape.w;
                    boxes.push_back(box);
                }
            }
        }
    }
    return boxes;
}

struct SSDResnet34Config {
    bool initialized = false;
    bool isNCHW;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    std::vector<PriorBox> priorBoxes;

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
    size_t nmsTopK = 200;
    size_t maxKeepBoxNum = 200;
    const size_t classNum = 81;
    std::string savePath = OUTPUT_IMAGE_DIR;
    std::vector<float> priorScales;

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
            iouThreshold = 0.5;
        } else {
            netDtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
            probThreshold = 0.5;
            iouThreshold = 0.5;
            input_scale = inTensor->get_scale();
        }
        if(!isNCHW){
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
            netFormat = FORMAT_RGB_PACKED;
            BM_ASSERT_EQ(inTensor->shape(3), 3);
        }

        // y=(x/255.0-mean)/std
        float scale_R = (1.0/255.0)/0.229;
        float scale_G = (1.0/255.0)/0.224;
        float scale_B = (1.0/255.0)/0.225;
        float bias_R = -0.485/0.229;
        float bias_G = -0.456/0.224;
        float bias_B = -0.406/0.225;

        float real_scale_R = scale_R * input_scale;
        float real_scale_G = scale_G * input_scale;
        float real_scale_B = scale_B * input_scale;
        float real_bias_R = bias_R * input_scale;
        float real_bias_G = bias_G * input_scale;
        float real_bias_B = bias_B * input_scale;
        ConvertAttr.alpha_0 = real_scale_R;
        ConvertAttr.beta_0 = real_bias_R;
        ConvertAttr.alpha_1 = real_scale_G;
        ConvertAttr.beta_1 = real_bias_G;
        ConvertAttr.alpha_2 = real_scale_B;
        ConvertAttr.beta_2 = real_bias_B;

        priorScales = {0.1, 0.1, 0.2, 0.2};
        ImageShape imageShape{1200,1200};
        std::vector<ImageShape> layerShapes = {{50,50}, {25,25}, {13,13}, {7,7}, {3,3}, {3,3}};
        std::vector<ImageShape> layerSteps = {{24,24}, {48,48}, {92,92}, {171,171}, {400,400}, {400,400}};
        std::vector<std::vector<float>> scales ={{0.07}, {0.15}, {0.33}, {0.51}, {0.69}, {0.87}};
        std::vector<std::vector<float>> extraScales = {{0.15}, {0.33}, {0.51}, {0.69}, {0.87}, {1.05}};
        std::vector<std::vector<float>> ratios = {{2}, {2,3}, {2,3}, {2,3}, {2}, {2}};

        BM_ASSERT_EQ(netHeight, imageShape.h);
        BM_ASSERT_EQ(netWidth, imageShape.w);

        priorBoxes = genPriorBox(imageShape, layerShapes, layerSteps, scales, extraScales, ratios, 0.5);

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
    //BM_ASSERT_EQ(inTensor->shape(3), 3);

    thread_local static SSDResnet34Config cfg;
    cfg.initialize(inTensor, ctx);

    auto alignedInputs = new std::vector<bm_image>;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs->push_back(image);
    }

    centralCropAndResize(ctx->handle, *alignedInputs, cfg.resizedImages, 1.0);
//    saveImage(cfg.resizedImages[0], "resize.jpg");

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

void center2Point(std::vector<DetectBox>& boxes){
    for(auto& box: boxes){
        box.xmin = box.xmin-box.xmax/2;
        box.ymin = box.ymin-box.ymax/2;
        box.xmax = box.xmin+box.xmax;
        box.ymax = box.ymin+box.ymax;
    }
}
void clipBoxs(std::vector<DetectBox>& boxes){
    for(auto& box: boxes){
        box.xmin = std::max(box.xmin,0.0f);
        box.ymin = std::max(box.ymin,0.0f);
        box.xmax = std::min(box.xmax,1.0f);
        box.ymax = std::min(box.ymax,1.0f);
        box.xmin = std::min(box.xmin, box.xmax);
        box.ymin = std::min(box.ymin, box.ymax);
    }
}

void softmax(float* scores, const int* shape, const size_t dim, int axis){
    size_t outer = 1;
    size_t inner = 1;
    while(axis<0) axis+=dim;
    for(size_t i=0; i<dim; i++){
        if(i<axis){ outer *= shape[i]; }
        if(i>axis){ inner *= shape[i]; }
    }
    auto len = shape[axis];
    for(size_t x=0; x<outer; x++){
        for(size_t y=0; y<inner; y++){
            size_t base = x*inner*len + y;
            float sum = 0;
            for(size_t z =0; z<len; z++){
                size_t offset = base + z*inner;
                scores[offset] = exp(scores[offset]);
                sum += scores[offset];
            }
            for(size_t z =0; z<len; z++){
                size_t offset = base + z*inner;
                scores[offset]/=sum;
            }
        }
    }
}

// shape: rawBoxData [batch, 4, boxNum], rawScoreData [boxNum]
std::vector<DetectBox> decodeBoxes(size_t batch, const float* rawBoxData,
                                   const std::vector<PriorBox>& anchorBox,
                                   const std::vector<float>& priorScales) {
    auto boxNum = anchorBox.size();
    std::vector<DetectBox> boxes(batch*boxNum);
    for(size_t b=0; b<batch; b++){
        auto boxOffset = b*boxNum;
        auto boxData = rawBoxData + b*4*boxNum;
        auto dataOffset = b*4*boxNum;
        // consider cpu cache
        auto locOffset = 0 * boxNum + dataOffset;
        for(size_t i=0; i<boxNum; i++){
            // decode location
            boxes[boxOffset+i].xmin = (boxData[locOffset+i] * anchorBox[i].w* priorScales[0] + anchorBox[i].cx);
        }
        locOffset = 1 * boxNum + dataOffset;
        for(size_t i=0; i<boxNum; i++){
            boxes[boxOffset+i].ymin = (boxData[locOffset+i] * anchorBox[i].h* priorScales[1] + anchorBox[i].cy);
        }
        locOffset = 2 * boxNum + dataOffset;
        for(size_t i=0; i<boxNum; i++){
            boxes[boxOffset+i].xmax = exp(boxData[locOffset+i] * priorScales[2]) * anchorBox[i].w;
        }
        locOffset = 3 * boxNum + dataOffset;
        for(size_t i=0; i<boxNum; i++){
            boxes[boxOffset+i].ymax = exp(boxData[locOffset+i] * priorScales[3]) * anchorBox[i].h;
        }
    }
    center2Point(boxes);
    return boxes;
}

// shape: boxes [batch*boxNum], rawScoreData [batch, classNum, boxNum]
// out class:(batch*boxes)
std::map<size_t, std::vector<std::vector<DetectBox>>> selectBoxes(
        const std::vector<DetectBox>& boxes, const float* rawScoreData,
        size_t batch, size_t classNum, size_t boxNum, float selectThresh) {
    std::map<size_t, std::vector<std::vector<DetectBox>>> result;
    for(size_t b=0; b<batch; b++){
        auto batchScoreOffset = b*classNum*boxNum;
        auto boxOffset = b*boxNum;
        for(size_t c=1; c<classNum; c++){
            result[c].resize(batch);
            auto& validBoxes = result[c][b];
            auto scoreData = rawScoreData + batchScoreOffset + c*boxNum;
            for(size_t n=0; n<boxNum; n++){
                auto& box = boxes[boxOffset + n];
                if(!box.isValid(1.0, 1.0)) continue;
                if(scoreData[n]<=selectThresh) continue;
                validBoxes.push_back(boxes[boxOffset + n]);
                validBoxes.back().confidence = scoreData[n];
                validBoxes.back().category = c;
                validBoxes.back().categoryName = globalLabelMap[c-1];
            }
        }
    }
    return result;
}

std::vector<DetectBox> sortBoxes(const std::vector<DetectBox>& boxes, size_t num_keep){
   return topkValues(boxes.data(), boxes.size(), num_keep);
}

std::vector<std::vector<DetectBox>> batchSortBoxes(const std::vector<std::vector<DetectBox>>& batchBoxes, size_t num_keep){
    std::vector<std::vector<DetectBox>> batchResult(batchBoxes.size());
    std::transform(batchBoxes.begin(), batchBoxes.end(), batchResult.begin(), [num_keep](const std::vector<DetectBox>& boxes) { return sortBoxes(boxes, num_keep); });
    return batchResult;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    postOut.rawIns = rawIn;
    auto pCfg = (SSDResnet34Config*)ctx->getConfigData();
    auto& cfg = *pCfg;
    BM_ASSERT_EQ(outTensors.size(),2);

    auto pInputImages = reinterpret_cast<std::vector<bm_image>*>(ctx->getPostExtra());
    size_t batch =    outTensors[0]->shape(0);
    size_t classNum = outTensors[0]->shape(1);
    size_t boxNum =   outTensors[0]->shape(2);
    BM_ASSERT_EQ(outTensors[1]->shape(0), batch);
    BM_ASSERT_EQ(outTensors[1]->shape(1), 4);
    BM_ASSERT_EQ(outTensors[1]->shape(2), boxNum);

    // [batch, classNum, boxNum]
    auto scoreData = outTensors[0]->get_float_data();
    // [batch, 4, boxNum]
    auto rawBoxData = outTensors[1]->get_float_data();

    // decode boxes to [0, 1]
    auto boxes = decodeBoxes(batch, rawBoxData, cfg.priorBoxes, cfg.priorScales);

    // softmax scores
    auto scoreShape = outTensors[0]->get_shape();
    softmax(scoreData, scoreShape->dims, scoreShape->num_dims, 1);

    auto classifiedBoxes = selectBoxes(boxes, scoreData, batch, classNum, boxNum, cfg.probThreshold);

    // NMS for result box
    auto& batchResult = postOut.results;
    batchResult.resize(batch);
    for(auto& r: batchResult){
        r.reserve(cfg.nmsTopK*classNum);
    }
    for(auto& cb: classifiedBoxes) {
        cb.second = batchNMS(cb.second, cfg.iouThreshold, cfg.nmsTopK);
        for(size_t b=0; b<batch; b++){
            auto& result = batchResult[b];
            result.insert(result.end(), cb.second[b].begin(), cb.second[b].end());
        }
    }
    for(size_t b=0; b<batch; b++){
        auto& result = batchResult[b];
        result = topkValues(result.data(), result.size(), cfg.maxKeepBoxNum);
        auto inputHeight = pInputImages->at(b).height;
        auto inputWidth = pInputImages->at(b).width;
        for(auto& r: result){
            r.xmin *= inputWidth;
            r.xmax *= inputWidth;
            r.ymin *= inputHeight;
            r.ymax *= inputHeight;
        }
    }

    // draw rectangle
    for(size_t b=0; b<batch; b++){
        auto name = baseName(rawIn[b]);
        drawDetectBoxEx(pInputImages->at(b), batchResult[b], globalGroundTruth[name], cfg.savePath+"/"+name);
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
    auto imageNum = out.rawIns.size();
    for(size_t i=0; i<imageNum; i++){
        auto name = baseName(out.rawIns[i]);
        if(!globalGroundTruth.count(name)) {
            BMLOG(WARNING, "cannot find %s in ground true info", name.c_str());
            continue;
        }
        std::string predName = OUTPUT_PREDICTION_DIR "/";
        predName += name+ ".txt";
        std::ofstream os(predName);
        auto& boxes = out.results[i];
        for(auto& box: boxes){
            os<<box<<std::endl;
        }

        std::string gtName = OUTPUT_GROUND_TRUTH_DIR "/";
        gtName += name + ".txt";
        std::ofstream gtOs(gtName);
        auto& gtBoxes = globalGroundTruth.at(name);
        for(auto& box: gtBoxes){
            if(!globalLabelSet.count(box.categoryName)) {
                BMLOG(WARNING, "current prediction does not cover category %s", box.categoryName.c_str());
                continue;
            }
            gtOs<<box<<std::endl;
        }
    }
    return true;
}

int main(int argc, char* argv[]){
    set_log_level(INFO);
    std::string topDir = "../";
    std::string dataPath = topDir + "data/coco/images";
    std::string bmodel = topDir + "models/ssd_resnet34/fp32.bmodel";
    std::string refFile = topDir+ "data/coco/instances_val2017.json";
    std::string labelFile = topDir + "data/coco/coco_val2017.names";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) refFile = argv[3];
    if(argc>4) labelFile = argv[4];

    mkdir(OUTPUT_DIR, 0777);
    mkdir(OUTPUT_IMAGE_DIR, 0777);
    mkdir(OUTPUT_PREDICTION_DIR, 0777);
    mkdir(OUTPUT_GROUND_TRUTH_DIR, 0777);
    globalLabelMap = loadLabels(labelFile);
    for(auto &p: globalLabelMap) globalLabelSet.insert(p.second);
    globalGroundTruth =  readCocoDatasetBBox(refFile);

    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    ProcessStatInfo info("ssdresnet34");
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const std::vector<std::string> names){
            return runner.push(names);
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
                info.show();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

