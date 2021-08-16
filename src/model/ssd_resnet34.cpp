#include <dirent.h>
#include<vector>
#include<map>
#include<thread>
#include<sys/stat.h>
#include <algorithm>
#include <fstream>
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

struct SSDResnet34Config {
    bool initialized = false;
    bool isNCHW;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;

    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;
    bmcv_convert_to_attr ConvertAttrMean;
    bmcv_convert_to_attr ConvertAttrStd;
    // use static to cache resizedImage, to avoid allocating memory everytime
    std::vector<bm_image> resizedImages;
    // bmcv_image_convert_to do not support RGB_PACKED format directly
    // use grayImages as a RGB_PACKED wrapper
    std::vector<bm_image> grayImages;
    // used as a wrapper of input tensor
    std::vector<bm_image> preOutImages;

    float probThreshold;
    float iouThreshold;
    const size_t classNum = 81;
    std::string savePath = OUTPUT_IMAGE_DIR;

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

        float alpha = 1.0;
        float mean_R = -0.485;
        float mean_G = -0.456;
        float mean_B = -0.406;
        float real_alpha = alpha * input_scale;
        float real_mean_R = mean_R * input_scale;
        float real_mean_G = mean_G * input_scale;
        float real_mean_B = mean_B * input_scale;
        ConvertAttrMean.alpha_0 = real_alpha;
        ConvertAttrMean.beta_0 = real_mean_R;
        ConvertAttrMean.alpha_1 = real_alpha;
        ConvertAttrMean.beta_1 = real_mean_G;
        ConvertAttrMean.alpha_2 = real_alpha;
        ConvertAttrMean.beta_2 = real_mean_B;

        float std_R = 1.0/0.229;
        float std_G = 1.0/0.224;
        float std_B = 1.0/0.225;
        float beta = 0;
        float real_std_R = std_R * input_scale;
        float real_std_G = std_G * input_scale;
        float real_std_B = std_B * input_scale;
        float real_beta = beta * input_scale;
        ConvertAttrStd.alpha_0 = real_std_R;
        ConvertAttrStd.beta_0 = real_beta;
        ConvertAttrStd.alpha_1 = real_std_G;
        ConvertAttrStd.beta_1 = real_beta;
        ConvertAttrStd.alpha_2 = real_std_B;
        ConvertAttrStd.beta_2 = real_beta;

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
    bmcv_color_t color = {128, 128, 128};

    aspectScaleAndPad(ctx->handle, *alignedInputs, cfg.resizedImages, color);
    //aspectResize(ctx->handle, *alignedInputs, cfg.resizedImages);
    saveImage(cfg.resizedImages[0], "resize.jpg");

    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);

    if(cfg.isNCHW){
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.resizedImages.data(), cfg.preOutImages.data());
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttrMean, cfg.preOutImages.data(), cfg.preOutImages.data());
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttrStd, cfg.preOutImages.data(), cfg.preOutImages.data());
    } else {
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.grayImages.data(), cfg.preOutImages.data());
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttrMean, cfg.preOutImages.data(), cfg.preOutImages.data());
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttrStd, cfg.preOutImages.data(), cfg.preOutImages.data());
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

struct Prior_Box {
  float anchor_cx;
  float anchor_cy;
  float anchor_w;
  float anchor_h;
};

bool get_prior_box(std::vector<Prior_Box> &prior_box) {

    int layer_num = 6;
    int ratio = 2;
    int ratio_six = 3;
    float offset = 0.5;
    int image_shape[2] = {1200, 1200};
    int shapes[layer_num][2] = {{50, 50}, {25, 25}, {13, 13}, {7, 7}, {3, 3}, {3, 3}};
    float anchor_scales[layer_num] = {0.07, 0.15, 0.33, 0.51, 0.69, 0.87};
    float extra_anchor_scales[layer_num] = {0.15, 0.33, 0.51, 0.69, 0.87, 1.05};
    int layer_steps[layer_num] = {24, 48, 92, 171, 400, 400};
    int num[layer_num] = {4, 6, 6, 6, 4, 4};

    int box_per_layer[layer_num] = {0};
    for (size_t i = 0; i < layer_num; i++) {
      box_per_layer[i] = shapes[i][0]*shapes[i][1]*num[i];
    }

    int priorbox_num = 0;
    for (size_t i = 0; i < layer_num; i++) {
      priorbox_num += box_per_layer[i];
    }

    int box_num_acc= 0;

    //float image_cx[priorbox_num] = {0.0};
    //float image_cy[priorbox_num] = {0.0};

    //vector<Prior_Box> prior_box(priorbox_num);

    for (size_t l = 0; l < layer_num; l++) {
      if (l != 0) box_num_acc += box_per_layer[l-1];
      if (num[l] == 4) {
        for (size_t i = 0; i < shapes[l][0]*shapes[l][1]; i++) {
          prior_box[box_num_acc + 4 * i].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 1].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 1].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 2].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 2].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 3].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 3].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
        }
      } else {
        for (size_t i = 0; i < shapes[l][0]*shapes[l][1]; i++) {
          prior_box[box_num_acc + 4 * i].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 1].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 1].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 2].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 2].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 3].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 3].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 4].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 4].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
          prior_box[box_num_acc + 4 * i + 5].anchor_cx = (i % shapes[l][0] + offset) * layer_steps[l] / image_shape[0];
          prior_box[box_num_acc + 4 * i + 5].anchor_cy = (i % shapes[l][1] + offset) * layer_steps[l] / image_shape[1];
        }
      }
    }

    //float image_w[priorbox_num] = {0.0};
    //float image_h[priorbox_num] = {0.0};
    box_num_acc = 0;

    for (size_t l = 0; l < layer_num; l++) {
      if(l != 0) box_num_acc += box_per_layer[l-1];
      if(num[l] == 4) {
        for (size_t i = 0; i < shapes[l][0]*shapes[l][1]; i++) {
          prior_box[box_num_acc + 4 * i].anchor_w = anchor_scales[l];
          prior_box[box_num_acc + 4 * i].anchor_h = anchor_scales[l];
          prior_box[box_num_acc + 4 * i + 1].anchor_w = sqrt(anchor_scales[l] * extra_anchor_scales[l]);
          prior_box[box_num_acc + 4 * i + 1].anchor_h = sqrt(anchor_scales[l] * extra_anchor_scales[l]);
          prior_box[box_num_acc + 4 * i + 2].anchor_w = anchor_scales[l] * sqrt(ratio);
          prior_box[box_num_acc + 4 * i + 2].anchor_h = anchor_scales[l] / sqrt(ratio);
          prior_box[box_num_acc + 4 * i + 3].anchor_w = anchor_scales[l] / sqrt(ratio);
          prior_box[box_num_acc + 4 * i + 3].anchor_h = anchor_scales[l] * sqrt(ratio);
        }
      } else {
        for (size_t i = 0; i < shapes[l][0]*shapes[l][1]; i++) {
          prior_box[box_num_acc + 6 * i].anchor_w = anchor_scales[l];
          prior_box[box_num_acc + 6 * i].anchor_h = anchor_scales[l];
          prior_box[box_num_acc + 6 * i + 1].anchor_w = sqrt(anchor_scales[l] * extra_anchor_scales[l]);
          prior_box[box_num_acc + 6 * i + 1].anchor_h = sqrt(anchor_scales[l] * extra_anchor_scales[l]);
          prior_box[box_num_acc + 6 * i + 2].anchor_w = anchor_scales[l] * sqrt(ratio);
          prior_box[box_num_acc + 6 * i + 2].anchor_h = anchor_scales[l] / sqrt(ratio);
          prior_box[box_num_acc + 6 * i + 3].anchor_w = anchor_scales[l] / sqrt(ratio);
          prior_box[box_num_acc + 6 * i + 3].anchor_h = anchor_scales[l] * sqrt(ratio);
          prior_box[box_num_acc + 6 * i + 4].anchor_w = anchor_scales[l] * sqrt(ratio_six);
          prior_box[box_num_acc + 6 * i + 4].anchor_h = anchor_scales[l] / sqrt(ratio_six);
          prior_box[box_num_acc + 6 * i + 5].anchor_w = anchor_scales[l] / sqrt(ratio_six);
          prior_box[box_num_acc + 6 * i + 5].anchor_h = anchor_scales[l] * sqrt(ratio_six);
        }
      }
    }
    //return prior_box;
    return true;
}


bool SSDResnet34BoxParse(DetectBox& box,
                         float* loc_data,
                         float* cls_data,
                         Prior_Box& priorbox,
                         size_t class_num,
                         float probThresh,
                         CoordConvertInfo& ci) {

    float prior_scaling[4] = {0.1, 0.1, 0.2, 0.2};

    float cls_min = 100.0;
    float cls_max = 0.0;

    for (size_t i = 0; i < class_num; i++) {
      if (cls_data[i] < cls_min) cls_min = cls_data[i];
      if (cls_data[i] > cls_max) cls_max = cls_data[i];
    }

    for (size_t i = 0; i < class_num; i++) {
      cls_data[i] = (cls_data[i] - cls_min) / (cls_max - cls_min);
    }

    box.category = argmax(cls_data, class_num);
    if(box.category == 0) return false;

    box.confidence = cls_data[box.category];
    box.category = box.category - 1;
    if(box.confidence <= probThresh) {
      return false;
    }

    // decode location
    float pred_cx = loc_data[0] * prior_scaling[0] * priorbox.anchor_w + priorbox.anchor_cx;
    float pred_cy = loc_data[1] * prior_scaling[1] * priorbox.anchor_h + priorbox.anchor_cy;
    float pred_w = exp(loc_data[2] * prior_scaling[2]) * priorbox.anchor_w;
    float pred_h = exp(loc_data[3] * prior_scaling[3]) * priorbox.anchor_h;

    // center -> point
    box.xmin = (pred_cx - pred_w / 2) * 1200;
    box.ymin = (pred_cy - pred_h / 2) * 1200;
    box.xmax = (pred_cx + pred_w / 2) * 1200;
    box.ymax = (pred_cy + pred_h / 2) * 1200;

    // restore original coordinates
    box.xmin = (box.xmin - ci.wOffset) * ci.ioRatio;
    box.xmax = (box.xmax - ci.wOffset) * ci.ioRatio;
    box.ymin = (box.ymin - ci.hOffset) * ci.ioRatio;
    box.ymax - (box.ymax - ci.hOffset) * ci.ioRatio;

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
    auto pCfg = (SSDResnet34Config*)ctx->getConfigData();
    auto& cfg = *pCfg;

    auto pInputImages = reinterpret_cast<std::vector<bm_image>*>(ctx->getPostExtra());
    size_t batch = outTensors[0]->shape(0);
    size_t prior_box_num = outTensors[0]->shape(2);
    size_t classNum = outTensors[0]->shape(1);

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

    std::vector<std::vector<DetectBox>> batchBoxInfos(batch, std::vector<DetectBox>(prior_box_num));

    std::vector<Prior_Box> prior_box(prior_box_num);
    if (get_prior_box(prior_box) == false)
      return false;

    //fill batchBoxInfo
    auto cls = outTensors[0]->get_float_data();
    auto loc = outTensors[1]->get_float_data();
    std::vector<int> batchIndice(batch, 0);
    for(size_t b=0; b<batch; b++) {
      auto& ci = coordInfos[b];
      for (size_t i = 0; i < prior_box_num; i++) {
         float loc_data[4];
         for (size_t l = 0; l < 4; l++) {
         //  loc_data[i] = loc[b][l][i];
           auto raw_loc_offset = l * prior_box_num + i;
           loc_data[l] = loc[raw_loc_offset];
         }
         float cls_data[classNum];
         for (size_t c = 0; c < classNum; c++) {
         //  cls_data[i] = cls[b][c][i];
           auto raw_cls_offset = c * prior_box_num + i;
           cls_data[c] = cls[raw_cls_offset];
         }
         auto& boxInfo = batchBoxInfos[b][batchIndice[b]];
         auto& priorbox = prior_box[i];
         if(SSDResnet34BoxParse(boxInfo, loc_data, cls_data, priorbox, classNum, cfg.probThreshold, ci)){
             batchIndice[b]++;
         }
      }
    }

    // resize box
    for(size_t b=0; b<batch; b++){
        batchBoxInfos[b].resize(batchIndice[b]);
    }

    // NMS for result box
    postOut.results = batchNMS(batchBoxInfos, cfg.iouThreshold);

    // draw rectangle
    for(size_t b=0; b<batch; b++){
        auto name = baseName(rawIn[b]);
        drawDetectBoxEx(pInputImages->at(b), postOut.results[b], globalGroundTruth[name], cfg.savePath+"/"+name);
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
    std::string bmodel = topDir + "models/ssdresnet34/fp32.bmodel";
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

