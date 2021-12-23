#include <dirent.h>
#include<vector>
#include<thread>
#include "BMDevicePool.h"
#include "BMDeviceUtils.h"
#include "BMImageUtils.h"
#include "bmcv_api.h"
#include <regex>

using namespace bm;
using InType = std::vector<std::string>;    
using ClassId = size_t;

struct PostOutType {
    InType rawIns;
    std::vector<std::vector<std::pair<ClassId, float>>> classAndScores;
};

struct ResNetConfig {
    bool initialized = false;
    bool isNCHW;
    size_t netBatch;
    size_t netHeight;
    size_t netWidth;
    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;
    bmcv_convert_to_attr ConvertAttr;
    // use static to cache resizedImage, to avoid allocating memory everytime
    std::vector<bm_image> cropedImages;
    std::vector<bm_image> preOutImages;

    void initialize(TensorPtr inTensor, ContextPtr ctx){
        if(initialized) return;
        initialized = true;
        netBatch = inTensor->shape(0);
        isNCHW = inTensor->shape(1) == 3;
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
        }else{
            netHeight = inTensor->shape(2);
            netWidth = inTensor->shape(3);
            netFormat = FORMAT_RGB_PLANAR; // for NHWC input

        }
        float input_scale = inTensor->get_scale();
        float scale = 1.0;
        float bias = 0;
        float real_scale = scale * input_scale;
        float real_bias = bias * input_scale;
        ConvertAttr.alpha_0 = real_scale;
        ConvertAttr.beta_0 = real_bias - 123.68;
        ConvertAttr.alpha_1 = real_scale;
        ConvertAttr.beta_1 = real_bias - 116.78;
        ConvertAttr.alpha_2 = real_scale;
        ConvertAttr.beta_2 = real_bias - 103.94;

        // new bm_image;
        cropedImages = ctx->allocAlignedImages(
                    netBatch, netHeight, netWidth, netFormat, DATA_TYPE_EXT_1N_BYTE);
        
        preOutImages = ctx->allocImagesWithoutMem(netBatch, netHeight, netWidth, netFormat, netDtype);
    }
};
/*
    @param: inTensor: input of model, vector of TensorPtr
*/
bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    thread_local static ResNetConfig cfg;
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);

    cfg.initialize(inTensor, ctx);

    std::vector<bm_image> alignedInputs;
    // after aspect preserve
    std::vector<bm_image> aspectResized;

    // std::regex r("ILSVRC2012_val_\\d+\\.JPEG");
    // std::smatch match; 
    // bool found = regex_search(in[0], match, r);

// TimeRecorder r;
// r.record("read");
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs.push_back(image);
    }
// r.record("resize and crop");

    // Original operations are aspect preserve resize then central crop 
    // But equivalent to cenrtal crop then resize to a square
    centralCrop(ctx->handle, alignedInputs, cfg.cropedImages); 

// r.record("attach memery");  
    auto mem = inTensor->get_device_mem();
    bm_image_attach_contiguous_mem(in.size(), cfg.preOutImages.data(), *mem);
// r.record("linear convert"); 
    if(cfg.isNCHW){
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, cfg.cropedImages.data(), cfg.preOutImages.data());
    } else {
        //to planar
        std::vector<bm_image> planarImage1, planarImage2;
        planarImage1 = ctx->allocImagesWithoutMem(
                            cfg.netBatch, cfg.netHeight, cfg.netWidth, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_FLOAT32, 1);
        planarImage2 = ctx->allocImagesWithoutMem(
                            cfg.netBatch, cfg.netHeight, cfg.netWidth, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_FLOAT32, 1);
        bmcv_image_storage_convert(ctx->handle, 1, cfg.cropedImages.data(), planarImage1.data());      
        bmcv_image_convert_to(ctx->handle, in.size(), cfg.ConvertAttr, planarImage1.data(), planarImage2.data());   
        bmcv_image_storage_convert(ctx->handle, 1, planarImage2.data(), cfg.preOutImages.data());

            // int* size = new int;

            // bm_image_get_byte_size(cfg.preOutImages[0], size);
            // auto buf2 = new void*[1];
            // buf2[0] = new unsigned char[*size];
            // bm_image_copy_device_to_host(cfg.preOutImages[0], buf2);
            // delete [] buf2;

        for(auto &p: planarImage1) {
            bm_image_destroy(p);
        }
         for(auto &p: planarImage2) {
            bm_image_destroy(p);
        }

    }
// r.record("destroy");    
    //destroy temporary bm_image
    for(auto &image: alignedInputs) {
        bm_image_destroy(image);
    }
    for(auto &ap: aspectResized) {
        bm_image_destroy(ap);
    }
// r.show();
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
        auto firstClass = classAndScores[0].first;
        auto firstScore = classAndScores[0].second;
        stat.samples++;
        stat.top1 += firstClass == realClass;
        for(auto& cs: classAndScores){
            if(cs.first == realClass){
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
    // std::string bmodel = topDir + "models/resnet50_v1/fix8b_4n.bmodel";
    std::string bmodel = topDir + "models/resnet101/fix8b_4n.bmodel";
    // std::string bmodel = topDir + "models/resnet50_v1/compilation_4n.bmodel";
    std::string refFile = topDir + "data/ILSVRC2012/val.txt";
    std::string labelFile = topDir + "data/ILSVRC2012/labels.txt";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    if(argc>3) refFile = argv[3];
    if(argc>4) labelFile = argv[4];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize = runner.getBatchSize();
    auto refMap = loadClassRefs(refFile, "");
    auto labelMap = loadLabels(labelFile);
    ProcessStatInfo info(bmodel);
    Top5AccuracyStat topStat;
    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const InType& imageFiles){
            runner.push(imageFiles);
            return true;
        });
        runner.join();
    });
    std::thread resultThread([&runner, &refMap, &labelMap, &info, batchSize](){
        PostOutType out;
        std::shared_ptr<ProcessStatus> status;
        Top5AccuracyStat stat;
        while(true){
            if (!runner.waitAndPop(out, status)) {
                info.show();
                stat.show();
                break;
            }
            info.update(status, batchSize);
            resultProcess(out, stat, refMap, labelMap);
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

