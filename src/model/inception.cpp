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

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    if(in.empty()) return false;
    BM_ASSERT_EQ(inTensors.size(), 1);
    auto inTensor = inTensors[0];
    BM_ASSERT_EQ(inTensor->dims(), 4);
    BM_ASSERT_EQ(inTensor->shape(3), 3);
    auto netBatch = inTensor->shape(0);
    auto netHeight = inTensor->shape(1);
    auto netWidth = inTensor->shape(2);
    auto netFormat = FORMAT_RGB_PACKED; // for NHWC input
    auto netDtype =  DATA_TYPE_EXT_FLOAT32;

    std::vector<bm_image> alignedInputs;
    for(auto imageName: in){
        auto image = readAlignedImage(ctx->handle, imageName);
        alignedInputs.push_back(image);
    }

    // just to keep the image info, and use tensor mem
    //static std::vector<bm_image> preOutImages = ctx->allocImagesWithoutMem(
    //            netBatch, netHeight, netWidth, netFormat, netDtype);
    thread_local static std::vector<bm_image> preOutImages;
    if(preOutImages.empty()){
        ctx->allocImagesWithoutMem(
                    netBatch, netHeight, netWidth*3, FORMAT_GRAY, netDtype);
        auto mem = inTensor->get_device_mem();
        bm_image_attach_contiguous_mem(in.size(), preOutImages.data(), *mem);
    }
    //BMLOG(INFO, "addr=%llx", bm_mem_get_device_addr(*mem));

    // use static to cache resizedImage, to avoid allocating memory everytime
    thread_local static std::vector<bm_image> resizedImages = ctx->allocAlignedImages(
                netBatch, netHeight, netWidth, netFormat, alignedInputs[0].data_type);

    // bmcv_image_convert_to do not support RGB_PACKED format directly
    static std::vector<bm_image> grayImages; 
    if(grayImages.empty()){ // run once
	    grayImages = ctx->allocImagesWithoutMem(
			    netBatch, netHeight, netWidth*3, FORMAT_GRAY, alignedInputs[0].data_type, 64);
	    bm_device_mem_t resizedMem;
	    bm_image_get_device_mem(resizedImages[0], &resizedMem);
	    bm_image_attach_contiguous_mem(grayImages.size(), grayImages.data(), resizedMem);
    }

    centralCropAndResize(ctx->handle, alignedInputs, resizedImages);

    //saveImage(resizedImages[0], "resize.jpg");
    //saveImage(grayImages[0], "gray.jpg");

    float input_scale = inTensor->get_scale();
    //
    input_scale = input_scale * 1/255 * 2.0;
    bmcv_convert_to_attr converto_attr;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = -1;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = -1;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = -1;

    //dumpImage(grayImages[0], "grayImage.txt");
    bmcv_image_convert_to(ctx->handle, in.size(), converto_attr, grayImages.data(), preOutImages.data());
    //inTensor->dumpData("tensor.txt");

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

bool resultProcess(const PostOutType& out, std::map<size_t, std::string> labelMap){
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
    std::string bmodel = "../compilation/compilation.bmodel";
    std::string labelFile = "../inception_label.txt";
    if(argc>1) dataPath = argv[1];
    if(argc>2) bmodel = argv[2];
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();
    auto labelMap = loadLabels(labelFile);


    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const InType& imageFiles){
            runner.push(imageFiles);
            return true;
        });
        runner.push({});
    });
    std::thread resultThread([&runner](){
        PostOutType out;
        while(true){
            while(!runner.pop(out)) {
                std::this_thread::yield();
            }
            if(!resultProcess(out, labelMap)){
                runner.stop();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}

