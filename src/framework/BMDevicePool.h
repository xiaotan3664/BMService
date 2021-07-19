#ifndef BMDEVICEPOOL_H
#define BMDEVICEPOOL_H
#include <vector>
#include <functional>
#include <algorithm>
#include <exception>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "BMDeviceUtils.h"
#include "BMPipelinePool.h"
#include "BMNetwork.h"
#include "bmlib_runtime.h"
#include "bmcv_api.h"

namespace bm {

class BMDeviceContext {

private:
    std::vector<bm_device_mem_t> mem_to_free;
    std::vector<std::vector<bm_image>> images_to_free;
    std::vector<bm_image> info_to_free;

public:
    DeviceId deviceId;
    bm_handle_t handle;
    void* pBMRuntime;
    std::shared_ptr<BMNetwork> net;
    size_t batchSize;

    BMDeviceContext(DeviceId deviceId, const std::string& bmodel);

    std::shared_ptr<BMNetwork> getNetwork() { return net; }
    size_t getBatchSize(){ return batchSize; }


    bm_device_mem_t allocDeviceMem(size_t bytes);

    void freeDeviceMem(bm_device_mem_t& mem);

    void allocMemForTensor(TensorPtr tensor);
    std::vector<bm_image> allocImagesWithoutMem(
            int num,
            int height,
            int width,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype, int align_bytes = 1);

    std::vector<bm_image> allocImages(int num, int height, int width,
                        bm_image_format_ext format,
                        bm_image_data_format_ext dtype,
                        int align_bytes = 1,
                        int heap_id = BMCV_HEAP_ANY);

    std::vector<bm_image> allocAlignedImages(
            int num,
            int height, int width,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int heap_id = BMCV_HEAP_ANY);

    void freeImages(std::vector<bm_image>& ref_images);
    ~BMDeviceContext();
};

using ContextPtr = typename std::shared_ptr<BMDeviceContext>;

struct ProcessStatus {

};

template<typename InType, typename OutType>
class BMDevicePool {
public:
    using ContextType = BMDeviceContext;

    struct _PreOutType {
        bool valid;
        InType in;
        TensorVec preOut;
    };

    struct _ForwardOutType {
        bool valid;
        InType in;
        TensorVec forwardOut;
    };

    struct _PostOutType {
        InType in;
        OutType out;
    };

    using RunnerType = BMPipelinePool<InType, _PostOutType, BMDeviceContext>;
    using RunnerPtr = std::shared_ptr<RunnerType>;
    using PreProcessFunc = std::function<bool(const InType&, const TensorVec&, ContextPtr)>;
    using PostProcessFunc = std::function<bool(const InType&, const TensorVec&, OutType&, ContextPtr)>;
    std::atomic_size_t atomicBatchSize;

    template<typename PreFuncType, typename PostFuncType>
    BMDevicePool(const std::string& bmodel, PreFuncType preProcessFunc, PostFuncType postProcessFunc,
                 std::vector<DeviceId> userDeviceIds={}): atomicBatchSize(0) {
        auto deviceIds = userDeviceIds;
        if(userDeviceIds.empty()){
           deviceIds = getAvailableDevices();
        }
        auto deviceNum = deviceIds.size();
        BM_ASSERT(deviceNum>0, "no device found");
        std::function<std::shared_ptr<ContextType>(size_t)>  contextInitializer = [deviceIds, bmodel, this](size_t i) {
            auto context = std::make_shared<ContextType>(deviceIds[i], bmodel);
            this->atomicBatchSize=context->getBatchSize();
            return context;
        };

        pool = std::make_shared<RunnerType>(deviceNum, contextInitializer);

        auto inQueue = pool->getInputQueue();
        inQueue->setMaxNode(deviceNum*4);

        PreProcessFunc preCoreFunc = preProcessFunc;
        PostProcessFunc postCoreFunc = postProcessFunc;
        std::function<bool(const InType&, _PreOutType&, ContextPtr ctx)> preFunc =
                [preCoreFunc] (const InType& in, _PreOutType& out, ContextPtr ctx){
            return preProcess(in, out, ctx, preCoreFunc);
        };
        std::function<std::vector<_PreOutType>(ContextPtr)> preCreateFunc = createPreProcessOutput;
        pool->addNode(preFunc, preCreateFunc);

        std::function<bool(const _PreOutType&, _ForwardOutType&, ContextPtr)> forwardFunc = forward;
        std::function<std::vector<_ForwardOutType>(ContextPtr)> createForwardFunc = createForwardOutput;
        pool->addNode(forwardFunc, createForwardFunc);

        std::function<bool(const _ForwardOutType&, _PostOutType&, ContextPtr)> postFunc =
                [postCoreFunc] (const _ForwardOutType& in, _PostOutType& out, ContextPtr ctx){
            return postProcess(in, out, ctx, postCoreFunc);
        };
        pool->addNode(postFunc);
    }

    void start() {
        pool->start();
    }

    void stop(){
        pool->stop();
    }

    virtual ~BMDevicePool() {
        stop();
    }

    void push(InType in){
        pool->push(in);
    }

    bool pop(OutType& out){
        _PostOutType postOut;
        bool res = pool->pop(postOut);
        out = postOut.out;
	return res;
    }

    static bool preProcess(const InType& in, _PreOutType& out, ContextPtr ctx, PreProcessFunc preCoreFunc) {
        out.in = in;
        out.valid = preCoreFunc(in, out.preOut, ctx);
        return true;
    }

    static std::vector<_PreOutType> createPreProcessOutput(ContextPtr ctx) {
        auto net = ctx->net;
        std::vector<_PreOutType> preOuts;
        for(size_t i=0; i<2; i++){
            _PreOutType preOut;
            preOut.preOut = net->createInputTensors();
            for(auto tensor: preOut.preOut){
                ctx->allocMemForTensor(tensor);
            }
            preOuts.push_back(std::move(preOut));
        }
        return preOuts;
    }

    static bool forward(const _PreOutType& in, _ForwardOutType& out, ContextPtr ctx) {
        out.in = in.in;
        out.valid = false;
        if(in.valid){
            out.valid = ctx->net->forward(in.preOut, out.forwardOut);
        }
        return true;
    }

    static std::vector<_ForwardOutType> createForwardOutput(ContextPtr ctx) {
        auto net = ctx->net;
        std::vector<_ForwardOutType> forwardOuts;
        for(size_t i=0; i<2; i++){
            _ForwardOutType forwardOut;
            forwardOut.forwardOut = net->createOutputTensors();
            for(auto tensor: forwardOut.forwardOut){
                ctx->allocMemForTensor(tensor);
            }
            forwardOuts.push_back(std::move(forwardOut));
        }
        return forwardOuts;
    }

    static bool postProcess(const _ForwardOutType& in, _PostOutType& out, ContextPtr ctx, PostProcessFunc postCoreFunc) {
        return postCoreFunc(in.in, in.forwardOut, out.out, ctx);
    }

    size_t getBatchSize(){
        return atomicBatchSize.load();
    }
private:
    RunnerPtr pool;
};

}

#endif // BMDEVICEPOOL_H
