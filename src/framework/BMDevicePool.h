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

extern const char* __phaseMap[];
class BMDeviceContext {

private:
    std::vector<bm_device_mem_t> mem_to_free;
    std::vector<std::vector<bm_image>> images_to_free;
    std::vector<bm_image> info_to_free;
    void* preExtra;
    void* postExtra;

public:
    DeviceId deviceId;
    bm_handle_t handle;
    void* pBMRuntime;
    std::shared_ptr<BMNetwork> net;
    size_t batchSize;
    void* configData;

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

    void setPreExtra(void* data){
        preExtra = data;
    }
    void* getPreExtra(){
        return preExtra;
    }

    void* getPostExtra(){
        return postExtra;
    }
    void setPostExtra(void* data){
        postExtra = data;
    }

    void freeImages(std::vector<bm_image>& ref_images);
    ~BMDeviceContext();
    void *getConfigData() const;
    void setConfigData(void *value);
};

using ContextPtr = typename std::shared_ptr<BMDeviceContext>;

struct ProcessStatus {
    DeviceId deviceId;
    bool valid;
    std::vector<std::chrono::steady_clock::time_point> starts;
    std::vector<std::chrono::steady_clock::time_point> ends;
    void reset(){
        starts.clear();
        ends.clear();
        valid = false;
    }
    void start(){
        starts.push_back(std::chrono::steady_clock::now());
        ends.push_back(starts.back());
    }
    void end(){
        ends.back() = std::chrono::steady_clock::now();
    }
    void show() {
        BMLOG(INFO, "device_id=%d, valid=%d, total=%dus", deviceId, valid, totalDuration());
        for(size_t i=0; i<starts.size(); i++){
            auto startStr = steadyToString(starts[i]);
            auto endStr = steadyToString(ends[i]);
            BMLOG(INFO, "  -> %s: duration=%dus",
                  __phaseMap[i],
                  usBetween(starts[i], ends[i]),
                  startStr.c_str(), endStr.c_str());
        }
    }
    size_t totalDuration() const {
        return usBetween(starts.front(), ends.back());
    }

    void increaseDuration(std::vector<size_t>& durations) {
    }
};

struct ProcessStatInfo {
    size_t totalDuration = 0;
    size_t numSamples = 0;
    std::map<size_t, size_t> deviceProcessNum;
    std::vector<size_t> durations;
    std::string name;
    std::chrono::steady_clock::time_point start;
    ProcessStatInfo(const std::string& name): name(name), start(std::chrono::steady_clock::now()){ }
    void update(const std::shared_ptr<ProcessStatus>& status) {
        if(status->valid){
            numSamples++;
            totalDuration += status->totalDuration();
            for(size_t i = durations.size(); i<status->starts.size(); i++){
                durations.push_back(0);
            }
            for(size_t i=0; i<status->starts.size(); i++){
                durations[i] += usBetween(status->starts[i], status->ends[i]);
            }
            deviceProcessNum[status->deviceId]++;
        }
    }

    void show() {
        auto end = std::chrono::steady_clock::now();
        auto totalUs = usBetween(start, end);
        BMLOG(INFO, "For model '%s'", name.c_str());
        BMLOG(INFO, "samples=%d: real_time=%gms, avg_real_time=%gms", numSamples, totalUs/1000.0, (float)totalUs/1000.0/numSamples);
        BMLOG(INFO, "            serialized_time=%gms, avg_serialized_time=%gms", totalDuration/1000.0, (float)totalDuration/1000.0/numSamples);
        for(size_t i=0; i<durations.size(); i++){
            BMLOG(INFO, "  -> total %s duration=%gms",
                  __phaseMap[i],
                  durations[i]/1000.0);
        }
        for(auto& p: deviceProcessNum){
            BMLOG(INFO, "  -> device #%d processes %d samples", p.first, p.second);
        }
    }
    ~ProcessStatInfo(){
    }
};

template<typename InType, typename OutType>
class BMDevicePool {
public:
    using ContextType = BMDeviceContext;

    struct _PreOutType {
        InType in;
        TensorVec preOut;
        std::shared_ptr<ProcessStatus> status;
        void* extra;
    };

    struct _ForwardOutType {
        InType in;
        TensorVec forwardOut;
        std::shared_ptr<ProcessStatus> status;
        void* extra;
    };

    struct _PostOutType {
        InType in;
        OutType out;
        std::shared_ptr<ProcessStatus> status;
    };

    using RunnerType = BMPipelinePool<InType, _PostOutType, BMDeviceContext>;
    using RunnerPtr = std::shared_ptr<RunnerType>;
    using PreProcessFunc = std::function<bool(const InType&, const TensorVec&, ContextPtr)>;
    using PostProcessFunc = std::function<bool(const InType&, const TensorVec&, OutType&, ContextPtr)>;
    std::atomic_size_t atomicBatchSize;

    template<typename PreFuncType, typename PostFuncType>
    BMDevicePool(const std::string& bmodel, PreFuncType preProcessFunc, PostFuncType postProcessFunc,
                 std::vector<DeviceId> userDeviceIds={}): atomicBatchSize(0) {
        deviceIds = userDeviceIds;
        if(userDeviceIds.empty()){
           deviceIds = getAvailableDevices();
        }
        auto localDeviceIds = deviceIds;
        auto deviceNum = deviceIds.size();
        BM_ASSERT(deviceNum>0, "no device found");
        std::string deviceStr = "";
        for(auto id: deviceIds){
            deviceStr += std::to_string(id)+" ";
        }
        BMLOG(INFO, "USING DEVICES: %s", deviceStr.c_str());
        std::function<std::shared_ptr<ContextType>(size_t)>  contextInitializer = [localDeviceIds, bmodel, this](size_t i) {
            auto context = std::make_shared<ContextType>(localDeviceIds[i], bmodel);
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

    void stop(int deviceId = -1){
        if(deviceId == -1) {
            pool->stop();
            return;
        }
        for(size_t index=0; index<deviceIds.size(); index++){
            if(deviceId == deviceIds[index]) {
                pool->stop(index);
                break;
            }
        }
    }

    virtual ~BMDevicePool() {
        stop();
    }

    bool canPush(){
        return pool->canPush();
    }

    void push(InType in){
        pool->push(in);
    }

    bool allStopped() {
        return pool->allStopped();
    }

    bool pop(OutType& out, std::shared_ptr<ProcessStatus>& status){
        _PostOutType postOut;
        bool res = pool->pop(postOut);
        if(res){
            status = postOut.status;
            out = postOut.out;
        }
        return res;
    }

    static bool preProcess(const InType& in, _PreOutType& out, ContextPtr ctx, PreProcessFunc preCoreFunc) {
        out.status = std::make_shared<ProcessStatus>();
        out.status->deviceId = ctx->deviceId;
        out.status->start();
        out.in = in;
        out.status->valid = preCoreFunc(in, out.preOut, ctx);
        out.status->end();
        out.extra = ctx->getPreExtra();
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
        out.status = std::move(in.status);
        out.in = in.in;
        if(out.status->valid){
            out.status->start();
            out.status->valid = ctx->net->forward(in.preOut, out.forwardOut);
            out.status->end();
        }
        out.extra = in.extra;
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
        out.status = std::move(in.status);
        ctx->setPostExtra(in.extra);
        if(out.status->valid){
            out.status->start();
            out.status->valid = postCoreFunc(in.in, in.forwardOut, out.out, ctx);
            out.status->end();
        }
        return true;
    }

    size_t getBatchSize(){
        return atomicBatchSize.load();
    }

    size_t deviceNum() const { return deviceIds.size(); }
private:
    RunnerPtr pool;
    std::vector<DeviceId> deviceIds;
};

}

#endif // BMDEVICEPOOL_H
