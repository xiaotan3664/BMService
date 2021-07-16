#ifndef BMDEVICEPOOL_H
#define BMDEVICEPOOL_H
#include <vector>
#include <functional>
#include <algorithm>
#include <exception>
#include "BMDeviceUtils.h"
#include "BMPipelinePool.h"
#include "BMNetwork.h"
#include "bmlib_runtime.h"
#include "bmcv_api.h"

namespace bm {

class BMDeviceContext {

private:
    std::vector<bm_device_mem_t> mem_to_free;

public:
    DeviceId deviceId;
    bm_handle_t handle;
    void* pBMRuntime;
    std::shared_ptr<BMNetwork> net;
    size_t batchSize;

    std::shared_ptr<BMNetwork> getNetwork() { return net; }

    BMDeviceContext(DeviceId deviceId, const std::string& bmodel):deviceId(deviceId), batchSize(batchSize) {
        batchSize = -1;
        auto status = bm_dev_request(&handle, deviceId);
        BM_ASSERT_EQ(status, BM_SUCCESS);
        pBMRuntime = bmrt_create(handle);
        BM_ASSERT(pBMRuntime != nullptr, "cannot create bmruntime handle");
        net = std::make_shared<BMNetwork>(pBMRuntime, bmodel);
        batchSize = net->getBatchSize();
    }

    bm_device_mem_t allocDeviceMem(size_t bytes) {
        bm_device_mem_t mem;
        if(bm_malloc_device_byte(handle, &mem, bytes) != BM_SUCCESS){
            BMLOG(FATAL, "cannot alloc device mem, size=%d", bytes);
        }
        mem_to_free.push_back(mem);
        return mem;
    }

    void freeDeviceMem(bm_device_mem_t& mem){
        auto iter = std::find_if(mem_to_free.begin(), mem_to_free.end(), [&mem](bm_device_mem_t& m){
            return bm_mem_get_device_addr(m) ==  bm_mem_get_device_addr(mem);
        });
        BM_ASSERT(iter != mem_to_free.end(), "cannot free mem!");
        bm_free_device(handle, mem);
        mem_to_free.erase(iter);
    }

    size_t getBatchSize(){
        return batchSize;
    }

    void allocMemForTensor(TensorPtr tensor){
        auto mem_size = tensor->get_mem_size();
        auto mem = allocDeviceMem(mem_size);
        tensor->set_device_mem(&mem);
    }

//    bm_image_t allocImage();
//    void freeImage(bm_image_t& image);
    ~BMDeviceContext() {
        auto mems = mem_to_free;
        for(auto m : mems){
            freeDeviceMem(m);
        }
        bmrt_destroy(pBMRuntime);
        bm_dev_free(handle);
    }
};

using ContextPtr = typename std::shared_ptr<BMDeviceContext>;

template<typename InType_, typename OutType_>
class BMDevicePool {
public:
    using InType = InType_;
    using OutType = OutType_;
    using ContextType = BMDeviceContext;
    using PoolType = BMPipelinePool<InType, OutType, ContextType>;
    using PoolPtr = std::shared_ptr<PoolType>;

    struct PreOutType {
        bool valid;
        InType in;
        TensorVec preOut;
    };

    struct ForwardOutType {
        bool valid;
        InType in;
        TensorVec forwardOut;
    };

    struct PostOutType {
        InType in;
        OutType out;
    };

    using RunnerType = BMDevicePool<InType, PostOutType>;
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

        pool = std::make_shared<PoolType>(deviceNum, contextInitializer);

        auto inQueue = pool->getInputQueue();
        inQueue->setMaxNode(deviceNum*4);

        PreProcessFunc preCoreFunc = preProcessFunc;
        PostProcessFunc postCoreFunc = postProcessFunc;
        std::function<bool(const InType&, PreOutType&, ContextPtr ctx)> preFunc =
                [preCoreFunc] (const InType& in, PreOutType& out, ContextPtr ctx){
            return preProcess(in, out, ctx, preCoreFunc);
        };
        std::function<std::vector<PreOutType>(ContextPtr)> preCreateFunc = createPreProcessOutput;
        pool->addNode(preFunc, preCreateFunc);

        std::function<bool(const PreOutType&, ForwardOutType&, ContextPtr)> forwardFunc = forward;
        std::function<std::vector<ForwardOutType>(ContextPtr)> createForwardFunc = createForwardOutput;
        pool->addNode(forwardFunc, createForwardFunc);

        std::function<bool(const ForwardOutType&, PostOutType&, ContextPtr)> postFunc =
                [postCoreFunc] (const ForwardOutType& in, PostOutType& out, ContextPtr ctx){
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
        return pool->pop(out);
    }

    static bool preProcess(const InType& in, PreOutType& out, ContextPtr ctx, PreProcessFunc preCoreFunc) {
        out.in = in;
        out.valid = preCoreFunc(in, out.preOut, ctx);
        return true;
    }

    static std::vector<PreOutType> createPreProcessOutput(ContextPtr ctx) {
        auto net = ctx->net;
        std::vector<PreOutType> preOuts;
        for(size_t i=0; i<2; i++){
            PreOutType preOut;
            preOut.preOut = net->createInputTensors();
            for(auto tensor: preOut.preOut){
                ctx->allocMemForTensor(tensor);
            }
            preOuts.push_back(std::move(preOut));
        }
        return preOuts;
    }

    static bool forward(const PreOutType& in, ForwardOutType& out, ContextPtr ctx) {
        out.in = in.in;
        out.valid = false;
        if(in.valid){
            out.valid = ctx->net->forward(in.preOut, out.forwardOut);
        }
        return true;
    }

    static std::vector<ForwardOutType> createForwardOutput(ContextPtr ctx) {
        auto net = ctx->net;
        std::vector<ForwardOutType> forwardOuts;
        for(size_t i=0; i<2; i++){
            ForwardOutType forwardOut;
            forwardOut.forwardOut = net->createOutputTensors();
            for(auto tensor: forwardOut.forwardOut){
                ctx->allocMemForTensor(tensor);
            }
            forwardOuts.push_back(std::move(forwardOut));
        }
        return forwardOuts;
    }

    static bool postProcess(const ForwardOutType& in, PostOutType& out, ContextPtr ctx, PostProcessFunc postCoreFunc) {
        return postCoreFunc(in.in, in.forwardOut, out.out, ctx);
    }

    size_t getBatchSize(){
        return atomicBatchSize.load();
    }
private:
    PoolPtr pool;
};

}

#endif // BMDEVICEPOOL_H
