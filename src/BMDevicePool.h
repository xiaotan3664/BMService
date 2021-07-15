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

    std::shared_ptr<BMNetwork> getNetwork() { return net; }

    BMDeviceContext(DeviceId deviceId, const std::string& bmodel=""):deviceId(deviceId) {
        auto status = bm_dev_request(&handle, deviceId);
        BM_ASSERT_EQ(status, BM_SUCCESS);
        pBMRuntime = bmrt_create(handle);
        BM_ASSERT(pBMRuntime != nullptr, "cannot create bmruntime handle");

        if(bmodel != ""){
            net = std::make_shared<BMNetwork>(pBMRuntime, bmodel);
        }
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


template<typename InType_, typename OutType_>
class BMDevicePoolBase {
public:
    using InType = InType_;
    using OutType = OutType_;
    using ContextType = BMDeviceContext;
    using PoolType = BMPipelinePool<InType, OutType, ContextType>;
    using PoolPtr = typename std::shared_ptr<PoolType>;
    std::atomic_bool done;
    std::function<InType()> dataFunc;

    std::vector<std::thread> dataThreads;
    std::vector<bool> finished;

    BMDevicePoolBase(const std::string& bmodel="", std::vector<DeviceId> userDeviceIds={}): done(false) {
        auto deviceIds = userDeviceIds;
        if(userIds.empty()){
           deviceIds = getAvailableDevices();
        }
        auto deviceNum = deviceIds.size();
        BM_ASSERT(deviceNum>0, "no device found");
        std::function<std::shared_ptr<ContextType>(size_t)>  contextInitializer = [deviceIds, bmodel](size_t i) {
            auto context = std::make_shared<ContextType>(deviceIds[i], bmodel);
            return context;
        };
        pool = std::make_shared<PoolType>(deviceNum, contextInitializer);
        auto inQueue = pool->getInputQueue();
        inQueue->setMaxNodes(deviceNum*3);
    }

    void generateData(size_t idx){
        auto inQueue = pool->getInputQueue();
        while(!done){
            if(inQueue->canPush()){
                try {
                    auto data = dataFunc(idx);
                    if(!data){
                        finished[idx] = true;
                        break;
                    }
                    inQueue->push(data);
                } catch (std::exception& e) {
                    done = true;
                    BMLOG(ERROR, "fail to generate data: %s", e.what());
                }
            }
        }
    }

    template<typename Func>
    void setDataGenerator(Func generator){
        dataFunc = generator;
    }

    template<typename NodeInType, typename NodeOutType, typename Container=std::vector<NodeOutType>>
    void addProcess(std::function<void(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> func,
                 std::function<Container(std::shared_ptr<ContextType>)> outResourceInitializer = nullptr){
                     pool->addNode(func, outResourceInitializer);
    }


    void start(size_t numDataThreads=1) {
        BM_ASSERT(!dataFunc, "call setDataGenerator first");
        BM_ASSERT(dataThreads.empty(), "device pool already started");
        done = false;
        finished.assign(numDataThreads, false);
        pool->start();
        for(size_t i=0; i<numDataThreads; i++){
            dataThreads.emplace_back(&BMDevicePoolBase::generateData, this, i);
        }
    }

    void stop(){
        done = true;
        for(auto& thread: dataThreads){
            if(thread.joinable()){
                thread.join();
            }
        }
        dataThreads.clear();
        pool.stop();
    }

    virtual ~BMDevicePoolBase() {
        stop();
    }

private:
    PoolPtr pool;
};

}

#endif // BMDEVICEPOOL_H
