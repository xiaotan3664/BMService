#ifndef BMPIPELINEPOOL_H
#define BMPIPELINEPOOL_H
#include <string>
#include <set>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <typeinfo>
#include <type_traits>
#include "BMLog.h"
#include "BMCommonUtils.h"
#include "BMQueue.h"

namespace bm {

class BMPipelineNodeBase {
private:
public:
    virtual ~BMPipelineNodeBase() {}
    virtual void start() = 0;
    virtual void setOutQueue(std::shared_ptr<BMQueueVoid>) = 0;
};

struct BMPipelineEmptyContext { };

template<typename InType, typename OutType, typename ContextType = BMPipelineEmptyContext>
class BMPipelineNodeImp: public Uncopiable, public BMPipelineNodeBase {
private:
    using InQueuePtr=std::shared_ptr<BMQueue<InType>>;
    using OutQueuePtr=std::shared_ptr<BMQueue<OutType>>;
    using TaskType =  std::function<bool (const InType&, OutType&, std::shared_ptr<ContextType>)>;

    std::shared_ptr<ContextType> context;
    TaskType taskFunc;
    InQueuePtr inFreeQueue;
    InQueuePtr inTaskQueue;
    OutQueuePtr outFreeQueue;
    OutQueuePtr outTaskQueue;
    std::thread innerThread;
    std::atomic_bool& done;

    void workThread(){
        if(!inTaskQueue) {
            done = true;
            BMLOG(FATAL, "[%d] no input task queue!", std::this_thread::get_id());
            return;
        }
        while(!done){
            OutType out;
            InType in;
            while(!done && outFreeQueue) {
                if(outFreeQueue->tryPop(out)) {
                    BMLOG(DEBUG, "[%d] got an output resource", std::this_thread::get_id());
                    break;
                }
                std::this_thread::yield();
            }
            bool finish = false;
            while(!done && !finish){
                while(!done) {
                    if(inTaskQueue->tryPop(in)) {
                        BMLOG(DEBUG, "[%d] got a task", std::this_thread::get_id());
                        break;
                    }
                    std::this_thread::yield();
                }
                try {
                    if(!done){
                        finish = taskFunc(in, out, context);
                        if(inFreeQueue) {
                            BMLOG(DEBUG, "[%d] return an input resource", std::this_thread::get_id());
                            inFreeQueue->push(in);
                        }
                    } else {
                        break;
                    }
                } catch(...){
                    done = true;
                }
            }
            if(!done){
                if(outTaskQueue) {
                    BMLOG(DEBUG, "[%d] put a task", std::this_thread::get_id());
                    outTaskQueue->push(out);
                }
            }
        }
        BMLOG(DEBUG, "[%d] leave thread", std::this_thread::get_id());
    }

public:
    BMPipelineNodeImp(TaskType taskFunc,
                      InQueuePtr inFreeQueue, InQueuePtr inTaskQueue,
                      OutQueuePtr outFreeQueue, OutQueuePtr outTaskQueue,
                      std::atomic_bool& done,
                      std::shared_ptr<ContextType> context
                      ):
        taskFunc(taskFunc),
        inFreeQueue(inFreeQueue), inTaskQueue(inTaskQueue),
        outFreeQueue(outFreeQueue), outTaskQueue(outTaskQueue),
        done(done),
        context(context)
    {}

    virtual void setOutQueue(std::shared_ptr<BMQueueVoid> outQueueVoid) override {
        auto outQueue = std::dynamic_pointer_cast<BMQueue<OutType>>(outQueueVoid);
        if(!outQueue){
            BMLOG(FATAL, "output queue set failed");
        }
        outTaskQueue = outQueue;
    }

    void start() override {
        innerThread = std::thread(&BMPipelineNodeImp<InType, OutType, ContextType>::workThread, this);
        BMLOG(DEBUG, "thread created id=%d", innerThread.get_id());
    }

    virtual ~BMPipelineNodeImp() {
        BMLOG(DEBUG, "thread=%d destructed", innerThread.get_id());
        if(innerThread.joinable()){
            innerThread.join();
        }
    }
 };

template<typename InType, typename OutType, typename ContextType=BMPipelineEmptyContext>
class BMPipeline: public Uncopiable {
private:
    std::shared_ptr<BMQueue<InType>> inQueue;
    std::shared_ptr<BMQueue<OutType>> outQueue;
    std::vector<std::shared_ptr<BMPipelineNodeBase>> pipelineNodes;
    std::shared_ptr<ContextType> context;
    std::atomic_bool done;
    std::shared_ptr<BMQueueVoid> lastOutResourceQueue;
    std::shared_ptr<BMQueueVoid> lastOutWorkQueue;
    std::string lastTypeName;
    std::string outTypeName;

public:
    BMPipeline(std::shared_ptr<ContextType> context = std::shared_ptr<ContextType>()):
        context(context),
        done(false)
    {
        setInputQueue(std::make_shared<BMQueue<InType>>());
        lastOutResourceQueue = std::shared_ptr<BMQueue<InType>>();
        if(!inQueue){
            BMLOG(FATAL, "Cannot create input queue!");
        }
        lastTypeName = typeid(InType).name();
        outTypeName = typeid(OutType).name();
        outQueue = std::shared_ptr<BMQueue<OutType>>();
    }

    void setInputQueue(std::shared_ptr<BMQueue<InType>> inQueue_){
        if(!pipelineNodes.empty()){
            BMLOG(FATAL, "input queue cannot be set after call addNode");
        }
        inQueue = inQueue_;
        lastOutWorkQueue = inQueue;
    }

    void setOutputQueue(std::shared_ptr<BMQueue<OutType>> outQueue){
        if(pipelineNodes.empty()){
            BMLOG(FATAL, "output queue cannot be set after call addNode");
        }
        lastOutWorkQueue = outQueue;
        pipelineNodes.back()->setOutQueue(outQueue);
    }

    template<typename NodeInType, typename NodeOutType, typename Container = std::vector<NodeOutType>>
    void addNode(std::function<NodeOutType(const NodeInType&, std::shared_ptr<ContextType>)> func,
                 Container outResource = {}) {
        std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> inner_func = [func](
                const NodeInType& in, NodeOutType& out, std::shared_ptr<ContextType> ctx){
            out =  std::move(func(in, ctx));
            return true;
        };
        addNode(inner_func, outResource);
    }

    template<typename NodeInType, typename NodeOutType, typename Container = std::vector<NodeOutType>>
    void addNode(std::function<NodeOutType(const NodeInType&)> func,
                 Container outResource = {}) {
        std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> inner_func = [func](
                const NodeInType& in, NodeOutType& out, std::shared_ptr<ContextType>){
            out =  std::move(func(in));
            return true;
        };
        addNode(inner_func, outResource);
    }

    template<typename NodeInType, typename NodeOutType, typename Container = std::vector<NodeOutType>>
    void addNode(std::function<bool(const NodeInType&, NodeOutType&)> func,
                 Container outResource = {}) {
        std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> inner_func = [func](
                const NodeInType& in, NodeOutType& out, std::shared_ptr<ContextType>){ return func(in, out); };
        addNode(inner_func, outResource);
    }

    template<typename NodeInType, typename NodeOutType, typename Container= std::vector<NodeOutType>>
    void addNode(std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> func,
                 Container outResource = {}) {
        auto inWorkQueue = std::dynamic_pointer_cast<BMQueue<NodeInType>>(lastOutWorkQueue);
        if(!inWorkQueue) {
            BMLOG(FATAL, "input type of the added node is wrong: %s is needed, but got %s", lastTypeName.c_str(),typeid(NodeInType).name());
        }
        auto inResourceQueue = std::dynamic_pointer_cast<BMQueue<NodeInType>>(lastOutResourceQueue);

        lastTypeName = typeid(NodeOutType).name();
        lastOutWorkQueue = std::shared_ptr<BMQueue<NodeOutType>>(new BMQueue<NodeOutType>());
        if(!outResource.empty()){
            lastOutResourceQueue = std::shared_ptr<BMQueue<NodeOutType>>(new BMQueue<NodeOutType>());
        } else {
            lastOutResourceQueue =  std::shared_ptr<BMQueueVoid>();
        }
        auto outWorkQueue = std::dynamic_pointer_cast<BMQueue<NodeOutType>>(lastOutWorkQueue);
        auto outResourceQueue = std::dynamic_pointer_cast<BMQueue<NodeOutType>>(lastOutResourceQueue);
        for(auto& out: outResource){
            outResourceQueue->push(out);
        }
        pipelineNodes.emplace_back(
                    new BMPipelineNodeImp<NodeInType, NodeOutType, ContextType>(func,
                                                                                inResourceQueue, inWorkQueue,
                                                                                outResourceQueue, outWorkQueue,
                                                                                done, context)
                    );
    }

    void start() {
        // connect last node
        done = false;
        outQueue = std::dynamic_pointer_cast<BMQueue<OutType>>(lastOutWorkQueue);
        if(!outQueue) {
            BMLOG(FATAL, "output type of the last node is wrong: %s is needed, but got %s", outTypeName.c_str(),lastTypeName.c_str());
        }
        if(lastOutResourceQueue) {
            BMLOG(FATAL, "output of pipeline should not be resource limited!");
        }
        for(auto& node: pipelineNodes){
            node->start();
        }
    }

    std::shared_ptr<ContextType> getContext() {
        return context;
    }

    void push(InType value){
        inQueue->push(value);
    }

    bool pop(OutType& value){
        if(!outQueue){
            BMLOG(FATAL, "pipeline is not started!");
        }
        return outQueue->tryPop(value);
    }

    bool isStopped(){
        return done;
    }

    void stop(){
        done = true;
    }

    ~BMPipeline(){
        BMLOG(DEBUG, "PIPELINE destructed!");
        stop();
    }
};

template <typename InType, typename OutType, typename ContextType = BMPipelineEmptyContext>
class BMPipelinePool: public Uncopiable
{
private:
   std::vector<std::unique_ptr<BMPipeline<InType, OutType, ContextType>>> pipelines;
   std::shared_ptr<BMQueue<InType>> inQueue;
   std::shared_ptr<BMQueue<OutType>> outQueue;
   std::function<void(std::shared_ptr<ContextType>)> contextDeinitializer;

public:
   BMPipelinePool(size_t num_pipeline = 1,
                  std::function<std::shared_ptr<ContextType>(size_t)> contextInitializer = nullptr,
                  std::function<void(std::shared_ptr<ContextType>)> contextDeinitializer = nullptr
                  ) {
       inQueue = std::make_shared<BMQueue<InType>>();
       outQueue = std::make_shared<BMQueue<OutType>>();
       for(size_t i=0; i<num_pipeline; i++){
           std::shared_ptr<ContextType> context;
           if(contextInitializer){
               context = contextInitializer(i);
           }
           pipelines.emplace_back(new BMPipeline<InType, OutType, ContextType>(context));
       }
       for(auto& pipeline: pipelines){
           pipeline->setInputQueue(inQueue);
       }
       this->contextDeinitializer = contextDeinitializer;
   }

    std::shared_ptr<BMQueue<InType>> getInputQueue(){
        return inQueue;
    }

    template<typename NodeInType, typename NodeOutType, typename Container = std::vector<NodeOutType>>
    void addNode(std::function<NodeOutType(const NodeInType&)> func,
                 std::function<Container(std::shared_ptr<ContextType>)> outResourceInitializer = nullptr) {
        std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> inner_func = [func](
                const NodeInType& in, NodeOutType& out, std::shared_ptr<ContextType>){
            out =  std::move(func(in));
            return true;
        };
        addNode(inner_func, outResourceInitializer);
    }

    template<typename NodeInType, typename NodeOutType, typename Container = std::vector<NodeOutType>>
    void addNode(std::function<NodeOutType(const NodeInType&, std::shared_ptr<ContextType>)> func,
                 std::function<Container(std::shared_ptr<ContextType>)> outResourceInitializer = nullptr) {
        std::function<NodeOutType(const NodeInType&, std::shared_ptr<ContextType>)> inner_func = [func](
                const NodeInType& in, NodeOutType& out, std::shared_ptr<ContextType> ctx){ out =  std::move(func(in, ctx)); };
        addNode(inner_func, outResourceInitializer);
    }

    template<typename NodeInType, typename NodeOutType, typename Container=std::vector<NodeOutType>>
    void addNode(std::function<bool(const NodeInType&, NodeOutType&)> func,
                 std::function<Container(std::shared_ptr<ContextType>)> outResourceInitializer = nullptr) {
        std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> inner_func = [func](
                const NodeInType& in, NodeOutType& out, std::shared_ptr<ContextType>){ return func(in, out); };
        addNode(inner_func, outResourceInitializer);
    }

    template<typename NodeInType, typename NodeOutType, typename Container=std::vector<NodeOutType>>
    void addNode(std::function<bool(const NodeInType&, NodeOutType&, std::shared_ptr<ContextType>)> func,
                 std::function<Container(std::shared_ptr<ContextType>)> outResourceInitializer = nullptr
                 ) {
       for(size_t i=0; i<pipelines.size(); i++){
           auto& pipeline = pipelines[i];
           if(!pipeline) continue;
           Container outResources;
           try {
               if(outResourceInitializer){
                   outResources = std::move(outResourceInitializer(pipeline->getContext()));
               }
               pipeline->addNode(func, outResources);
           } catch (...) {
               BMLOG(WARNING, "pipeline #%d is not created!", i);
               contextDeinitializer(pipeline->getContext());
               pipeline.reset();
           }
       }
    }

    void start(){
       for(auto& pipeline: pipelines){
           if(pipeline) {
               pipeline->setOutputQueue(outQueue);
               pipeline->start();
           }
       }
    }
    bool allStopped(){
       for(auto& pipeline: pipelines){
           if(!pipeline->isStopped()){
               return false;
           }
       }
       return true;
    }

    void stop(int index = -1){
        if(index == -1){
            for(auto& pipeline: pipelines){
                if(pipeline) pipeline->stop();
            }
        } else if(index<pipelines.size()){
            pipelines[index]->stop();
        }
    }

    bool canPush(){
        return inQueue->canPush();
    }
    void push(InType in) {
        inQueue->push(in);
    }

    bool pop(OutType& out) {
        return outQueue->tryPop(out);
    }

    ~BMPipelinePool(){
        if(contextDeinitializer){
            for(auto& pipeline: pipelines){
                if(!pipeline) continue;
                contextDeinitializer(pipeline->getContext());
            }
        }

    }
};

}

#endif // BMPIPELINEPOOL_H
