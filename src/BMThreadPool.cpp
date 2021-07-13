#include<algorithm>
#include "BMThreadPool.h"
#include "BMLog.h"

namespace bm {

bool BMThreadPool::popLocalWork(BMTask &task)
{
    return localQueue && localQueue->tryPop(task);
}

bool BMThreadPool::popGlobalWork(BMTask &task)
{
    return globalQueue.tryPop(task);
}

bool BMThreadPool::stealOtherWork(BMTask &task)
{
    size_t numThread = allLocalQueues.size();
    for(size_t i=0; i<numThread; i++){
        size_t currentIndex = (threadIndex+i+1)%numThread;
        if(allLocalQueues[currentIndex]->trySteal(task)){
            return true;
        }
    }
    return false;
}

void BMThreadPool::workThread(size_t index) {
    threadIndex = index;
    localQueue = allLocalQueues[threadIndex].get();
    BMLOG(DEBUG, "begin thread id=%d, index=%d", std::this_thread::get_id(), threadIndex);
    while (!done) {
        runPendingTask();
    }
    BMLOG(DEBUG, "end thread id=%d", std::this_thread::get_id());
}

void BMThreadPool::runPendingTask()
{
        BMTask task;
        if(globalQueue.tryPop(task)) {
            BMLOG(DEBUG, "[%d] get a task", std::this_thread::get_id());
            task();
        } else {
            std::this_thread::yield();
        }
}

BMThreadPool::BMThreadPool(size_t num_thread): done(false), joiner(threads) {
    localQueue = nullptr;
    try {
        for(size_t i = 0; i<num_thread; i++){
            allLocalQueues.emplace_back(new BMWorkStealingQueue<BMTask>);
            threads.emplace_back(&BMThreadPool::workThread, this, i);
            BMLOG(DEBUG, "create thread id=%d", threads.back().get_id());
        }
    } catch(...){
        done = true;
        throw;
    }
}

BMThreadPool::~BMThreadPool(){
    done = true;
}

BMThreadPool::__ThreadsJoiner::__ThreadsJoiner(BMThreadPool::Threads &ts): threads(ts) {}

void BMThreadPool::__ThreadsJoiner::join(){
    std::for_each(threads.begin(), threads.end(), [](std::thread& t){
        if(t.joinable()){
            BMLOG(DEBUG, "join thread id=%d", t.get_id());
            t.join();
        }
    });
}

BMThreadPool::__ThreadsJoiner::~__ThreadsJoiner(){
    join();
}
thread_local BMWorkStealingQueue<BMTask>* BMThreadPool::localQueue = nullptr;
thread_local size_t BMThreadPool::threadIndex = 0;

}
