#ifndef INCEPTION_V3_H
#define INCEPTION_V3_H
#include <vector>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include "BMDevicePool.h"

namespace bm
{
/*

class Inception
{
public:
    std::thread dataThread;

    using InType = std::string;

    struct PreprocessOutType {
        std::vector<std::string> imageNames;
    };

    struct ForwardOutType {
        std::vector<std::string> imageNames;
    };

    struct PostprocessOutType {
        std::vector<std::string> imageNames;
        std::vector<int> classIndice;
    };

    using RunnerType = BMDevicePool<InType, PostprocessOutType>;
    using ContextPtr = typename std::shared_ptr<BMDeviceContext>;


public:
    Inception(const std::string& bmodelPath, std::vector<DeviceId> ids= {0}): done(false),runner(bmodelPath, ids) {

        std::function<bool(const InType&, PreprocessOutType&, ContextPtr ctx)> preFunc = preProcess;
        std::function<std::vector<PreprocessOutType>(ContextPtr)> preCreateFunc = createPreProcessOutput;
        runner.addProcess(preFunc, preCreateFunc);

        std::function<bool(const PreprocessOutType&, ForwardOutType&, ContextPtr)> forwardFunc = forward;
        std::function<std::vector<ForwardOutType>(ContextPtr)> createForwardFunc = createForwardOutput;
        runner.addProcess(forwardFunc, createForwardFunc);

        std::function<bool(const ForwardOutType&, PostprocessOutType&, ContextPtr)> postFunc = postProcess;
        runner.addProcess(postFunc);
    };

    static bool preProcess(const InType& in, PreprocessOutType& out, ContextPtr ctx);
    static std::vector<PreprocessOutType> createPreProcessOutput(ContextPtr ctx);

    static bool forward(const PreprocessOutType& in, ForwardOutType& out, ContextPtr ctx);
    static std::vector<ForwardOutType> createForwardOutput(ContextPtr ctx);

    static bool postProcess(const ForwardOutType& in, PostprocessOutType& out, ContextPtr ctx);

    void generateData(const std::string &dataset) {
        forEachFile(dataset, [this] (const std::string& imagePath){
            if(!this->isDone()) return false;
            this->runner.push(imagePath);
            return true;
        });
        this->runner.push(std::string(""));
    }

    void start(const std::string &dataset) {
        done = false;
        runner.start();
        dataThread = std::move(std::thread(&Inception::generateData, this, dataset));
    }

    void stop(){
        done = true;
        dataThread.join();
        runner.stop();
    }

    ~Inception() {
        stop();
    }
    bool isDone() { return done; }

private:
    std::atomic_bool done;
    RunnerType runner;
};
*/

}

#endif
