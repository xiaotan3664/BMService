#include<iostream>
#include <vector>
#include <future>
#include <functional>
#include <unistd.h>
#include "BMLog.h"
#include "BMThreadPool.h"
#include "BMPipelinePool.h"

using namespace std;
using namespace bm;


int main(){
    set_log_level(DEBUG);
    BMPipelinePool<int, int> pool(2);
    std::function<float(const int&)> func = [](int a){ return a+1.0; };
    std::function<void(const float& a, int& b)> func1 = [](const float& a, int& b){
        sleep(1);
//        std::this_thread::sleep_for(chrono::seconds(1));
        b = (int)a*1.8;
    };

    std::function<void(const int& a, int& b)> func2 = [](const int& a, int& b){
        sleep(1);
        b = a*2;
    };

//    pool.addNode([](const int& a) -> float{ return a+0.1;}, resource1);
    pool.addNode(func);
    pool.addNode(func1);
    pool.addNode(func2);
//    pool.addNode([](Context&, const float& a) { return a*2.5;});
//    pool.addNode([](Context&, const float& a) { return (int)a;});
    pool.start();
    vector<int> inputs{1,2,3,4,5,6};
    for(auto& i: inputs){
        pool.push(i);
    }
    this_thread::sleep_for(chrono::seconds(1));
    for(size_t i=0; i<inputs.size(); i++){
        int value;
        while(!pool.pop(value)) {
            this_thread::yield();
        }
        BMLOG(INFO, "o=%d", value);
    }
    return 0;
}

int main_pipe(){
    set_log_level(DEBUG);
    BMPipeline<int, int> pipeline{};
    std::function<float(const int&)> func = [](int a){  return a+1.0; };
    std::function<void(const float& a, int& b)> func1 = [](const float& a, int& b){
        sleep(1);
//        std::this_thread::sleep_for(chrono::seconds(1));
        b = (int)a*1.8;
    };

    std::function<void(const int& a, int& b)> func2 = [](const int& a, int& b){
        sleep(1);
        b = a*2;
    };

    vector<float> resource1{1,2,3};
    vector<int> resource2{1,2,3};
//    pipeline.addNode([](const int& a) -> float{ return a+0.1;}, resource1);
    pipeline.addNode(func, resource1);
    pipeline.addNode(func1, resource2);
    pipeline.addNode(func2);
//    pipeline.addNode([](Context&, const float& a) { return a*2.5;});
//    pipeline.addNode([](Context&, const float& a) { return (int)a;});
    pipeline.start();
    vector<int> inputs{1,2,3,4,5,6};
    for(auto& i: inputs){
        pipeline.push(i);
    }
    this_thread::sleep_for(chrono::seconds(1));
    for(size_t i=0; i<inputs.size(); i++){
        int value;
        while(!pipeline.pop(value)) {
            this_thread::yield();
        }
        BMLOG(INFO, "o=%d", value);
    }
    return 0;
}


int increase(int i){
    return i+10;
}

int main_1()
{
    set_log_level(DEBUG);
    BMLOG(INFO, "Hello");
    int num_thread = 3;
    BMThreadPool pool(num_thread);
    vector<int> inputs{1,2,3,4,5,6};
    vector<future<int>> output_futures;
    for(auto& i: inputs){
        output_futures.emplace_back(pool.submit(increase, i));
    }

    for(auto& of: output_futures){
        BMLOG(INFO, "o=%d", of.get());
    }

    return 0;
}
