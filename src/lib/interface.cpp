#include <map>
#include <memory>
#include <string.h>
#include "bmruntime_interface.h"
#include "BMDevicePool.h"
#include "BMLog.h"
#include "interface.h"

using namespace bm;

size_t dtype_len(unsigned int t) {
    if(t == BM_FLOAT32 || t == BM_INT32 || t== BM_UINT32){
        return 4;
    } else if(t == BM_UINT16 || t==BM_INT16 || t==BM_FLOAT16){
        return 2;
    } else if(t == BM_UINT8 || t == BM_UINT8){
        return 1;
    } else {
        BMLOG(FATAL, "Not support dtype=%d", t);
    }
}

static size_t elem_num(const unsigned int* shape, unsigned int dims){
    size_t elem = 1;
    for(size_t i=0; i<dims; i++){
        elem *= shape[i];
    }
    return elem;
}

struct InputType {
    bool release_inside = false;
    unsigned int id = 0;
    unsigned num = 0;
    tensor_data_t* tensors = nullptr;
};

struct OutputType {
    unsigned int id = 0;
    unsigned num = 0;
    tensor_data_t* tensors = nullptr;
};

bool preProcess(const InputType& input, const TensorVec& inTensors, ContextPtr ctx);
bool postProcess(const InputType& input, const TensorVec& outTensors, OutputType& postOut, ContextPtr ctx);

std::vector<DeviceId> globalDevices;
using GeneralRunner = BMDevicePool<InputType, OutputType>;
struct RunnerInfo {
    RunnerInfo(const char* bmodel):
        task_id(INVALID_TASK_ID), runner(bmodel, preProcess, postProcess, globalDevices), status(bmodel) {
        runner.start();
        status.start();
    }
    unsigned int nextId() {
        task_id++;
        if(task_id == INVALID_TASK_ID) task_id++;
        return task_id;
    }

    unsigned int task_id;

    GeneralRunner runner;
    ProcessStatInfo status;
};

std::map<unsigned int, std::shared_ptr<RunnerInfo>> globalRunnerInfos;

bool preProcess(const InputType& input, const TensorVec& inTensors, ContextPtr ctx){
    if(input.num == 0){
        return false;
    }
    BM_ASSERT_EQ(input.num, inTensors.size());
    for(size_t i=0; i<input.num; i++){
        size_t in_mem_size = elem_num(input.tensors[i].shape, input.tensors[i].dims) * dtype_len(input.tensors[i].dtype);
        BM_ASSERT_EQ(inTensors[i]->get_dtype(), input.tensors[i].dtype);
        BM_ASSERT_LE(in_mem_size, inTensors[i]->get_mem_size());
        inTensors[i]->fill_device_mem(input.tensors[i].data, in_mem_size);
    }
    return true;
}

bool postProcess(const InputType& input, const TensorVec& outTensors, OutputType& postOut, ContextPtr ctx){
    if(input.num == 0) {
        postOut.num = 0;
        throw "finished"; // to stop current pipeline
    }
    postOut.id = input.id;
    if(input.release_inside){
        for(size_t i=0; i<input.num; i++){
            delete [] input.tensors[i].data;
        }
        delete []input.tensors;
    }

    size_t outNum = outTensors.size();
    postOut.num = outNum;
    postOut.tensors = new tensor_data_t[outNum];
    for(size_t i=0; i<outNum; i++){
        postOut.tensors[i].dims = outTensors[i]->dims();
        for(size_t d=0; d<postOut.tensors[i].dims; d++){
            postOut.tensors[i].shape[d] = outTensors[i]->shape(d);
        }
        postOut.tensors[i].dtype = outTensors[i]->get_dtype();
        auto mem_size = outTensors[i]->get_mem_size();
        postOut.tensors[i].data = new unsigned char[mem_size];
        auto fill_size = outTensors[i]->fill_host_mem(postOut.tensors[i].data, mem_size);
        BM_ASSERT_EQ(fill_size, mem_size);
    }
    return true;
}

unsigned int runner_start(const char *bmodel) {
    set_env_log_level();
    unsigned int runner_id = 0;
    while(globalRunnerInfos.count(runner_id)) runner_id++;
    globalRunnerInfos[runner_id] = std::make_shared<RunnerInfo>(bmodel);
    return runner_id;
}

void runner_stop(unsigned int runner_id) {
    if(!globalRunnerInfos.count(runner_id)) return;
    globalRunnerInfos[runner_id]->runner.stop();
    globalRunnerInfos.erase(runner_id);
}

void runner_show_status(unsigned int runner_id)
{
    if(!globalRunnerInfos.count(runner_id)) return;
    globalRunnerInfos[runner_id]->status.show();
}

unsigned int runner_put_input(unsigned runner_id, unsigned int input_num, const tensor_data_t *input_tensors, int need_copy)
{
    if(!globalRunnerInfos.count(runner_id)) return -1;
    InputType input;
    input.id = globalRunnerInfos[runner_id]->nextId();
    input.release_inside = need_copy;
    input.num = input_num;
    if(input_num != 0){
        if(need_copy){
            input.tensors = new tensor_data_t[input_num];
            memcpy(input.tensors, input_tensors, sizeof(tensor_data_t)*input_num);
            for(size_t i = 0; i<input.num; i++){
                auto mem_size = dtype_len(input_tensors[i].dtype) * elem_num(input_tensors[i].shape, input_tensors[i].dims);
                input.tensors[i].data = new unsigned char[mem_size];
                memcpy(input.tensors[i].data, input_tensors[i].data, mem_size);
            }
        } else {
            input.tensors = (tensor_data_t*)input_tensors;
        }
    } else {
        input.tensors = nullptr;
    }
    globalRunnerInfos[runner_id]->runner.push(input);
    return input.id;
}


int runner_all_stopped(size_t runner_id){
    if(!globalRunnerInfos.count(runner_id)) return true;
    return globalRunnerInfos[runner_id]->runner.allStopped();
}

static tensor_data_t *__runner_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid, bool is_async){
    if(!globalRunnerInfos.count(runner_id)) return nullptr;
    auto& info = globalRunnerInfos[runner_id];
    OutputType output;
    std::shared_ptr<ProcessStatus> status;
    bool ok = false;
    do{
       ok = info->runner.pop(output, status);
       if(ok || is_async){
           break;
       } else {
           std::this_thread::yield();
       }
    } while(1);
    if(!ok) return nullptr;

    *task_id = output.id;
    *output_num = output.num;
    *is_valid = status->valid;
    info->status.update(status);
    return output.tensors;
}

tensor_data_t *runner_try_to_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid)
{
    return __runner_get_output(runner_id, task_id, output_num, is_valid, true);
}


tensor_data_t *runner_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid)
{
    return __runner_get_output(runner_id, task_id, output_num, is_valid, false);
}

unsigned int runner_release_output(unsigned int output_num, const tensor_data_t *output_data){
    for(size_t i=0; i<output_num; i++){
        delete [] output_data[i].data;
    }
    delete []output_data;
}

int runner_empty(unsigned int runner_id)
{
    if(!globalRunnerInfos.count(runner_id)) return true;
    return globalRunnerInfos[runner_id]->runner.empty();
}

void runner_use_devices(const unsigned *device_ids, unsigned num)
{
    globalDevices.assign(device_ids, device_ids+num);
}

unsigned int available_devices(unsigned int *devices, unsigned int maxNum)
{
    auto deviceIds = getAvailableDevices();
    unsigned int realNum = maxNum>deviceIds.size()?deviceIds.size(): maxNum;
    for(size_t i =0; i<realNum; i++){
        devices[i] = deviceIds[i];
    }
    return realNum;
}
