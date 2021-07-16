#ifndef BMNETWORK_H
#define BMNETWORK_H
#include <string>
#include <vector>
#include <cassert>
#include <memory>
#include "BMLog.h"
#include "BMCommonUtils.h"
#include "bmruntime_interface.h"

namespace bm {

class BMTensor{
public:
    BMTensor(bm_handle_t handle, const char *name, float scale, bm_tensor_t* tensor, bool relase):
        m_handle(handle), m_name(name), m_cpu_data(nullptr), m_scale(scale), m_tensor(tensor), need_release(relase) {
    }

    void set_device_mem(bm_device_mem_t *mem) { this->m_tensor->device_mem = *mem; }
    const bm_device_mem_t* get_device_mem() { return &this->m_tensor->device_mem; }

    // basic attribute of tensor
    bm_store_mode_t get_store_mode() const { return m_tensor->st_mode; };
    const bm_shape_t* get_shape() const { return &m_tensor->shape; }
    bm_data_type_t get_dtype() const { return m_tensor->dtype; }
    float get_scale() const { return m_scale; }

    size_t get_mem_size() const;

    bm_tensor_t* raw_tensor() { return m_tensor; }
    float *get_cpu_data();

    virtual ~BMTensor();

private:
    bm_handle_t  m_handle;
    std::string m_name;
    float* m_cpu_data;
    float m_scale;
    bool need_release;
    bm_tensor_t *m_tensor;
};

using TensorPtr = std::shared_ptr<BMTensor>;
using TensorVec = std::vector<TensorPtr>;

class BMNetwork : public Uncopiable {
    const bm_net_info_t *m_netinfo;
    bm_handle_t  m_handle;
    std::string bmodelPath;
    void *m_bmrt;
    size_t batchSize;

public:
    BMNetwork(void *bmrt, const std::string& name);
    void showInfo();

    ~BMNetwork() { }
    size_t getBatchSize(){ return batchSize; }

    TensorVec createOutputTensors();
    TensorVec createInputTensors();
    int forward(TensorVec inTensors, TensorVec outTensors);
};

}

#endif //BMNETWORK_H
