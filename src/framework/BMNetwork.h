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
        m_handle(handle), m_name(name), m_raw_data(nullptr), m_raw_size(0), m_float_data(nullptr), m_scale(scale), m_tensor(tensor), need_release(relase) {
    }

    void set_device_mem(bm_device_mem_t *mem) { this->m_tensor->device_mem = *mem; }
    const bm_device_mem_t* get_device_mem() { return &this->m_tensor->device_mem; }

    bool fill_device_mem(const void* data, size_t len, size_t mem_offset = 0);

    // basic attribute of tensor
    bm_store_mode_t get_store_mode() const { return m_tensor->st_mode; };
    const bm_shape_t* get_shape() const { return &m_tensor->shape; }
    bm_data_type_t get_dtype() const { return m_tensor->dtype; }
    size_t get_dtype_len() const;
    void set_shape(const unsigned int* shape, size_t dims){
        m_tensor->shape.num_dims = dims;
        for(size_t i=0; i<dims; i++){
            m_tensor->shape.dims[i] = shape[i];
        }
    }
    size_t shape(int dim) const {
        while(dim<0) dim+=m_tensor->shape.num_dims;
        return dim<m_tensor->shape.num_dims? m_tensor->shape.dims[dim]:1; }
    ssize_t partial_shape_count(size_t begin, size_t end);
    size_t dims() const { return m_tensor->shape.num_dims; }
    float get_scale() const { return m_scale; }

    size_t get_mem_size() const;
    size_t get_elem_num() const;

    bm_tensor_t* raw_tensor() { return m_tensor; }
    unsigned char* get_raw_data();
    float *get_float_data();
    size_t fill_host_mem(void* ptr, size_t len);

    virtual ~BMTensor();
    void dumpData(const char* name);

private:
    bm_handle_t  m_handle;
    std::string m_name;
    unsigned char* m_raw_data;
    size_t m_raw_size;
    float* m_float_data;
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
    std::vector<std::string> m_network_names;

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
