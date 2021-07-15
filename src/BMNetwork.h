#ifndef BMNETWORK_H
#define BMNETWORK_H
#include <string>
#include <cassert>
#include <unordered_map>
#include <memory>
#include "BMLog.h"
#include "BMCommonUtils.h"
#include "bmruntime_interface.h"

namespace bm {

class BMTensor{
    /**
     *  members from bm_tensor {
     *  bm_data_type_t dtype;
        bm_shape_t shape;
        bm_device_mem_t device_mem;
        bm_store_mode_t st_mode;
        }
     */
    bm_handle_t  m_handle;

    std::string m_name;
    float* m_cpu_data;
    float m_scale;
    bm_tensor_t *m_tensor;

public:
    BMTensor(bm_handle_t handle, const char *name, float scale, bm_tensor_t* tensor):
        m_handle(handle), m_name(name), m_cpu_data(nullptr), m_scale(scale), m_tensor(tensor) {
    }

    virtual ~BMTensor() {
        if (m_cpu_data != NULL) {
            delete [] m_cpu_data;
            m_cpu_data = NULL;
        }
    }

    int set_device_mem(bm_device_mem_t *mem){
        this->m_tensor->device_mem = *mem;
        return 0;
    }

    const bm_device_mem_t* get_device_mem() {
        return &this->m_tensor->device_mem;
    }

    float *get_cpu_data() {
        if (m_cpu_data == NULL) {
            bm_status_t ret;
            float *pFP32 = nullptr;
            int count = bmrt_shape_count(&m_tensor->shape);
            if (m_tensor->dtype == BM_FLOAT32) {
                pFP32 = new float[count];
                assert(pFP32 != nullptr);
                ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
                assert(BM_SUCCESS ==ret);
            }else if (BM_INT8 == m_tensor->dtype) {
                int tensor_size = bmrt_tensor_bytesize(m_tensor);
                int8_t *pU8 = new int8_t[tensor_size];
                assert(pU8 != nullptr);
                pFP32 = new float[count];
                assert(pFP32 != nullptr);
                ret = bm_memcpy_d2s_partial(m_handle, pU8, m_tensor->device_mem, tensor_size);
                assert(BM_SUCCESS ==ret);
                for(int i = 0;i < count; ++ i) {
                    pFP32[i] = pU8[i] * m_scale;
                }
                delete [] pU8;
            }else{
                BMLOG(FATAL, "NOT support dtype=%d", m_tensor->dtype);
            }

            m_cpu_data = pFP32;
        }

        return m_cpu_data;
    }

    const bm_shape_t* get_shape() const {
        return &m_tensor->shape;
    }

    bm_data_type_t get_dtype() const {
        return m_tensor->dtype;
    }

    float get_scale() const {
        return m_scale;
    }

};

class BMNetwork : public Uncopiable {
    const bm_net_info_t *m_netinfo;
    bm_tensor_t* m_inputTensors;
    bm_tensor_t* m_outputTensors;
    bm_handle_t  m_handle;
    void *m_bmrt;

    std::unordered_map<std::string, bm_tensor_t*> m_mapInputs;
    std::unordered_map<std::string, bm_tensor_t*> m_mapOutputs;

public:
    BMNetwork(void *bmrt, const std::string& name):m_bmrt(bmrt) {
        m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
        m_netinfo = bmrt_get_network_info(bmrt, name.c_str());

        m_inputTensors = new bm_tensor_t[m_netinfo->input_num];
        m_outputTensors = new bm_tensor_t[m_netinfo->output_num];
        for(int i = 0; i < m_netinfo->input_num; ++i) {
            m_inputTensors[i].dtype = m_netinfo->input_dtypes[i];
            m_inputTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
            m_inputTensors[i].st_mode = BM_STORE_1N;
            m_inputTensors[i].device_mem = bm_mem_null();
        }

        for(int i = 0; i < m_netinfo->output_num; ++i) {
            m_outputTensors[i].dtype = m_netinfo->output_dtypes[i];
            m_outputTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
            m_outputTensors[i].st_mode = BM_STORE_1N;
            m_outputTensors[i].device_mem = bm_mem_null();
        }

        assert(m_netinfo->stage_num == 1);
    }

    ~BMNetwork() {
        //Free input tensors
        delete [] m_inputTensors;
        //Free output tensors
        for(int i = 0; i < m_netinfo->output_num; ++i) {
            if (m_outputTensors[i].device_mem.size != 0) {
                bm_free_device(m_handle, m_outputTensors[i].device_mem);
            }
        }
        delete []m_outputTensors;
    }

    int inputTensorNum() {
        return m_netinfo->input_num;
    }

    std::shared_ptr<BMTensor> inputTensor(int index){
        assert(index < m_netinfo->input_num);
        return std::make_shared<BMTensor>(m_handle, m_netinfo->input_names[index],
                m_netinfo->input_scales[index], &m_inputTensors[index]);
    }

    std::vector<std::shared_ptr<BMTensor>> inputTensors(){
        std::vector<std::shared_ptr<BMTensor>> inputs;
        for(int i=0; i<inputTensorNum(); i++){
            inputs.push_back(inputTensor(i));
        }
        return inputs;
    }

    std::vector<std::shared_ptr<BMTensor>> outputTensors(){
        std::vector<std::shared_ptr<BMTensor>> outputs;
        for(int i=0; i<m_netinfo->output_num; i++){
            outputs.push_back(outputTensor(i));
        }
        return outputs;
    }

    int outputTensorNum() {
        return m_netinfo->output_num;
    }

    std::shared_ptr<BMTensor> outputTensor(int index){
        assert(index < m_netinfo->output_num);
        return std::make_shared<BMTensor>(m_handle, m_netinfo->output_names[index],
                m_netinfo->output_scales[index], &m_outputTensors[index]);
    }

    int forward() {

        bool user_mem = false; // if false, bmrt will alloc mem every time.
        if (m_outputTensors->device_mem.size != 0) {
            // if true, bmrt don't alloc mem again.
            user_mem = true;
        }

        bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, m_inputTensors, m_netinfo->input_num,
                m_outputTensors, m_netinfo->output_num, user_mem, false);
        BM_ASSERT(ok, "bmrt_launch_tensor_ex failed");

#if 0
        for(int i = 0;i < m_netinfo->output_num; ++i) {
            auto tensor = m_outputTensors[i];
            BMLOG(INFO, "output_tensor [%d] size=%d", i, bmrt_tensor_device_size(&tensor));
        }
#endif

        return 0;
    }
};

}

#endif //BMNETWORK_H
